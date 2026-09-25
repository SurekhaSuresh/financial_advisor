"""Deterministic integration tests for the complete evidence-backed workflow.

These tests intentionally use the production WorkflowEngine, SQLite repository,
LanceDB retrievers, evidence-selection pipeline, web page processing, and
ScenarioRunner. Only model decisions and external HTTP provider responses are
scripted, making the tests repeatable without API keys or network access.
"""

import asyncio
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from uuid import UUID

from financial_advisor.agents.advisor.contracts import (
    AdvisorRecommendationDraft,
    AdvisorResearchPlanDraft,
)
from financial_advisor.agents.advisor.workflow import AdvisorWorkflowCoordinator
from financial_advisor.agents.analyst.contracts import AnalystResearchDraft, create_research_brief
from financial_advisor.agents.analyst.execution import AnalystResearchExecutor
from financial_advisor.agents.analyst.workflow import AnalystWorkflowCoordinator
from financial_advisor.agents.client.contracts import (
    ClientReviewDraft,
    ClientReviewRequest,
    create_client_review,
)
from financial_advisor.domain import (
    AdvisorDecisionType,
    CitedText,
    ClientMessage,
    ClientProfile,
    ClientReview,
    EventType,
    Finding,
    RecommendationOption,
    ResearchRefinement,
    RiskTolerance,
    SessionState,
    WebResearchMode,
    WebResearchScope,
)
from financial_advisor.persistence import SessionRepository
from financial_advisor.retrieval.knowledge_base.ingestion import (
    KnowledgeSource,
    SourceSection,
    chunk_sections,
    write_lancedb,
)
from financial_advisor.retrieval.knowledge_base.retrievers import (
    LanceKeywordRetriever,
    LanceVectorRetriever,
)
from financial_advisor.retrieval.pipeline import EvidenceRetrievalPipeline
from financial_advisor.retrieval.web.discovery import (
    CircuitBreaker,
    SearchProviderName,
    WebResearchError,
    WebResearchService,
    WebSearchRequest,
    WebSearchResponse,
    WebSearchResult,
)
from financial_advisor.retrieval.web.fetch import RawDocumentResponse, WebPageFetcher
from financial_advisor.retrieval.web.passage_ranker import InMemoryWebVectorRetriever
from financial_advisor.retrieval.web.pipeline import LiveWebResearchPipeline
from financial_advisor.scenario import ScenarioRunner
from financial_advisor.workflow import WorkflowEngine


class StableWhitespaceCodec:
    """Small deterministic codec that preserves words for parser/chunker integration tests."""

    def __init__(self) -> None:
        self._token_ids_by_word: dict[str, int] = {}
        self._words_by_token_id: dict[int, str] = {}

    def encode(self, text: str) -> list[int]:
        token_ids: list[int] = []
        for word in text.split():
            if word not in self._token_ids_by_word:
                token_id = len(self._token_ids_by_word)
                self._token_ids_by_word[word] = token_id
                self._words_by_token_id[token_id] = word
            token_ids.append(self._token_ids_by_word[word])
        return token_ids

    def decode(self, token_ids: Sequence[int]) -> str:
        return " ".join(self._words_by_token_id[token_id] for token_id in token_ids)


class KeywordEmbedder:
    """Offline deterministic vectors used only by this integration fixture corpus."""

    _TERM_GROUPS = (
        ("home", "down", "payment", "liquidity", "cash"),
        ("student", "loan", "debt", "repayment"),
        ("risk", "diversification", "invest", "portfolio"),
        ("emergency", "reserve", "budget", "saving"),
    )

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            words = set(text.casefold().replace("-", " ").split())
            vectors.append(
                [float(sum(word in words for word in group)) + 0.01 for group in self._TERM_GROUPS]
            )
        return vectors


class FixtureCrossEncoder:
    """Stable pair scorer while the production reranking implementation remains exercised."""

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        query_words = set(query.casefold().split())
        return [
            float(len(query_words & set(document.casefold().split()))) + (index / 1_000)
            for index, document in enumerate(documents, start=1)
        ]


class FixtureProvider:
    """One scripted Exa or Brave discovery boundary with no HTTP dependency."""

    def __init__(
        self,
        name: SearchProviderName,
        outcome: WebSearchResponse | Exception,
    ) -> None:
        self._name = name
        self._outcome = outcome
        self.requests: list[WebSearchRequest] = []

    @property
    def name(self) -> SearchProviderName:
        return self._name

    def search(self, request: WebSearchRequest) -> WebSearchResponse:
        self.requests.append(request)
        if isinstance(self._outcome, Exception):
            raise self._outcome
        return self._outcome


class FixtureDocumentTransport:
    """Return original-page fixture bytes after real discovery and fetch policy logic."""

    def __init__(self, documents: Mapping[str, RawDocumentResponse]) -> None:
        self._documents = dict(documents)
        self.calls: list[str] = []

    def fetch(
        self,
        url: str,
        headers: Mapping[str, str],
        *,
        timeout_seconds: float,
        max_bytes: int,
    ) -> RawDocumentResponse:
        del headers, timeout_seconds, max_bytes
        self.calls.append(url)
        return self._documents[url]


class ScriptedClient:
    """Provide fixed Client decisions while real bounded review handling executes."""

    def __init__(self, reviews: list[ClientReviewDraft] | None = None) -> None:
        self._reviews = iter(reviews or [ClientReviewDraft(accepted=True)])

    async def open_conversation(self, profile: ClientProfile, session_id: UUID) -> ClientMessage:
        return ClientMessage(
            session_id=session_id,
            text=(
                f"How should I balance my {profile.primary_goal} goal, student loan, "
                "and investment risk?"
            ),
            follow_up_number=0,
        )

    async def review_recommendation(self, request: ClientReviewRequest) -> ClientReview:
        return create_client_review(request, draft=next(self._reviews))


class ScriptedAdvisor:
    """Return valid structured planning and response drafts without a live Gemini call."""

    def __init__(self, plans: list[AdvisorResearchPlanDraft]) -> None:
        self._plans = iter(plans)

    async def plan(self, request: object) -> AdvisorResearchPlanDraft:
        del request
        return next(self._plans)

    async def recommend(
        self, request: object, *, repair: bool = False
    ) -> AdvisorRecommendationDraft:
        del repair
        evidence_id = request.research_brief.evidence[0].evidence_id  # type: ignore[attr-defined]
        return AdvisorRecommendationDraft(
            summary=CitedText(
                text="Use a goal-based approach that weighs liquidity, debt, and investment risk.",
                evidence_ids=[evidence_id],
            ),
            options=[
                RecommendationOption(
                    title="Protect the near-term goal first",
                    description=CitedText(
                        text=(
                            "Keep the amount needed for a five-year goal aligned with its timeline."
                        ),
                        evidence_ids=[evidence_id],
                    ),
                    suitability="Useful for a client balancing a shorter goal and moderate risk.",
                )
            ],
            rationale=[
                CitedText(
                    text="Liquidity, debt costs, and market risk are separate planning tradeoffs.",
                    evidence_ids=[evidence_id],
                )
            ],
            assumptions=["The stated five-year home-purchase goal remains unchanged."],
            risks=[
                CitedText(
                    text="Market losses can affect money invested for a shorter-term goal.",
                    evidence_ids=[evidence_id],
                )
            ],
            next_steps=["Confirm the home-goal amount before making an allocation decision."],
        )


class ScriptedAnalystSynthesis:
    """Build a valid brief from the exact server-selected evidence it receives."""

    def __init__(self, *, should_fail: bool = False) -> None:
        self._should_fail = should_fail

    async def research(self, request: object):  # type: ignore[no-untyped-def]
        if self._should_fail:
            raise RuntimeError("Fixture Analyst synthesis outage.")
        evidence = request.evidence  # type: ignore[attr-defined]
        return create_research_brief(
            request,  # type: ignore[arg-type]
            AnalystResearchDraft(
                findings=[
                    Finding(
                        statement=(
                            "The selected evidence supports comparing liquidity, debt, and risk."
                        ),
                        evidence_ids=[item.evidence_id for item in evidence[:2]],
                    )
                ],
                caveats=["Educational information only; not individualized financial advice."],
            ),
        )


def _maya_profile() -> ClientProfile:
    return ClientProfile(
        client_id="maya-chen-integration",
        name="Maya Chen",
        age=38,
        risk_tolerance=RiskTolerance.MODERATE,
        emergency_fund_months=6,
        retirement_savings=Decimal("120000.00"),
        brokerage_savings=Decimal("35000.00"),
        student_loan_balance=Decimal("18000.00"),
        student_loan_rate_percent=Decimal("5.8"),
        primary_goal="five-year home purchase",
        goal_time_horizon_years=5,
    )


def _delegate_plan(
    *,
    refinement: ResearchRefinement | None = None,
    research_question: str | None = None,
) -> AdvisorResearchPlanDraft:
    return AdvisorResearchPlanDraft(
        decision_type=AdvisorDecisionType.DELEGATE_RESEARCH,
        summary=("Retrieve stable guidance and current source material for the stated tradeoffs."),
        progress_summary=(
            "I’m gathering relevant guidance before preparing an educational response."
        ),
        research_question=research_question
        or (
            "How should a moderate-risk investor balance a five-year home goal, student loan, "
            "liquidity, and investment risk?"
        ),
        include_local_hybrid=True,
        web_scopes=(WebResearchScope(mode=WebResearchMode.BROAD_WEB),),
        refinement=refinement,
    )


def _real_evidence_pipeline(
    tmp_path: Path,
) -> tuple[
    EvidenceRetrievalPipeline,
    FixtureProvider,
    FixtureProvider,
    FixtureDocumentTransport,
]:
    """Build production retrieval components over temporary, fully local fixture data."""

    codec = StableWhitespaceCodec()
    embedder = KeywordEmbedder()
    source = KnowledgeSource(
        id="integration_guidance",
        publisher="Investor education fixture",
        title="Planning a short-term financial goal",
        url="https://www.investor.gov/integration-guidance",
        content_type="html",
        topics=["home_goal", "risk", "debt"],
    )
    chunks = chunk_sections(
        source,
        [
            SourceSection(
                "Time horizon and liquidity",
                (
                    "A five year home down payment goal needs liquidity and lower volatility "
                    "than a long term portfolio."
                ),
                1,
            ),
            SourceSection(
                "Debt and investment tradeoffs",
                (
                    "Student loan repayment reduces debt while diversified investment portfolios "
                    "can fluctuate with market risk."
                ),
                2,
            ),
        ],
        codec,
        max_tokens=120,
        min_tokens=0,
        overlap_tokens=10,
        source_content_hash="integration-curated-document-v1",
        retrieved_at=datetime(2026, 9, 20, tzinfo=UTC),
    )
    database_path = tmp_path / "lancedb"
    write_lancedb(database_path, chunks, embedder.embed([chunk.embedding_text for chunk in chunks]))

    result = WebSearchResult(
        title="Current public guidance fixture",
        url="https://example.org/current-guidance",
        snippet="Fixture discovery result only.",
    )
    exa = FixtureProvider(
        SearchProviderName.EXA,
        WebResearchError("Fixture Exa outage.", retryable=False),
    )
    brave = FixtureProvider(
        SearchProviderName.BRAVE,
        WebSearchResponse(provider=SearchProviderName.BRAVE, results=[result]),
    )
    discovery = WebResearchService(
        [exa, brave],
        {
            SearchProviderName.EXA: CircuitBreaker(),
            SearchProviderName.BRAVE: CircuitBreaker(),
        },
        max_attempts=1,
        sleep=lambda _: None,
    )
    transport = FixtureDocumentTransport(
        {
            str(result.url): RawDocumentResponse(
                final_url=str(result.url),
                content_type="text/html",
                body=(
                    b"<html><body><h1>Current planning context</h1>"
                    b"<p>Cash reserves protect a near-term home purchase goal from market "
                    b"volatility. "
                    b"Debt repayment and investment risk should be considered together.</p>"
                    b"</body></html>"
                ),
            )
        }
    )
    live_web = LiveWebResearchPipeline(
        discovery,
        WebPageFetcher(transport=transport, max_attempts=1),
        codec,
        InMemoryWebVectorRetriever(embedder),
        usable_page_target=1,
    )
    return (
        EvidenceRetrievalPipeline(
            LanceVectorRetriever(database_path, embedder),
            LanceKeywordRetriever(database_path),
            live_web,
            FixtureCrossEncoder(),
            embedder,
        ),
        exa,
        brave,
        transport,
    )


def _runner(
    tmp_path: Path,
    *,
    plans: list[AdvisorResearchPlanDraft],
    client_reviews: list[ClientReviewDraft] | None = None,
    analyst_synthesis_should_fail: bool = False,
) -> tuple[
    ScenarioRunner,
    SessionRepository,
    FixtureProvider,
    FixtureProvider,
    FixtureDocumentTransport,
]:
    repository = SessionRepository(tmp_path / "financial_advisor.db")
    workflow = WorkflowEngine(repository)
    pipeline, exa, brave, transport = _real_evidence_pipeline(tmp_path)
    analyst = AnalystWorkflowCoordinator(
        workflow,
        AnalystResearchExecutor(
            pipeline,
            ScriptedAnalystSynthesis(should_fail=analyst_synthesis_should_fail),
        ),
    )
    return (
        ScenarioRunner(
            workflow,
            ScriptedClient(client_reviews),
            AdvisorWorkflowCoordinator(workflow, ScriptedAdvisor(plans)),
            analyst,
        ),
        repository,
        exa,
        brave,
        transport,
    )


def _assert_persisted_replay(repository: SessionRepository, completed) -> None:  # type: ignore[no-untyped-def]
    """Compare persisted replay data without retaining transient fetched-page bytes."""

    restored = repository.load(completed.session_id)
    assert restored is not None
    assert restored.session_id == completed.session_id
    assert restored.trace_id == completed.trace_id
    assert restored.client_profile == completed.client_profile
    assert restored.state == completed.state
    assert restored.follow_up_count == completed.follow_up_count
    assert restored.events == completed.events
    assert restored.messages == completed.messages
    assert restored.research_plans == completed.research_plans
    assert restored.analyst_tasks == completed.analyst_tasks
    assert restored.research_briefs == completed.research_briefs
    assert restored.recommendations == completed.recommendations
    assert restored.client_reviews == completed.client_reviews
    assert [item.task_id for item in restored.retrieval_traces] == [
        item.task_id for item in completed.retrieval_traces
    ]
    assert [item.retrieval.model_dump(mode="json") for item in restored.retrieval_traces] == [
        item.retrieval.model_dump(mode="json") for item in completed.retrieval_traces
    ]


def test_integration_real_retrieval_pipeline_persists_three_agent_resolution(
    tmp_path: Path,
) -> None:
    """Exercise real LanceDB, RRF/reranking/MMR, web processing, SQLite, and the workflow."""

    runner, repository, exa, brave, transport = _runner(
        tmp_path,
        plans=[
            _delegate_plan(),
            AdvisorResearchPlanDraft(
                decision_type=AdvisorDecisionType.ANSWER_FROM_STATE,
                summary="The validated brief now answers the Client question.",
            ),
        ],
    )

    completed = asyncio.run(runner.run(_maya_profile()))
    retrieval = completed.retrieval_traces[0].retrieval

    assert completed.state is SessionState.RESOLVED
    _assert_persisted_replay(repository, completed)
    assert retrieval.local_vector is not None and retrieval.local_vector.candidates
    assert retrieval.local_bm25 is not None and retrieval.local_bm25.candidates
    assert retrieval.selection is not None
    assert retrieval.selection.fusion.input_candidate_count >= 3
    assert retrieval.selection.reranking.candidates
    assert retrieval.selection.diversity_selection.selected_candidates
    assert retrieval.web_researches[0].trace.content_processing[0].chunk_count > 0
    # The real service falls back to fixture Brave after fixture Exa fails.
    assert exa.requests and brave.requests
    assert transport.calls == ["https://example.org/current-guidance"]
    assert completed.recommendations[0].evidence_ids


def test_integration_real_retrieval_supports_one_refinement_then_stops_at_plan_budget(
    tmp_path: Path,
) -> None:
    """A second precise plan is persisted; no third plan can be requested for the same question."""

    runner, repository, _, _, _ = _runner(
        tmp_path,
        plans=[
            _delegate_plan(),
            _delegate_plan(
                research_question=(
                    "How should a moderate-risk investor compare the 5.25 percent student-loan "
                    "paydown tradeoff against a five-year home down-payment savings goal?"
                ),
                refinement=ResearchRefinement(
                    material_gap="Compare the debt tradeoff more explicitly.",
                    why_current_evidence_cannot_answer_gap="The first brief was broad.",
                    authorized_retrieval_changes=(
                        "Use a more precise question over the same sources."
                    ),
                )
            ),
        ],
    )

    completed = asyncio.run(runner.run(_maya_profile()))

    assert completed.state is SessionState.RESOLVED
    assert len(completed.research_plans) == 2
    assert len(completed.analyst_tasks) == 2
    assert len(completed.retrieval_traces) == 2
    assert len(completed.research_briefs) == 2
    _assert_persisted_replay(repository, completed)
    assert completed.research_plans[-1].refinement is not None


def test_integration_real_retrieval_persists_analyst_failure_and_escalation(
    tmp_path: Path,
) -> None:
    """Evidence retrieval completes before a scripted synthesis outage safely escalates."""

    runner, repository, _, _, _ = _runner(
        tmp_path,
        plans=[_delegate_plan()],
        analyst_synthesis_should_fail=True,
    )

    completed = asyncio.run(runner.run(_maya_profile()))

    assert completed.state is SessionState.ESCALATED
    assert len(completed.retrieval_traces) == 1
    assert completed.research_briefs == []
    assert completed.recommendations == []
    assert EventType.TOOL_FAILED in [event.event_type for event in completed.events]
    assert EventType.SESSION_ESCALATED in [event.event_type for event in completed.events]
    _assert_persisted_replay(repository, completed)


def test_integration_real_retrieval_closes_after_the_follow_up_budget(
    tmp_path: Path,
) -> None:
    """Two scripted follow-ups use validated state; no third model review is called."""

    runner, repository, _, _, _ = _runner(
        tmp_path,
        plans=[
            _delegate_plan(),
            AdvisorResearchPlanDraft(
                decision_type=AdvisorDecisionType.ANSWER_FROM_STATE,
                summary="The brief is sufficient for the opening question.",
            ),
            AdvisorResearchPlanDraft(
                decision_type=AdvisorDecisionType.ANSWER_FROM_STATE,
                summary="The validated brief directly answers the liquidity follow-up.",
            ),
            AdvisorResearchPlanDraft(
                decision_type=AdvisorDecisionType.ANSWER_FROM_STATE,
                summary="The validated brief directly answers the goal-timing follow-up.",
            ),
        ],
        client_reviews=[
            ClientReviewDraft(
                accepted=False,
                follow_up_question="How does liquidity affect the five-year goal?",
            ),
            ClientReviewDraft(
                accepted=False,
                follow_up_question="How does that change the timing decision?",
            ),
        ],
    )

    completed = asyncio.run(runner.run(_maya_profile()))

    assert completed.state is SessionState.RESOLVED
    assert completed.follow_up_count == 2
    assert len(completed.messages) == 3
    assert len(completed.recommendations) == 3
    assert len(completed.retrieval_traces) == 1
    assert completed.client_reviews[-1].accepted is True
    _assert_persisted_replay(repository, completed)
