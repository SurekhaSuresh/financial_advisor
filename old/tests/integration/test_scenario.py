"""Deterministic end-to-end tests for the Client–Advisor–Analyst scenario runner."""

import asyncio
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import httpx

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
    Evidence,
    Finding,
    RecommendationOption,
    ResearchRefinement,
    RiskTolerance,
    SessionState,
    WebResearchMode,
    WebResearchScope,
)
from financial_advisor.main import create_app
from financial_advisor.persistence import SessionRepository
from financial_advisor.retrieval.contracts import (
    CrossEncoderRerankingTrace,
    EvidenceSelectionTrace,
    MmrSelectionTrace,
    ReciprocalRankFusionTrace,
)
from financial_advisor.retrieval.knowledge_base.retrievers import (
    KeywordRetrievalTrace,
    VectorRetrievalTrace,
)
from financial_advisor.retrieval.pipeline import (
    CombinedEvidenceRetrievalTrace,
    EvidenceRetrievalRequest,
    EvidenceRetrievalStatus,
)
from financial_advisor.scenario import ScenarioRunner
from financial_advisor.workflow import WorkflowEngine


def _maya_profile() -> ClientProfile:
    """Return the synthetic profile used by the full scenario test."""

    return ClientProfile(
        client_id="maya-chen",
        name="Maya Chen",
        age=38,
        risk_tolerance=RiskTolerance.MODERATE,
        emergency_fund_months=6,
        retirement_savings=Decimal("120000.00"),
        brokerage_savings=Decimal("35000.00"),
        student_loan_balance=Decimal("18000.00"),
        student_loan_rate_percent=Decimal("5.8"),
        primary_goal="Buy a home",
        goal_time_horizon_years=5,
    )


class FakeClientService:
    """Emulate a Client opening and acceptance without a live model invocation."""

    def __init__(self, reviews: list[ClientReviewDraft] | None = None) -> None:
        self._reviews = iter(reviews or [ClientReviewDraft(accepted=True)])

    async def open_conversation(self, profile: ClientProfile, session_id) -> ClientMessage:  # type: ignore[no-untyped-def]
        return ClientMessage(
            session_id=session_id,
            text=(
                f"How should I use a $15,000 bonus while keeping my {profile.primary_goal} "
                "goal on track?"
            ),
            follow_up_number=0,
        )

    async def review_recommendation(self, request: ClientReviewRequest) -> ClientReview:
        return create_client_review(
            request,
            draft=next(self._reviews),
        )


class FakeAdvisorService:
    """Return deterministic valid plans and evidence-cited recommendation wording."""

    def __init__(self, plans: list[AdvisorResearchPlanDraft] | None = None) -> None:
        self._plans = iter(
            plans
            or [
                AdvisorResearchPlanDraft(
                    decision_type=AdvisorDecisionType.DELEGATE_RESEARCH,
                    summary=(
                        "Compare liquidity, debt repayment, and diversified investing education."
                    ),
                    research_question=(
                        "Compare preserving liquidity, student-loan repayment, and diversified "
                        "investing for this five-year home goal."
                    ),
                    include_local_hybrid=True,
                ),
                AdvisorResearchPlanDraft(
                    decision_type=AdvisorDecisionType.ANSWER_FROM_STATE,
                    summary="The validated brief fully answers the Client question.",
                ),
            ]
        )

    async def plan(self, request):  # type: ignore[no-untyped-def]
        del request
        return next(self._plans)

    async def recommend(self, request, *, repair: bool = False):  # type: ignore[no-untyped-def]
        del repair
        evidence_id = request.research_brief.evidence[0].evidence_id
        return AdvisorRecommendationDraft(
            summary=CitedText(
                text=(
                    "A balanced approach can preserve home-goal liquidity while reviewing "
                    "debt tradeoffs."
                ),
                evidence_ids=[evidence_id],
            ),
            options=[
                RecommendationOption(
                    title="Split the bonus deliberately",
                    description=CitedText(
                        text=(
                            "Reserve the near-term goal portion before considering longer-term "
                            "investing."
                        ),
                        evidence_ids=[evidence_id],
                    ),
                    suitability="Matches the stated five-year goal and moderate risk tolerance.",
                )
            ],
            rationale=[
                CitedText(
                    text=(
                        "Diversification and liquidity are distinct considerations for "
                        "goal-based planning."
                    ),
                    evidence_ids=[evidence_id],
                )
            ],
            assumptions=["The stated home-purchase time horizon remains five years."],
            risks=[
                CitedText(
                    text=(
                        "Investments can fluctuate and may not suit money needed on a short "
                        "timeline."
                    ),
                    evidence_ids=[evidence_id],
                )
            ],
            next_steps=["Confirm the amount needed for the home goal before acting."],
        )


class FakeEvidenceRetriever:
    """Return one deterministic completed retrieval trace for the authorized task."""

    def __init__(
        self,
        statuses: list[EvidenceRetrievalStatus] | None = None,
    ) -> None:
        self._statuses = iter(statuses or [EvidenceRetrievalStatus.COMPLETED])

    def retrieve(self, request: EvidenceRetrievalRequest) -> CombinedEvidenceRetrievalTrace:
        status = next(self._statuses)
        if status is EvidenceRetrievalStatus.INSUFFICIENT_EVIDENCE:
            return CombinedEvidenceRetrievalTrace(
                query=request.query,
                local_vector=VectorRetrievalTrace(
                    query=request.query, candidate_limit=20, candidates=[]
                ),
                local_bm25=KeywordRetrievalTrace(
                    query=request.query, candidate_limit=20, candidates=[]
                ),
                status=EvidenceRetrievalStatus.INSUFFICIENT_EVIDENCE,
                reason="The selected retrieval sources returned no usable supporting evidence.",
            )
        evidence = Evidence(
            title="Investor education",
            publisher="Investor.gov",
            url="https://www.investor.gov/",
            retrieved_at=datetime(2026, 9, 19, tzinfo=UTC),
            excerpt="Diversification can help manage investment risk.",
            source_type="vector_store",
        )
        return CombinedEvidenceRetrievalTrace(
            query=request.query,
            local_vector=VectorRetrievalTrace(
                query=request.query, candidate_limit=20, candidates=[]
            ),
            local_bm25=KeywordRetrievalTrace(
                query=request.query, candidate_limit=20, candidates=[]
            ),
            status=EvidenceRetrievalStatus.COMPLETED,
            selection=EvidenceSelectionTrace(
                fusion=ReciprocalRankFusionTrace(
                    rank_constant=60,
                    input_candidate_count=1,
                    canonical_candidate_count=1,
                    candidates=[],
                ),
                reranking=CrossEncoderRerankingTrace(
                    query=request.query,
                    input_candidate_count=1,
                    candidates=[],
                ),
                diversity_selection=MmrSelectionTrace(
                    requested_limit=1,
                    relevance_weight=0.7,
                    input_candidate_count=1,
                    selected_candidates=[],
                ),
            ),
            evidence=[evidence],
        )


class FakeAnalystSynthesisService:
    """Create a brief from exactly the evidence the deterministic retriever selected."""

    def __init__(self, *, should_fail: bool = False) -> None:
        self._should_fail = should_fail

    async def research(self, request):  # type: ignore[no-untyped-def]
        if self._should_fail:
            raise RuntimeError("Simulated Analyst model outage.")
        return create_research_brief(
            request,
            AnalystResearchDraft(
                findings=[
                    Finding(
                        statement="Diversification can help manage investment risk.",
                        evidence_ids=[request.evidence[0].evidence_id],
                    )
                ],
                caveats=["This is educational guidance, not a personalized allocation."],
            ),
        )


def test_scenario_runner_persists_a_complete_three_agent_resolution(tmp_path: Path) -> None:
    """The real runner/coordinators/SQLite store complete a valid three-agent trajectory."""

    repository = SessionRepository(tmp_path / "financial_advisor.db")
    workflow = WorkflowEngine(repository)
    analyst = AnalystWorkflowCoordinator(
        workflow,
        AnalystResearchExecutor(FakeEvidenceRetriever(), FakeAnalystSynthesisService()),
    )
    runner = ScenarioRunner(
        workflow,
        FakeClientService(),
        AdvisorWorkflowCoordinator(workflow, FakeAdvisorService()),
        analyst,
    )

    completed = asyncio.run(runner.run(_maya_profile()))
    restored = repository.load(completed.session_id)

    assert restored is not None
    assert restored == completed
    assert completed.state is SessionState.RESOLVED
    assert len(completed.messages) == 1
    assert len(completed.research_plans) == 1
    assert len(completed.analyst_tasks) == 1
    assert len(completed.retrieval_traces) == 1
    assert len(completed.research_briefs) == 1
    assert len(completed.recommendations) == 1
    assert completed.client_reviews[-1].accepted is True
    assert completed.recommendations[0].evidence_ids == [
        completed.research_briefs[0].evidence[0].evidence_id
    ]
    assert [event.sequence for event in completed.events] == list(
        range(1, len(completed.events) + 1)
    )
    assert {event.actor for event in completed.events} >= {
        "client",
        "advisor",
        "analyst",
        "workflow_engine",
    }


def test_evidence_miss_is_persisted_then_replanned_once_with_web_research(tmp_path: Path) -> None:
    """A local-only evidence miss is visible and receives one bounded web-enabled replan."""

    repository = SessionRepository(tmp_path / "financial_advisor.db")
    workflow = WorkflowEngine(repository)
    advisor = FakeAdvisorService(
        plans=[
            AdvisorResearchPlanDraft(
                decision_type=AdvisorDecisionType.DELEGATE_RESEARCH,
                summary="Start with stable local financial education.",
                research_question="Find stable guidance for the five-year goal.",
                include_local_hybrid=True,
            ),
            AdvisorResearchPlanDraft(
                decision_type=AdvisorDecisionType.DELEGATE_RESEARCH,
                summary="Add current broad-web context after the local evidence miss.",
                research_question="Find current context for the five-year goal.",
                include_local_hybrid=True,
                web_scopes=(WebResearchScope(mode=WebResearchMode.BROAD_WEB),),
            ),
        ]
    )
    analyst = AnalystWorkflowCoordinator(
        workflow,
        AnalystResearchExecutor(
            FakeEvidenceRetriever(
                [
                    EvidenceRetrievalStatus.INSUFFICIENT_EVIDENCE,
                    EvidenceRetrievalStatus.COMPLETED,
                ]
            ),
            FakeAnalystSynthesisService(),
        ),
    )
    completed = asyncio.run(
        ScenarioRunner(
            workflow,
            FakeClientService(),
            AdvisorWorkflowCoordinator(workflow, advisor),
            analyst,
        ).run(_maya_profile())
    )
    restored = repository.load(completed.session_id)

    assert restored == completed
    assert completed.state is SessionState.RESOLVED
    assert len(completed.research_plans) == 2
    assert completed.research_plans[0].web_scopes == ()
    assert completed.research_plans[1].web_scopes[0].mode is WebResearchMode.BROAD_WEB
    assert [trace.retrieval.status for trace in completed.retrieval_traces] == [
        EvidenceRetrievalStatus.INSUFFICIENT_EVIDENCE,
        EvidenceRetrievalStatus.COMPLETED,
    ]
    assert any(
        "Advisor may create one web-enabled replan" in event.summary
        for event in completed.events
    )


def test_analyst_model_failure_is_persisted_and_escalated(tmp_path: Path) -> None:
    """A synthesis outage records the completed evidence trace, failure, and terminal outcome."""

    repository = SessionRepository(tmp_path / "financial_advisor.db")
    workflow = WorkflowEngine(repository)
    analyst = AnalystWorkflowCoordinator(
        workflow,
        AnalystResearchExecutor(
            FakeEvidenceRetriever(), FakeAnalystSynthesisService(should_fail=True)
        ),
    )
    completed = asyncio.run(
        ScenarioRunner(
            workflow,
            FakeClientService(),
            AdvisorWorkflowCoordinator(workflow, FakeAdvisorService()),
            analyst,
        ).run(_maya_profile())
    )
    restored = repository.load(completed.session_id)

    assert restored == completed
    assert completed.state is SessionState.ESCALATED
    assert len(completed.retrieval_traces) == 1
    assert completed.research_briefs == []
    assert completed.recommendations == []
    assert completed.client_reviews == []
    assert EventType.TOOL_FAILED in [event.event_type for event in completed.events]
    assert EventType.SESSION_ESCALATED in [event.event_type for event in completed.events]

    app = create_app(repository=repository)

    async def inspect_failure() -> tuple[httpx.Response, httpx.Response, httpx.Response]:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return (
                await client.get(f"/sessions/{completed.session_id}"),
                await client.get(f"/traces/{completed.trace_id}/events"),
                await client.get("/inspection/sqlite/tables/session_events/rows"),
            )

    session_response, trace_response, sqlite_response = asyncio.run(inspect_failure())
    assert session_response.json()["state"] == "escalated"
    assert session_response.json()["terminal_response"] is not None
    assert "no investment recommendation was generated" in session_response.json()[
        "terminal_response"
    ]
    assert EventType.TOOL_FAILED.value in {
        event["event_type"] for event in trace_response.json()["events"]
    }
    assert EventType.SESSION_ESCALATED.value in {
        row["event_type"] for row in sqlite_response.json()["rows"]
    }


def test_failed_refinement_uses_the_first_validated_brief_with_a_caveat(tmp_path: Path) -> None:
    """A second-plan evidence miss does not discard usable first-plan research."""

    repository = SessionRepository(tmp_path / "financial_advisor.db")
    workflow = WorkflowEngine(repository)
    advisor = FakeAdvisorService(
        plans=[
            AdvisorResearchPlanDraft(
                decision_type=AdvisorDecisionType.DELEGATE_RESEARCH,
                summary="Research the original home-goal question.",
                research_question="Research the home-goal tradeoffs.",
                include_local_hybrid=True,
            ),
            AdvisorResearchPlanDraft(
                decision_type=AdvisorDecisionType.DELEGATE_RESEARCH,
                summary="Refine the evidence for the remaining debt comparison gap.",
                research_question="Refine the student-loan tradeoff for the home goal.",
                include_local_hybrid=True,
                refinement=ResearchRefinement(
                    material_gap="The first brief did not fully compare the debt tradeoff.",
                    why_current_evidence_cannot_answer_gap="The first evidence set is broad.",
                    authorized_retrieval_changes="Run a more specific local-hybrid query.",
                ),
            ),
        ]
    )
    analyst = AnalystWorkflowCoordinator(
        workflow,
        AnalystResearchExecutor(
            FakeEvidenceRetriever(
                [
                    EvidenceRetrievalStatus.COMPLETED,
                    EvidenceRetrievalStatus.INSUFFICIENT_EVIDENCE,
                ]
            ),
            FakeAnalystSynthesisService(),
        ),
    )

    completed = asyncio.run(
        ScenarioRunner(
            workflow,
            FakeClientService(),
            AdvisorWorkflowCoordinator(workflow, advisor),
            analyst,
        ).run(_maya_profile())
    )

    assert completed.state is SessionState.RESOLVED
    assert len(completed.research_plans) == 2
    assert len(completed.research_briefs) == 1
    assert completed.recommendations[0].limitations[-1].startswith(
        "The attempted refined research was unavailable"
    )
    assert any(
        "Refined evidence retrieval was unavailable" in event.summary
        for event in completed.events
    )


def test_follow_up_budget_automatically_closes_after_two_questions(tmp_path: Path) -> None:
    """The Client may ask two related follow-ups; no third Client model review is called."""

    repository = SessionRepository(tmp_path / "financial_advisor.db")
    workflow = WorkflowEngine(repository)
    advisor = FakeAdvisorService(
        plans=[
            AdvisorResearchPlanDraft(
                decision_type=AdvisorDecisionType.DELEGATE_RESEARCH,
                summary="Research the opening planning question.",
                research_question="Research the home-goal planning question.",
                include_local_hybrid=True,
            ),
            AdvisorResearchPlanDraft(
                decision_type=AdvisorDecisionType.ANSWER_FROM_STATE,
                summary="The opening research brief fully answers the original question.",
            ),
            AdvisorResearchPlanDraft(
                decision_type=AdvisorDecisionType.ANSWER_FROM_STATE,
                summary="The validated brief directly answers the liquidity follow-up.",
            ),
            AdvisorResearchPlanDraft(
                decision_type=AdvisorDecisionType.ANSWER_FROM_STATE,
                summary="The validated brief directly answers the goal-timing follow-up.",
            ),
        ]
    )
    client = FakeClientService(
        reviews=[
            ClientReviewDraft(
                accepted=False,
                follow_up_question="Can you explain the liquidity tradeoff?",
            ),
            ClientReviewDraft(
                accepted=False,
                follow_up_question="How does that affect my goal timing?",
            ),
        ]
    )
    analyst = AnalystWorkflowCoordinator(
        workflow,
        AnalystResearchExecutor(FakeEvidenceRetriever(), FakeAnalystSynthesisService()),
    )
    completed = asyncio.run(
        ScenarioRunner(
            workflow,
            client,
            AdvisorWorkflowCoordinator(workflow, advisor),
            analyst,
        ).run(_maya_profile())
    )

    assert completed.state is SessionState.RESOLVED
    assert completed.follow_up_count == 2
    assert len(completed.messages) == 3
    assert len(completed.recommendations) == 3
    assert len(completed.research_plans) == 1
    assert len(completed.analyst_tasks) == 1
    assert len(completed.retrieval_traces) == 1
    assert len(completed.client_reviews) == 3
    assert completed.client_reviews[-1].accepted is True
    assert "follow-up limit is complete" in completed.client_reviews[-1].message
