"""Tests for SQLite-backed workflow-session persistence."""

import sqlite3
from decimal import Decimal
from pathlib import Path

from financial_advisor.domain import (
    AdvisorDecision,
    AdvisorDecisionType,
    AdvisorResearchPlan,
    AnalystTask,
    CitedText,
    ClientMessage,
    ClientProfile,
    ClientReview,
    Evidence,
    EvidenceCitation,
    Finding,
    Recommendation,
    RecommendationOption,
    ResearchBrief,
    RiskTolerance,
    SessionState,
    WebResearchMode,
    WebResearchScope,
    to_analyst_profile_context,
)
from financial_advisor.persistence import SessionRepository
from financial_advisor.retrieval import KeywordRetrievalTrace, VectorRetrievalTrace
from financial_advisor.retrieval.pipeline import (
    CombinedEvidenceRetrievalTrace,
    EvidenceRetrievalStatus,
)
from financial_advisor.retrieval.web.discovery import SearchProviderName, WebSearchResponse
from financial_advisor.retrieval.web.fetch import (
    FetchedWebDocument,
    WebFetchStatus,
    WebPageFetchOutcome,
)
from financial_advisor.retrieval.web.passage_ranker import WebVectorRetrievalTrace
from financial_advisor.retrieval.web.pipeline import LiveWebResearchTrace
from financial_advisor.workflow import WorkflowEngine, WorkflowSession


def make_profile() -> ClientProfile:
    """Return a synthetic client profile for a persistence round trip."""

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


def resolve_session(engine: WorkflowEngine) -> WorkflowSession:
    """Complete a representative Client–Advisor–Analyst workflow."""

    session = engine.start_session(make_profile())
    engine.submit_opening_message(
        session,
        ClientMessage(
            session_id=session.session_id,
            text="How should I use my $15,000 bonus?",
            follow_up_number=0,
        ),
    )
    question = "Compare debt repayment with preserving liquidity."
    plan = AdvisorResearchPlan(
        session_id=session.session_id,
        client_message_id=session.messages[-1].message_id,
        attempt_number=1,
        question=question,
        web_scopes=(
            WebResearchScope(
                mode=WebResearchMode.AUTHORITATIVE_DOMAIN,
                approved_domain_ids=("consumerfinance_gov",),
            ),
        ),
        rationale="Current educational evidence is required.",
    )
    engine.apply_advisor_decision(
        session,
        AdvisorDecision(
            decision_type=AdvisorDecisionType.DELEGATE_RESEARCH,
            summary="Current educational evidence is required.",
            research_plan=plan,
        ),
    )
    task = AnalystTask(
        session_id=session.session_id,
        question=question,
        research_plan=plan,
        client_context=to_analyst_profile_context(session.client_profile),
    )
    engine.submit_analyst_task(session, task)
    evidence = Evidence(
        title="Emergency fund guidance",
        publisher="Consumer Financial Protection Bureau",
        url="https://www.consumerfinance.gov/",
        excerpt="Emergency savings can help absorb unplanned expenses.",
        source_type="web",
    )
    engine.submit_research_brief(
        session,
        ResearchBrief(
            task_id=task.task_id,
            findings=[
                Finding(
                    statement="Emergency savings can help absorb unplanned expenses.",
                    evidence_ids=[evidence.evidence_id],
                )
            ],
            evidence=[evidence],
            caveats=["Current rates require live verification."],
        ),
    )
    engine.submit_recommendation(
        session,
        Recommendation(
            session_id=session.session_id,
            summary=CitedText(
                text="Keep the home-goal portion liquid and consider partial debt repayment.",
                evidence_ids=[evidence.evidence_id],
            ),
            options=[
                RecommendationOption(
                    title="Split the bonus",
                    description=CitedText(
                        text="Reserve part for the home goal and apply part to debt.",
                        evidence_ids=[evidence.evidence_id],
                    ),
                    suitability="Balances liquidity and debt reduction.",
                )
            ],
            rationale=[
                CitedText(
                    text="Maintains liquidity for the stated home-purchase goal.",
                    evidence_ids=[evidence.evidence_id],
                )
            ],
            assumptions=["Emergency savings remains available."],
            risks=[
                CitedText(
                    text="Market investments can lose value before the home goal.",
                    evidence_ids=[evidence.evidence_id],
                )
            ],
            next_steps=["Confirm the home-purchase timeline."],
            evidence_ids=[evidence.evidence_id],
            citations=[
                EvidenceCitation(
                    evidence_id=evidence.evidence_id,
                    title=evidence.title,
                    publisher=evidence.publisher,
                    url=evidence.url,
                )
            ],
            educational_disclaimer=(
                "Educational information only; not individualized financial advice."
            ),
        ),
    )
    engine.submit_client_review(
        session,
        ClientReview(
            session_id=session.session_id,
            follow_up_number=0,
            accepted=True,
            message="This plan addresses my goal. Thank you.",
        ),
    )
    return session


def test_repository_reconstructs_a_completed_session_exactly(tmp_path: Path) -> None:
    """SQLite retains the state, artifacts, and ordered trace for replay."""

    database_path = tmp_path / "financial_advisor.db"
    repository = SessionRepository(database_path)
    session = resolve_session(WorkflowEngine(repository))

    restored = repository.load(session.session_id)

    assert restored == session
    assert restored.state is SessionState.RESOLVED
    assert repository.load_by_trace_id(session.trace_id) == session
    assert {event.trace_id for event in restored.events} == {session.trace_id}
    assert [event.sequence for event in restored.events] == list(range(1, len(restored.events) + 1))
    with sqlite3.connect(database_path) as connection:
        trace_rows = connection.execute(
            """
            SELECT sequence, actor, event_type, summary
            FROM session_events WHERE trace_id = ? ORDER BY sequence
            """,
            (str(session.trace_id),),
        ).fetchall()
    assert len(trace_rows) == len(session.events)
    assert trace_rows[0][1:3] == ("workflow_engine", "session_started")


def test_repository_returns_none_for_an_unknown_session(tmp_path: Path) -> None:
    """Callers can distinguish an absent session from a failed database read."""

    repository = SessionRepository(tmp_path / "financial_advisor.db")
    repository.initialize()

    assert repository.load(resolve_session(WorkflowEngine()).session_id) is None


def test_repository_persists_a_task_linked_retrieval_trace_and_tool_events(tmp_path: Path) -> None:
    """A trace replay includes retrieval details, not only the final citations."""

    repository = SessionRepository(tmp_path / "financial_advisor.db")
    engine = WorkflowEngine(repository)
    session = engine.start_session(make_profile())
    engine.submit_opening_message(
        session,
        ClientMessage(
            session_id=session.session_id,
            text="How should I use my $15,000 bonus?",
            follow_up_number=0,
        ),
    )
    question = "Compare debt repayment with preserving liquidity."
    plan = AdvisorResearchPlan(
        session_id=session.session_id,
        client_message_id=session.messages[-1].message_id,
        attempt_number=1,
        question=question,
        web_scopes=(
            WebResearchScope(
                mode=WebResearchMode.AUTHORITATIVE_DOMAIN,
                approved_domain_ids=("consumerfinance_gov",),
            ),
        ),
        rationale="Current educational evidence is required.",
    )
    engine.apply_advisor_decision(
        session,
        AdvisorDecision(
            decision_type=AdvisorDecisionType.DELEGATE_RESEARCH,
            summary="Current educational evidence is required.",
            research_plan=plan,
        ),
    )
    task = AnalystTask(
        session_id=session.session_id,
        question=question,
        research_plan=plan,
        client_context=to_analyst_profile_context(session.client_profile),
    )
    engine.submit_analyst_task(session, task)

    engine.begin_evidence_retrieval(session, task.task_id)
    engine.record_evidence_retrieval(session, task.task_id, insufficient_retrieval_trace())

    restored = repository.load(session.session_id)

    assert restored.session_id == session.session_id
    assert len(restored.retrieval_traces) == 1
    assert restored.retrieval_traces[0].task_id == task.task_id
    assert (
        restored.retrieval_traces[0].retrieval.status
        is EvidenceRetrievalStatus.INSUFFICIENT_EVIDENCE
    )
    assert (
        restored.retrieval_traces[0]
        .retrieval.live_web.page_fetches[0]
        .document
        .content
        is None
    )
    assert [event.event_type.value for event in restored.events[-2:]] == [
        "tool_started",
        "tool_completed",
    ]
    with sqlite3.connect(repository._database_path) as connection:  # noqa: SLF001
        row = connection.execute(
            """
            SELECT trace_id, task_id, status FROM evidence_retrieval_traces
            WHERE session_id = ?
            """,
            (str(session.session_id),),
        ).fetchone()
    assert row == (str(session.trace_id), str(task.task_id), "insufficient_evidence")


def insufficient_retrieval_trace() -> CombinedEvidenceRetrievalTrace:
    """Return a valid no-evidence trace without calling a provider or model."""

    query = "Compare debt repayment with preserving liquidity."
    return CombinedEvidenceRetrievalTrace(
        query=query,
        local_vector=VectorRetrievalTrace(query=query, candidate_limit=20, candidates=[]),
        local_bm25=KeywordRetrievalTrace(query=query, candidate_limit=20, candidates=[]),
        live_web=LiveWebResearchTrace(
            discovery=WebSearchResponse(provider=SearchProviderName.EXA, results=[]),
            discovered_result_count=0,
            usable_page_target=4,
            attempted_page_count=1,
            usable_page_count=0,
            page_fetches=[
                WebPageFetchOutcome(
                    discovered_url="https://www.investor.gov/example",
                    status=WebFetchStatus.FETCHED,
                    attempt_count=1,
                    document=FetchedWebDocument(
                        title="Example investor guidance",
                        discovered_url="https://www.investor.gov/example",
                        final_url="https://www.investor.gov/example",
                        content_type="text/html",
                        content=b"<html><body>Transient page content.</body></html>",
                    ),
                )
            ],
            content_processing=[],
            web_vector_retrieval=WebVectorRetrievalTrace(
                query=query,
                fetched_chunk_count=0,
                candidate_limit=10,
                candidates=[],
            ),
        ),
        status=EvidenceRetrievalStatus.INSUFFICIENT_EVIDENCE,
        insufficient_evidence_reason="No evidence candidates were available for this test.",
    )
