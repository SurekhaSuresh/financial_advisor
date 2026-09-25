"""Tests for the Analyst's required evidence-before-synthesis execution path."""

import asyncio
from datetime import UTC, datetime
from decimal import Decimal
from uuid import uuid4

import pytest

from financial_advisor.agents.analyst.contracts import (
    AnalystResearchDraft,
    AnalystResearchRequest,
    create_research_brief,
)
from financial_advisor.agents.analyst.execution import (
    AnalystResearchExecutor,
    AnalystResearchUnavailable,
)
from financial_advisor.agents.analyst.workflow import AnalystWorkflowCoordinator
from financial_advisor.domain import (
    AdvisorDecision,
    AdvisorDecisionType,
    AdvisorResearchPlan,
    AnalystProfileContext,
    AnalystTask,
    ApprovedDomainId,
    ClientMessage,
    ClientProfile,
    Evidence,
    Finding,
    RiskTolerance,
    SessionState,
    WebResearchMode,
    WebResearchScope,
    to_analyst_profile_context,
)
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
from financial_advisor.retrieval.web.discovery import (
    SearchProviderName,
    WebSearchResponse,
)
from financial_advisor.retrieval.web.passage_ranker import WebVectorRetrievalTrace
from financial_advisor.retrieval.web.pipeline import LiveWebResearchTrace
from financial_advisor.workflow import WorkflowEngine


def _task(mode: WebResearchMode = WebResearchMode.BROAD_WEB) -> AnalystTask:
    session_id = uuid4()
    question = "How should I diversify for a seven-year goal?"
    scope = WebResearchScope(
        mode=mode,
        approved_domain_ids=(
            ("investor_gov",) if mode is WebResearchMode.AUTHORITATIVE_DOMAIN else ()
        ),
    )
    return AnalystTask(
        session_id=session_id,
        question=question,
        research_plan=AdvisorResearchPlan(
            session_id=session_id,
            client_message_id=uuid4(),
            attempt_number=1,
            question=question,
            web_scopes=(scope,),
            rationale="Current evidence is needed.",
        ),
        client_context=AnalystProfileContext(
            age=32,
            risk_tolerance=RiskTolerance.MODERATE,
            emergency_fund_months=4,
            retirement_savings=Decimal("30000"),
            brokerage_savings=Decimal("10000"),
            student_loan_balance=Decimal("12000"),
            student_loan_rate_percent=Decimal("5.25"),
            primary_goal="Buy a home in seven years",
            goal_time_horizon_years=7,
        ),
    )


def _active_workflow_task() -> tuple[WorkflowEngine, object, AnalystTask]:
    profile = ClientProfile(
        client_id="maya-chen",
        name="Maya Chen",
        age=32,
        risk_tolerance=RiskTolerance.MODERATE,
        emergency_fund_months=4,
        retirement_savings=Decimal("30000"),
        brokerage_savings=Decimal("10000"),
        student_loan_balance=Decimal("12000"),
        student_loan_rate_percent=Decimal("5.25"),
        primary_goal="Buy a home in seven years",
        goal_time_horizon_years=7,
    )
    workflow = WorkflowEngine()
    session = workflow.start_session(profile)
    workflow.submit_opening_message(
        session,
        ClientMessage(
            session_id=session.session_id,
            text="How should I diversify for a seven-year goal?",
            follow_up_number=0,
        ),
    )
    question = "How should I diversify for a seven-year goal?"
    plan = AdvisorResearchPlan(
        session_id=session.session_id,
        client_message_id=session.messages[-1].message_id,
        attempt_number=1,
        question=question,
        web_scopes=(WebResearchScope(mode=WebResearchMode.BROAD_WEB),),
        rationale="Current evidence is needed.",
    )
    workflow.apply_advisor_decision(
        session,
        AdvisorDecision(
            decision_type=AdvisorDecisionType.DELEGATE_RESEARCH,
            summary="Current evidence is needed.",
            research_plan=plan,
        ),
    )
    task = AnalystTask(
        session_id=session.session_id,
        question=question,
        research_plan=plan,
        client_context=to_analyst_profile_context(profile),
    )
    workflow.submit_analyst_task(session, task)
    return workflow, session, task


def _base_trace(query: str) -> dict[str, object]:
    return {
        "query": query,
        "local_vector": VectorRetrievalTrace(query=query, candidate_limit=20, candidates=[]),
        "local_bm25": KeywordRetrievalTrace(query=query, candidate_limit=20, candidates=[]),
        "live_web": LiveWebResearchTrace(
            discovery=WebSearchResponse(provider=SearchProviderName.EXA, results=[]),
            discovered_result_count=0,
            usable_page_target=4,
            attempted_page_count=0,
            usable_page_count=0,
            page_fetches=[],
            content_processing=[],
            web_vector_retrieval=WebVectorRetrievalTrace(
                query=query,
                fetched_chunk_count=0,
                candidate_limit=10,
                candidates=[],
            ),
        ),
    }


def _completed_retrieval(query: str) -> CombinedEvidenceRetrievalTrace:
    evidence = Evidence(
        title="Investor education",
        publisher="Investor.gov",
        url="https://www.investor.gov/",
        retrieved_at=datetime(2026, 9, 18, tzinfo=UTC),
        excerpt="Diversification can help manage investment risk.",
        source_type="web",
    )
    return CombinedEvidenceRetrievalTrace(
        **_base_trace(query),
        status=EvidenceRetrievalStatus.COMPLETED,
        selection=EvidenceSelectionTrace(
            fusion=ReciprocalRankFusionTrace(
                rank_constant=60,
                input_candidate_count=1,
                canonical_candidate_count=1,
                candidates=[],
            ),
            reranking=CrossEncoderRerankingTrace(
                query=query,
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


def _insufficient_retrieval(query: str) -> CombinedEvidenceRetrievalTrace:
    return CombinedEvidenceRetrievalTrace(
        **_base_trace(query),
        status=EvidenceRetrievalStatus.INSUFFICIENT_EVIDENCE,
        insufficient_evidence_reason="No evidence was available.",
    )


class FakeEvidenceRetriever:
    def __init__(self, retrieval: CombinedEvidenceRetrievalTrace) -> None:
        self.retrieval = retrieval
        self.request: EvidenceRetrievalRequest | None = None

    def retrieve(self, request: EvidenceRetrievalRequest) -> CombinedEvidenceRetrievalTrace:
        self.request = request
        return self.retrieval


class FakeAnalystSynthesisService:
    def __init__(self) -> None:
        self.request: AnalystResearchRequest | None = None

    async def research(self, request: AnalystResearchRequest):  # type: ignore[no-untyped-def]
        self.request = request
        return create_research_brief(
            request,
            AnalystResearchDraft(
                findings=[
                    Finding(
                        statement="Diversification can help manage investment risk.",
                        evidence_ids=[request.evidence[0].evidence_id],
                    )
                ],
                caveats=["The evidence does not establish a personalized allocation."],
            ),
        )


def test_executor_retrieves_before_structured_analyst_synthesis() -> None:
    task = _task()
    retriever = FakeEvidenceRetriever(_completed_retrieval(task.question))
    synthesis = FakeAnalystSynthesisService()
    executor = AnalystResearchExecutor(retriever, synthesis)

    result = asyncio.run(executor.execute(task))

    assert retriever.request is not None
    assert retriever.request.query == task.question
    assert retriever.request.include_local_hybrid is True
    assert retriever.request.web_scopes[0].mode is WebResearchMode.BROAD_WEB
    assert retriever.request.web_scopes[0].approved_domain_ids == ()
    assert synthesis.request is not None
    assert synthesis.request.evidence == result.retrieval.evidence
    assert result.research_brief.task_id == task.task_id


def test_executor_uses_approved_domain_ids_for_authoritative_research() -> None:
    task = _task(WebResearchMode.AUTHORITATIVE_DOMAIN)
    retriever = FakeEvidenceRetriever(_completed_retrieval(task.question))
    synthesis = FakeAnalystSynthesisService()
    executor = AnalystResearchExecutor(retriever, synthesis)

    asyncio.run(executor.execute(task))

    assert retriever.request is not None
    assert retriever.request.web_scopes[0].approved_domain_ids == (
        ApprovedDomainId.INVESTOR_GOV,
    )


def test_executor_does_not_invoke_model_when_evidence_is_insufficient() -> None:
    task = _task()
    retriever = FakeEvidenceRetriever(_insufficient_retrieval(task.question))
    synthesis = FakeAnalystSynthesisService()
    executor = AnalystResearchExecutor(retriever, synthesis)

    with pytest.raises(AnalystResearchUnavailable, match="No evidence was available"):
        asyncio.run(executor.execute(task))

    assert synthesis.request is None


def test_workflow_coordinator_persists_retrieval_before_submitting_brief() -> None:
    workflow, session, task = _active_workflow_task()
    retriever = FakeEvidenceRetriever(_completed_retrieval(task.question))
    executor = AnalystResearchExecutor(retriever, FakeAnalystSynthesisService())
    coordinator = AnalystWorkflowCoordinator(workflow, executor)

    result = asyncio.run(coordinator.execute(session, task))

    assert result.research_brief == session.research_briefs[0]
    assert session.retrieval_traces[0].retrieval == result.retrieval
    assert session.state is SessionState.ADVISOR_PROPOSES


def test_workflow_coordinator_escalates_when_evidence_is_insufficient() -> None:
    workflow, session, task = _active_workflow_task()
    retriever = FakeEvidenceRetriever(_insufficient_retrieval(task.question))
    executor = AnalystResearchExecutor(retriever, FakeAnalystSynthesisService())
    coordinator = AnalystWorkflowCoordinator(workflow, executor)

    with pytest.raises(AnalystResearchUnavailable):
        asyncio.run(coordinator.execute(session, task))

    assert (
        session.retrieval_traces[0].retrieval.status
        is EvidenceRetrievalStatus.INSUFFICIENT_EVIDENCE
    )
    assert session.state is SessionState.ESCALATED
