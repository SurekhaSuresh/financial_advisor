"""Tests for trusted Advisor plans and client-facing source limitations."""

import asyncio
from decimal import Decimal
from uuid import uuid4

import pytest

from financial_advisor.agents.advisor.contracts import (
    AdvisorRecommendationDraft,
    AdvisorResearchPlanDraft,
    create_recommendation,
)
from financial_advisor.agents.advisor.prompt import (
    ADVISOR_PLANNER_VALIDATION_INSTRUCTIONS,
    ADVISOR_RESPONSE_INSTRUCTION,
)
from financial_advisor.agents.advisor.workflow import AdvisorWorkflowCoordinator
from financial_advisor.domain import (
    AdvisorDecisionType,
    CitedText,
    ClientMessage,
    ClientProfile,
    Evidence,
    Finding,
    RecommendationOption,
    ResearchBrief,
    RiskTolerance,
    SessionState,
    WebResearchMode,
    WebResearchScope,
)
from financial_advisor.workflow import WorkflowEngine, WorkflowSession


class FakeAdvisorService:
    """Supply deterministic Advisor drafts without calling a live model."""

    def __init__(self, plans: list[AdvisorResearchPlanDraft]) -> None:
        self._plans = iter(plans)

    async def plan(self, _request):  # type: ignore[no-untyped-def]
        return next(self._plans)

    async def recommend(self, request, *, repair=False):  # type: ignore[no-untyped-def]
        del repair
        evidence_id = request.research_brief.evidence[0].evidence_id
        return AdvisorRecommendationDraft(
            summary=CitedText(
                text="Use the available evidence as education, not a guarantee.",
                evidence_ids=[evidence_id],
            ),
            options=[
                RecommendationOption(
                    title="Review the evidence-backed options",
                    description=CitedText(
                        text="Compare the tradeoffs before acting.",
                        evidence_ids=[evidence_id],
                    ),
                    suitability="Keeps the decision aligned with the stated goal.",
                )
            ],
            rationale=[
                CitedText(
                    text="The Analyst brief supports this general comparison.",
                    evidence_ids=[evidence_id],
                ),
                CitedText(text="Unsupported rationale.", evidence_ids=[uuid4()]),
            ],
            assumptions=["The profile facts remain current."],
            risks=[CitedText(text="Market outcomes are uncertain.", evidence_ids=[evidence_id])],
            next_steps=["Review the cited sources and confirm current details."],
        )


def _profile() -> ClientProfile:
    return ClientProfile(
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


def _open_session(workflow: WorkflowEngine) -> WorkflowSession:
    session = workflow.start_session(_profile())
    workflow.submit_opening_message(
        session,
        ClientMessage(
            session_id=session.session_id,
            text="How should I balance investing with a seven-year home goal?",
            follow_up_number=0,
        ),
    )
    return session


def test_advisor_creates_a_persisted_plan_then_one_web_enabled_replan() -> None:
    workflow = WorkflowEngine()
    session = _open_session(workflow)
    coordinator = AdvisorWorkflowCoordinator(
        workflow,
        FakeAdvisorService(
            [
                AdvisorResearchPlanDraft(
                    decision_type=AdvisorDecisionType.DELEGATE_RESEARCH,
                    summary="Start with stable investor guidance.",
                    research_question=session.messages[-1].text,
                    include_local_hybrid=True,
                ),
                AdvisorResearchPlanDraft(
                    decision_type=AdvisorDecisionType.DELEGATE_RESEARCH,
                    summary="Local evidence was insufficient; check current guidance.",
                    research_question=session.messages[-1].text,
                    include_local_hybrid=True,
                    web_scopes=(WebResearchScope(mode=WebResearchMode.BROAD_WEB),),
                ),
            ]
        ),
    )

    first_task = asyncio.run(coordinator.assess(session, session.messages[-1]))

    assert first_task is not None
    assert first_task.research_plan.attempt_number == 1
    assert first_task.research_plan.include_local_hybrid is True
    assert first_task.research_plan.web_scopes == ()
    assert session.research_plans == [first_task.research_plan]
    assert workflow.return_to_advisor_for_replan(session, first_task.task_id) is True
    assert session.state is SessionState.ADVISOR_ASSESSES

    second_task = asyncio.run(coordinator.assess(session, session.messages[-1]))

    assert second_task is not None
    assert second_task.research_plan.attempt_number == 2
    assert second_task.research_plan.web_scopes[0].mode is WebResearchMode.BROAD_WEB
    assert session.research_plans == [first_task.research_plan, second_task.research_plan]


def test_advisor_skips_an_identical_completed_refinement_scope() -> None:
    """A duplicate refinement uses the validated brief instead of spending plan budget."""

    workflow = WorkflowEngine()
    session = _open_session(workflow)
    coordinator = AdvisorWorkflowCoordinator(
        workflow,
        FakeAdvisorService(
            [
                AdvisorResearchPlanDraft(
                    decision_type=AdvisorDecisionType.DELEGATE_RESEARCH,
                    summary="Check current mortgage-rate context.",
                    research_question=session.messages[-1].text,
                    include_local_hybrid=True,
                    web_scopes=(WebResearchScope(mode=WebResearchMode.BROAD_WEB),),
                ),
                AdvisorResearchPlanDraft(
                    decision_type=AdvisorDecisionType.DELEGATE_RESEARCH,
                    summary="Repeat the same research scope.",
                    research_question=session.messages[-1].text,
                    include_local_hybrid=True,
                    web_scopes=(WebResearchScope(mode=WebResearchMode.BROAD_WEB),),
                ),
            ]
        ),
    )
    first_task = asyncio.run(coordinator.assess(session, session.messages[-1]))
    assert first_task is not None

    evidence = Evidence(
        title="Current guidance",
        publisher="Investor.gov",
        url="https://www.investor.gov/",
        excerpt="A five-year goal requires attention to risk and liquidity.",
        source_type="vector_store",
    )
    session.evidence_ids.add(evidence.evidence_id)
    workflow.submit_research_brief(
        session,
        ResearchBrief(
            task_id=first_task.task_id,
            findings=[
                Finding(
                    statement="A five-year goal requires attention to risk and liquidity.",
                    evidence_ids=[evidence.evidence_id],
                )
            ],
            evidence=[evidence],
            caveats=["Educational guidance only."],
        ),
    )
    assert workflow.begin_research_brief_review(session, first_task.task_id) is True

    duplicate_task = asyncio.run(coordinator.assess(session, session.messages[-1]))

    assert duplicate_task is None
    assert session.state is SessionState.ADVISOR_PROPOSES
    assert session.research_plans == [first_task.research_plan]
    assert any(
        "repeats an already completed research scope" in event.summary
        for event in session.events
    )


def test_advisor_attaches_analyst_source_limitations_to_client_recommendation() -> None:
    workflow = WorkflowEngine()
    session = _open_session(workflow)
    coordinator = AdvisorWorkflowCoordinator(workflow, FakeAdvisorService([]))
    evidence = Evidence(
        title="Investor education",
        publisher="Investor.gov",
        url="https://www.investor.gov/",
        excerpt="Diversification can help manage investment risk.",
        source_type="vector_store",
    )
    # This focused response test begins at the state reached after Analyst submission.
    session.state = SessionState.ADVISOR_PROPOSES
    session.evidence_ids.add(evidence.evidence_id)
    brief = ResearchBrief(
        task_id=session.active_task_id or uuid4(),
        findings=[
            Finding(
                statement="Diversification can help manage investment risk.",
                evidence_ids=[evidence.evidence_id],
            )
        ],
        evidence=[evidence],
        caveats=["Broad web research was unavailable for this run."],
    )

    recommendation = asyncio.run(coordinator.respond(session, brief, session.messages[-1]))

    assert recommendation.limitations == ["Broad web research was unavailable for this run."]
    assert session.recommendations[-1].limitations == recommendation.limitations
    assert recommendation.citations[0].evidence_id == evidence.evidence_id
    assert str(recommendation.citations[0].url) == "https://www.investor.gov/"
    assert recommendation.integrity_warnings[0].actor == "advisor"
    assert recommendation.integrity_warnings[0].claim_type == "rationale"
    assert any(event.event_type.value == "claim_filtered" for event in session.events)


def test_recommendation_removes_internal_evidence_identifiers_from_client_prose() -> None:
    """The UI receives clean claim text and separately renders validated source links."""

    evidence = Evidence(
        title="Investor education",
        publisher="Investor.gov",
        url="https://www.investor.gov/",
        excerpt="Diversification can help manage investment risk.",
        source_type="vector_store",
    )
    brief = ResearchBrief(
        task_id=uuid4(),
        findings=[
            Finding(
                statement="Diversification can help manage investment risk.",
                evidence_ids=[evidence.evidence_id],
            )
        ],
        evidence=[evidence],
        caveats=["Educational guidance only."],
    )
    draft = AdvisorRecommendationDraft(
        summary=CitedText(
            text=f"Diversification can manage risk [{evidence.evidence_id}].",
            evidence_ids=[evidence.evidence_id],
        ),
        options=[
            RecommendationOption(
                title="Diversify",
                description=CitedText(
                    text=f"Use a diversified approach [{evidence.evidence_id}].",
                    evidence_ids=[evidence.evidence_id],
                ),
                suitability="Matches a moderate risk tolerance.",
            )
        ],
        rationale=[CitedText(text="Evidence supports this.", evidence_ids=[evidence.evidence_id])],
        assumptions=["The profile remains current."],
        risks=[CitedText(text="Markets fluctuate.", evidence_ids=[evidence.evidence_id])],
        next_steps=["Review the source."],
    )

    recommendation = create_recommendation(uuid4(), brief, draft)

    assert recommendation.summary.text == "Diversification can manage risk."
    assert "[" not in recommendation.options[0].description.text


def test_advisor_recommendation_caps_cited_point_budget() -> None:
    """A detailed response remains bounded even when a model overproduces claims."""

    evidence_id = uuid4()
    cited_text = CitedText(text="Supported by selected evidence.", evidence_ids=[evidence_id])
    option = RecommendationOption(
        title="Consider this option",
        description=cited_text,
        suitability="Suitable when it fits the stated goal.",
    )

    with pytest.raises(ValueError, match="at most 10 cited supporting points"):
        AdvisorRecommendationDraft(
            summary=cited_text,
            options=[option, option, option],
            rationale=[cited_text] * 5,
            assumptions=["The supplied profile remains current."],
            risks=[cited_text] * 3,
            next_steps=["Review the cited source before acting."],
        )


def test_advisor_response_prompt_targets_rich_bounded_guidance() -> None:
    """Prompt policy preserves recommendation depth without encouraging filler."""

    assert "up to 10 material, cited decision points" in ADVISOR_RESPONSE_INSTRUCTION
    assert "no more than five genuinely different choices" in ADVISOR_RESPONSE_INSTRUCTION
    assert "Return fewer rather than pad" in ADVISOR_RESPONSE_INSTRUCTION


def test_advisor_planner_prompt_requires_safe_client_progress_wording() -> None:
    """Planner output includes one safe status update without exposing internals."""

    assert "Always populate progress_summary" in ADVISOR_PLANNER_VALIDATION_INSTRUCTIONS
    assert (
        "For ESCALATE, also populate escalation_summary"
        in ADVISOR_PLANNER_VALIDATION_INSTRUCTIONS
    )
    assert (
        "must not mention\nAgents, tools, retrieval channels"
        in ADVISOR_PLANNER_VALIDATION_INSTRUCTIONS
    )


def test_advisor_response_prompt_requires_a_natural_client_orientation() -> None:
    """The response starts with useful context, not an abrupt generated conclusion."""

    assert "Begin summary with one brief, natural orientation" in ADVISOR_RESPONSE_INSTRUCTION
    assert "Do not use a generic greeting" in ADVISOR_RESPONSE_INSTRUCTION
