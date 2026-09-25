"""Tests for deterministic session orchestration without any LLM dependency."""

from decimal import Decimal

import pytest

from financial_advisor.domain import (
    AdvisorDecision,
    AdvisorDecisionType,
    AdvisorResearchPlan,
    AnalystTask,
    CitedText,
    ClientMessage,
    ClientProfile,
    ClientReview,
    ErrorCode,
    EventType,
    Evidence,
    EvidenceCitation,
    Finding,
    Recommendation,
    RecommendationOption,
    ResearchBrief,
    RiskTolerance,
    ScenarioComparison,
    SessionState,
    WebResearchMode,
    WebResearchScope,
    to_analyst_profile_context,
)
from financial_advisor.workflow import (
    MAX_FOLLOW_UPS,
    WorkflowEngine,
    WorkflowSession,
    WorkflowViolation,
)


def make_profile() -> ClientProfile:
    """Return the deterministic synthetic profile used in workflow tests."""

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


def open_session(engine: WorkflowEngine) -> WorkflowSession:
    """Start Maya's session and route her opening question to the Advisor."""

    session = engine.start_session(make_profile())
    engine.submit_opening_message(
        session,
        ClientMessage(
            session_id=session.session_id,
            text="How should I use my $15,000 bonus?",
            follow_up_number=0,
        ),
    )
    return session


def delegate_research(engine: WorkflowEngine, session: WorkflowSession) -> AnalystTask:
    """Move a session from Advisor assessment into a valid Analyst task."""

    question = "Compare debt repayment with preserving liquidity for a home goal."
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
            progress_summary="I’m comparing the relevant educational tradeoffs.",
            research_plan=plan,
        ),
    )
    assert any(
        event.summary == "Current educational evidence is required." for event in session.events
    )
    assert any(
        event.event_type is EventType.CLIENT_PROGRESS_CREATED
        and event.summary == "I’m comparing the relevant educational tradeoffs."
        for event in session.events
    )
    task = AnalystTask(
        session_id=session.session_id,
        question=question,
        research_plan=plan,
        client_context=to_analyst_profile_context(session.client_profile),
    )
    engine.submit_analyst_task(session, task)
    return task


def return_research_brief(
    engine: WorkflowEngine, session: WorkflowSession, task: AnalystTask
) -> Evidence:
    """Return a valid, evidence-backed Analyst result to the Advisor."""

    evidence = Evidence(
        title="An essential guide to building an emergency fund",
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
            scenario_comparison=[
                ScenarioComparison(
                    scenario="Split the bonus between debt and home savings",
                    summary="Balances debt reduction with near-term liquidity.",
                    benefits=["Keeps funds available for the home goal."],
                    tradeoffs=["Debt remains outstanding for longer."],
                    evidence_ids=[evidence.evidence_id],
                )
            ],
            evidence=[evidence],
            caveats=["Current rates and tax rules require live verification."],
        ),
    )
    return evidence


def make_recommendation(session: WorkflowSession, evidence: Evidence) -> Recommendation:
    """Build a client-facing recommendation that cites session evidence."""

    return Recommendation(
        session_id=session.session_id,
        summary=CitedText(
            text="Keep the home-goal portion liquid and evaluate a partial debt payment.",
            evidence_ids=[evidence.evidence_id],
        ),
        options=[
            RecommendationOption(
                title="Split the bonus",
                description=CitedText(
                    text="Reserve part for the home goal and apply part to debt.",
                    evidence_ids=[evidence.evidence_id],
                ),
                suitability="Balances Maya's five-year goal and existing debt.",
            )
        ],
        rationale=[
            CitedText(
                text="Maintains liquidity for the stated home-purchase goal.",
                evidence_ids=[evidence.evidence_id],
            )
        ],
        assumptions=["Maya's emergency fund remains available."],
        risks=[
            CitedText(
                text="Market investments can lose value before the home goal.",
                evidence_ids=[evidence.evidence_id],
            )
        ],
        next_steps=["Confirm the target home-purchase timeline."],
        evidence_ids=[evidence.evidence_id],
        citations=[
            EvidenceCitation(
                evidence_id=evidence.evidence_id,
                title=evidence.title,
                publisher=evidence.publisher,
                url=evidence.url,
            )
        ],
        educational_disclaimer="Educational information only; not individualized financial advice.",
    )


def reach_client_review(engine: WorkflowEngine) -> tuple[WorkflowSession, Evidence]:
    """Build a session through the first Advisor recommendation."""

    session = open_session(engine)
    task = delegate_research(engine, session)
    evidence = return_research_brief(engine, session, task)
    engine.submit_recommendation(session, make_recommendation(session, evidence))
    return session, evidence


def test_evidence_backed_session_reaches_resolved_state() -> None:
    """The happy path follows the complete Client–Advisor–Analyst lifecycle."""

    engine = WorkflowEngine()
    session, _ = reach_client_review(engine)

    engine.submit_client_review(
        session,
        ClientReview(
            session_id=session.session_id,
            follow_up_number=0,
            accepted=True,
            message="This plan addresses my goal. Thank you.",
        ),
    )

    assert session.state is SessionState.RESOLVED
    assert [event.sequence for event in session.events] == list(range(1, len(session.events) + 1))


def test_advisor_can_answer_from_existing_state_without_an_analyst_task() -> None:
    """A follow-up supported by session state does not require another task."""

    engine = WorkflowEngine()
    session = open_session(engine)

    engine.apply_advisor_decision(
        session,
        AdvisorDecision(
            decision_type=AdvisorDecisionType.ANSWER_FROM_STATE,
            summary="Existing validated session evidence answers the follow-up.",
        ),
    )

    assert session.state is SessionState.ADVISOR_PROPOSES
    assert session.active_task_id is None


def test_analyst_task_rejects_an_unapproved_client_context() -> None:
    """The workflow prevents accidental or fabricated context from reaching the Analyst."""

    engine = WorkflowEngine()
    session = open_session(engine)
    question = "Compare available options."
    plan = AdvisorResearchPlan(
        session_id=session.session_id,
        client_message_id=session.messages[-1].message_id,
        attempt_number=1,
        question=question,
        web_scopes=(WebResearchScope(mode=WebResearchMode.BROAD_WEB),),
        rationale="Research is needed.",
    )
    engine.apply_advisor_decision(
        session,
        AdvisorDecision(
            decision_type=AdvisorDecisionType.DELEGATE_RESEARCH,
            summary="Research is needed.",
            research_plan=plan,
        ),
    )
    unapproved_context = to_analyst_profile_context(session.client_profile).model_copy(
        update={"emergency_fund_months": 0}
    )
    task = AnalystTask(
        session_id=session.session_id,
        question=question,
        research_plan=plan,
        client_context=unapproved_context,
    )

    with pytest.raises(WorkflowViolation, match="approved profile context"):
        engine.submit_analyst_task(session, task)


def test_recommendation_rejects_evidence_missing_from_the_session() -> None:
    """The Advisor cannot cite evidence that the Analyst did not return."""

    engine = WorkflowEngine()
    session = open_session(engine)
    engine.apply_advisor_decision(
        session,
        AdvisorDecision(
            decision_type=AdvisorDecisionType.ANSWER_FROM_STATE,
            summary="Existing state is sufficient.",
        ),
    )
    unknown_evidence = Evidence(
        title="Unknown evidence",
        publisher="Test publisher",
        excerpt="No evidence was returned in this session.",
        source_type="vector_store",
    )

    with pytest.raises(WorkflowViolation, match="only evidence available"):
        engine.submit_recommendation(session, make_recommendation(session, unknown_evidence))


def test_third_client_follow_up_escalates_and_closes_the_session() -> None:
    """The workflow enforces the two-follow-up limit independently of agent behavior."""

    engine = WorkflowEngine()
    session, evidence = reach_client_review(engine)

    for follow_up_number in range(1, MAX_FOLLOW_UPS + 1):
        engine.submit_client_review(
            session,
            ClientReview(
                session_id=session.session_id,
                follow_up_number=follow_up_number,
                accepted=False,
                message="I have another question.",
            ),
        )
        engine.apply_advisor_decision(
            session,
            AdvisorDecision(
                decision_type=AdvisorDecisionType.ANSWER_FROM_STATE,
                summary="Prior validated evidence answers the follow-up.",
            ),
        )
        engine.submit_recommendation(session, make_recommendation(session, evidence))

    with pytest.raises(WorkflowViolation) as error:
        engine.submit_client_review(
            session,
            ClientReview(
                session_id=session.session_id,
                follow_up_number=MAX_FOLLOW_UPS + 1,
                accepted=False,
                message="I have one more question.",
            ),
        )

    assert error.value.error.code is ErrorCode.FOLLOW_UP_LIMIT_REACHED
    assert session.state is SessionState.ESCALATED


def test_advisor_safety_escalation_creates_a_client_visible_terminal_response() -> None:
    """A safety boundary is both persisted workflow state and client-facing wording."""

    engine = WorkflowEngine()
    session = open_session(engine)

    engine.apply_advisor_decision(
        session,
        AdvisorDecision(
            decision_type=AdvisorDecisionType.ESCALATE,
            summary="A specific-security recommendation requires professional review.",
            progress_summary="I need to keep this decision within safe educational guidance.",
            escalation_summary=(
                "I can’t recommend whether to buy Nvidia with your brokerage savings. "
                "I can help explain general diversification, risk, and time-horizon factors, "
                "or you can consult a qualified financial professional for a specific decision."
            ),
        ),
    )

    assert session.state is SessionState.ESCALATED
    assert session.terminal_response == (
        "I can’t recommend whether to buy Nvidia with your brokerage savings. "
        "I can help explain general diversification, risk, and time-horizon factors, "
        "or you can consult a qualified financial professional for a specific decision."
    )
