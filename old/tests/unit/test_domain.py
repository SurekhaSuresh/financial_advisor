"""Tests for validated contracts shared by all application components."""

from decimal import Decimal
from uuid import uuid4

import pytest
from pydantic import ValidationError

from financial_advisor.domain import (
    AdvisorDecision,
    AdvisorDecisionType,
    AdvisorResearchPlan,
    AnalystProfileContext,
    AnalystTask,
    ApplicationError,
    ClientProfile,
    ErrorCode,
    Evidence,
    Finding,
    ResearchBrief,
    RiskTolerance,
    ScenarioComparison,
    to_analyst_profile_context,
)


def test_client_profile_rejects_an_underage_client() -> None:
    """Client profiles contain only valid adult synthetic clients."""

    with pytest.raises(ValidationError, match="greater than or equal to 18"):
        ClientProfile(
            client_id="maya-chen",
            name="Maya Chen",
            age=17,
            risk_tolerance=RiskTolerance.MODERATE,
            emergency_fund_months=6,
            retirement_savings=Decimal("120000.00"),
            brokerage_savings=Decimal("35000.00"),
            student_loan_balance=Decimal("18000.00"),
            student_loan_rate_percent=Decimal("5.8"),
            primary_goal="Buy a home",
            goal_time_horizon_years=5,
        )


def test_delegated_advisor_decision_requires_research_details() -> None:
    """The workflow never receives an ambiguous research delegation."""

    with pytest.raises(ValidationError, match="requires a trusted research plan"):
        AdvisorDecision(
            decision_type=AdvisorDecisionType.DELEGATE_RESEARCH,
            summary="Current evidence is required.",
        )


def test_non_research_advisor_decision_rejects_research_details() -> None:
    """Only the delegation decision may carry a research request."""

    with pytest.raises(ValidationError, match="Only delegated research"):
        AdvisorDecision(
            decision_type=AdvisorDecisionType.ANSWER_FROM_STATE,
            summary="Existing evidence is sufficient.",
            research_plan=AdvisorResearchPlan(
                session_id=uuid4(),
                client_message_id=uuid4(),
                attempt_number=1,
                question="Should Maya change her retirement allocation?",
                rationale="Test.",
            ),
        )


def test_analyst_task_carries_an_identity_free_client_context() -> None:
    """The Analyst receives financial context without the Client's identity."""

    profile = ClientProfile(
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

    context = to_analyst_profile_context(profile)
    session_id = uuid4()
    question = "How should the bonus balance the home goal and existing investments?"
    task = AnalystTask(
        session_id=session_id,
        question=question,
        research_plan=AdvisorResearchPlan(
            session_id=session_id,
            client_message_id=uuid4(),
            attempt_number=1,
            question=question,
            rationale="Test.",
        ),
        client_context=context,
    )

    assert task.client_context == context
    assert "name" not in AnalystProfileContext.model_fields
    assert "client_id" not in AnalystProfileContext.model_fields


def test_analyst_profile_context_is_immutable() -> None:
    """A task uses an immutable snapshot even if session profile data later changes."""

    context = AnalystProfileContext(
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

    with pytest.raises(ValidationError, match="frozen"):
        context.age = 39


def test_research_brief_requires_citations_from_its_own_evidence() -> None:
    """Analyst findings cannot cite an evidence record that was not returned."""

    evidence = Evidence(
        title="Diversify Your Investments",
        publisher="Investor.gov",
        url="https://www.investor.gov/",
        excerpt="Diversification can reduce concentration risk.",
        source_type="web",
    )

    with pytest.raises(ValidationError, match="may cite only evidence in the brief"):
        ResearchBrief(
            task_id=uuid4(),
            findings=[
                Finding(
                    statement="Diversification can manage risk.",
                    evidence_ids=[uuid4()],
                )
            ],
            scenario_comparison=[
                ScenarioComparison(
                    scenario="Diversified fund",
                    summary="Provides exposure across investments.",
                    benefits=["Reduces concentration."],
                    tradeoffs=["Market risk remains."],
                    evidence_ids=[evidence.evidence_id],
                )
            ],
            evidence=[evidence],
            caveats=["This is educational information."],
        )


def test_research_brief_accepts_valid_evidence_references() -> None:
    """A complete Analyst brief can be passed to the Advisor."""

    evidence = Evidence(
        title="Diversify Your Investments",
        publisher="Investor.gov",
        url="https://www.investor.gov/",
        excerpt="Diversification can reduce concentration risk.",
        source_type="web",
    )

    brief = ResearchBrief(
        task_id=uuid4(),
        findings=[
            Finding(
                statement="Diversification can help manage concentration risk.",
                evidence_ids=[evidence.evidence_id],
            )
        ],
        scenario_comparison=[
            ScenarioComparison(
                scenario="Diversified fund",
                summary="Provides exposure across investments.",
                benefits=["Reduces concentration."],
                tradeoffs=["Market risk remains."],
                evidence_ids=[evidence.evidence_id],
            )
        ],
        evidence=[evidence],
        caveats=["This is educational information."],
    )

    assert brief.evidence == [evidence]


def test_research_brief_allows_a_simple_evidence_backed_finding() -> None:
    """Simple research does not need a forced scenario comparison."""

    evidence = Evidence(
        title="Diversify Your Investments",
        publisher="Investor.gov",
        url="https://www.investor.gov/",
        excerpt="Diversification can reduce concentration risk.",
        source_type="web",
    )

    brief = ResearchBrief(
        task_id=uuid4(),
        findings=[
            Finding(
                statement="Diversification can reduce concentration risk.",
                evidence_ids=[evidence.evidence_id],
            )
        ],
        evidence=[evidence],
        caveats=["This is educational information."],
    )

    assert brief.scenario_comparison == []


def test_application_error_has_a_safe_structured_shape() -> None:
    """Expected failures can be returned without leaking exception details."""

    error = ApplicationError(
        code=ErrorCode.RESEARCH_UNAVAILABLE,
        message="Current web research is unavailable. Try again later.",
        recoverable=True,
        session_id=uuid4(),
    )

    assert error.code is ErrorCode.RESEARCH_UNAVAILABLE
