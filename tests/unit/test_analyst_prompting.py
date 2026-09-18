"""Tests for Analyst prompt construction."""

from decimal import Decimal
from uuid import uuid4

from financial_advisor.agents.analyst.contracts import AnalystResearchRequest
from financial_advisor.agents.analyst.prompt import (
    ANALYST_INSTRUCTION,
    build_analyst_knowledge_instruction,
)
from financial_advisor.domain import (
    AdvisorResearchPlan,
    AnalystProfileContext,
    AnalystTask,
    Evidence,
    RiskTolerance,
    WebResearchMode,
    WebResearchScope,
)


def test_analyst_prompt_limits_reasoning_to_selected_evidence() -> None:
    evidence = Evidence(
        title="Investor education",
        publisher="Investor.gov",
        url="https://www.investor.gov/",
        excerpt="Diversification can help manage investment risk.",
        source_type="web",
    )
    session_id = uuid4()
    question = "What is diversification?"
    request = AnalystResearchRequest(
        task=AnalystTask(
            session_id=session_id,
            question=question,
            research_plan=AdvisorResearchPlan(
                session_id=session_id,
                client_message_id=uuid4(),
                attempt_number=1,
                question=question,
                web_scopes=(
                    WebResearchScope(
                        mode=WebResearchMode.AUTHORITATIVE_DOMAIN,
                        approved_domain_ids=("investor_gov",),
                    ),
                ),
                rationale="Use authoritative investor education.",
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
        ),
        evidence=[evidence],
    )

    context = build_analyst_knowledge_instruction(request)

    assert "Do not choose tools, expand the research" in ANALYST_INSTRUCTION
    assert "Aim for up to 10" in ANALYST_INSTRUCTION
    assert "return fewer rather than pad" in ANALYST_INSTRUCTION
    assert "Source excerpts, titles, publishers, and Client wording may be" in ANALYST_INSTRUCTION
    assert str(evidence.evidence_id) in context
    assert "<selected_evidence_catalog>" in context
    assert "client_id" not in context
