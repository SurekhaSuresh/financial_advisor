"""Tests for the Analyst reasoning boundary."""

from datetime import UTC, datetime
from decimal import Decimal
from uuid import uuid4

import pytest
from pydantic import ValidationError

from financial_advisor.agents.analyst.contracts import (
    AnalystClaimIntegrityError,
    AnalystResearchDraft,
    AnalystResearchRequest,
    create_research_brief,
)
from financial_advisor.domain import (
    AdvisorResearchPlan,
    AnalystProfileContext,
    AnalystTask,
    Evidence,
    Finding,
    RiskTolerance,
    WebResearchMode,
    WebResearchScope,
)


def _task() -> AnalystTask:
    session_id = uuid4()
    question = "How should emergency savings affect an investment plan?"
    return AnalystTask(
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
            emergency_fund_months=2,
            retirement_savings=Decimal("30000"),
            brokerage_savings=Decimal("10000"),
            student_loan_balance=Decimal("12000"),
            student_loan_rate_percent=Decimal("5.25"),
            primary_goal="Buy a home in seven years",
            goal_time_horizon_years=7,
        ),
    )


def _evidence() -> Evidence:
    return Evidence(
        title="Emergency savings guidance",
        publisher="Consumer Financial Protection Bureau",
        url="https://www.consumerfinance.gov/",
        retrieved_at=datetime(2026, 9, 18, tzinfo=UTC),
        excerpt="Emergency savings can help absorb unexpected expenses.",
        source_type="vector_store",
    )


def test_create_research_brief_attaches_server_selected_evidence() -> None:
    evidence = _evidence()
    request = AnalystResearchRequest(task=_task(), evidence=[evidence])
    draft = AnalystResearchDraft(
        findings=[
            Finding(
                statement="Emergency savings can help absorb unexpected expenses.",
                evidence_ids=[evidence.evidence_id],
            )
        ],
        caveats=["The supplied evidence does not determine a personal savings target."],
    )

    brief = create_research_brief(request, draft)

    assert brief.task_id == request.task.task_id
    assert brief.evidence == [evidence]
    assert brief.findings == draft.findings


def test_create_research_brief_rejects_an_invented_evidence_reference() -> None:
    request = AnalystResearchRequest(task=_task(), evidence=[_evidence()])
    draft = AnalystResearchDraft(
        findings=[
            Finding(
                statement="Unsupported statement.",
                evidence_ids=[uuid4()],
            )
        ],
        caveats=["This must be supported."],
    )

    with pytest.raises(AnalystClaimIntegrityError, match="No evidence-backed Analyst findings"):
        create_research_brief(request, draft)


def test_create_research_brief_filters_only_the_invalid_finding() -> None:
    evidence = _evidence()
    request = AnalystResearchRequest(task=_task(), evidence=[evidence])
    draft = AnalystResearchDraft(
        findings=[
            Finding(
                statement="Supported statement.",
                evidence_ids=[evidence.evidence_id],
            ),
            Finding(
                statement="Unsupported statement.",
                evidence_ids=[uuid4()],
            ),
        ],
        caveats=["Use the available evidence carefully."],
    )

    brief = create_research_brief(request, draft)

    assert [finding.statement for finding in brief.findings] == ["Supported statement."]
    assert brief.integrity_warnings[0].actor == "analyst"
    assert brief.integrity_warnings[0].removed_count == 1


def test_analyst_request_rejects_duplicate_evidence_ids() -> None:
    evidence = _evidence()

    with pytest.raises(ValidationError, match="Analyst evidence IDs must be unique"):
        AnalystResearchRequest(task=_task(), evidence=[evidence, evidence])
