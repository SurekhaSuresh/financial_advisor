"""Tests for shared application contracts."""

from decimal import Decimal
from uuid import uuid4

import pytest
from pydantic import ValidationError

from financial_advisor.contracts import (
    CitedText,
    ClientAction,
    ClientProfile,
    ClientResult,
    Evidence,
    EvidenceCitation,
    Recommendation,
    RecommendationOption,
    ResearchTask,
    RetrievalChannel,
    RetrievalPath,
)


def profile() -> ClientProfile:
    return ClientProfile(
        name="Maya Chen",
        age=38,
        risk_tolerance="moderate",
        emergency_fund_months=6,
        retirement_savings=Decimal("120000"),
        brokerage_savings=Decimal("35000"),
        student_loan_balance=Decimal("18000"),
        student_loan_rate_percent=Decimal("5.8"),
        primary_goal="Buy a home",
        goal_time_horizon_years=5,
    )


def evidence() -> Evidence:
    return Evidence(
        evidence_id=uuid4(),
        title="Assessing Your Risk Tolerance",
        publisher="Investor.gov",
        url="https://www.investor.gov/example",
        text="Shorter time horizons generally call for lower volatility.",
        source=RetrievalChannel.VECTOR,
    )


def test_client_acceptance_has_a_meaningful_message() -> None:
    result = ClientResult(
        action=ClientAction.ACCEPT,
        message="This answers my question and gives me a clear next step.",
    )

    assert result.action is ClientAction.ACCEPT


def test_research_task_requires_unique_retrieval_paths() -> None:
    with pytest.raises(ValidationError, match="each retrieval path only once"):
        ResearchTask(
            question="Research current guidance.",
            client_profile_json=profile().model_dump_json(),
            retrieval_paths=[
                RetrievalPath.WEB,
                RetrievalPath.WEB,
            ],
        )


def test_recommendation_uses_server_created_citations() -> None:
    selected = evidence()
    cited = CitedText(
        text="Keep near-term goal funds in lower-volatility assets.",
        evidence_ids=[selected.evidence_id],
    )

    recommendation = Recommendation(
        summary=cited,
        options=[RecommendationOption(title="Preserve liquidity", description=cited)],
        assumptions=["The five-year time horizon remains unchanged."],
        risks=[cited],
        next_steps=["Set a target down-payment amount."],
        citations=[
            EvidenceCitation(
                evidence_id=selected.evidence_id,
                title=selected.title,
                publisher=selected.publisher,
                url=selected.url,
            )
        ],
        limitations=["Tax information was not supplied."],
    )

    assert recommendation.recommendation_id
    assert "educational_disclaimer" not in recommendation.model_dump()
    assert "rationale" not in recommendation.model_dump()
