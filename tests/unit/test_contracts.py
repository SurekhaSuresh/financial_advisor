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
    Finding,
    Recommendation,
    RecommendationOption,
    ResearchBrief,
    ResearchTask,
    RetrievalChannel,
    RetrievalPath,
    ScenarioComparison,
    create_recommendation,
    create_research_brief,
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


def brief() -> ResearchBrief:
    selected = evidence()
    return ResearchBrief(
        task_id=uuid4(),
        findings=[
            Finding(
                text="A five-year goal generally favors lower volatility.",
                evidence_ids=[selected.evidence_id],
            )
        ],
        scenario_comparisons=[
            ScenarioComparison(
                scenario="Preserve home savings",
                summary="Keep the near-term goal funded with less volatile assets.",
                benefits=["Protects near-term liquidity."],
                tradeoffs=["May have lower expected returns."],
                evidence_ids=[selected.evidence_id],
            )
        ],
        evidence=[selected],
        limitations=["Tax information was not supplied."],
    )


def test_client_acceptance_has_a_meaningful_message() -> None:
    result = ClientResult(
        action=ClientAction.ACCEPT,
        message="This answers my question and gives me a clear next step.",
    )

    assert result.action is ClientAction.ACCEPT


def test_refinement_requires_a_previous_brief_and_material_gap() -> None:
    with pytest.raises(ValidationError):
        ResearchTask(
            question="Research the missing point.",
            client_profile=profile(),
            retrieval_paths=[RetrievalPath.LOCAL_HYBRID],
            material_gaps=["Compare the liquidity trade-off."],
        )

    with pytest.raises(ValidationError):
        ResearchTask(
            question="Research the missing point.",
            client_profile=profile(),
            retrieval_paths=[RetrievalPath.LOCAL_HYBRID],
            previous_brief=brief(),
        )


def test_research_task_requires_unique_retrieval_paths() -> None:
    with pytest.raises(ValidationError, match="each retrieval path only once"):
        ResearchTask(
            question="Research current guidance.",
            client_profile=profile(),
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


def test_create_research_brief_keeps_only_supported_claims() -> None:
    selected = evidence()
    task = ResearchTask(
        question="Compare liquidity and debt repayment.",
        client_profile=profile(),
        retrieval_paths=[RetrievalPath.LOCAL_HYBRID],
    )
    supported = Finding(
        text="Preserve liquidity for the near-term goal.",
        evidence_ids=[selected.evidence_id],
    )
    unsupported = Finding(text="Unsupported finding.", evidence_ids=[uuid4()])

    result = create_research_brief(
        task,
        findings=[supported, unsupported],
        scenario_comparisons=[
            ScenarioComparison(
                scenario="Unsupported comparison",
                summary="This comparison has no selected evidence.",
                benefits=["Unknown benefit."],
                tradeoffs=["Unknown trade-off."],
                evidence_ids=[uuid4()],
            )
        ],
        evidence=[selected],
        limitations=[],
    )

    assert result.findings == [supported]
    assert result.scenario_comparisons == []
    assert result.task_id == task.task_id


def test_create_research_brief_requires_one_supported_finding() -> None:
    task = ResearchTask(
        question="Compare liquidity and debt repayment.",
        client_profile=profile(),
        retrieval_paths=[RetrievalPath.LOCAL_HYBRID],
    )

    with pytest.raises(ValueError, match="no supported findings"):
        create_research_brief(
            task,
            findings=[Finding(text="Unsupported finding.", evidence_ids=[uuid4()])],
            scenario_comparisons=[],
            evidence=[evidence()],
            limitations=[],
        )


def test_create_recommendation_keeps_only_supported_claims() -> None:
    research = brief()
    evidence_id = research.evidence[0].evidence_id
    supported = CitedText(text="Preserve near-term liquidity.", evidence_ids=[evidence_id])
    unsupported = CitedText(text="Unsupported claim.", evidence_ids=[uuid4()])

    result = create_recommendation(
        research,
        summary=supported,
        options=[
            RecommendationOption(title="Supported", description=supported),
            RecommendationOption(title="Unsupported", description=unsupported),
        ],
        assumptions=["The five-year goal remains unchanged."],
        risks=[supported, unsupported],
        next_steps=["Confirm the required down payment."],
    )

    assert [option.title for option in result.options] == ["Supported"]
    assert result.risks == [supported]
    assert [citation.evidence_id for citation in result.citations] == [evidence_id]
    assert result.limitations == research.limitations


def test_create_recommendation_rejects_an_unsupported_summary() -> None:
    research = brief()
    unsupported = CitedText(text="Unsupported summary.", evidence_ids=[uuid4()])
    supported = CitedText(
        text="Supported claim.",
        evidence_ids=[research.evidence[0].evidence_id],
    )

    with pytest.raises(ValueError, match="summary is not supported"):
        create_recommendation(
            research,
            summary=unsupported,
            options=[RecommendationOption(title="Supported", description=supported)],
            assumptions=["The goal remains unchanged."],
            risks=[supported],
            next_steps=["Confirm the goal amount."],
        )
