"""Tests for shared application contracts."""

from uuid import uuid4

import pytest
from pydantic import ValidationError

from financial_advisor.contracts import (
    CitedText,
    ClientResult,
    Evidence,
    EvidenceCitation,
    Recommendation,
    ResearchTask,
)


def evidence() -> Evidence:
    return Evidence(
        evidence_id=uuid4(),
        title="Assessing Your Risk Tolerance",
        publisher="Investor.gov",
        url="https://www.investor.gov/example",
        text="Shorter time horizons generally call for lower volatility.",
        source="vector",
    )


def test_client_acceptance_has_a_meaningful_message() -> None:
    result = ClientResult(
        action="accept",
        message="This answers my question and gives me a clear next step.",
    )

    assert result.action == "accept"


def test_research_task_rejects_unknown_retrieval_paths() -> None:
    with pytest.raises(ValidationError):
        ResearchTask(
            question="Research current guidance.",
            retrieval_paths=["unsupported"],  # type: ignore[list-item]
        )


def test_recommendation_uses_server_created_citations() -> None:
    selected = evidence()
    cited = CitedText(
        text="Keep near-term goal funds in lower-volatility assets.",
        evidence_ids=[selected.evidence_id],
    )

    recommendation = Recommendation(
        summary=cited,
        options={"Preserve liquidity": cited},
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

    assert "educational_disclaimer" not in recommendation.model_dump()
    assert "rationale" not in recommendation.model_dump()
