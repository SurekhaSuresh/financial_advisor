"""Tests for Client ADK construction and bounded resilience without live calls."""

import asyncio
from uuid import uuid4

import pytest

from financial_advisor.agents.client.adk import (
    AdkClientService,
    ClientModelError,
    create_client_opening_agent,
    create_client_review_agent,
)
from financial_advisor.agents.client.contracts import ClientOpeningDraft, ClientReviewDraft
from financial_advisor.config import GeminiSettings
from financial_advisor.domain import ClientProfile, RiskTolerance


def test_client_agents_have_separate_typed_opening_and_review_outputs() -> None:
    """Client question generation and recommendation review remain distinct roles."""

    settings = GeminiSettings(client_model="test-client-model")

    opening = create_client_opening_agent(settings)
    review = create_client_review_agent(settings)

    assert opening.model == "test-client-model"
    assert opening.output_schema is ClientOpeningDraft
    assert opening.output_key == "client_opening_draft"
    assert review.output_schema is ClientReviewDraft
    assert review.output_key == "client_review_draft"
    assert opening.tools == []
    assert review.tools == []


def test_client_uses_configured_fallback_after_primary_failures(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """A primary outage tries the fallback model once rather than waiting indefinitely."""

    service = AdkClientService(GeminiSettings(client_primary_attempts=2))
    attempted_models: list[object] = []

    async def attempt(agent, _key, _context, parser):  # type: ignore[no-untyped-def]
        attempted_models.append(agent.model)
        if agent is not service._fallback_opening_agent:
            raise ClientModelError("primary unavailable")
        return parser({"question": "How should I think about diversification?"})

    monkeypatch.setattr(service, "_attempt", attempt)

    draft = asyncio.run(
        service._run_with_policy(
            primary_agent=service._opening_agent,
            fallback_agent=service._fallback_opening_agent,
            output_key="opening",
            context="test",
            repair_instruction="repair",
            parser=ClientOpeningDraft.model_validate,
        )
    )

    assert draft.question == "How should I think about diversification?"
    assert attempted_models == [
        service._opening_agent.model,
        service._opening_agent.model,
        service._fallback_opening_agent.model,
    ]


def test_client_turn_timeout_becomes_a_typed_failure(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """The ScenarioRunner can safely escalate instead of awaiting a hung Client call."""

    service = AdkClientService(GeminiSettings(client_turn_timeout_seconds=0.01))

    async def never_returns(**_kwargs):  # type: ignore[no-untyped-def]
        await asyncio.sleep(1)

    monkeypatch.setattr(service, "_run_with_policy", never_returns)
    profile = ClientProfile(
        client_id="test-client",
        name="Test Client",
        age=38,
        risk_tolerance=RiskTolerance.MODERATE,
        emergency_fund_months=6,
        retirement_savings=120_000,
        brokerage_savings=35_000,
        student_loan_balance=18_000,
        student_loan_rate_percent=5.8,
        primary_goal="Buy a home",
        goal_time_horizon_years=5,
    )

    with pytest.raises(ClientModelError, match="deadline exceeded"):
        asyncio.run(service.open_conversation(profile, uuid4()))
