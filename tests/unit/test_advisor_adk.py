"""Tests for Advisor ADK resilience without a live model call."""

import asyncio
from uuid import uuid4

import pytest

from financial_advisor.agents.advisor.adk import AdkAdvisorService, AdvisorModelError
from financial_advisor.agents.advisor.contracts import (
    AdvisorPlanningRequest,
    AdvisorResearchPlanDraft,
)
from financial_advisor.config import GeminiSettings
from financial_advisor.domain import (
    AdvisorDecisionType,
    AnalystProfileContext,
    ClientMessage,
    RiskTolerance,
)


def test_advisor_uses_configured_fallback_after_primary_failures(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """A primary Advisor outage uses the bounded fallback model once."""

    service = AdkAdvisorService(GeminiSettings(advisor_primary_attempts=2))
    attempted_models: list[object] = []

    async def attempt(agent, _key, _context, parser):  # type: ignore[no-untyped-def]
        attempted_models.append(agent.model)
        if agent is not service._fallback_planning_agent:
            raise AdvisorModelError("primary unavailable")
        return parser(
            {
                "decision_type": AdvisorDecisionType.ANSWER_FROM_STATE,
                "summary": "Validated state directly answers the question.",
                "progress_summary": "I’m preparing a response from the available information.",
            }
        )

    monkeypatch.setattr(service, "_attempt", attempt)

    draft = asyncio.run(
        service._run_with_policy(
            primary_agent=service._planning_agent,
            fallback_agent=service._fallback_planning_agent,
            output_key="planning",
            context="test",
            repair_instruction="repair",
            parser=AdvisorResearchPlanDraft.model_validate,
        )
    )

    assert draft.decision_type is AdvisorDecisionType.ANSWER_FROM_STATE
    assert attempted_models == [
        service._planning_agent.model,
        service._planning_agent.model,
        service._fallback_planning_agent.model,
    ]


def test_advisor_turn_timeout_becomes_a_typed_failure(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """The workflow can safely escalate instead of awaiting a hung Advisor call."""

    service = AdkAdvisorService(GeminiSettings(advisor_turn_timeout_seconds=0.01))

    async def never_returns(**_kwargs):  # type: ignore[no-untyped-def]
        await asyncio.sleep(1)

    monkeypatch.setattr(service, "_run_with_policy", never_returns)

    with pytest.raises(AdvisorModelError, match="deadline exceeded"):
        asyncio.run(service.plan(_planning_request()))


def _planning_request() -> AdvisorPlanningRequest:
    """A minimal request is enough because the timeout happens before prompt use."""

    session_id = uuid4()
    return AdvisorPlanningRequest(
        client_message=ClientMessage(
            session_id=session_id,
            text="How should I diversify?",
            follow_up_number=0,
        ),
        client_context=AnalystProfileContext(
            age=38,
            risk_tolerance=RiskTolerance.MODERATE,
            emergency_fund_months=6,
            retirement_savings=120_000,
            brokerage_savings=35_000,
            student_loan_balance=18_000,
            student_loan_rate_percent=5.8,
            primary_goal="Buy a home",
            goal_time_horizon_years=5,
        ),
    )
