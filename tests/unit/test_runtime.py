import asyncio
from collections.abc import AsyncIterator
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from uuid import uuid4

from google.adk.agents import RunConfig
from google.adk.events import Event, EventActions
from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService
from google.genai import types

from financial_advisor.agents.advisor.agent import FINAL_ADVISOR_RECOMMENDATION_STATE_KEY
from financial_advisor.config import MAX_INVOCATION_LLM_CALLS
from financial_advisor.contracts import (
    CitedText,
    ClientProfile,
    EvidenceCitation,
    Recommendation,
    RecommendationOption,
)
from financial_advisor.retrieval.pipeline import RetrievalPipeline
from financial_advisor.runtime import APP_NAME, create_runner, run_conversation


def client_profile() -> ClientProfile:
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


def recommendation() -> Recommendation:
    evidence_id = uuid4()
    supported_text = CitedText(
        text="Preserve liquidity for the home purchase.",
        evidence_ids=[evidence_id],
    )
    return Recommendation(
        summary=supported_text,
        options=[RecommendationOption(title="Preserve liquidity", description=supported_text)],
        assumptions=["The five-year goal remains unchanged."],
        risks=[supported_text],
        next_steps=["Confirm the required down payment."],
        citations=[
            EvidenceCitation(
                evidence_id=evidence_id,
                title="Liquidity guidance",
                publisher="Investor.gov",
                url="https://www.investor.gov/example",
            )
        ],
    )


class StubRunner:
    def __init__(
        self,
        session_service: DatabaseSessionService,
        *,
        result: Recommendation | None = None,
        error: Exception | None = None,
    ) -> None:
        self.session_service = session_service
        self.result = result
        self.error = error
        self.run_config: RunConfig | None = None

    async def run_async(
        self,
        *,
        user_id: str,
        session_id: str,
        invocation_id: str,
        new_message: types.Content,
        run_config: RunConfig,
    ) -> AsyncIterator[Event]:
        del new_message
        self.run_config = run_config
        if self.error is not None:
            raise self.error

        session = await self.session_service.get_session(
            app_name=APP_NAME,
            user_id=user_id,
            session_id=session_id,
        )
        assert session is not None
        state_delta = (
            {
                FINAL_ADVISOR_RECOMMENDATION_STATE_KEY: self.result.model_dump(
                    mode="json"
                )
            }
            if self.result is not None
            else {}
        )
        event = Event(
            invocation_id=invocation_id,
            author="advisor_agent",
            actions=EventActions(state_delta=state_delta),
        )
        await self.session_service.append_event(session, event)
        yield event


def test_runner_persists_session_events_and_state(tmp_path: Path) -> None:
    database_url = f"sqlite+aiosqlite:///{tmp_path / 'sessions.db'}"
    runner = create_runner(
        "test-model",
        cast(RetrievalPipeline, SimpleNamespace()),
        database_url,
    )

    async def verify_persistence() -> None:
        session = await runner.session_service.create_session(
            app_name=APP_NAME,
            user_id="test-user",
            session_id="test-session",
        )
        await runner.session_service.append_event(
            session,
            Event(
                invocation_id="test-invocation",
                author="advisor_agent",
                actions=EventActions(state_delta={"status": "active"}),
            ),
        )
        await runner.session_service.close()

        recovered_service = DatabaseSessionService(db_url=database_url)
        recovered_session = await recovered_service.get_session(
            app_name=APP_NAME,
            user_id="test-user",
            session_id="test-session",
        )

        assert recovered_session is not None
        assert recovered_session.state == {"status": "active"}
        assert len(recovered_session.events) == 1
        assert recovered_session.events[0].invocation_id == "test-invocation"
        await recovered_service.close()

    asyncio.run(verify_persistence())


def test_conversation_returns_the_persisted_recommendation(tmp_path: Path) -> None:
    database_url = f"sqlite+aiosqlite:///{tmp_path / 'conversation.db'}"
    session_service = DatabaseSessionService(db_url=database_url)
    expected_recommendation = recommendation()
    runner = StubRunner(session_service, result=expected_recommendation)

    async def run() -> None:
        result = await run_conversation(
            cast(Runner, runner),
            client_profile(),
            user_id="test-user",
            session_id="test-session",
        )

        assert result == expected_recommendation
        assert runner.run_config is not None
        assert runner.run_config.max_llm_calls == MAX_INVOCATION_LLM_CALLS
        await session_service.close()

    asyncio.run(run())


def test_conversation_without_a_recommendation_is_not_a_failure(tmp_path: Path) -> None:
    database_url = f"sqlite+aiosqlite:///{tmp_path / 'no-recommendation.db'}"
    session_service = DatabaseSessionService(db_url=database_url)
    runner = StubRunner(session_service)

    async def run() -> None:
        result = await run_conversation(
            cast(Runner, runner),
            client_profile(),
            user_id="test-user",
            session_id="test-session",
        )
        stored_session = await session_service.get_session(
            app_name=APP_NAME,
            user_id="test-user",
            session_id="test-session",
        )

        assert result is None
        assert stored_session is not None
        assert all(event.error_code is None for event in stored_session.events)
        await session_service.close()

    asyncio.run(run())


def test_conversation_persists_unexpected_failures(tmp_path: Path) -> None:
    database_url = f"sqlite+aiosqlite:///{tmp_path / 'failure.db'}"
    session_service = DatabaseSessionService(db_url=database_url)
    runner = StubRunner(session_service, error=ConnectionError("model unavailable"))

    async def run() -> None:
        result = await run_conversation(
            cast(Runner, runner),
            client_profile(),
            user_id="test-user",
            session_id="test-session",
        )
        stored_session = await session_service.get_session(
            app_name=APP_NAME,
            user_id="test-user",
            session_id="test-session",
        )

        assert result is None
        assert stored_session is not None
        assert stored_session.events[-1].error_code == "ConnectionError"
        assert stored_session.events[-1].error_message == "model unavailable"
        assert stored_session.events[-1].content is None
        await session_service.close()

    asyncio.run(run())
