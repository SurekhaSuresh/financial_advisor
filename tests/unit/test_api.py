import asyncio
from collections.abc import AsyncIterator
from decimal import Decimal
from pathlib import Path
from typing import cast
from uuid import uuid4

import httpx
from google.adk.agents import RunConfig
from google.adk.events import Event, EventActions
from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService
from google.genai import types

from financial_advisor.agents.advisor.agent import (
    ADVISOR_RECOMMENDATION_HISTORY_STATE_KEY,
    CLIENT_RESULT_HISTORY_STATE_KEY,
    CONVERSATION_STATUS_STATE_KEY,
    FINAL_ADVISOR_RECOMMENDATION_STATE_KEY,
)
from financial_advisor.api import create_app
from financial_advisor.config import ApplicationSettings
from financial_advisor.contracts import (
    CitedText,
    ClientAction,
    ClientProfile,
    ClientResult,
    ConversationStatus,
    EvidenceCitation,
    Recommendation,
    RecommendationOption,
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


class ApiStubRunner:
    def __init__(self, session_service: DatabaseSessionService) -> None:
        self.session_service = session_service

    async def run_async(
        self,
        *,
        user_id: str,
        session_id: str,
        invocation_id: str,
        new_message: types.Content,
        run_config: RunConfig,
    ) -> AsyncIterator[Event]:
        del new_message, run_config
        session = await self.session_service.get_session(
            app_name="financial_advisor",
            user_id=user_id,
            session_id=session_id,
        )
        assert session is not None
        progress_event = Event(
            invocation_id=invocation_id,
            author="advisor_agent",
            content=types.Content(
                role="model",
                parts=[
                    types.Part.from_text(
                        text="I have reviewed the research and am preparing the recommendation."
                    ),
                    types.Part.from_function_call(
                        name="finalize_recommendation",
                        args={"draft_json": "{}"},
                    )
                ],
            ),
        )
        await self.session_service.append_event(session, progress_event)
        yield progress_event

        client_question = ClientResult(
            action=ClientAction.QUESTION,
            message="How should I preserve liquidity for my home purchase?",
        )
        client_acceptance = ClientResult(
            action=ClientAction.ACCEPT,
            message="The recommendation answers my question.",
        )
        advisor_recommendation = recommendation()
        event = Event(
            invocation_id=invocation_id,
            author="advisor_agent",
            actions=EventActions(
                state_delta={
                    CLIENT_RESULT_HISTORY_STATE_KEY: [
                        client_question.model_dump(mode="json"),
                        client_acceptance.model_dump(mode="json"),
                    ],
                    ADVISOR_RECOMMENDATION_HISTORY_STATE_KEY: [
                        advisor_recommendation.model_dump(mode="json")
                    ],
                    FINAL_ADVISOR_RECOMMENDATION_STATE_KEY: (
                        advisor_recommendation.model_dump(mode="json")
                    ),
                    CONVERSATION_STATUS_STATE_KEY: ConversationStatus.RESOLVED.value,
                }
            ),
        )
        await self.session_service.append_event(session, event)
        yield event


def test_api_starts_lists_and_replays_adk_sessions(tmp_path: Path) -> None:
    database_url = f"sqlite+aiosqlite:///{tmp_path / 'sessions.db'}"
    session_service = DatabaseSessionService(db_url=database_url)
    app = create_app(
        settings=ApplicationSettings(
            _env_file=None,
            google_api_key="test-key",
            session_database_url=database_url,
            knowledge_database_path=None,
        ),
        runner=cast(Runner, ApiStubRunner(session_service)),
    )

    async def request_api() -> None:
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://test",
            ) as client:
                started = await client.post("/sessions", json=profile().model_dump(mode="json"))
                session_id = started.json()["session_id"]

                for _attempt in range(10):
                    snapshot = await client.get(f"/sessions/{session_id}")
                    if snapshot.json()["status"] == "resolved":
                        break
                    await asyncio.sleep(0.01)

                history = await client.get("/sessions")
                tables = await client.get("/inspection/sqlite/tables")
                event_stream = await client.get(
                    f"/sessions/{session_id}/events/stream"
                )

                assert started.status_code == 202
                assert snapshot.status_code == 200
                assert snapshot.json()["client_profile"]["name"] == "Maya Chen"
                assert snapshot.json()["client_results"][0]["action"] == "question"
                assert snapshot.json()["recommendations"]
                assert snapshot.json()["progress_updates"] == [
                    "I have reviewed the research and am preparing the recommendation."
                ]
                assert snapshot.json()["events"][0]["author"] == "advisor_agent"
                assert history.json()[0]["session_id"] == session_id
                assert history.json()[0]["client_question"] == (
                    "How should I preserve liquidity for my home purchase?"
                )
                assert {item["table_name"] for item in tables.json()} >= {
                    "sessions",
                    "events",
                }
                assert "event: session_event" in event_stream.text
                assert "event: session_complete" in event_stream.text

    asyncio.run(request_api())
