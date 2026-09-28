import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import cast
from unittest.mock import Mock
from uuid import uuid4

import httpx
import pytest
from google.adk.agents import RunConfig
from google.adk.events import Event, EventActions
from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService
from google.genai import types

from financial_advisor import api as api_module
from financial_advisor.api import create_app
from financial_advisor.config import (
    CONVERSATION_STATUS_STATE_KEY,
    CURRENT_CLIENT_QUESTION_STATE_KEY,
    ApplicationSettings,
)
from financial_advisor.contracts import (
    CitedText,
    ClientResult,
    EvidenceCitation,
    Recommendation,
)


def recommendation() -> Recommendation:
    evidence_id = uuid4()
    supported_text = CitedText(
        text="Preserve liquidity for the home purchase.",
        evidence_ids=[evidence_id],
    )
    return Recommendation(
        summary=supported_text,
        options={"Preserve liquidity": supported_text},
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
            author="advisor_agent",
            content=types.Content(
                role="model",
                parts=[
                    types.Part.from_text(
                        text="I have reviewed the research and am preparing the recommendation."
                    ),
                    types.Part.from_function_call(
                        name="finalize_recommendation",
                        args={"recommendation_draft_json": "{}"},
                    ),
                ],
            ),
        )
        await self.session_service.append_event(session, progress_event)
        yield progress_event

        client_question = ClientResult(
            action="question",
            message="How should I preserve liquidity for my home purchase?",
        )
        client_acceptance = ClientResult(
            action="accept",
            message="The recommendation answers my question.",
        )
        advisor_recommendation = recommendation()
        event = Event(
            author="advisor_agent",
            content=types.Content(
                role="user",
                parts=[
                    types.Part.from_function_response(
                        name="client_agent",
                        response=client_question.model_dump(mode="json"),
                    ),
                    types.Part.from_function_response(
                        name="finalize_recommendation",
                        response={"message": "Redraft the recommendation."},
                    ),
                    types.Part.from_function_response(
                        name="finalize_recommendation",
                        response=advisor_recommendation.model_dump(mode="json"),
                    ),
                    types.Part.from_function_response(
                        name="client_agent",
                        response=client_acceptance.model_dump(mode="json"),
                    ),
                ],
            ),
            actions=EventActions(
                state_delta={
                    CURRENT_CLIENT_QUESTION_STATE_KEY: client_question.message,
                    CONVERSATION_STATUS_STATE_KEY: "resolved",
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
            session_database_path=tmp_path / "sessions.db",
            knowledge_database_path=tmp_path / "missing-knowledge",
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
                started = await client.post("/sessions")
                session_id = started.json()

                for _attempt in range(10):
                    snapshot = await client.get(f"/sessions/{session_id}")
                    if snapshot.json()["status"] == "resolved":
                        break
                    await asyncio.sleep(0.01)

                history = await client.get("/sessions")
                tables = await client.get("/inspection/sqlite/tables")
                knowledge = await client.get("/inspection/knowledge")
                event_stream = await client.get(f"/sessions/{session_id}/events/stream")

                assert started.status_code == 202
                assert snapshot.status_code == 200
                assert snapshot.json()["client_results"][0]["action"] == "question"
                assert len(snapshot.json()["client_results"]) == 2
                assert len(snapshot.json()["recommendations"]) == 1
                assert snapshot.json()["progress_updates"] == [
                    "I have reviewed the research and am preparing the recommendation."
                ]
                assert snapshot.json()["events"][0]["author"] == "advisor_agent"
                assert history.json()[0]["session_id"] == session_id
                assert history.json()[0]["client_question"] == (
                    "How should I preserve liquidity for my home purchase?"
                )
                assert {"sessions", "events"} <= set(tables.json())
                assert knowledge.status_code == 400
                assert knowledge.json()["detail"].startswith("Knowledge store does not exist:")
                assert "data: session_event" in event_stream.text
                assert "data: session_complete" in event_stream.text

    asyncio.run(request_api())


def test_api_allows_only_one_background_conversation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_service = DatabaseSessionService(
        db_url=f"sqlite+aiosqlite:///{tmp_path / 'sessions.db'}"
    )
    started = asyncio.Event()
    finish = asyncio.Event()

    async def slow_conversation(*_args: object) -> None:
        started.set()
        await finish.wait()

    monkeypatch.setattr(api_module, "run_conversation", slow_conversation)
    app = create_app(
        settings=ApplicationSettings(
            _env_file=None,
            google_api_key="test-key",
            session_database_path=tmp_path / "sessions.db",
            knowledge_database_path=tmp_path / "missing-knowledge",
        ),
        runner=cast(Runner, Mock(session_service=session_service)),
    )

    async def request_api() -> None:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            assert (await client.post("/sessions")).status_code == 202
            await started.wait()
            assert (await client.post("/sessions")).status_code == 409
            finish.set()
            await asyncio.sleep(0)

    asyncio.run(request_api())
