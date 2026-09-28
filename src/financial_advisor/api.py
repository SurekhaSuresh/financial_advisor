"""FastAPI endpoints used by the local demonstration UI."""

import asyncio
import sqlite3
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from http import HTTPStatus
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from google.adk.runners import Runner
from google.adk.sessions import Session

from financial_advisor.config import (
    CLIENT_AGENT,
    CONVERSATION_STATUS_STATE_KEY,
    CURRENT_CLIENT_QUESTION_STATE_KEY,
    FINALIZE_RECOMMENDATION,
    ApplicationSettings,
)
from financial_advisor.contracts import (
    ClientResult,
    ConversationStatus,
    Recommendation,
    SessionSnapshot,
    SessionSummary,
)
from financial_advisor.inspection import (
    KnowledgeStoreInspection,
    inspect_knowledge_chunks,
)
from financial_advisor.runtime import (
    APP_NAME,
    create_runner,
    run_conversation,
)

DEMO_USER_ID = "demo-user"
SESSION_EVENT_POLL_SECONDS = 0.25


def create_app(
    settings: ApplicationSettings | None = None,
    runner: Runner | None = None,
) -> FastAPI:
    """Create the API, optionally using an injected runner for tests."""

    settings = settings or ApplicationSettings()
    runner = runner or create_runner(settings)
    app = FastAPI(title="Financial Advisor")
    active_conversation: asyncio.Task[None] | None = None

    @app.post("/sessions", status_code=HTTPStatus.ACCEPTED)
    async def start_conversation() -> str:
        nonlocal active_conversation
        if active_conversation is not None and not active_conversation.done():
            raise HTTPException(HTTPStatus.CONFLICT)

        session_id = str(uuid4())
        await runner.session_service.create_session(
            app_name=APP_NAME,
            user_id=DEMO_USER_ID,
            session_id=session_id,
            state={
                CONVERSATION_STATUS_STATE_KEY: ConversationStatus.ACTIVE.value,
            },
        )
        active_conversation = asyncio.create_task(
            run_conversation(runner, DEMO_USER_ID, session_id)
        )
        return session_id

    @app.get("/sessions")
    async def list_sessions(limit: int = 50) -> list[SessionSummary]:
        sessions = await runner.session_service.list_sessions(
            app_name=APP_NAME,
            user_id=DEMO_USER_ID,
        )
        return sorted(
            [
                SessionSummary(
                    session_id=session.id,
                    client_question=session.state.get(CURRENT_CLIENT_QUESTION_STATE_KEY),
                    updated_at=datetime.fromtimestamp(session.last_update_time, UTC),
                    status=ConversationStatus(session.state[CONVERSATION_STATUS_STATE_KEY]),
                )
                for session in sessions.sessions
            ],
            key=lambda item: item.updated_at,
            reverse=True,
        )[:limit]

    @app.get("/sessions/{session_id}")
    async def get_session(session_id: str) -> SessionSnapshot:
        session = await _load_session(runner, session_id)
        client_results = []
        recommendations = []
        progress_updates = []
        for event in session.events:
            if event.author in {"advisor_agent", "runtime"} and event.content is not None:
                progress = " ".join(
                    part.text.strip()
                    for part in event.content.parts or []
                    if part.text and not part.thought
                )
                if progress:
                    progress_updates.append(progress)

            for response in event.get_function_responses():
                if response.name == CLIENT_AGENT:
                    client_results.append(ClientResult.model_validate(response.response))
                elif (
                    response.name == FINALIZE_RECOMMENDATION
                    and isinstance(response.response, dict)
                    and "citations" in response.response
                ):
                    recommendations.append(Recommendation.model_validate(response.response))

        return SessionSnapshot(
            session_id=session.id,
            status=ConversationStatus(session.state[CONVERSATION_STATUS_STATE_KEY]),
            client_results=client_results,
            recommendations=recommendations,
            progress_updates=progress_updates,
            state=session.state,
            events=[event.model_dump(mode="json", exclude_none=True) for event in session.events],
        )

    @app.get("/sessions/{session_id}/events/stream")
    async def stream_session_events(session_id: str) -> StreamingResponse:
        await _load_session(runner, session_id)

        async def session_updates() -> AsyncIterator[str]:
            event_count = 0
            while True:
                session = await _load_session(runner, session_id)
                if len(session.events) > event_count:
                    event_count = len(session.events)
                    yield "data: session_event\n\n"
                if session.state[CONVERSATION_STATUS_STATE_KEY] != ConversationStatus.ACTIVE:
                    yield "data: session_complete\n\n"
                    return
                await asyncio.sleep(SESSION_EVENT_POLL_SECONDS)

        return StreamingResponse(session_updates(), media_type="text/event-stream")

    @app.get("/inspection/sqlite/tables")
    def inspect_sqlite_tables() -> list[str]:
        if not settings.session_database_path.exists():
            return []
        with sqlite3.connect(settings.session_database_path) as connection:
            return [
                row[0]
                for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
            ]

    @app.get("/inspection/sqlite/tables/{table_name}/rows")
    def inspect_sqlite_rows(
        table_name: str,
        limit: int = 50,
    ) -> list[dict[str, object]]:
        if not settings.session_database_path.exists():
            raise HTTPException(HTTPStatus.NOT_FOUND, "Session database is unavailable.")
        with sqlite3.connect(settings.session_database_path) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(f'SELECT * FROM "{table_name}" LIMIT ?', (limit,))
            return [dict(row) for row in rows]

    @app.get("/inspection/knowledge")
    def inspect_knowledge_store(
        publisher: str | None = None,
        topic: str | None = None,
        metadata_field: str | None = None,
        metadata_value: str | None = None,
        min_token_count: int | None = None,
        max_token_count: int | None = None,
        offset: int = 0,
        limit: int = 50,
    ) -> KnowledgeStoreInspection:
        try:
            return inspect_knowledge_chunks(
                settings.knowledge_database_path,
                publisher=publisher,
                topic=topic,
                metadata_field=metadata_field,
                metadata_value=metadata_value,
                min_token_count=min_token_count,
                max_token_count=max_token_count,
                offset=offset,
                limit=limit,
            )
        except (FileNotFoundError, ValueError) as error:
            raise HTTPException(HTTPStatus.BAD_REQUEST, str(error)) from error

    return app


async def _load_session(runner: Runner, session_id: str) -> Session:
    session = await runner.session_service.get_session(
        app_name=APP_NAME,
        user_id=DEMO_USER_ID,
        session_id=session_id,
    )
    if session is None:
        raise HTTPException(HTTPStatus.NOT_FOUND)
    return session


app = create_app()
