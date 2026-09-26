"""FastAPI endpoints used by the local demonstration UI."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from google.adk.events import Event
from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService, Session

from financial_advisor.agents.advisor.agent import (
    ADVISOR_RECOMMENDATION_HISTORY_STATE_KEY,
    CLIENT_RESULT_HISTORY_STATE_KEY,
    CONVERSATION_STATUS_STATE_KEY,
)
from financial_advisor.config import ApplicationSettings
from financial_advisor.contracts import (
    ClientProfile,
    ClientResult,
    ConversationResult,
    ConversationStarted,
    ConversationStatus,
    PersistedSessionEvent,
    Recommendation,
    SessionSnapshot,
    SessionSummary,
)
from financial_advisor.inspection import (
    KnowledgeChunkPage,
    KnowledgeStoreInspector,
    KnowledgeStoreSummary,
    SqliteInspector,
    SqliteTableRows,
    SqliteTableSummary,
)
from financial_advisor.runtime import (
    APP_NAME,
    CLIENT_PROFILE_STATE_KEY,
    create_application_runner,
    create_conversation_session,
    run_conversation,
)

DEMO_USER_ID = "demo-user"
SESSION_EVENT_POLL_SECONDS = 0.25


def create_app(
    settings: ApplicationSettings | None = None,
    runner: Runner | None = None,
) -> FastAPI:
    """Create the API, optionally using an injected runner for tests."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        active_settings = settings or ApplicationSettings()
        active_runner = runner or create_application_runner(active_settings)
        app.state.settings = active_settings
        app.state.runner = active_runner
        app.state.conversation_tasks = set()
        yield

        tasks: set[asyncio.Task[ConversationResult]] = app.state.conversation_tasks
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        if isinstance(active_runner.session_service, DatabaseSessionService):
            await active_runner.session_service.close()

    app = FastAPI(title="Financial Advisor", version="0.1.0", lifespan=lifespan)

    @app.get("/health", tags=["system"])
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post(
        "/sessions",
        response_model=ConversationStarted,
        status_code=202,
        tags=["conversations"],
    )
    async def start_conversation(client_profile: ClientProfile) -> ConversationStarted:
        active_runner: Runner = app.state.runner
        session_id = await create_conversation_session(
            active_runner,
            client_profile,
            DEMO_USER_ID,
        )
        task = asyncio.create_task(
            run_conversation(
                active_runner,
                client_profile,
                DEMO_USER_ID,
                session_id,
            )
        )
        tasks: set[asyncio.Task[ConversationResult]] = app.state.conversation_tasks
        tasks.add(task)
        task.add_done_callback(tasks.discard)
        return ConversationStarted(
            session_id=session_id,
            status=ConversationStatus.ACTIVE,
        )

    @app.get(
        "/sessions",
        response_model=list[SessionSummary],
        tags=["conversations"],
    )
    async def list_sessions(
        limit: int = Query(default=50, ge=1, le=100),
    ) -> list[SessionSummary]:
        active_runner: Runner = app.state.runner
        response = await active_runner.session_service.list_sessions(
            app_name=APP_NAME,
            user_id=DEMO_USER_ID,
        )
        summaries = [
            _session_summary(session)
            for session in response.sessions
            if CLIENT_PROFILE_STATE_KEY in session.state
        ]
        return sorted(summaries, key=lambda item: item.updated_at, reverse=True)[:limit]

    @app.get(
        "/sessions/{session_id}",
        response_model=SessionSnapshot,
        tags=["conversations"],
    )
    async def get_session(session_id: str) -> SessionSnapshot:
        session = await _load_session(app.state.runner, session_id)
        return _session_snapshot(session)

    @app.get("/sessions/{session_id}/events/stream", tags=["conversations"])
    async def stream_session_events(
        session_id: str,
        after_sequence: int = Query(default=0, ge=0),
    ) -> StreamingResponse:
        await _load_session(app.state.runner, session_id)

        async def encoded_events() -> AsyncIterator[str]:
            next_sequence = after_sequence + 1
            while True:
                session = await _load_session(app.state.runner, session_id)
                for sequence, event in enumerate(session.events, start=1):
                    if sequence < next_sequence:
                        continue
                    persisted_event = _persisted_event(sequence, event)
                    yield (
                        f"id: {sequence}\nevent: session_event\n"
                        f"data: {persisted_event.model_dump_json()}\n\n"
                    )
                    next_sequence = sequence + 1

                if _conversation_status(session) is not ConversationStatus.ACTIVE:
                    yield "event: session_complete\ndata: {}\n\n"
                    return
                await asyncio.sleep(SESSION_EVENT_POLL_SECONDS)

        return StreamingResponse(encoded_events(), media_type="text/event-stream")

    @app.get(
        "/inspection/sqlite/tables",
        response_model=list[SqliteTableSummary],
        tags=["inspection"],
    )
    def inspect_sqlite_tables() -> list[SqliteTableSummary]:
        active_settings: ApplicationSettings = app.state.settings
        return SqliteInspector(active_settings.session_database_url).tables()

    @app.get(
        "/inspection/sqlite/tables/{table_name}/rows",
        response_model=SqliteTableRows,
        tags=["inspection"],
    )
    def inspect_sqlite_rows(
        table_name: str,
        limit: int = Query(default=50, ge=1, le=100),
    ) -> SqliteTableRows:
        active_settings: ApplicationSettings = app.state.settings
        try:
            return SqliteInspector(active_settings.session_database_url).rows(
                table_name,
                limit=limit,
            )
        except ValueError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.get(
        "/inspection/knowledge/summary",
        response_model=KnowledgeStoreSummary,
        tags=["inspection"],
    )
    def inspect_knowledge_summary() -> KnowledgeStoreSummary:
        try:
            return _knowledge_inspector(app).summary()
        except FileNotFoundError as error:
            raise HTTPException(
                status_code=503,
                detail="Local knowledge is unavailable.",
            ) from error

    @app.get(
        "/inspection/knowledge/chunks",
        response_model=KnowledgeChunkPage,
        tags=["inspection"],
    )
    def inspect_knowledge_chunks(
        publisher: str | None = None,
        topic: str | None = None,
        metadata_field: str | None = None,
        metadata_value: str | None = None,
        min_token_count: int | None = Query(default=None, ge=1),
        max_token_count: int | None = Query(default=None, ge=1),
        offset: int = Query(default=0, ge=0),
        limit: int = Query(default=50, ge=1, le=100),
    ) -> KnowledgeChunkPage:
        try:
            return _knowledge_inspector(app).chunks(
                publisher=publisher,
                topic=topic,
                metadata_field=metadata_field,
                metadata_value=metadata_value,
                min_token_count=min_token_count,
                max_token_count=max_token_count,
                offset=offset,
                limit=limit,
            )
        except FileNotFoundError as error:
            raise HTTPException(
                status_code=503,
                detail="Local knowledge is unavailable.",
            ) from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    return app


async def _load_session(runner: Runner, session_id: str) -> Session:
    session = await runner.session_service.get_session(
        app_name=APP_NAME,
        user_id=DEMO_USER_ID,
        session_id=session_id,
    )
    if session is None:
        raise HTTPException(status_code=404, detail="Session was not found.")
    return session


def _session_summary(session: Session) -> SessionSummary:
    stored_client_results = session.state.get(CLIENT_RESULT_HISTORY_STATE_KEY, [])
    client_question = (
        ClientResult.model_validate(stored_client_results[0]).message
        if stored_client_results
        else None
    )
    return SessionSummary(
        session_id=session.id,
        client_question=client_question,
        updated_at=datetime.fromtimestamp(session.last_update_time, UTC),
        status=_conversation_status(session),
    )


def _session_snapshot(session: Session) -> SessionSnapshot:
    return SessionSnapshot(
        session_id=session.id,
        client_profile=ClientProfile.model_validate(
            session.state[CLIENT_PROFILE_STATE_KEY]
        ),
        status=_conversation_status(session),
        updated_at=datetime.fromtimestamp(session.last_update_time, UTC),
        client_results=[
            ClientResult.model_validate(result)
            for result in session.state.get(CLIENT_RESULT_HISTORY_STATE_KEY, [])
        ],
        recommendations=[
            Recommendation.model_validate(recommendation)
            for recommendation in session.state.get(
                ADVISOR_RECOMMENDATION_HISTORY_STATE_KEY,
                [],
            )
        ],
        progress_updates=_progress_updates(session),
        state=session.state,
        events=[
            _persisted_event(sequence, event)
            for sequence, event in enumerate(session.events, start=1)
        ],
    )


def _progress_updates(session: Session) -> list[str]:
    """Return the Advisor's persisted client-facing progress summaries."""

    updates: list[str] = []
    for event in session.events:
        if event.author != "advisor_agent" or event.content is None:
            continue
        summary = " ".join(
            part.text.strip()
            for part in event.content.parts or []
            if part.text and not part.thought
        )
        if summary:
            updates.append(summary)
    return updates


def _persisted_event(sequence: int, event: Event) -> PersistedSessionEvent:
    return PersistedSessionEvent(
        sequence=sequence,
        event_id=event.id,
        timestamp=datetime.fromtimestamp(event.timestamp, UTC),
        author=event.author,
        event=event.model_dump(mode="json", exclude_none=True),
    )


def _conversation_status(session: Session) -> ConversationStatus:
    return ConversationStatus(
        session.state.get(
            CONVERSATION_STATUS_STATE_KEY,
            ConversationStatus.ACTIVE.value,
        )
    )


def _knowledge_inspector(app: FastAPI) -> KnowledgeStoreInspector:
    settings: ApplicationSettings = app.state.settings
    if settings.knowledge_database_path is None:
        raise HTTPException(status_code=503, detail="Local knowledge is not configured.")
    return KnowledgeStoreInspector(settings.knowledge_database_path)


app = create_app()
