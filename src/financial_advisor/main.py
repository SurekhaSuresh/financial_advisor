"""FastAPI application entry point and observable scenario endpoints."""

from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from uuid import UUID

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse

from financial_advisor.api import (
    ScenarioExecutionManager,
    ScenarioStarted,
    SessionSnapshot,
    SessionSummary,
    SqliteTableRows,
    SqliteTableSummary,
    TraceEvents,
    TraceEventStream,
)
from financial_advisor.config import get_settings
from financial_advisor.domain import ClientProfile
from financial_advisor.inspection import (
    KnowledgeChunkPage,
    KnowledgeStoreInspector,
    KnowledgeStoreSummary,
    KnowledgeStoreUnavailable,
)
from financial_advisor.persistence import SessionRepository
from financial_advisor.runtime import LocalRuntime, RuntimeConfigurationError, build_local_runtime


def _missing_runtime(detail: str) -> HTTPException:
    """Return a clear configuration error without exposing implementation details."""

    return HTTPException(status_code=503, detail=detail)


def create_app(
    *,
    repository: SessionRepository | None = None,
    execution_manager: ScenarioExecutionManager | None = None,
    configure_local_runtime: bool = False,
) -> FastAPI:
    """Create the API application with optional runtime dependencies for local execution."""

    settings = get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        if configure_local_runtime:
            inspection_repository = SessionRepository(settings.storage.sqlite_database_path)
            inspection_repository.initialize()
            app.state.inspection_repository = inspection_repository
            try:
                runtime = build_local_runtime(settings, repository=inspection_repository)
            except RuntimeConfigurationError as error:
                app.state.runtime_error = str(error)
            else:
                app.state.local_runtime = runtime
        yield

    app = FastAPI(title="Financial Advisor", version="0.1.0", lifespan=lifespan)

    def active_runtime() -> LocalRuntime | None:
        """Return the started real runtime only when module startup configured one."""

        return getattr(app.state, "local_runtime", None)

    def active_repository() -> SessionRepository | None:
        """Prefer injected dependencies, then use the local runtime after startup."""

        runtime = active_runtime()
        started_repository = getattr(app.state, "inspection_repository", None)
        return repository or (runtime.repository if runtime is not None else started_repository)

    def active_execution_manager() -> ScenarioExecutionManager | None:
        """Prefer an injected test/runtime manager, then use the started local runtime."""

        runtime = active_runtime()
        return execution_manager or (runtime.execution_manager if runtime is not None else None)

    @app.get("/health", tags=["system"])
    def health() -> dict[str, str]:
        """Report that the API process is available."""

        return {"status": "ok", "environment": settings.app_env}

    @app.post("/sessions", response_model=ScenarioStarted, status_code=202, tags=["scenarios"])
    async def start_scenario(profile: ClientProfile) -> ScenarioStarted:
        """Start a Client-led scenario and immediately return its replay identifiers."""

        manager = active_execution_manager()
        if manager is None:
            detail = getattr(
                app.state,
                "runtime_error",
                "Scenario execution runtime is not configured.",
            )
            raise _missing_runtime(detail)
        return manager.start(profile)

    @app.get("/sessions/{session_id}", response_model=SessionSnapshot, tags=["sessions"])
    def get_session(session_id: UUID) -> SessionSnapshot:
        """Return one persisted session and all artifacts needed for completed-session replay."""

        active_repository_value = active_repository()
        if active_repository_value is None:
            raise _missing_runtime("Session persistence is not configured.")
        session = active_repository_value.load(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session was not found.")
        return SessionSnapshot.from_session(session)

    @app.get("/sessions", response_model=list[SessionSummary], tags=["sessions"])
    def list_sessions(limit: int = Query(default=50, ge=1, le=100)) -> list[SessionSummary]:
        """List recent persisted sessions for local completed-conversation replay."""

        active_repository_value = active_repository()
        if active_repository_value is None:
            raise _missing_runtime("Session persistence is not configured.")
        return [
            SessionSummary.from_session(session)
            for session in active_repository_value.list_sessions(limit=limit)
        ]

    @app.get("/traces/{trace_id}", response_model=SessionSnapshot, tags=["traces"])
    def get_trace(trace_id: UUID) -> SessionSnapshot:
        """Resolve one trace identifier to its complete persisted root session."""

        active_repository_value = active_repository()
        if active_repository_value is None:
            raise _missing_runtime("Session persistence is not configured.")
        session = active_repository_value.load_by_trace_id(trace_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Trace was not found.")
        return SessionSnapshot.from_session(session)

    @app.get("/traces/{trace_id}/events", response_model=TraceEvents, tags=["traces"])
    def get_trace_events(trace_id: UUID) -> TraceEvents:
        """Return the ordered persisted event log for one trace investigation."""

        active_repository_value = active_repository()
        if active_repository_value is None:
            raise _missing_runtime("Session persistence is not configured.")
        session = active_repository_value.load_by_trace_id(trace_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Trace was not found.")
        return TraceEvents(trace_id=trace_id, events=session.events)

    @app.get("/traces/{trace_id}/events/stream", tags=["traces"])
    async def stream_trace_events(
        trace_id: UUID,
        after_sequence: int = Query(default=0, ge=0),
    ) -> StreamingResponse:
        """Stream new persisted trace events in Server-Sent Events format until terminal."""

        active_repository_value = active_repository()
        if active_repository_value is None:
            raise _missing_runtime("Session persistence is not configured.")
        if active_repository_value.load_by_trace_id(trace_id) is None:
            raise HTTPException(status_code=404, detail="Trace was not found.")

        stream = TraceEventStream(active_repository_value)

        async def encoded_events() -> AsyncIterator[str]:
            async for event in stream.events(trace_id, after_sequence=after_sequence):
                yield (
                    f"id: {event.sequence}\nevent: trace_event\n"
                    f"data: {event.model_dump_json()}\n\n"
                )
            yield "event: trace_complete\ndata: {}\n\n"

        return StreamingResponse(encoded_events(), media_type="text/event-stream")

    @app.get(
        "/inspection/sqlite/tables",
        response_model=list[SqliteTableSummary],
        tags=["inspection"],
    )
    def inspect_sqlite_tables() -> list[SqliteTableSummary]:
        """List allowlisted SQLite tables and row counts for read-only inspection."""

        active_repository_value = active_repository()
        if active_repository_value is None:
            raise _missing_runtime("Session persistence is not configured.")
        return [
            SqliteTableSummary(table_name=table_name, row_count=row_count)
            for table_name, row_count in active_repository_value.inspect_tables().items()
        ]

    @app.get(
        "/inspection/sqlite/tables/{table_name}/rows",
        response_model=SqliteTableRows,
        tags=["inspection"],
    )
    def inspect_sqlite_table_rows(
        table_name: str, limit: int = Query(default=50, ge=1, le=100)
    ) -> SqliteTableRows:
        """Return bounded values for one explicitly allowlisted SQLite table."""

        active_repository_value = active_repository()
        if active_repository_value is None:
            raise _missing_runtime("Session persistence is not configured.")
        try:
            columns, rows = active_repository_value.inspect_table_rows(table_name, limit=limit)
        except ValueError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        return SqliteTableRows(table_name=table_name, columns=columns, rows=rows)

    def knowledge_inspector() -> KnowledgeStoreInspector:
        """Create a read-only inspector without loading embedding or ADK models."""

        return KnowledgeStoreInspector(settings.storage.lancedb_database_path)

    @app.get(
        "/inspection/knowledge/summary",
        response_model=KnowledgeStoreSummary,
        tags=["inspection"],
    )
    def inspect_knowledge_summary() -> KnowledgeStoreSummary:
        """Return publisher/topic distribution for the ingested local corpus."""

        try:
            return knowledge_inspector().summary()
        except KnowledgeStoreUnavailable as error:
            raise _missing_runtime(str(error)) from error

    @app.get(
        "/inspection/knowledge/chunks",
        response_model=KnowledgeChunkPage,
        tags=["inspection"],
    )
    def inspect_knowledge_chunks(
        publisher: str | None = Query(default=None, max_length=300),
        topic: str | None = Query(default=None, max_length=100),
        source_id: str | None = Query(default=None, max_length=300),
        chunk_id: str | None = Query(default=None, max_length=500),
        metadata_field: str | None = Query(default=None, max_length=100),
        metadata_value: str | None = Query(default=None, max_length=500),
        min_token_count: int | None = Query(default=None, ge=1),
        max_token_count: int | None = Query(default=None, ge=1),
        offset: int = Query(default=0, ge=0),
        limit: int = Query(default=50, ge=1, le=100),
    ) -> KnowledgeChunkPage:
        """Browse bounded human-readable chunks with optional metadata filters."""

        try:
            return knowledge_inspector().chunks(
                publisher=publisher,
                topic=topic,
                source_id=source_id,
                chunk_id=chunk_id,
                metadata_field=metadata_field,
                metadata_value=metadata_value,
                min_token_count=min_token_count,
                max_token_count=max_token_count,
                offset=offset,
                limit=limit,
            )
        except KnowledgeStoreUnavailable as error:
            raise _missing_runtime(str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    return app


app = create_app(configure_local_runtime=True)
