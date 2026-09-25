"""SQLite persistence for replayable Financial Advisor workflow sessions."""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar, cast
from uuid import UUID

from pydantic import BaseModel

from financial_advisor.domain import (
    AdvisorResearchPlan,
    AnalystTask,
    ClientMessage,
    ClientProfile,
    ClientReview,
    Recommendation,
    ResearchBrief,
    SessionEvent,
    SessionState,
)
from financial_advisor.retrieval.pipeline import TaskEvidenceRetrievalTrace

if TYPE_CHECKING:
    from financial_advisor.workflow import WorkflowSession


ModelT = TypeVar("ModelT", bound=BaseModel)

INSPECTABLE_SQLITE_TABLES = (
    "sessions",
    "session_events",
    "client_messages",
    "research_plans",
    "analyst_tasks",
    "evidence_retrieval_traces",
    "research_briefs",
    "evidence",
    "recommendations",
    "client_reviews",
)


class SessionRepository:
    """Stores and reconstructs the complete trusted state of workflow sessions."""

    def __init__(self, database_path: Path) -> None:
        self._database_path = database_path

    def initialize(self) -> None:
        """Create the SQLite schema if it does not already exist."""

        self._database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    trace_id TEXT NOT NULL UNIQUE,
                    client_profile_json TEXT NOT NULL,
                    state TEXT NOT NULL,
                    follow_up_count INTEGER NOT NULL,
                    active_task_id TEXT,
                    terminal_response TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS session_events (
                    event_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES sessions(session_id),
                    trace_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    actor TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    event_json TEXT NOT NULL,
                    UNIQUE (session_id, sequence)
                );

                CREATE TABLE IF NOT EXISTS client_messages (
                    message_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES sessions(session_id),
                    position INTEGER NOT NULL,
                    message_json TEXT NOT NULL,
                    UNIQUE (session_id, position)
                );

                CREATE TABLE IF NOT EXISTS analyst_tasks (
                    task_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES sessions(session_id),
                    position INTEGER NOT NULL,
                    task_json TEXT NOT NULL,
                    UNIQUE (session_id, position)
                );

                CREATE TABLE IF NOT EXISTS research_plans (
                    plan_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES sessions(session_id),
                    position INTEGER NOT NULL,
                    plan_json TEXT NOT NULL,
                    UNIQUE (session_id, position)
                );

                CREATE TABLE IF NOT EXISTS research_briefs (
                    task_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES sessions(session_id),
                    position INTEGER NOT NULL,
                    brief_json TEXT NOT NULL,
                    UNIQUE (session_id, position)
                );

                CREATE TABLE IF NOT EXISTS evidence_retrieval_traces (
                    task_id TEXT PRIMARY KEY REFERENCES analyst_tasks(task_id),
                    session_id TEXT NOT NULL REFERENCES sessions(session_id),
                    trace_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    recorded_at TEXT NOT NULL,
                    retrieval_trace_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS evidence (
                    evidence_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES sessions(session_id),
                    task_id TEXT NOT NULL REFERENCES analyst_tasks(task_id),
                    evidence_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS recommendations (
                    recommendation_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES sessions(session_id),
                    position INTEGER NOT NULL,
                    recommendation_json TEXT NOT NULL,
                    UNIQUE (session_id, position)
                );

                CREATE TABLE IF NOT EXISTS client_reviews (
                    session_id TEXT NOT NULL REFERENCES sessions(session_id),
                    position INTEGER NOT NULL,
                    review_json TEXT NOT NULL,
                    PRIMARY KEY (session_id, position)
                );

                CREATE INDEX IF NOT EXISTS idx_session_events_trace_sequence
                ON session_events (trace_id, sequence);

                CREATE INDEX IF NOT EXISTS idx_evidence_retrieval_traces_trace_task
                ON evidence_retrieval_traces (trace_id, task_id);
                """
            )
            columns = {
                cast(str, row["name"])
                for row in connection.execute("PRAGMA table_info(sessions)").fetchall()
            }
            if "terminal_response" not in columns:
                connection.execute("ALTER TABLE sessions ADD COLUMN terminal_response TEXT")

    def save(self, session: WorkflowSession) -> None:
        """Atomically persist all known session artifacts without rewriting events."""

        session_id = str(session.session_id)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO sessions (
                    session_id, trace_id, client_profile_json, state,
                    follow_up_count, active_task_id, terminal_response
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    trace_id = excluded.trace_id,
                    client_profile_json = excluded.client_profile_json,
                    state = excluded.state,
                    follow_up_count = excluded.follow_up_count,
                    active_task_id = excluded.active_task_id,
                    terminal_response = excluded.terminal_response,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    session_id,
                    str(session.trace_id),
                    session.client_profile.model_dump_json(),
                    session.state.value,
                    session.follow_up_count,
                    str(session.active_task_id) if session.active_task_id is not None else None,
                    session.terminal_response,
                ),
            )
            self._save_events(connection, session_id, session.events)
            self._save_models(
                connection, "client_messages", "message_id", session_id, session.messages
            )
            self._save_models(
                connection, "research_plans", "plan_id", session_id, session.research_plans
            )
            self._save_models(
                connection, "analyst_tasks", "task_id", session_id, session.analyst_tasks
            )
            self._save_retrieval_traces(
                connection,
                session_id,
                str(session.trace_id),
                session.retrieval_traces,
            )
            self._save_briefs(connection, session_id, session.research_briefs)
            self._save_models(
                connection,
                "recommendations",
                "recommendation_id",
                session_id,
                session.recommendations,
            )
            self._save_reviews(connection, session_id, session.client_reviews)

    def load(self, session_id: UUID) -> WorkflowSession | None:
        """Reconstruct a session and its ordered history from SQLite."""

        from financial_advisor.workflow import WorkflowSession

        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT trace_id, client_profile_json, state, follow_up_count, active_task_id,
                       terminal_response
                FROM sessions WHERE session_id = ?
                """,
                (str(session_id),),
            ).fetchone()
            if row is None:
                return None

            session = WorkflowSession(
                session_id=session_id,
                trace_id=UUID(row["trace_id"]),
                client_profile=ClientProfile.model_validate_json(row["client_profile_json"]),
                state=SessionState(row["state"]),
                follow_up_count=row["follow_up_count"],
                active_task_id=UUID(row["active_task_id"]) if row["active_task_id"] else None,
                terminal_response=row["terminal_response"],
                events=self._load_events(connection, session_id),
                messages=self._load_models(
                    connection, session_id, "client_messages", "message_json", ClientMessage
                ),
                research_plans=self._load_models(
                    connection, session_id, "research_plans", "plan_json", AdvisorResearchPlan
                ),
                analyst_tasks=self._load_models(
                    connection, session_id, "analyst_tasks", "task_json", AnalystTask
                ),
                retrieval_traces=self._load_retrieval_traces(connection, session_id),
                research_briefs=self._load_models(
                    connection, session_id, "research_briefs", "brief_json", ResearchBrief
                ),
                recommendations=self._load_models(
                    connection,
                    session_id,
                    "recommendations",
                    "recommendation_json",
                    Recommendation,
                ),
                client_reviews=self._load_models(
                    connection, session_id, "client_reviews", "review_json", ClientReview
                ),
            )
            session.evidence_ids = {
                evidence.evidence_id
                for brief in session.research_briefs
                for evidence in brief.evidence
            }
            return session

    def load_by_trace_id(self, trace_id: UUID) -> WorkflowSession | None:
        """Load the one root workflow session associated with a trace identifier."""

        with self._connect() as connection:
            row = connection.execute(
                "SELECT session_id FROM sessions WHERE trace_id = ?", (str(trace_id),)
            ).fetchone()
        return self.load(UUID(row["session_id"])) if row is not None else None

    def list_sessions(self, *, limit: int = 50) -> list[WorkflowSession]:
        """Return recent sessions for local replay selection without exposing SQL to routes."""

        if not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100.")
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT session_id FROM sessions
                ORDER BY updated_at DESC, created_at DESC, session_id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        sessions: list[WorkflowSession] = []
        for row in rows:
            session = self.load(UUID(row["session_id"]))
            if session is not None:
                sessions.append(session)
        return sessions

    def inspect_tables(self) -> dict[str, int]:
        """Return allowlisted SQLite table row counts for the local inspector."""

        with self._connect() as connection:
            return {
                table_name: cast(
                    int,
                    connection.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0],
                )
                for table_name in INSPECTABLE_SQLITE_TABLES
            }

    def inspect_table_rows(
        self, table_name: str, *, limit: int = 50
    ) -> tuple[list[str], list[dict[str, object]]]:
        """Return a bounded raw table view from a fixed safe allowlist."""

        if table_name not in INSPECTABLE_SQLITE_TABLES:
            raise ValueError("Table is not available for inspection.")
        if not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100.")
        with self._connect() as connection:
            columns = [
                cast(str, row["name"])
                for row in connection.execute(f"PRAGMA table_info({table_name})").fetchall()
            ]
            rows = connection.execute(f"SELECT * FROM {table_name} LIMIT ?", (limit,)).fetchall()
        return columns, [dict(row) for row in rows]

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    @staticmethod
    def _save_events(
        connection: sqlite3.Connection, session_id: str, events: list[SessionEvent]
    ) -> None:
        connection.executemany(
            """
            INSERT OR IGNORE INTO session_events (
                event_id, session_id, trace_id, sequence, actor, event_type,
                summary, created_at, event_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    str(event.event_id),
                    session_id,
                    str(event.trace_id),
                    event.sequence,
                    event.actor,
                    event.event_type.value,
                    event.summary,
                    event.created_at.isoformat(),
                    event.model_dump_json(),
                )
                for event in events
            ],
        )

    @staticmethod
    def _save_models(
        connection: sqlite3.Connection,
        table: str,
        id_column: str,
        session_id: str,
        models: (
            Sequence[ClientMessage]
            | Sequence[AdvisorResearchPlan]
            | Sequence[AnalystTask]
            | Sequence[Recommendation]
        ),
    ) -> None:
        payload_column = {
            "client_messages": "message_json",
            "research_plans": "plan_json",
            "analyst_tasks": "task_json",
            "recommendations": "recommendation_json",
        }[table]
        connection.executemany(
            f"""
            INSERT OR IGNORE INTO {table} ({id_column}, session_id, position, {payload_column})
            VALUES (?, ?, ?, ?)
            """,
            [
                (str(getattr(model, id_column)), session_id, position, model.model_dump_json())
                for position, model in enumerate(models, start=1)
            ],
        )

    @staticmethod
    def _save_briefs(
        connection: sqlite3.Connection, session_id: str, briefs: list[ResearchBrief]
    ) -> None:
        connection.executemany(
            """
            INSERT OR IGNORE INTO research_briefs (task_id, session_id, position, brief_json)
            VALUES (?, ?, ?, ?)
            """,
            [
                (str(brief.task_id), session_id, position, brief.model_dump_json())
                for position, brief in enumerate(briefs, start=1)
            ],
        )
        connection.executemany(
            """
            INSERT OR IGNORE INTO evidence (evidence_id, session_id, task_id, evidence_json)
            VALUES (?, ?, ?, ?)
            """,
            [
                (str(item.evidence_id), session_id, str(brief.task_id), item.model_dump_json())
                for brief in briefs
                for item in brief.evidence
            ],
        )

    @staticmethod
    def _save_retrieval_traces(
        connection: sqlite3.Connection,
        session_id: str,
        trace_id: str,
        retrieval_traces: Sequence[TaskEvidenceRetrievalTrace],
    ) -> None:
        connection.executemany(
            """
            INSERT OR IGNORE INTO evidence_retrieval_traces (
                task_id, session_id, trace_id, status, recorded_at, retrieval_trace_json
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    str(item.task_id),
                    session_id,
                    trace_id,
                    item.retrieval.status.value,
                    item.recorded_at.isoformat(),
                    item.model_dump_json(),
                )
                for item in retrieval_traces
            ],
        )

    @staticmethod
    def _save_reviews(
        connection: sqlite3.Connection, session_id: str, reviews: list[ClientReview]
    ) -> None:
        connection.executemany(
            """
            INSERT OR IGNORE INTO client_reviews (session_id, position, review_json)
            VALUES (?, ?, ?)
            """,
            [
                (session_id, position, review.model_dump_json())
                for position, review in enumerate(reviews, start=1)
            ],
        )

    @staticmethod
    def _load_models(
        connection: sqlite3.Connection,
        session_id: UUID,
        table: str,
        payload_column: str,
        model_type: type[ModelT],
    ) -> list[ModelT]:
        rows = connection.execute(
            f"SELECT {payload_column} FROM {table} WHERE session_id = ? ORDER BY position",
            (str(session_id),),
        ).fetchall()
        return [model_type.model_validate_json(row[payload_column]) for row in rows]

    @staticmethod
    def _load_retrieval_traces(
        connection: sqlite3.Connection, session_id: UUID
    ) -> list[TaskEvidenceRetrievalTrace]:
        rows = connection.execute(
            """
            SELECT retrieval_trace_json FROM evidence_retrieval_traces
            WHERE session_id = ? ORDER BY recorded_at, task_id
            """,
            (str(session_id),),
        ).fetchall()
        return [
            TaskEvidenceRetrievalTrace.model_validate_json(row["retrieval_trace_json"])
            for row in rows
        ]

    @staticmethod
    def _load_events(connection: sqlite3.Connection, session_id: UUID) -> list[SessionEvent]:
        rows = connection.execute(
            """
            SELECT event_json FROM session_events
            WHERE session_id = ? ORDER BY sequence
            """,
            (str(session_id),),
        ).fetchall()
        return [SessionEvent.model_validate_json(row["event_json"]) for row in rows]
