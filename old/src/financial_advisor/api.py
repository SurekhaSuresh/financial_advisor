"""Transport-neutral API execution and read-model helpers.

FastAPI routes call these small application services; agent decisions and workflow
transitions remain in their existing modules.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Protocol
from uuid import UUID

from pydantic import BaseModel, Field

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
from financial_advisor.persistence import SessionRepository
from financial_advisor.retrieval.pipeline import TaskEvidenceRetrievalTrace
from financial_advisor.workflow import (
    GENERIC_ESCALATION_RESPONSE,
    WorkflowEngine,
    WorkflowSession,
)


class ScenarioService(Protocol):
    """The asynchronous scenario behavior exposed by the API layer."""

    async def run(
        self, profile: ClientProfile, *, session: WorkflowSession | None = None
    ) -> WorkflowSession:
        """Run a complete bounded scenario, optionally from an already-created session."""


class SessionSnapshot(BaseModel):
    """Complete replayable session payload returned by inspection endpoints."""

    session_id: UUID
    trace_id: UUID
    client_profile: ClientProfile
    state: SessionState
    terminal_response: str | None
    follow_up_count: int = Field(ge=0)
    active_task_id: UUID | None
    events: list[SessionEvent]
    messages: list[ClientMessage]
    research_plans: list[AdvisorResearchPlan]
    analyst_tasks: list[AnalystTask]
    retrieval_traces: list[TaskEvidenceRetrievalTrace]
    research_briefs: list[ResearchBrief]
    recommendations: list[Recommendation]
    client_reviews: list[ClientReview]

    @classmethod
    def from_session(cls, session: WorkflowSession) -> SessionSnapshot:
        """Create a response model without exposing the workflow's internal evidence-ID set."""

        return cls(
            session_id=session.session_id,
            trace_id=session.trace_id,
            client_profile=session.client_profile,
            state=session.state,
            terminal_response=_terminal_response(session),
            follow_up_count=session.follow_up_count,
            active_task_id=session.active_task_id,
            events=session.events,
            messages=session.messages,
            research_plans=session.research_plans,
            analyst_tasks=session.analyst_tasks,
            retrieval_traces=session.retrieval_traces,
            research_briefs=session.research_briefs,
            recommendations=session.recommendations,
            client_reviews=session.client_reviews,
        )


def _terminal_response(session: WorkflowSession) -> str | None:
    """Return server-owned client wording for a terminal safe-failure outcome."""

    if session.state is not SessionState.ESCALATED:
        return None
    return session.terminal_response or GENERIC_ESCALATION_RESPONSE


class ScenarioStarted(BaseModel):
    """Stable identifiers returned immediately after a scenario is scheduled."""

    session_id: UUID
    trace_id: UUID
    state: SessionState


class SessionSummary(BaseModel):
    """Small session record used by the local conversation-history sidebar."""

    session_id: UUID
    trace_id: UUID
    client_name: str
    primary_goal: str
    opening_question: str | None
    started_at: datetime
    state: SessionState

    @classmethod
    def from_session(cls, session: WorkflowSession) -> SessionSummary:
        """Create a concise session-list entry from trusted persisted state."""

        return cls(
            session_id=session.session_id,
            trace_id=session.trace_id,
            client_name=session.client_profile.name,
            primary_goal=session.client_profile.primary_goal,
            opening_question=session.messages[0].text if session.messages else None,
            started_at=session.events[0].created_at,
            state=session.state,
        )


class TraceEvents(BaseModel):
    """Ordered events for a replayable workflow trace."""

    trace_id: UUID
    events: list[SessionEvent]


class SqliteTableSummary(BaseModel):
    """One allowlisted SQLite table and its current row count."""

    table_name: str
    row_count: int = Field(ge=0)


class SqliteTableRows(BaseModel):
    """Bounded raw SQLite row values for local development inspection."""

    table_name: str
    columns: list[str]
    rows: list[dict[str, object]]


class ScenarioExecutionManager:
    """Start one persisted scenario and retain its local background task for inspection."""

    def __init__(self, workflow: WorkflowEngine, scenario: ScenarioService) -> None:
        self._workflow = workflow
        self._scenario = scenario
        self._tasks: dict[UUID, asyncio.Task[WorkflowSession]] = {}

    def start(self, profile: ClientProfile) -> ScenarioStarted:
        """Persist a session first, then schedule its agent conversation in the background."""

        session = self._workflow.start_session(profile)
        task = asyncio.create_task(self._scenario.run(profile, session=session))
        self._tasks[session.session_id] = task
        task.add_done_callback(lambda completed: self._complete_task(session, completed))
        return ScenarioStarted(
            session_id=session.session_id,
            trace_id=session.trace_id,
            state=session.state,
        )

    def _complete_task(self, session: WorkflowSession, task: asyncio.Task[WorkflowSession]) -> None:
        """Collect task failures and persist a safe terminal outcome for unexpected ones."""

        self._tasks.pop(session.session_id, None)
        if task.cancelled():
            return
        try:
            task.result()
        except Exception:
            self._workflow.escalate_unexpected_runtime_failure(session)


class TraceEventStream:
    """Poll persisted ordered events so any UI can replay an active local trace."""

    def __init__(
        self, repository: SessionRepository, *, poll_interval_seconds: float = 0.25
    ) -> None:
        if poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be greater than zero.")
        self._repository = repository
        self._poll_interval_seconds = poll_interval_seconds

    async def events(
        self, trace_id: UUID, *, after_sequence: int = 0
    ) -> AsyncIterator[SessionEvent]:
        """Yield new persisted events until the associated session reaches a terminal state."""

        if after_sequence < 0:
            raise ValueError("after_sequence must not be negative.")
        next_sequence = after_sequence + 1
        while True:
            session = self._repository.load_by_trace_id(trace_id)
            if session is None:
                return
            for event in session.events:
                if event.sequence >= next_sequence:
                    yield event
                    next_sequence = event.sequence + 1
            if session.state in {SessionState.RESOLVED, SessionState.ESCALATED}:
                return
            await asyncio.sleep(self._poll_interval_seconds)
