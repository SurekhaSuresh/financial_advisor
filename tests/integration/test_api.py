"""Tests for observable FastAPI session and trace endpoints."""

import asyncio
from decimal import Decimal
from pathlib import Path
from uuid import UUID

import httpx

from financial_advisor.api import ScenarioExecutionManager, TraceEventStream
from financial_advisor.domain import (
    AdvisorDecision,
    AdvisorDecisionType,
    ClientMessage,
    ClientProfile,
    RiskTolerance,
    SessionState,
)
from financial_advisor.main import create_app
from financial_advisor.persistence import SessionRepository
from financial_advisor.workflow import WorkflowEngine


def make_profile() -> ClientProfile:
    """Return the synthetic profile accepted by the scenario-start endpoint."""

    return ClientProfile(
        client_id="maya-chen",
        name="Maya Chen",
        age=38,
        risk_tolerance=RiskTolerance.MODERATE,
        emergency_fund_months=6,
        retirement_savings=Decimal("120000.00"),
        brokerage_savings=Decimal("35000.00"),
        student_loan_balance=Decimal("18000.00"),
        student_loan_rate_percent=Decimal("5.8"),
        primary_goal="Buy a home",
        goal_time_horizon_years=5,
    )


def saved_session(engine: WorkflowEngine):  # type: ignore[no-untyped-def]
    """Persist a nonterminal trace so API reads have ordered event data to return."""

    session = engine.start_session(make_profile())
    engine.submit_opening_message(
        session,
        ClientMessage(
            session_id=session.session_id,
            text="How should I use my $15,000 bonus?",
            follow_up_number=0,
        ),
    )
    return session


class FakeScenarioService:
    """Completes the pre-created session without invoking external models in API tests."""

    async def run(self, profile, *, session=None):  # type: ignore[no-untyped-def]
        assert session is not None
        return session


class FailingScenarioService:
    """Raises unexpectedly to verify the API execution boundary remains inspectable."""

    async def run(self, profile, *, session=None):  # type: ignore[no-untyped-def]
        raise RuntimeError("Unexpected test failure")


def test_session_and_trace_endpoints_return_persisted_replay_data(tmp_path: Path) -> None:
    """The API resolves both session and trace identifiers to the stored trajectory."""

    repository = SessionRepository(tmp_path / "financial_advisor.db")
    session = saved_session(WorkflowEngine(repository))
    app = create_app(repository=repository)

    async def request_data() -> tuple[
        httpx.Response, httpx.Response, httpx.Response, httpx.Response
    ]:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return (
                await client.get(f"/sessions/{session.session_id}"),
                await client.get("/sessions"),
                await client.get(f"/traces/{session.trace_id}"),
                await client.get(f"/traces/{session.trace_id}/events"),
            )

    session_response, list_response, trace_response, events_response = asyncio.run(request_data())

    assert session_response.status_code == 200
    assert session_response.json()["state"] == "advisor_assesses"
    assert list_response.status_code == 200
    assert list_response.json()[0]["session_id"] == str(session.session_id)
    assert list_response.json()[0]["opening_question"] == "How should I use my $15,000 bonus?"
    assert list_response.json()[0]["started_at"]
    assert trace_response.status_code == 200
    assert trace_response.json()["session_id"] == str(session.session_id)
    assert events_response.status_code == 200
    assert [event["sequence"] for event in events_response.json()["events"]] == list(
        range(1, len(session.events) + 1)
    )


def test_start_scenario_returns_persisted_identifiers_immediately(tmp_path: Path) -> None:
    """An API caller can subscribe to a known trace while a scenario task is scheduled."""

    repository = SessionRepository(tmp_path / "financial_advisor.db")
    workflow = WorkflowEngine(repository)
    app = create_app(
        repository=repository,
        execution_manager=ScenarioExecutionManager(workflow, FakeScenarioService()),
    )

    async def start() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post("/sessions", json=make_profile().model_dump(mode="json"))

    response = asyncio.run(start())

    assert response.status_code == 202
    payload = response.json()
    assert payload["state"] == "new"
    assert repository.load_by_trace_id(payload["trace_id"]) is not None


def test_unexpected_background_scenario_failure_becomes_a_safe_terminal_session(
    tmp_path: Path,
) -> None:
    """Unexpected task errors do not leave an active persisted session stranded."""

    repository = SessionRepository(tmp_path / "financial_advisor.db")
    manager = ScenarioExecutionManager(WorkflowEngine(repository), FailingScenarioService())

    async def start_and_wait() -> UUID:
        started = manager.start(make_profile())
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        return started.session_id

    session = repository.load(asyncio.run(start_and_wait()))

    assert session is not None
    assert session.state is SessionState.ESCALATED
    assert session.terminal_response is not None
    assert session.events[-1].event_type.value == "session_escalated"


def test_unconfigured_execution_route_is_explicit(tmp_path: Path) -> None:
    """A health-only application never pretends it can run agent scenarios."""

    app = create_app(repository=SessionRepository(tmp_path / "financial_advisor.db"))

    async def start() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post("/sessions", json=make_profile().model_dump(mode="json"))

    response = asyncio.run(start())

    assert response.status_code == 503


def test_event_stream_resumes_after_a_seen_sequence_and_stops_at_terminal_state(
    tmp_path: Path,
) -> None:
    """SSE support can replay only unseen durable events after a reconnect."""

    repository = SessionRepository(tmp_path / "financial_advisor.db")
    workflow = WorkflowEngine(repository)
    session = saved_session(workflow)
    workflow.escalate_advisor_assessment(session, reason="Test terminal outcome.")

    async def collect_unseen_sequences() -> list[int]:
        return [
            event.sequence
            async for event in TraceEventStream(repository, poll_interval_seconds=0.001).events(
                session.trace_id,
                after_sequence=1,
            )
        ]

    assert asyncio.run(collect_unseen_sequences()) == list(range(2, len(session.events) + 1))


def test_session_endpoint_returns_the_persisted_client_safe_escalation_response(
    tmp_path: Path,
) -> None:
    """The UI receives a safe response, not only an internal escalation event."""

    repository = SessionRepository(tmp_path / "financial_advisor.db")
    workflow = WorkflowEngine(repository)
    session = saved_session(workflow)
    workflow.apply_advisor_decision(
        session,
        AdvisorDecision(
            decision_type=AdvisorDecisionType.ESCALATE,
            summary="Specific-security advice requires professional review.",
        ),
    )
    app = create_app(repository=repository)

    async def get_session() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.get(f"/sessions/{session.session_id}")

    response = asyncio.run(get_session())

    assert response.status_code == 200
    assert "can’t recommend whether to buy or sell" in response.json()["terminal_response"]
