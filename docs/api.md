# API and Trace Replay

The FastAPI layer is a transport boundary. It does not make financial decisions
or call retrieval tools directly. It schedules a scenario through the runtime;
`ScenarioRunner` coordinates the three agents, and `WorkflowEngine` validates
and persists every transition.

## Endpoints

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Confirms that the API process is available. |
| `POST /sessions` | Starts a Client-led scenario and returns `202 Accepted` with a stable session ID and trace ID. |
| `GET /sessions` | Lists recent persisted conversations for local replay. |
| `GET /sessions/{session_id}` | Returns the persisted session and every replay artifact. |
| `GET /traces/{trace_id}` | Resolves a trace ID to the same replayable root session. |
| `GET /traces/{trace_id}/events` | Returns the complete ordered event history for investigation. |
| `GET /traces/{trace_id}/events/stream` | Emits new persisted events as Server-Sent Events until the scenario resolves or escalates. |
| `GET /inspection/sqlite/tables` | Lists allowlisted SQLite tables and counts for local inspection. |
| `GET /inspection/sqlite/tables/{table_name}/rows` | Returns up to 100 rows from one allowlisted SQLite table. |
| `GET /inspection/knowledge/summary` | Summarizes the curated LanceDB corpus. |
| `GET /inspection/knowledge/chunks` | Browses source-attributed knowledge chunks with optional metadata filters. |

FastAPI also serves this API contract interactively at `GET /docs` while the
local server is running.

## Start and inspect a scenario without the UI

`POST /sessions` accepts a validated synthetic `ClientProfile`. This example
starts the same asynchronous lifecycle that the React UI starts:

```bash
curl -X POST http://127.0.0.1:8000/sessions \
  -H 'content-type: application/json' \
  -d '{
    "client_id":"maya-chen",
    "name":"Maya Chen",
    "age":38,
    "risk_tolerance":"moderate",
    "emergency_fund_months":6,
    "retirement_savings":"120000.00",
    "brokerage_savings":"35000.00",
    "student_loan_balance":"18000.00",
    "student_loan_rate_percent":"5.8",
    "primary_goal":"Buy a home",
    "goal_time_horizon_years":5
  }'
```

The `202 Accepted` response contains `session_id` and `trace_id`. Substitute
either returned identifier in the following read-only calls:

```bash
curl http://127.0.0.1:8000/sessions/{session_id}
curl http://127.0.0.1:8000/traces/{trace_id}/events
curl -N http://127.0.0.1:8000/traces/{trace_id}/events/stream
```

`GET /sessions` lists recent persisted conversations. All API routes are
documented in the local OpenAPI UI at `http://127.0.0.1:8000/docs`.

## Why return IDs immediately?

An agent scenario can take longer than a normal HTTP request because it may run local
retrieval, live web research, and several bounded model calls. The API creates and
persists the session before scheduling the runner. The caller therefore receives the
session and trace identifiers immediately, then can read or stream the durable event
history while the scenario proceeds.

## Replay and observability

SQLite is the source of truth for a completed trajectory. Events are ordered by a
per-session sequence number and indexed by trace ID. The SSE endpoint polls that
persisted record; it does not stream unvalidated model tokens or transient in-memory
agent state. A UI can reconnect with `after_sequence` and continue from the next event
without replaying events it has already displayed.

Expected failure paths are durable, inspectable workflow outcomes rather than hidden
logs. An insufficient local-only retrieval is stored before the Advisor receives one
bounded replan opportunity. An Analyst synthesis outage stores the completed
retrieval trace, then records `tool_failed` and `session_escalated`. API clients can
inspect either outcome through the session snapshot, trace-event endpoints, or the
allowlisted SQLite inspector.

If an unexpected exception escapes the bounded runner itself, the API task-completion
boundary consumes it and records a generic `session_escalated` outcome. This prevents
a persisted session from remaining indefinitely active while keeping internal error
details out of the Client response and replay payload.

## Runtime composition

`create_app()` accepts explicit repository and scenario-execution dependencies. This
makes health checks and API tests independent of model credentials, a LanceDB corpus,
and live web providers. The module-level application builds the real local runtime at
startup: SQLite repository, workflow engine, Client/Advisor/Analyst services, LanceDB
retrievers, reranker, and Exa/Brave live research pipeline. SQLite inspection starts
first, so previously completed sessions remain viewable even when live setup is
incomplete. The live runtime validates the Gemini, Exa, and Brave keys plus the local
LanceDB directory; a configuration problem is explicit rather than causing the API to
pretend that a new scenario can run.
