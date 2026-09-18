# Local UI

The React UI is a local application and observability layer. It does not make
agent decisions or reconstruct workflow state in the browser. The FastAPI API and
SQLite repository remain the trusted execution and replay boundaries.

## Run locally

In one terminal, start the API:

```bash
uv run uvicorn financial_advisor.main:app --reload --host 127.0.0.1 --port 8000
```

In another terminal, start the React development server:

```bash
cd frontend
npm ci
npm run dev
```

Open `http://127.0.0.1:5173`.

Vite proxies `/api/*` to FastAPI at `http://127.0.0.1:8000`, so browser calls stay
same-origin during local development without weakening the API with broad CORS rules.

## What the UI shows

- **Conversation history**: recent persisted sessions from `GET /sessions`.
- **Synthetic client profile**: the scenario profile being used by the Client Agent.
- **Conversation**: Client messages and Advisor recommendations from persisted state.
- **Workflow**: safe, ordered persisted workflow updates associated with the
  current trace. These include deterministic event summaries and constrained
  Advisor progress summaries; they are not hidden chain-of-thought.
- **Validated sources**: server-created citations from the current Advisor recommendation.
- **Safe terminal outcomes**: an escalated scenario clearly explains that no unsupported
  recommendation was generated and directs the reviewer to the recorded timeline.
- **Observability**: read-only pages for **Trace Explorer**, **SQLite Store**,
  and **Knowledge Store**. The Knowledge Store browses LanceDB chunks with
  publisher, topic, metadata, and token-window filters. It intentionally omits
  embedding vectors.

## Live update and replay

Starting Maya’s scenario calls `POST /sessions`. The API returns the identifiers
immediately, and the UI opens the trace SSE endpoint. For each new persisted event,
the UI reloads the current session snapshot from FastAPI. This keeps the browser as a
viewer of trusted server state rather than a second workflow engine.

Before the first validated Client message arrives, the conversation shows a
frontend-only “Maya is preparing her question…” placeholder. It is not a stored
message and is replaced as soon as the persisted Client opening message arrives.

When the Advisor delegates or reassesses research, its structured planning
output supplies one constrained, client-readable `progress_summary`. The server
persists it as `client_progress_created`, SSE delivers it, and the conversation
renders it below the active Client message. It describes the next review step
without exposing hidden reasoning, retrieval internals, or an unestablished
recommendation.

If the workflow reaches `ESCALATED`, the response is different: the server
persists a safe terminal `terminal_response` and the conversation renders that
Advisor message. It explains that an evidence-backed recommendation was not
generated and points the Client to an appropriate next step; it is not generated
by a fallback LLM call.

Selecting an existing session does not rerun agents. It reads the completed or
in-progress session from SQLite for replay.

## Failure-path visibility

The same replay surfaces show normal and failure trajectories. For example, a
local-evidence miss records its retrieval result and the Advisor's one permitted
web-enabled replan; a model-synthesis failure records `tool_failed` and
`session_escalated`. The Workflow panel renders these durable events as they arrive.
The Observability pages can then show the underlying `session_events`,
`evidence_retrieval_traces`, and related rows without offering arbitrary SQL.
