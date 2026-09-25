# Financial Advisor

Financial Advisor is a local multi-agent implementation of a simulated financial-advisory conversation. A Client Agent, Advisor Agent, and Analyst Agent collaborate toward an evidence-backed resolution.

## Core workflow

```text
Client Agent → Advisor Agent → Analyst Agent
                 ↑               │
                 └── cited brief ┘
                      │
                      ▼
                 Advisor Agent → Client Agent
```

The Advisor is the only communication bridge. The Client and Analyst cannot communicate directly.

The Advisor authorizes the required deterministic retrieval operations; the
application retrieves, validates, and selects evidence before the Analyst
synthesizes a structured brief. The Advisor is then the only agent that turns
that brief into a client-facing, cited educational recommendation.

The included scenario simulates Maya Chen balancing a five-year home-purchase
goal, student-loan obligations, savings, and a moderate risk tolerance.

## Tech stack

| Component | Technology | Use in Financial Advisor |
| --- | --- | --- |
| Runtime | Python 3.12 | Local application runtime managed with `uv`. |
| API | FastAPI + Pydantic Settings | Provides scenario start, session/trace inspection, SSE trace replay, and environment-backed settings. |
| UI | React + TypeScript + Vite | Runs the synthetic scenario, presents citations, and replays persisted trace events. |
| Domain contracts | Pydantic v2 | Validates the agent messages, research artifacts, evidence, recommendations, and trace events. |
| Workflow | Custom Python state machine | Enforces allowed session transitions, agent-action ordering, evidence citations, and the two-follow-up limit. |
| Persistence | SQLite via Python standard library | Stores replayable sessions, ordered events, messages, research artifacts, evidence, recommendations, and client reviews. |
| Knowledge ingestion | LanceDB + BGE small embeddings | Builds a local vector table from approved Investor.gov, FINRA, CFPB, and FDIC source snapshots. |
| Quality checks | pytest, Ruff, mypy | Runs focused tests, integration tests, linting, and static type checks. |

## Capability coverage

| Required capability | Implementation |
| --- | --- |
| Simulated Client with a financial profile | A typed synthetic profile includes age, risk tolerance, savings, debt, emergency fund, goal, and time horizon. The Client LLM opens and reviews a bounded conversation. |
| Advisor as the only communication bridge | The Workflow Engine permits Client ↔ Advisor and Advisor ↔ Analyst exchanges only. The Advisor plans research, receives the Analyst brief, and writes the Client response. |
| Analyst access to internet and knowledge store | An Advisor-authorized plan triggers LanceDB hybrid retrieval and, when needed, Exa/Brave live web discovery plus original-page processing. The Analyst receives only selected, source-attributed evidence. |
| Agents work toward a resolution | The bounded runner persists every transition and ends in `RESOLVED` when the Client accepts or `ESCALATED` when safe evidence-backed guidance cannot be produced. |
| Prompting choices and tradeoffs | Typed, least-privilege prompts, structured output, evidence-ID grounding, and deterministic enforcement are documented in [Prompt engineering](docs/prompt-engineering.md). |
| Framework assumptions and first-principles boundaries | [Framework responsibilities](docs/framework-assumptions.md) and [architecture decisions](docs/architecture-decisions.md) distinguish framework-provided plumbing from application-owned policy and safety. |
| Pythonic quality controls | Typed Pydantic contracts, focused modules, dependency injection at API/runtime boundaries, pytest, Ruff, mypy, and deterministic integration tests. |

## Project organization

```text
src/financial_advisor/
├─ config.py                 environment settings and validated operating policies
├─ domain.py                 Pydantic business contracts
├─ workflow.py               deterministic agent-permission state machine
├─ scenario.py               bounded Client → Advisor → Analyst → Advisor runner
├─ persistence.py            SQLite session and trajectory repository
├─ api.py                    scenario scheduling and persisted trace-stream helpers
├─ main.py                   FastAPI routes and application factory
├─ agents/
│  ├─ advisor/               trusted plan creation, ADK planning/response, and prompt
│  ├─ analyst/               required evidence retrieval, ADK synthesis, and prompt
│  └─ client/                ADK synthetic Client opening and bounded-review roles
└─ retrieval/
   ├─ content_processing/    shared HTML/PDF parsing and token-window chunking
   ├─ knowledge_base/        stable-source ingestion and LanceDB retrievers
   ├─ web/                   live discovery, provider adapters, fetching, page processing,
   │                         passage ranking, and live-web pipeline
   ├─ contracts.py           normalized retrieval candidates and trace models
   ├─ reranker.py            local ONNX cross-encoder adapter
   ├─ selection.py           RRF fusion, reranking, and MMR selection
   ├─ evidence.py            selected candidate to clean Analyst citation conversion
   └─ pipeline.py            three-channel evidence coordinator
frontend/                    React scenario runner and trace-replay interface
scripts/                     source snapshot fetching and LanceDB ingestion commands
data/                        versioned source manifest; generated local data is ignored by Git
```

`config.py` distinguishes deployment settings from operating policy. API keys,
models, and paths come from the environment. Validated policy objects centralize
retrieval limits, chunk sizes, retry behavior, timeouts, circuit-breaker values,
workflow follow-up limits, source-fetch behavior, and HTTP client identity.
Safety invariants such as allowed URL schemes and supported page content types
remain enforced in code. Nested environment overrides use double underscores,
for example `WEB_RESEARCH__PAGE_TIMEOUT_SECONDS=15`. Provider base URLs are
also configurable; provider-specific API paths and request/response translation
remain in their adapters.

## Run locally

### 1. Install dependencies and configure local credentials

```bash
uv sync --all-groups
cp .env.example .env
```

Set non-empty values for `GOOGLE_API_KEY`, `EXA_API_KEY`, and
`BRAVE_SEARCH_API_KEY` in `.env`. Do not commit `.env`.

### 2. Build the local knowledge base

The public repository contains the source manifest, not the downloaded source
snapshots or generated LanceDB table. Fetch the approved snapshots, then ingest
them once. Re-run ingestion whenever the chunking or embedding policy changes.

```bash
uv run python scripts/fetch_knowledge_sources.py
HF_HOME="$PWD/.local/model_cache" XDG_CACHE_HOME="$PWD/.local/cache" \
  HF_HUB_DISABLE_XET=1 uv run python scripts/ingest_knowledge.py
```

If `uv` is unavailable on a local macOS setup after creating `.venv`, the same
script can be run with `.venv/bin/python scripts/ingest_knowledge.py`.

### 3. Start the API

```bash
uv run uvicorn financial_advisor.main:app --reload --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/health` to verify the application process. FastAPI's
interactive API contract is available at `http://127.0.0.1:8000/docs`.

The API supports session and trace inspection, including an SSE event stream. See
[API and trace replay](docs/api.md) for its endpoint contracts and replay model.

### 4. Start the UI and run the scenario

In a second terminal:

```bash
cd frontend
npm ci
npm run dev
```

Open `http://127.0.0.1:5173`. The Vite development server proxies API calls to
FastAPI at `http://127.0.0.1:8000`. Select **Start Scenario** in the left sidebar.
The UI immediately receives a session ID and trace ID, streams persisted
workflow events, shows a concise client-readable Advisor progress update, and
renders the completed Client/Advisor conversation. Use
**Observability** to inspect the same session, trace, SQLite records, and
LanceDB chunk metadata without rerunning the scenario.

If one of the local runtime prerequisites is absent, `POST /sessions` returns a clear
`503`; health and inspection of existing SQLite sessions remain available.

The approved-source manifest contains 50 public sources. The generated chunk
count is intentionally not committed because it changes when the manifest,
parser, chunking policy, or embedding model changes. Inspect the current local
corpus through the Knowledge Store UI or `GET /inspection/knowledge/summary`.

## What starts and what runs

When Uvicorn starts the backend, FastAPI runs its application lifespan:

```text
Uvicorn → FastAPI application → initialize SQLite inspection repository
        → construct the local runtime (LanceDB retrievers, reranker,
          web providers, and typed Client/Advisor/Analyst services)
        → accept HTTP requests
```

Starting a scenario does not hold the browser request open. `POST /sessions`
persists a new session, returns its session and trace IDs with `202 Accepted`,
and schedules the bounded runner. The runner executes Client → Advisor →
deterministic retrieval (when authorized) → Analyst → Advisor → Client review.
The UI follows the same persisted events through SSE until the session resolves
or escalates safely.

## Request the API directly

The React UI is the normal interactive path, but the FastAPI API is also independently
usable. With the backend running, this starts the same scenario lifecycle with
a synthetic profile:

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

The response contains `session_id` and `trace_id`. Retrieve the durable replay
record with `GET /sessions/{session_id}`, or follow new workflow events with
`GET /traces/{trace_id}/events/stream`. See [API and trace replay](docs/api.md)
for all endpoints and request details.

## Verify the project

```bash
uv run pytest
uv run ruff check .
uv run mypy src
cd frontend && npm run build
```

The scenario tests run the real Scenario Runner, coordinators, workflow state
machine, and SQLite persistence with scripted model/retrieval boundaries. They
prove core three-agent orchestration without requiring live Gemini or
web-provider availability.

Run the deterministic integration suite separately when changing workflow or
retrieval behavior:

```bash
uv run pytest tests/integration/test_workflow_integration.py
```

Those tests use a temporary LanceDB corpus and SQLite database, real vector and
BM25 retrieval, RRF, cross-encoder reranking, MMR, page parsing/chunking, and
the real Scenario Runner. They replace only live Gemini decisions and external
Exa/Brave HTTP responses with typed scripted fixtures. They cover a complete
resolution, one bounded refinement, and automatic closure after the Client's
two-follow-up budget.

Tests are organized by scope: `tests/unit/` contains focused component and
contract tests; `tests/integration/` contains API, persistence, scenario, and
end-to-end workflow tests.

Use the live UI scenario separately to validate the non-deterministic boundary:
actual Gemini decisions, live Exa/Brave and original-page requests, SSE event
delivery, and browser rendering.

## Documentation

- [Domain contracts and workflow states](docs/domain-contracts.md)
- [Architecture decisions and operating constraints](docs/architecture-decisions.md)
- [Framework responsibilities and assumptions](docs/framework-assumptions.md)
- [Model routing](docs/model-routing.md)
- [Prompt engineering and agent boundaries](docs/prompt-engineering.md)
- [Advisor Agent](docs/advisor-agent.md)
- [Analyst Agent](docs/analyst-agent.md)
- [Client Agent](docs/client-agent.md)
- [API and trace replay](docs/api.md)
- [Local UI](docs/ui.md)
- [SQLite and LanceDB inspection](docs/inspection.md)
- [Knowledge-source manifest](data/knowledge_sources.yaml)
- [Knowledge ingestion](docs/knowledge-ingestion.md)
- [Local retrieval](docs/retrieval.md)

## Principles

- Synthetic data only
- Evidence-backed recommendations
- Explicit agent permissions and bounded follow-ups
- No trade execution or brokerage access
- Transparent assumptions, risks, and limitations
