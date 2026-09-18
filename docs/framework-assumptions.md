# Framework Responsibilities and Assumptions

Frameworks reduce implementation work, but they do not define this application's
business rules. This note makes that boundary explicit.

## FastAPI

FastAPI provides HTTP routing, request parsing, response serialization, OpenAPI
documentation, and the application lifespan hook. It does not decide whether a
financial question is safe, invoke retrieval, run agent workflow transitions,
or create evidence citations.

The application uses FastAPI only as a transport boundary. `ScenarioRunner` and
`WorkflowEngine` own orchestration and transition validation; the API returns a
persisted session/trace identifier immediately and streams persisted events
through SSE.

## Pydantic

Pydantic validates types, field bounds, and model-level invariants at process
boundaries. It does not establish factual correctness, citation entailment, or
financial suitability.

The application therefore uses Pydantic for typed contracts and adds separate
deterministic rules for state transitions, evidence identity, source policy,
claim filtering, and terminal safe failures.

## Google ADK and Gemini

Google ADK provides the `LlmAgent` abstraction, model transport, ephemeral
in-memory model sessions, output-schema integration, and configured HTTP retry
behavior. ADK does not persist the business conversation, authorize research,
enforce plan/follow-up budgets, validate citations, or provide an auditable
workflow trace.

Each ADK invocation uses a new ephemeral in-memory session. SQLite remains the
source of truth for the durable Client/Advisor/Analyst trajectory. The
application wraps ADK with total deadlines, primary retries, one repair path,
fallback-model use, and a safe escalation outcome.

## LanceDB and FastEmbed

LanceDB stores generated knowledge chunks and vectors, supplies exact vector
search and BM25 full-text search, and manages its local index artifacts.
FastEmbed supplies local BGE embeddings and the ONNX cross-encoder runtime.

Neither framework decides what evidence reaches a model. The application owns
document parsing, source-aware chunking, candidate normalization, RRF fusion,
cross-encoder ordering, MMR diversity selection, canonical deduplication, and
the final `Evidence` contract.

## React and Vite

React renders API-derived session state and Vite provides the local development
server and API proxy. The browser does not reconstruct workflow state, make
agent decisions, or store authoritative conversation artifacts. It receives
the session identifier, subscribes to server-sent persisted events, and reloads
the server-owned session snapshot for display and replay.

## Design consequence

The framework choices accelerate plumbing while deterministic application code
owns the decisions that must be inspectable: permissions, evidence boundaries,
source policy, safety limits, persistence, and replay.
