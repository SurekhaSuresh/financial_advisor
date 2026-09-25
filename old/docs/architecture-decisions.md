# Architecture Decisions

This document records accepted technical decisions and the operating constraints they create in Financial Advisor.

## ADR-001: Google ADK agents with a custom workflow engine

**Status:** Accepted

- Google ADK runs typed Client, Advisor, and Analyst model invocations.
- Deterministic Python services execute retrieval, evidence selection, and
  workflow transitions; models do not receive direct retrieval-tool authority.
- The custom Workflow Engine owns session state, allowed transitions, follow-up limits, event recording, and permission checks.
- An agent may request an action; only the Workflow Engine executes an allowed transition.

## ADR-002: LanceDB for stable knowledge retrieval

**Status:** Accepted

- Source documents are fetched and ingested by an offline workflow.
- Chunks, embeddings, and source metadata are stored in a local LanceDB table.
- The Analyst receives only server-selected `Evidence` assembled by the
  deterministic retrieval pipeline; it does not query LanceDB directly.
- Generated LanceDB data is local and excluded from version control.

## ADR-003: Live web research with provider failover

**Status:** Accepted

- The deterministic retrieval pipeline calls a typed `WebResearchProvider` interface
  under an Advisor-authorized plan; the Analyst receives only selected Evidence.
- Exa is the primary discovery provider; Brave is the fallback provider.
- Exa is preferred for natural-language discovery; Brave provides an
  independently operated search index, reducing single-provider dependency.
- Provider SDKs are not coupled to the domain: the application normalizes
  discovered URLs, fetches original pages itself, and retains provenance.
- Authoritative/current questions use an approved-domain search policy; broader research uses regular web search followed by source evaluation.
- The system discovers up to eight URLs, attempts them in provider rank order as
  needed, and stops after four pages yield usable parsed chunks or results end.
- A provider receives bounded retries before failover. If both providers fail, the Advisor receives a structured research-unavailable result.
- Each provider has an in-process closed/open/half-open circuit breaker. It
  resets on process restart; a concurrent, multi-instance deployment would add
  shared locking and tokenized probe leases.

## ADR-004: Source-aware chunking and exact initial vector retrieval

**Status:** Accepted

- The stable corpus is defined in `data/knowledge_sources.yaml`; current values, rules, and market data remain live-web responsibilities.
- Ingestion preserves headings, lists, tables, pages, and source metadata.
- Knowledge-base chunks are nominally at most 500 BGE-tokenizer tokens with a
  60-token overlap only within a split heading section. A final tail below 150
  tokens may merge with its preceding chunk only in the same section.
- Fetched web pages are session-scoped evidence and use nominally 350-token
  chunks with a 40-token same-section overlap. A final tail below 120 tokens
  may merge only with its preceding chunk from the same section.
- LanceDB uses exact cosine vector retrieval and BM25 full-text retrieval, then rank fusion, cross-encoder reranking, and diversity selection.
- ANN indexing is deferred until a measured corpus-size or query-latency need exists.

## ADR-005: Persisted, replayable session traces

**Status:** Accepted

- SQLite stores append-only, ordered events for every session.
- Each event includes session ID, trace ID, sequence, timestamp, actor, event type, state transition, and structured payload.
- The UI streams active events with SSE and reads completed sessions from SQLite.
- Replay renders saved events without rerunning models or web research.
- Secrets, authorization headers, and hidden chain-of-thought are not stored.

## ADR-006: Model routing by agent responsibility

**Status:** Accepted

- Client: `gemini-3.5-flash-lite`, with `gemini-3.5-flash-lite` fallback.
- Advisor and Analyst: `gemini-3.5-flash`, with `gemini-3.5-flash-lite` fallback.
- Model IDs are environment-backed, pinned, and validated at startup.
- Fallback follows bounded transient-failure retries. Terminal model failure is
  persisted as a safe workflow outcome; detailed per-attempt telemetry is a
  production observability extension.
- Model fallback does not change an agent's role, tools, permissions, contract, or workflow state.

## ADR-007: Deterministic evidence assembly before Analyst reasoning

**Status:** Accepted

- Retrieval selection produces an inspectable trace containing rank and model-scoring details.
- A deterministic server-side assembler creates the smaller `Evidence` contract passed to the Analyst.
- Evidence IDs are derived from canonical source provenance so citations are stable and server-owned.
- The Analyst receives source facts and excerpts only; it never receives or invents retrieval scores, ranks, embeddings, or channel contributions.
- The final `ResearchBrief` validates that every finding and scenario references only evidence IDs supplied by the server.

## ADR-008: Persist task-scoped retrieval trajectories for replay

**Status:** Accepted

- The workflow records a `TOOL_STARTED` event before retrieval and a safe `TOOL_COMPLETED` or `TOOL_FAILED` event afterward.
- `evidence_retrieval_traces` stores the complete retrieval JSON with its session ID, trace ID, task ID, status, and recording time.
- The detailed trace is retained for inspection; the Analyst receives only the assembled `Evidence` records.
- A recorded retrieval trace cannot be overwritten for the same Analyst task, preserving replayable history.

## ADR-009: Retrieval package ownership and centralized operating policy

**Status:** Accepted

- `retrieval/content_processing` owns source-format parsing and token-aware chunking shared by stable and live sources.
- `retrieval/knowledge_base` owns stable-corpus ingestion and LanceDB vector/BM25 retrieval.
- `retrieval/web` owns live-web discovery, provider adapters, page fetching, URL policy, page processing, and request-scoped passage ranking.
- Cross-channel contracts, reranking, RRF/MMR selection, evidence assembly, and the three-channel pipeline remain at the `retrieval` package level.
- `config.py` centralizes validated operating policies, local model identities/cache location, default artifact paths, source-fetch behavior, HTTP client identity, and provider base URLs. Provider endpoint paths remain adapter constants, source URLs remain versioned manifest data, and safety invariants remain enforced code contracts.

## ADR-010: Deterministic integration tests around real retrieval and workflow components

**Status:** Accepted

- Integration tests run the production Workflow Engine, Scenario Runner,
  SQLite repository, LanceDB vector and BM25 retrievers, RRF, reranking, MMR,
  live-page parsing/chunking, and the complete evidence pipeline.
- Tests use temporary local LanceDB and SQLite artifacts, so no project data is
  changed and no credentials are required.
- Client, Advisor, and Analyst LLM outputs are typed scripted fixtures. Exa and
  Brave discovery responses and original page documents are fixture boundaries.
- This makes workflow limits, retrieval composition, persistence, refinement,
  and safe escalation behavior repeatable. Separate live UI runs validate the
  real Gemini, provider, SSE, and browser boundaries.

## ADR-011: Safe completion boundary for background scenario execution

**Status:** Accepted

- The API persists a session before scheduling its scenario task, allowing the
  caller to subscribe immediately using stable session and trace IDs.
- Known failures are handled within the Client, Advisor, Analyst, and retrieval
  workflows using their specific terminal paths.
- If an unexpected exception escapes that boundary, the background-task
  completion handler consumes it and persists a generic safe escalation rather
  than leaving the session indefinitely active or emitting an unobserved task
  exception.
- The persisted event exposes a safe error category for replay without leaking
  implementation details to the Client.

## ADR-012: Bounded local latency and cost rather than open-ended autonomy

**Status:** Accepted

- Each model operation has a total deadline, bounded primary attempts, one
  schema-repair attempt, and one fallback-model attempt.
- Retrieval limits bound vector/BM25 candidates, final evidence, live discovery
  URLs, usable fetched pages, page bytes, provider attempts, and web scopes.
- The workflow bounds Client follow-ups and research plans per Client question;
  web scopes run sequentially to keep the persisted trajectory deterministic.
- This MVP intentionally does not implement production token-cost metering,
  request-level latency percentiles, or a total scenario deadline. Those need
  provider billing telemetry and centralized tracing rather than local process
  state. The existing per-stage limits prevent unbounded work in the meantime.
