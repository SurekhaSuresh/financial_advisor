# Analyst Agent

## Purpose

The Analyst is an internal research agent. It receives a persisted,
Advisor-authorized task and server-selected evidence. It returns a structured
`ResearchBrief` to the Advisor; it never addresses the Client directly.

## Execution flow

```text
Persisted AdvisorResearchPlan + AnalystTask
  → AnalystResearchExecutor creates EvidenceRetrievalRequest
  → Lance vector + BM25 and selected live-web scopes retrieve evidence
  → deterministic RRF, cross-encoder reranking, and MMR select citations
  → AnalystResearchRequest contains selected Evidence and source limitations
  → Google ADK LlmAgent returns AnalystResearchDraft
  → server validates citations and builds ResearchBrief
```

The Analyst LLM is not given retrieval tools. The Advisor chooses the bounded
source policy; deterministic application code executes it and records the full
trajectory. This makes source choice reviewable and prevents the Analyst from
silently expanding an authoritative search into broad web research.

`AnalystResearchDraft` contains reasoning fields only: findings, scenario
comparisons, calculations, and caveats. The server attaches the original
evidence objects to the final `ResearchBrief`. The model may cite an existing
`evidence_id`, but cannot replace a source, fabricate evidence, or omit the
retrieval pipeline's provenance.

## Advisor-directed research plans

The Advisor LLM returns a typed plan draft. Server-side validation then creates
an immutable `AdvisorResearchPlan`, persisted before the Analyst task begins.
The plan can authorize local hybrid retrieval, authoritative-domain web research
with approved IDs, broad-web research, or a combination of those options.

At most two web scopes run sequentially. Any selected source may be unavailable
or return no usable evidence; the Analyst proceeds when at least one selected
source provides evidence and receives deterministic caveats for unavailable
sources. When every selected source has no usable evidence, the model is not
invoked.

If the first local-only plan returns insufficient evidence, the workflow records
that trace and returns to the Advisor for one web-enabled replan. The maximum
is two plans per client question; it never loops indefinitely.

For a second plan requested after an adequate first brief, the executor merges
the new final selected evidence with prior final selected evidence using exact
provenance identity. It deliberately excludes raw retrieval candidates that
were not selected after RRF, reranking, and MMR. The Analyst therefore receives
only validated, reviewable evidence from the first plan plus the fresh selected
evidence, and produces a consolidated second `ResearchBrief`.

## ADK runtime policy

`create_analyst_agent()` creates an explicit Google ADK `LlmAgent` with a
pinned Gemini model, low-temperature generation, bounded output, structured
output schema, and no agent transfers or tools.

`AdkAnalystService` applies a task-wide deadline. Within that deadline it tries
the primary model a small configured number of times, allows one structured
output repair after malformed output, and then tries the configured fallback
model. If all paths fail, the workflow records the successful retrieval trace
and safely escalates rather than fabricating an analysis.

The application's workflow state and SQLite store—not ADK session memory—are
the source of truth for permissions, replay, and conversation history. ADK
sessions are ephemeral transport for a single model attempt.
