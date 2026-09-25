# Local Knowledge Retrieval

## Current capability

The retrieval package has clear ownership boundaries:

```text
content_processing → shared parse and chunk mechanics
knowledge_base     → stable curated corpus and LanceDB candidate retrieval
web                → live discovery, original-page processing, and passage ranking
selection          → cross-channel RRF, cross-encoder reranking, and MMR
evidence           → selected candidates to clean Analyst citations
pipeline           → one combined local-vector + BM25 + live-web path
```

The first retrieval layer performs read-only exact cosine vector search over the
local `knowledge_chunks` LanceDB table.

```text
question
→ local BGE query embedding → exact cosine candidates
→ LanceDB BM25 full-text candidates
→ fetched live-web chunks → in-memory BGE cosine candidates
→ normalize to one candidate contract → reciprocal-rank fusion
→ fused candidates with source provenance and channel contributions
```

Each candidate retains the chunk body, heading path, source URL, publisher,
topics, ordering information, rank, and cosine distance. This makes the result
inspectable without relying on opaque model output.

Exact vector search is intentional at the current small local corpus scale: it keeps
retrieval simple, deterministic, and easy to evaluate. Approximate-nearest-
neighbor indexing is a future scale optimization, not a current need.

BM25 uses LanceDB's local full-text index over the chunk `text` field. It uses
English stemming and stop-word removal, and is created on demand if a corpus
re-ingestion replaced the generated table. It complements semantic search by
matching precise financial terms and identifiers.

Vector cosine distance and BM25 score have different meanings and scales, so
they are not added or normalized together. Reciprocal-rank fusion (RRF) combines
only each candidate's rank from each channel, using `1 / (60 + rank)`. A chunk
returned by both channels receives one contribution from each; its fused result
retains all contributing channel records for inspection. Exact evidence identity
is storage-independent: `canonical_candidate_id` hashes the document version
hash plus normalized chunk text. This lets an unchanged curated snapshot and
the same page fetched from the web collapse into one candidate, while a changed
page version remains distinct. An accidental duplicate within one channel is
counted only at its better rank.

RRF candidates are then reranked by the local ONNX cross-encoder
`Xenova/ms-marco-MiniLM-L-6-v2`. Unlike bi-encoder vector search, it receives a
question and one candidate chunk together and returns a query-specific relevance
score. The reranking trace preserves the input count, reranked order, score, and
the fused candidate with its original channel contributions. Model artifacts are
cached locally during setup so the application does not depend on a model request
while the application is running.

The final selection uses deterministic maximal marginal relevance (MMR). It
selects up to five chunks greedily with a 0.7 relevance weight and 0.3 redundancy
penalty. Cross-encoder scores are normalized only within the current candidate
set because they are relative relevance outputs, not probabilities. BGE cosine
similarity supplies the redundancy penalty. Internal candidates reuse their
already stored LanceDB vectors in memory; only candidates without a stored vector
(such as future live-web evidence) are embedded for this step. Raw vectors are
excluded from serialized traces. The trace records selection order, MMR score,
normalized relevance, and maximum similarity to previously selected evidence.

Local vector and BM25 candidate retrievers remain independently testable. The
production coordinator combines them with live-web candidates before one shared
selection pass, rather than selecting local evidence in a separate local-only
pipeline.

## Live-web candidate retrieval

Search-provider results identify pages, not the most relevant passage on a
page. After original HTML/PDF pages are fetched, parsed, and token-windowed,
`InMemoryWebVectorRetriever` embeds the question and all request-scoped web
chunks in one local BGE batch. It then ranks chunks by exact cosine similarity,
deduplicates exact canonical IDs, and returns up to ten
`RetrievalEvidenceCandidate` records with `retrieval_channel="live_web"`.

The web retriever uses the same embedding representation and candidate contract
as local retrieval, but it intentionally does not write time-sensitive fetched
content into the persistent LanceDB corpus. Its top candidates are ready for the
same later fusion, cross-encoder reranking, and MMR selection stages.

`LiveWebResearchPipeline` composes discovery, sequential original-page fetches,
structure-aware parsing, token-window chunking, and in-memory ranking. It stops
early once four pages have yielded usable chunks or the bounded discovered result
list is exhausted. It records discovery, every page-fetch outcome, parse/chunk
outcome, and ranked web candidates in one trace. A failed page is recorded and
skipped so later discovered URLs can still contribute evidence.

## Combined evidence selection

`EvidenceRetrievalPipeline` collects local vector, local BM25, and live-web
`RetrievalEvidenceCandidate` records before applying one shared selection path:

```text
internal vector candidates + internal BM25 candidates + live-web candidates
→ reciprocal-rank fusion
→ local cross-encoder reranking
→ deterministic MMR diversity selection
```

This ordering prevents local evidence from being selected before current web
evidence is considered. The combined trace preserves each channel's independent
retrieval trace plus the one shared fusion, reranking, and selection trace.

If every channel returns zero candidates, the pipeline records
`status="insufficient_evidence"` with no selection trace. It does not invoke
RRF, the cross-encoder, or MMR, and downstream agents must not create an
evidence-backed investment recommendation from that result.

## Analyst evidence boundary

After MMR selection, `assemble_selected_evidence` deterministically converts
each selected candidate into an `Evidence` record. The Analyst receives those
records—not retrieval objects or their selection scores. An evidence record
contains only a stable server-owned evidence ID, source title, publisher, URL,
available publication date, retrieval time, excerpt, and source type.

Ranks, vector distances, BM25 scores, RRF scores, cross-encoder scores, MMR
scores, channel-contribution details, and embeddings remain in the retrieval
trace for inspection. They are implementation signals, not source facts the
Analyst should reason from or repeat to a client.

The workflow persists that detailed trace in SQLite before Analyst reasoning.
Each record is linked to its session, workflow trace, and Analyst task. The
workflow event timeline records retrieval start and completion or unexpected
failure, allowing a completed session to be replayed without rerunning web
search or models.

Live-web publication dates flow from discovery through fetching and chunking to
the final citation. The live page fetch timestamp is preserved as its retrieval
time. A stable local snapshot currently records the time it was selected for a
research task; snapshot-capture timestamp propagation is a future provenance
enhancement.

## Advisor-directed web scopes

`EvidenceRetrievalRequest` is constructed deterministically from the trusted
`AdvisorResearchPlan` before retrieval begins. The plan is the only source of
authorization for the local-hybrid flag and up to two web scopes.

The Advisor, not the Analyst model, will choose whether a research plan needs
local knowledge, live web research, or both. A plan may contain at most two
live-web scopes:

1. **Authoritative-domain scope** — one request constrained to one or more
   server-owned approved-domain identifiers, such as `investor_gov`, `sec_gov`,
   `finra_org`, `irs_gov`, or `treasury_gov`.
2. **Broad-web scope** — one unconstrained request for current context when a
   domain-restricted source alone would be incomplete.

Local hybrid retrieval is enabled or disabled by the Advisor's research plan.
Every selected source is attempted, but no selected source is a hard gate: if
at least one selected source yields usable evidence, the pipeline may continue
with the available evidence. It persists a structured outcome for every source
that was unavailable or returned no usable evidence, so downstream agents must
disclose that limitation rather than imply complete coverage.

The Advisor selects approved identifiers; it never creates raw domain strings.
The server resolves those identifiers through its configured policy before the
provider request is built. This keeps source policy reviewable and prevents an
LLM from widening an authoritative search silently.

When a plan has both scopes, the application runs them **sequentially**. Each
call has an independent `LiveWebResearchTrace`, including its discovery,
fetch, parse, chunk, and ranking outcomes. Their candidates then join any
enabled Lance vector and BM25 candidate lists in the existing shared selection
path:

```text
optional local vector + optional local BM25
  + authoritative-web candidates + broad-web candidates
→ RRF when multiple candidate channels exist
→ cross-encoder reranking
→ MMR evidence selection
```

Candidate provenance identifies its scope, for example
`authoritative_web` or `broad_web`, rather than treating every live result as
one undifferentiated channel. That makes it possible to inspect exactly which
scope supplied an eventual citation.

Sequential scope execution intentionally requires no new concurrency mechanism.
The existing in-process provider-health circuit breaker remains shared: if an
authoritative request exhausts Exa and opens its circuit, the immediately
following broad request sees that provider state and uses the configured
failover path. Parallel web scopes are not part of the MVP.
