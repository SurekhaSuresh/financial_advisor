# Local Knowledge Ingestion

## Purpose

Financial Advisor uses a small local knowledge base for stable educational
guidance. It complements, rather than replaces, later live-web research for
current rules, rates, and market information.

## Source control

`data/knowledge_sources.yaml` is the explicit allowlist. It currently contains
50 Investor.gov, FINRA, CFPB, and FDIC sources covering investing basics, risk,
asset allocation, diversification, saving, debt, consumer protections, and
financial-professional due diligence.

The fetch script stores source snapshots and metadata containing the source URL,
fetch time, and SHA-256 digest in `data/source_snapshots/`. Snapshots are
generated local data and are excluded from Git.

## Ingestion pipeline

```text
approved source manifest
→ public source snapshot
→ HTML hierarchical heading sections or PDF page sections
→ BGE-token windows
→ local BGE embeddings
→ LanceDB knowledge_chunks table
```

HTML parsing retains the full heading path (for example, `Investing basics >
Risk`) with its associated paragraphs. PDF parsing retains page position. The
chunker uses the `BAAI/bge-small-en-v1.5` tokenizer, limits each section window
to 500 tokens **including the title and heading-path context supplied to the
embedding model**, and overlaps by up to 60 body tokens only when a long section
is split. It never overlaps content between two headings or pages. This keeps
each retrieved chunk focused while preserving neighboring context when a single
topic is too large for one embedding window. An unusually long heading can
reduce the overlap only when necessary to leave a positive body-window advance.

The chunker also has a 150-token minimum for curated sources (120 tokens for
live-web pages). If the final window of a section falls below that threshold,
it is merged into its immediate preceding window from the same section. The
merged window may modestly exceed the nominal maximum; that is intentional—a
coherent passage is more useful for retrieval than an isolated, weak tail.
The merge reconstructs the combined token range, so overlapping tokens are not
duplicated. A small standalone heading or PDF page remains separate rather than
being merged across a topic boundary.

Each LanceDB row includes the vector plus source ID, title, publisher, URL,
topics, heading path, positions, token count, and chunk text. BGE embeddings
have 384 dimensions. The generated chunk count changes with the source
manifest and ingestion policy, so inspect the active corpus through
`GET /inspection/knowledge/summary` rather than relying on a checked-in count.

## Commands

```bash
uv run python scripts/fetch_knowledge_sources.py
HF_HOME="$PWD/.local/model_cache" XDG_CACHE_HOME="$PWD/.local/cache" \
HF_HUB_DISABLE_XET=1 uv run python scripts/ingest_knowledge.py
```

To fetch a verified incremental batch rather than re-fetch every snapshot,
repeat `--source-id` for its explicit manifest IDs:

```bash
uv run python scripts/fetch_knowledge_sources.py \
  --source-id investor_gov_what_is_risk \
  --source-id fdic_understanding_deposit_insurance
```

The fetcher retries transient network and rate-limit responses. For an audit of
an entire batch without stopping at its first failure, add `--continue-on-error`;
it reports every failed source and exits unsuccessfully so an incomplete corpus
cannot be mistaken for a successful refresh.

The cache variables keep downloaded model artifacts inside the project’s ignored
`.local/` directory. Retrieval, rank fusion, reranking, and diversity selection
are deliberately not part of ingestion; they are implemented in the retrieval
stage.
