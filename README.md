# Financial Advisor

A small multi-agent financial-planning exercise built with Google ADK.

The Advisor is the root agent and the only communication bridge. It calls the
Client and Analyst through `AgentTool`; those agents never call each other.

## Active code

```text
src/financial_advisor/
├── contracts.py       shared data contracts
├── config.py          application and retrieval settings
├── runtime.py         runner construction and conversation execution
├── api.py             UI endpoints, live events, and session replay
├── inspection.py      read-only SQLite and LanceDB inspection
├── agents/
│   ├── advisor/
│   │   ├── agent.py   root agent and deterministic recommendation finalization
│   │   └── prompt.py  Advisor instructions
│   ├── analyst/
│   │   ├── agent.py   Analyst AgentTool target with pre-model retrieval
│   │   └── prompt.py  Analyst instructions
│   └── client.py      Client AgentTool target and instructions
└── retrieval/
    ├── content_processing.py  shared HTML/PDF parsing and token chunking
    ├── ranking.py      shared RRF, reranking, and MMR stages
    ├── pipeline.py     executes the Advisor-selected retrieval paths
    ├── text_models.py  local embedding, tokenizer, and reranker
    ├── hybrid_search.py  LanceDB vector and BM25 search
    └── web/
        ├── providers.py   discovery, failover, and URL policy
        ├── page_fetch.py  bounded original-page fetching
        └── web_search.py  web page processing and passage ranking

tests/unit/
├── test_client_agent.py
├── test_analyst_agent.py
├── test_advisor_agent.py
├── test_retrieval_pipeline.py
├── test_runtime.py
└── test_api.py

frontend/
├── src/App.tsx        conversation and observability UI
├── src/api.ts         backend requests
└── src/types.ts       UI response contracts

knowledge_base/
├── offline_ingestion.py  offline knowledge-store build
└── data/
    ├── sources.yaml      approved local-knowledge sources
    ├── snapshots/        downloaded sources and integrity metadata
    └── lancedb/          generated vector and full-text indexes
```

The Advisor selects `local_hybrid`, `web`, or both in
`ResearchTask.retrieval_paths`. The Analyst executes that plan deterministically
before any model call. Empty retrieval returns `success=false` without
calling the Analyst model. All retrieved candidates share one RRF →
cross-encoder reranking → MMR evidence-selection pass.

## Verify

```bash
uv run pytest
uv run ruff check src tests
uv run mypy src/financial_advisor
cd frontend && npm run build
```

## Run locally

Create `.env` from `.env.example`, then start the API and UI in separate terminals:

```bash
uv run python knowledge_base/offline_ingestion.py
uv run uvicorn financial_advisor.api:app --reload
cd frontend && npm run dev
```
