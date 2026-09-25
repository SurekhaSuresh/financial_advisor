# Financial Advisor

A small multi-agent financial-planning exercise built with Google ADK.

The Advisor is the root agent and the only communication bridge. It calls the
Client and Analyst through `AgentTool`; those agents never call each other.

## Active code

```text
src/financial_advisor/
├── contracts.py       shared data contracts
├── config.py          application and retrieval settings
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
    ├── knowledge_base/
    │   ├── hybrid_search.py      LanceDB vector and BM25 search
    │   └── offline_ingestion.py  offline knowledge-store build
    └── web/
        ├── providers.py   discovery, failover, and URL policy
        ├── page_fetch.py  bounded original-page fetching
        └── web_search.py  web page processing and passage ranking

tests/unit/
├── test_models.py
├── test_client_agent.py
└── test_analyst_agent.py
```

The system is being rebuilt contract-first. The previous implementation is
available under `old/` as a behavioral reference and is not part of the active
application or test suite.

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
```
