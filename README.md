# Financial Advisor

A small multi-agent financial-planning exercise built with Google ADK.

The Advisor is the root agent and the only communication bridge. It calls the
Client and Analyst through `AgentTool`; those agents never call each other.

## Active code

```text
src/financial_advisor/
├── models.py          shared agent contracts and trusted result creation
├── agents/
    ├── client.py      Client AgentTool target
    └── analyst.py     Analyst AgentTool target with pre-model retrieval
└── retrieval/
    ├── models.py       two small retrieval contracts
    ├── documents.py    HTML/PDF parsing and token chunking
    ├── knowledge.py    LanceDB vector and BM25 search
    ├── providers.py    web discovery, failover, and URL policy
    ├── web_fetch.py    bounded original-page fetching
    ├── web.py          web page processing and passage ranking
    ├── ranking.py      separate RRF, reranking, and MMR stages
    ├── pipeline.py     executes the Advisor-selected retrieval paths
    ├── local_models.py local embedding, tokenizer, and reranker
    └── ingestion.py    offline knowledge-store build

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
