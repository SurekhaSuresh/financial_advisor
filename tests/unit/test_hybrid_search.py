from pathlib import Path

import lancedb
from lancedb.index import FTS

from financial_advisor.config import KNOWLEDGE_TEXT_INDEX_NAME
from financial_advisor.contracts import RetrievalChannel
from financial_advisor.retrieval.knowledge_base.hybrid_search import LocalHybridRetriever


def test_local_hybrid_retriever_returns_both_candidate_channels(
    tmp_path: Path,
) -> None:
    knowledge_table = lancedb.connect(str(tmp_path)).create_table(
        "knowledge_chunks",
        data=[
            {
                "chunk_id": "guide:1:1",
                "canonical_candidate_id": "home",
                "source_id": "guide",
                "source_title": "Home savings",
                "publisher": "Investor.gov",
                "source_url": "https://example.com/home",
                "topics": ["saving"],
                "heading_path": ["Saving"],
                "section_position": 1,
                "chunk_position": 1,
                "token_count": 3,
                "text": "home savings liquidity",
                "vector": [1.0, 0.0],
            },
            {
                "chunk_id": "risk:1:1",
                "canonical_candidate_id": "risk",
                "source_id": "risk",
                "source_title": "Investment risk",
                "publisher": "FINRA",
                "source_url": "https://example.com/risk",
                "topics": ["risk"],
                "heading_path": ["Risk"],
                "section_position": 1,
                "chunk_position": 1,
                "token_count": 4,
                "text": "diversification and investment risk",
                "vector": [0.0, 1.0],
            },
        ],
    )
    knowledge_table.create_index(
        "text",
        config=FTS(stem=True, remove_stop_words=True),
        name=KNOWLEDGE_TEXT_INDEX_NAME,
    )
    retriever = LocalHybridRetriever(
        tmp_path,
        embed=lambda _texts: [[1.0, 0.0]],
        candidate_limit=2,
    )

    candidates = retriever.search("home savings")

    assert [item.retrieval_channel for item in candidates[:2]] == [
        RetrievalChannel.VECTOR,
        RetrievalChannel.VECTOR,
    ]
    assert any(
        item.canonical_candidate_id == "home"
        and item.retrieval_channel is RetrievalChannel.KEYWORD
        for item in candidates
    )
    assert all(
        item.cosine_distance is not None
        for item in candidates
        if item.retrieval_channel is RetrievalChannel.VECTOR
    )
    assert all(
        item.bm25_score is not None
        for item in candidates
        if item.retrieval_channel is RetrievalChannel.KEYWORD
    )
