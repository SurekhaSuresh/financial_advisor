from financial_advisor.contracts import RetrievalChannel, RetrievedEvidenceCandidate
from financial_advisor.retrieval.ranking import (
    reciprocal_rank_fusion,
    rerank_candidates,
    select_candidates,
)


def candidate(
    canonical_candidate_id: str,
    retrieval_channel: RetrievalChannel,
    rank: int,
    text: str,
    vector: list[float],
) -> RetrievedEvidenceCandidate:
    return RetrievedEvidenceCandidate.model_validate(
        {
            "chunk_id": canonical_candidate_id,
            "canonical_candidate_id": canonical_candidate_id,
            "source_id": "test",
            "source_title": canonical_candidate_id,
            "publisher": "Publisher",
            "source_url": f"https://example.com/{canonical_candidate_id}",
            "topics": ["testing"],
            "heading_path": ["Test"],
            "section_position": 1,
            "chunk_position": 1,
            "token_count": len(text),
            "text": text,
            "retrieval_channel": retrieval_channel,
            "rank": rank,
            "cosine_distance": (
                0.2 if retrieval_channel is not RetrievalChannel.KEYWORD else None
            ),
            "bm25_score": (2.0 if retrieval_channel is RetrievalChannel.KEYWORD else None),
            "vector": vector,
        }
    )


def test_selection_fuses_duplicate_channels_then_keeps_diverse_evidence() -> None:
    candidates = [
        candidate("cash", RetrievalChannel.VECTOR, 1, "cash", [1.0, 0.0]),
        candidate("cash", RetrievalChannel.KEYWORD, 1, "cash", [1.0, 0.0]),
        candidate("cash-copy", RetrievalChannel.WEB, 1, "cash copy", [0.99, 0.01]),
        candidate("risk", RetrievalChannel.WEB, 2, "risk", [0.0, 1.0]),
    ]

    selected = select_candidates(
        "planning",
        candidates,
        rerank=lambda _query, documents: [1.0, 0.95, 0.9][: len(documents)],
        embed=lambda _texts: [],
        limit=2,
        relevance_weight=0.5,
    )

    assert [item.canonical_candidate_id for item in selected] == ["cash", "risk"]


def test_ranking_stages_are_independently_readable() -> None:
    candidates = [
        candidate("shared", RetrievalChannel.VECTOR, 2, "shared", [1.0, 0.0]),
        candidate("shared", RetrievalChannel.KEYWORD, 1, "shared", [1.0, 0.0]),
        candidate("other", RetrievalChannel.WEB, 1, "other", [0.0, 1.0]),
    ]

    fused = reciprocal_rank_fusion(candidates)
    reranked = rerank_candidates(
        "planning",
        fused,
        rerank=lambda _query, documents: [float(text == "other") for text in documents],
    )

    assert [item.canonical_candidate_id for item in fused] == ["shared", "other"]
    assert fused[0].cosine_distance == 0.2
    assert fused[0].bm25_score == 2.0
    assert [item.canonical_candidate_id for item, _score in reranked] == [
        "other",
        "shared",
    ]


def test_selection_rejects_incomplete_reranker_output() -> None:
    item = candidate("cash", RetrievalChannel.VECTOR, 1, "cash", [1.0, 0.0])

    try:
        select_candidates(
            "planning",
            [item],
            rerank=lambda _query, _documents: [],
            embed=lambda _texts: [],
        )
    except ValueError as error:
        assert "one score per candidate" in str(error)
    else:
        raise AssertionError("Incomplete reranker output must fail.")
