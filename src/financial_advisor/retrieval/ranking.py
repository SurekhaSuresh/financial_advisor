"""Deterministic ranking stages shared by local and web retrieval."""

from collections.abc import Sequence
from math import sqrt

from financial_advisor.config import (
    DEFAULT_EVIDENCE_LIMIT,
    DEFAULT_MMR_RELEVANCE_WEIGHT,
    DEFAULT_RRF_RANK_CONSTANT,
    Rerank,
)
from financial_advisor.contracts import (
    RetrievalChannel,
    RetrievedEvidenceCandidate,
)

RerankedCandidate = tuple[RetrievedEvidenceCandidate, float]


def reciprocal_rank_fusion(
    retrieved_candidates: Sequence[RetrievedEvidenceCandidate],
) -> list[RetrievedEvidenceCandidate]:
    """Merge duplicate candidates and order them by combined channel rank."""

    best_candidates_by_channel = _deduplicate_candidates_within_channels(retrieved_candidates)
    scored_fused_candidates: list[tuple[RetrievedEvidenceCandidate, float]] = []
    for best_candidates_per_id in best_candidates_by_channel.values():
        fused_candidate = next(iter(best_candidates_per_id.values()))
        rrf_score = sum(
            1 / (DEFAULT_RRF_RANK_CONSTANT + candidate.rank)
            for candidate in best_candidates_per_id.values()
        )
        scored_fused_candidates.append((fused_candidate, rrf_score))

    scored_fused_candidates.sort(key=lambda item: (-item[1], item[0].canonical_candidate_id))
    return [candidate for candidate, _score in scored_fused_candidates]

# Example: A1/A2/A3 from vector, B1/B2 from keyword, and C1/C2 from web
# all have the same canonical ID and therefore represent the same content.
#
# Vector ranks:  A1=1, A2=4, A3=7  → keep A1
# Keyword ranks: B1=2, B2=5        → keep B1
# Web ranks:     C1=3, C2=6        → keep C1
#
# candidates_by_canonical_id becomes:
# {
#     "same-canonical-id": {
#         RetrievalChannel.VECTOR: A1,
#         RetrievalChannel.KEYWORD: B1,
#         RetrievalChannel.WEB: C1,
#     }
# }
#
# Each channel retains only its best-ranked duplicate. 

def _deduplicate_candidates_within_channels(
    retrieved_candidates: Sequence[RetrievedEvidenceCandidate],
) -> dict[str, dict[RetrievalChannel, RetrievedEvidenceCandidate]]:
    """Keep the best-ranked candidate per canonical ID and retrieval channel."""

    candidates_by_canonical_id: dict[
        str,
        dict[RetrievalChannel, RetrievedEvidenceCandidate],
    ] = {}
    for retrieved_candidate in retrieved_candidates:
        best_candidates_by_channel = candidates_by_canonical_id.setdefault(
            retrieved_candidate.canonical_candidate_id,
            {},
        )
        existing_candidate_for_channel = best_candidates_by_channel.get(
            retrieved_candidate.retrieval_channel
        )
        if (
            existing_candidate_for_channel is None
            or retrieved_candidate.rank < existing_candidate_for_channel.rank
        ):
            best_candidates_by_channel[retrieved_candidate.retrieval_channel] = retrieved_candidate
    return candidates_by_canonical_id


def select_candidates_with_mmr(
    reranked_candidates: Sequence[RerankedCandidate],
    *,
    selection_limit: int = DEFAULT_EVIDENCE_LIMIT,
    relevance_weight: float = DEFAULT_MMR_RELEVANCE_WEIGHT,
) -> list[RetrievedEvidenceCandidate]:
    """Use maximal marginal relevance to avoid redundant final evidence."""

    if not reranked_candidates:
        return []

    raw_scores = [score for _candidate, score in reranked_candidates]
    lowest_relevance = min(raw_scores)
    relevance_range = max(raw_scores) - lowest_relevance
    remaining_candidates = []
    for candidate, score in reranked_candidates:
        norm_score = 1.0 if relevance_range == 0 else (score - lowest_relevance) / relevance_range
        remaining_candidates.append((candidate, norm_score))

    selected_candidates: list[RetrievedEvidenceCandidate] = []
    while remaining_candidates and len(selected_candidates) < selection_limit:
        scored_candidates = []
        for candidate, relevance_score in remaining_candidates:
            similarity = 0.0
            for selected_candidate in selected_candidates:
                similarity = max(
                    similarity,
                    cosine_similarity(candidate.vector, selected_candidate.vector),
                )
            mmr_score = relevance_weight * relevance_score - (1 - relevance_weight) * similarity
            scored_candidates.append((mmr_score, (candidate, relevance_score)))

        best_choice = max(scored_candidates, key=lambda item: item[0])[1]
        remaining_candidates.remove(best_choice)
        selected_candidates.append(best_choice[0])
    return selected_candidates


def select_evidence_candidates(
    query: str,
    retrieved_candidates: Sequence[RetrievedEvidenceCandidate],
    rerank: Rerank,
    *,
    limit: int = DEFAULT_EVIDENCE_LIMIT,
    relevance_weight: float = DEFAULT_MMR_RELEVANCE_WEIGHT,
) -> list[RetrievedEvidenceCandidate]:
    """Run RRF, cross-encoder reranking, and MMR in that order."""

    query = query.strip()
    if not query:
        raise ValueError("Evidence-selection query must not be blank.")
    fused_candidates = reciprocal_rank_fusion(retrieved_candidates)
    if not fused_candidates:
        return []

    # Rerank fused candidates with the cross-encoder.
    reranker_scores = rerank(query, [candidate.text for candidate in fused_candidates])
    reranked_candidates = sorted(
        zip(fused_candidates, reranker_scores, strict=True),
        key=lambda item: (-item[1], item[0].canonical_candidate_id),
    )

    return select_candidates_with_mmr(
        reranked_candidates,
        selection_limit=limit,
        relevance_weight=relevance_weight,
    )


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Return cosine similarity for two non-zero vectors of equal size."""

    left_size = sqrt(sum(value * value for value in left))
    right_size = sqrt(sum(value * value for value in right))
    if left_size == 0 or right_size == 0:
        return 0.0
    dot_product = sum(
        left_value * right_value for left_value, right_value in zip(left, right, strict=True)
    )
    return dot_product / (left_size * right_size)
