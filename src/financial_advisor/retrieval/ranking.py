"""Deterministic ranking stages shared by local and web retrieval."""

from collections.abc import Sequence
from math import sqrt

from financial_advisor.config import (
    DEFAULT_EVIDENCE_LIMIT,
    DEFAULT_MMR_RELEVANCE_WEIGHT,
    DEFAULT_RRF_RANK_CONSTANT,
)
from financial_advisor.contracts import (
    Rerank,
    RetrievalChannel,
    RetrievedEvidenceCandidate,
)

RerankedCandidate = tuple[RetrievedEvidenceCandidate, float]
ChannelCandidates = dict[RetrievalChannel, RetrievedEvidenceCandidate]


def reciprocal_rank_fusion(
    retrieved_candidates: Sequence[RetrievedEvidenceCandidate],
    *,
    rank_constant: int = DEFAULT_RRF_RANK_CONSTANT,
) -> list[RetrievedEvidenceCandidate]:
    """Merge duplicate candidates and order them by combined channel rank."""

    if rank_constant <= 0:
        raise ValueError("RRF rank constant must be positive.")

    candidates_by_source = _deduplicate_candidates_within_channels(retrieved_candidates)
    scored_fused_candidates: list[tuple[RetrievedEvidenceCandidate, float]] = []
    for candidates_by_channel in candidates_by_source.values():
        vector_candidate = candidates_by_channel.get(RetrievalChannel.VECTOR)
        keyword_candidate = candidates_by_channel.get(RetrievalChannel.KEYWORD)
        fused_candidate = vector_candidate or next(iter(candidates_by_channel.values()))

        if keyword_candidate is not None:
            fused_candidate = fused_candidate.model_copy(
                update={"bm25_score": keyword_candidate.bm25_score}
            )

        rrf_score = sum(
            1 / (rank_constant + candidate.rank)
            for candidate in candidates_by_channel.values()
        )
        scored_fused_candidates.append((fused_candidate, rrf_score))

    scored_fused_candidates.sort(
        key=lambda item: (-item[1], item[0].canonical_candidate_id)
    )
    return [candidate for candidate, _score in scored_fused_candidates]


def _deduplicate_candidates_within_channels(
    retrieved_candidates: Sequence[RetrievedEvidenceCandidate],
) -> dict[str, ChannelCandidates]:
    """Group matching sources, keeping only the best duplicate per channel."""

    candidates_by_source: dict[str, ChannelCandidates] = {}
    for retrieved_candidate in retrieved_candidates:
        candidates_by_channel = candidates_by_source.setdefault(
            retrieved_candidate.canonical_candidate_id,
            {},
        )
        existing_candidate_for_channel = candidates_by_channel.get(
            retrieved_candidate.retrieval_channel
        )
        if (
            existing_candidate_for_channel is None
            or retrieved_candidate.rank < existing_candidate_for_channel.rank
        ):
            candidates_by_channel[retrieved_candidate.retrieval_channel] = retrieved_candidate
    return candidates_by_source


def rerank_candidates(
    query: str,
    fused_candidates: Sequence[RetrievedEvidenceCandidate],
    rerank: Rerank,
) -> list[RerankedCandidate]:
    """Order fused candidates by cross-encoder relevance."""

    if not fused_candidates:
        return []
    relevance_scores = rerank(query, [candidate.text for candidate in fused_candidates])
    if len(relevance_scores) != len(fused_candidates):
        raise ValueError("Reranker must return one score per candidate.")

    reranked_candidates = list(zip(fused_candidates, relevance_scores, strict=True))
    reranked_candidates.sort(key=lambda item: (-item[1], item[0].canonical_candidate_id))
    return reranked_candidates


def select_candidates_with_mmr(
    reranked_candidates: Sequence[RerankedCandidate],
    *,
    selection_limit: int = DEFAULT_EVIDENCE_LIMIT,
    relevance_weight: float = DEFAULT_MMR_RELEVANCE_WEIGHT,
) -> list[RetrievedEvidenceCandidate]:
    """Use maximal marginal relevance to avoid redundant final evidence."""

    if selection_limit <= 0:
        raise ValueError("MMR candidate limit must be positive.")
    if not 0 <= relevance_weight <= 1:
        raise ValueError("MMR relevance weight must be between zero and one.")
    if not reranked_candidates:
        return []

    evidence_candidates = [candidate for candidate, _score in reranked_candidates]
    relevance_scores = [score for _candidate, score in reranked_candidates]
    minimum_score = min(relevance_scores)
    score_range = max(relevance_scores) - minimum_score
    normalized_relevance_scores = [
        1.0 if score_range == 0 else (score - minimum_score) / score_range
        for score in relevance_scores
    ]
    candidate_vectors = [candidate.vector for candidate in evidence_candidates]

    selected_indices: list[int] = []
    remaining_indices = set(range(len(evidence_candidates)))
    maximum_similarities = [0.0] * len(evidence_candidates)
    while remaining_indices and len(selected_indices) < selection_limit:
        mmr_scores = {
            index: (
                relevance_weight * normalized_relevance_scores[index]
                - (1 - relevance_weight) * maximum_similarities[index]
            )
            for index in remaining_indices
        }
        selected_index = max(
            remaining_indices,
            key=lambda index: (mmr_scores[index], -index),
        )
        selected_indices.append(selected_index)
        remaining_indices.remove(selected_index)

        selected_vector = candidate_vectors[selected_index]
        for candidate_index in remaining_indices:
            similarity = cosine_similarity(
                candidate_vectors[candidate_index],
                selected_vector,
            )
            maximum_similarities[candidate_index] = max(
                maximum_similarities[candidate_index],
                similarity,
            )
    return [evidence_candidates[index] for index in selected_indices]


def select_evidence_candidates(
    query: str,
    retrieved_candidates: Sequence[RetrievedEvidenceCandidate],
    rerank: Rerank,
    *,
    limit: int = DEFAULT_EVIDENCE_LIMIT,
    rank_constant: int = DEFAULT_RRF_RANK_CONSTANT,
    relevance_weight: float = DEFAULT_MMR_RELEVANCE_WEIGHT,
) -> list[RetrievedEvidenceCandidate]:
    """Run RRF, cross-encoder reranking, and MMR in that order."""

    query = query.strip()
    if not query:
        raise ValueError("Evidence-selection query must not be blank.")
    fused_candidates = reciprocal_rank_fusion(
        retrieved_candidates,
        rank_constant=rank_constant,
    )
    reranked_candidates = rerank_candidates(query, fused_candidates, rerank)
    return select_candidates_with_mmr(
        reranked_candidates,
        selection_limit=limit,
        relevance_weight=relevance_weight,
    )


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Return cosine similarity for two non-zero vectors of equal size."""

    if len(left) != len(right):
        raise ValueError("Candidate vectors must have the same dimension.")
    left_size = sqrt(sum(value * value for value in left))
    right_size = sqrt(sum(value * value for value in right))
    if left_size == 0 or right_size == 0:
        raise ValueError("Candidate vectors must not be zero vectors.")
    dot_product = sum(
        left_value * right_value
        for left_value, right_value in zip(left, right, strict=True)
    )
    return dot_product / (left_size * right_size)
