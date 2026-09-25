"""Deterministic ranking stages shared by local and web retrieval."""

from collections.abc import Callable, Sequence
from math import sqrt

from financial_advisor.config import (
    DEFAULT_EVIDENCE_LIMIT,
    DEFAULT_MMR_RELEVANCE_WEIGHT,
    DEFAULT_RRF_RANK_CONSTANT,
)
from financial_advisor.contracts import RetrievalChannel, RetrievedEvidenceCandidate

Embed = Callable[[Sequence[str]], list[list[float]]]
Rerank = Callable[[str, Sequence[str]], list[float]]
ScoredCandidate = tuple[RetrievedEvidenceCandidate, float]


def reciprocal_rank_fusion(
    candidates: Sequence[RetrievedEvidenceCandidate],
    *,
    rank_constant: int = DEFAULT_RRF_RANK_CONSTANT,
) -> list[RetrievedEvidenceCandidate]:
    """Merge duplicate candidates and order them by combined channel rank."""

    if rank_constant <= 0:
        raise ValueError("RRF rank constant must be positive.")

    candidates_by_canonical_id: dict[
        str,
        dict[RetrievalChannel, RetrievedEvidenceCandidate],
    ] = {}
    for candidate in candidates:
        channel_candidates = candidates_by_canonical_id.setdefault(
            candidate.canonical_candidate_id,
            {},
        )
        current = channel_candidates.get(candidate.retrieval_channel)
        if current is None or candidate.rank < current.rank:
            channel_candidates[candidate.retrieval_channel] = candidate

    fused_candidates: list[tuple[RetrievedEvidenceCandidate, float]] = []
    for channel_candidates in candidates_by_canonical_id.values():
        representative = max(
            channel_candidates.values(),
            key=lambda candidate: candidate.retrieval_channel is RetrievalChannel.VECTOR,
        )
        vector_candidate = channel_candidates.get(RetrievalChannel.VECTOR)
        keyword_candidate = channel_candidates.get(RetrievalChannel.KEYWORD)
        retrieval_scores: dict[str, float | None] = {}
        if vector_candidate is not None:
            retrieval_scores["cosine_distance"] = vector_candidate.cosine_distance
        if keyword_candidate is not None:
            retrieval_scores["bm25_score"] = keyword_candidate.bm25_score
        representative = representative.model_copy(update=retrieval_scores)
        rrf_score = sum(
            1 / (rank_constant + candidate.rank) for candidate in channel_candidates.values()
        )
        fused_candidates.append((representative, rrf_score))

    fused_candidates.sort(key=lambda item: (-item[1], item[0].canonical_candidate_id))
    return [candidate for candidate, _score in fused_candidates]


def rerank_candidates(
    query: str,
    candidates: Sequence[RetrievedEvidenceCandidate],
    rerank: Rerank,
) -> list[ScoredCandidate]:
    """Order fused candidates by cross-encoder relevance."""

    if not candidates:
        return []
    relevance_scores = rerank(query, [candidate.text for candidate in candidates])
    if len(relevance_scores) != len(candidates):
        raise ValueError("Reranker must return one score per candidate.")

    ranked_candidates = list(zip(candidates, relevance_scores, strict=True))
    ranked_candidates.sort(key=lambda item: (-item[1], item[0].canonical_candidate_id))
    return ranked_candidates


def select_diverse_candidates(
    ranked_candidates: Sequence[ScoredCandidate],
    embed: Embed,
    *,
    limit: int = DEFAULT_EVIDENCE_LIMIT,
    relevance_weight: float = DEFAULT_MMR_RELEVANCE_WEIGHT,
) -> list[RetrievedEvidenceCandidate]:
    """Use maximal marginal relevance to avoid redundant final evidence."""

    if limit <= 0 or not 0 <= relevance_weight <= 1:
        raise ValueError("Invalid MMR selection arguments.")
    if not ranked_candidates:
        return []

    candidates = [candidate for candidate, _score in ranked_candidates]
    relevance = _normalize([score for _candidate, score in ranked_candidates])
    vectors = _candidate_vectors(candidates, embed)

    selected_indices: list[int] = []
    remaining_indices = set(range(len(candidates)))
    while remaining_indices and len(selected_indices) < limit:
        selected_vectors = [vectors[index] for index in selected_indices]
        winner = min(
            remaining_indices,
            key=lambda index: (
                -(
                    relevance_weight * relevance[index]
                    - (1 - relevance_weight) * _maximum_similarity(vectors[index], selected_vectors)
                ),
                index,
                candidates[index].canonical_candidate_id,
            ),
        )
        selected_indices.append(winner)
        remaining_indices.remove(winner)
    return [candidates[index] for index in selected_indices]


def select_candidates(
    query: str,
    candidates: Sequence[RetrievedEvidenceCandidate],
    rerank: Rerank,
    embed: Embed,
    *,
    limit: int = DEFAULT_EVIDENCE_LIMIT,
    rank_constant: int = DEFAULT_RRF_RANK_CONSTANT,
    relevance_weight: float = DEFAULT_MMR_RELEVANCE_WEIGHT,
) -> list[RetrievedEvidenceCandidate]:
    """Run RRF, cross-encoder reranking, and MMR in that order."""

    query = query.strip()
    if not query:
        raise ValueError("Evidence-selection query must not be blank.")
    fused = reciprocal_rank_fusion(candidates, rank_constant=rank_constant)
    reranked = rerank_candidates(query, fused, rerank)
    return select_diverse_candidates(
        reranked,
        embed,
        limit=limit,
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
    return sum(a * b for a, b in zip(left, right, strict=True)) / (left_size * right_size)


def _candidate_vectors(
    candidates: Sequence[RetrievedEvidenceCandidate],
    embed: Embed,
) -> list[list[float]]:
    vectors = [candidate.vector for candidate in candidates]
    missing_indices = [index for index, vector in enumerate(vectors) if vector is None]
    if missing_indices:
        generated_vectors = embed([candidates[index].embedding_text for index in missing_indices])
        if len(generated_vectors) != len(missing_indices):
            raise ValueError("Embedder must return one vector per candidate.")
        for index, vector in zip(missing_indices, generated_vectors, strict=True):
            vectors[index] = vector
    return [vector for vector in vectors if vector is not None]


def _normalize(scores: Sequence[float]) -> list[float]:
    minimum, maximum = min(scores), max(scores)
    if minimum == maximum:
        return [1.0] * len(scores)
    return [(score - minimum) / (maximum - minimum) for score in scores]


def _maximum_similarity(
    candidate_vector: Sequence[float],
    selected_vectors: Sequence[Sequence[float]],
) -> float:
    if not selected_vectors:
        return 0.0
    return max(
        0.0,
        max(
            cosine_similarity(candidate_vector, selected_vector)
            for selected_vector in selected_vectors
        ),
    )
