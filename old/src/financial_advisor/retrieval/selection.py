"""Deterministic RRF fusion, cross-encoder ranking, and MMR selection."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from financial_advisor.config import get_settings
from financial_advisor.retrieval.contracts import (
    CrossEncoderRerankingTrace,
    EvidenceSelectionTrace,
    FusedRetrievalCandidate,
    MmrSelectedCandidate,
    MmrSelectionTrace,
    QueryDocumentReranker,
    QueryEmbedder,
    ReciprocalRankFusionTrace,
    RerankedRetrievalCandidate,
    RetrievalEvidenceCandidate,
)
from financial_advisor.retrieval.similarity import cosine_similarity

_POLICY = get_settings().retrieval
DEFAULT_RRF_RANK_CONSTANT = _POLICY.rrf_rank_constant
DEFAULT_FINAL_EVIDENCE_LIMIT = _POLICY.final_evidence_limit
DEFAULT_MMR_RELEVANCE_WEIGHT = _POLICY.mmr_relevance_weight


@dataclass(frozen=True)
class _MmrScoredOption:
    """Internal score details for one candidate during one MMR selection round."""

    candidate_index: int
    mmr_score: float
    maximum_similarity: float | None


def fuse_retrieval_candidates(
    candidates: Sequence[RetrievalEvidenceCandidate],
    *,
    rank_constant: int = DEFAULT_RRF_RANK_CONSTANT,
) -> ReciprocalRankFusionTrace:
    """Fuse channel rankings by RRF without comparing their raw score scales.

    At most one contribution per `(canonical_candidate_id, retrieval_channel)`
    is counted: if a caller accidentally supplies a duplicate, only its better
    rank contributes. Expected vector and BM25 result lists already contain
    unique chunks, while future web retrieval will canonicalize URLs before it
    reaches this boundary.
    """

    if rank_constant <= 0:
        raise ValueError("rank_constant must be positive.")

    grouped: dict[str, dict[str, RetrievalEvidenceCandidate]] = {}
    for candidate in candidates:
        by_channel = grouped.setdefault(candidate.canonical_candidate_id, {})
        existing = by_channel.get(candidate.retrieval_channel)
        if existing is None or candidate.rank < existing.rank:
            by_channel[candidate.retrieval_channel] = candidate

    fused_candidates = []
    for canonical_candidate_id, by_channel in grouped.items():
        contributions = list(by_channel.values())
        representative = by_channel.get("internal_vector") or contributions[0]
        fused_candidates.append(
            FusedRetrievalCandidate(
                canonical_candidate_id=canonical_candidate_id,
                rrf_score=sum(1 / (rank_constant + candidate.rank) for candidate in contributions),
                # Canonical internal chunks have identical content across channels.
                # Prefer the vector record for a clear, fixed trace policy.
                representative=representative,
                contributions=contributions,
            )
        )

    fused_candidates.sort(
        key=lambda candidate: (-candidate.rrf_score, candidate.canonical_candidate_id)
    )
    return ReciprocalRankFusionTrace(
        rank_constant=rank_constant,
        input_candidate_count=len(candidates),
        canonical_candidate_count=len(fused_candidates),
        candidates=fused_candidates,
    )


def rerank_fused_candidates(
    query: str,
    candidates: Sequence[FusedRetrievalCandidate],
    reranker: QueryDocumentReranker,
) -> CrossEncoderRerankingTrace:
    """Score each query-chunk pair, then rank by cross-encoder relevance."""

    normalized_query = query.strip()
    if not normalized_query:
        raise ValueError("A reranking query must not be blank.")

    documents = [candidate.representative.text for candidate in candidates]
    scores = reranker.score(normalized_query, documents)
    if len(scores) != len(candidates):
        raise ValueError("Reranker must return one score for every supplied candidate.")

    scored_candidates = sorted(
        zip(candidates, scores, strict=True),
        key=lambda item: (-item[1], item[0].canonical_candidate_id),
    )
    reranked_candidates = [
        RerankedRetrievalCandidate(
            rank=rank,
            cross_encoder_score=score,
            candidate=candidate,
        )
        for rank, (candidate, score) in enumerate(scored_candidates, start=1)
    ]
    return CrossEncoderRerankingTrace(
        query=normalized_query,
        input_candidate_count=len(candidates),
        candidates=reranked_candidates,
    )


def select_mmr_candidates(
    candidates: Sequence[RerankedRetrievalCandidate],
    embedder: QueryEmbedder,
    *,
    selection_limit: int = DEFAULT_FINAL_EVIDENCE_LIMIT,
    relevance_weight: float = DEFAULT_MMR_RELEVANCE_WEIGHT,
) -> MmrSelectionTrace:
    """Select high-relevance evidence while penalizing similar selected chunks."""

    if selection_limit <= 0:
        raise ValueError("selection_limit must be positive.")
    if not 0 <= relevance_weight <= 1:
        raise ValueError("relevance_weight must be between 0 and 1.")
    if not candidates:
        return MmrSelectionTrace(
            requested_limit=selection_limit,
            relevance_weight=relevance_weight,
            input_candidate_count=0,
            selected_candidates=[],
        )

    document_vectors = _resolve_diversity_vectors(candidates, embedder)
    normalized_relevances = _normalize_scores(
        [candidate.cross_encoder_score for candidate in candidates]
    )

    remaining_indices = set(range(len(candidates)))
    selected_indices: list[int] = []
    selected_candidates: list[MmrSelectedCandidate] = []
    while remaining_indices and len(selected_candidates) < selection_limit:
        scored_options = [
            _score_mmr_option(
                candidate_index=index,
                document_vectors=document_vectors,
                selected_indices=selected_indices,
                normalized_relevance=normalized_relevances[index],
                relevance_weight=relevance_weight,
            )
            for index in remaining_indices
        ]
        winner = _select_best_mmr_option(scored_options, candidates)
        selected_index = winner.candidate_index
        remaining_indices.remove(selected_index)
        selected_indices.append(selected_index)
        selected_candidates.append(
            MmrSelectedCandidate(
                selection_order=len(selected_candidates) + 1,
                mmr_score=winner.mmr_score,
                normalized_relevance=normalized_relevances[selected_index],
                maximum_similarity_to_selected=winner.maximum_similarity,
                candidate=candidates[selected_index],
            )
        )

    return MmrSelectionTrace(
        requested_limit=selection_limit,
        relevance_weight=relevance_weight,
        input_candidate_count=len(candidates),
        selected_candidates=selected_candidates,
    )


def select_evidence_candidates(
    query: str,
    candidates: Sequence[RetrievalEvidenceCandidate],
    reranker: QueryDocumentReranker,
    diversity_embedder: QueryEmbedder,
    *,
    selection_limit: int = DEFAULT_FINAL_EVIDENCE_LIMIT,
    relevance_weight: float = DEFAULT_MMR_RELEVANCE_WEIGHT,
) -> EvidenceSelectionTrace:
    """Fuse all channels, rerank once, and select diverse final evidence."""

    normalized_query = query.strip()
    if not normalized_query:
        raise ValueError("An evidence-selection query must not be blank.")
    fusion = fuse_retrieval_candidates(candidates)
    reranking = rerank_fused_candidates(normalized_query, fusion.candidates, reranker)
    diversity_selection = select_mmr_candidates(
        reranking.candidates,
        diversity_embedder,
        selection_limit=selection_limit,
        relevance_weight=relevance_weight,
    )
    return EvidenceSelectionTrace(
        fusion=fusion,
        reranking=reranking,
        diversity_selection=diversity_selection,
    )


def _diversity_text(candidate: RerankedRetrievalCandidate) -> str:
    """Match the ingestion embedding template for candidates lacking stored vectors."""

    representative = candidate.candidate.representative
    return (
        f"Source: {representative.source_title}\n"
        f"Section: {' > '.join(representative.heading_path)}\n\n"
        f"{representative.text}"
    )


def _score_mmr_option(
    *,
    candidate_index: int,
    document_vectors: Sequence[Sequence[float]],
    selected_indices: Sequence[int],
    normalized_relevance: float,
    relevance_weight: float,
) -> _MmrScoredOption:
    """Compute relevance-minus-redundancy for one remaining candidate."""

    maximum_similarity = _maximum_redundancy_similarity(
        candidate_vector=document_vectors[candidate_index],
        selected_vectors=[document_vectors[index] for index in selected_indices],
    )
    mmr_score = relevance_weight * normalized_relevance - (1 - relevance_weight) * (
        maximum_similarity or 0.0
    )
    return _MmrScoredOption(
        candidate_index=candidate_index,
        mmr_score=mmr_score,
        maximum_similarity=maximum_similarity,
    )


def _select_best_mmr_option(
    options: Sequence[_MmrScoredOption], candidates: Sequence[RerankedRetrievalCandidate]
) -> _MmrScoredOption:
    """Choose highest MMR score, then cross-encoder rank and canonical ID."""

    return min(
        options,
        key=lambda option: (
            -option.mmr_score,
            candidates[option.candidate_index].rank,
            candidates[option.candidate_index].candidate.canonical_candidate_id,
        ),
    )


def _resolve_diversity_vectors(
    candidates: Sequence[RerankedRetrievalCandidate], embedder: QueryEmbedder
) -> list[list[float]]:
    """Reuse private LanceDB vectors and embed only candidates that lack one."""

    resolved_vectors = [item.candidate.representative.embedding for item in candidates]
    missing_indices = [index for index, vector in enumerate(resolved_vectors) if vector is None]
    if missing_indices:
        generated_vectors = embedder.embed(
            [_diversity_text(candidates[index]) for index in missing_indices]
        )
        if len(generated_vectors) != len(missing_indices):
            raise ValueError("Embedder must return one vector for every supplied candidate.")
        for index, vector in zip(missing_indices, generated_vectors, strict=True):
            resolved_vectors[index] = vector
    return [vector for vector in resolved_vectors if vector is not None]


def _normalize_scores(scores: Sequence[float]) -> list[float]:
    """Min-max normalize one reranking run without treating its scores as probabilities."""

    minimum = min(scores)
    maximum = max(scores)
    if minimum == maximum:
        return [1.0] * len(scores)
    return [(score - minimum) / (maximum - minimum) for score in scores]


def _maximum_redundancy_similarity(
    candidate_vector: Sequence[float], selected_vectors: Sequence[Sequence[float]]
) -> float | None:
    """Return the largest non-negative similarity to previously selected evidence."""

    if not selected_vectors:
        return None
    highest_similarity = max(
        cosine_similarity(candidate_vector, selected_vector) for selected_vector in selected_vectors
    )
    return max(0.0, highest_similarity)
