"""Pydantic contracts exchanged across retrieval channels and selection stages."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, Protocol

from pydantic import BaseModel, Field

from financial_advisor.retrieval.content_processing.chunking import SourceEvidenceChunk


class QueryEmbedder(Protocol):
    """The minimal embedding behavior required by vector retrieval."""

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class QueryDocumentReranker(Protocol):
    """The pair-scoring behavior required by the reranking stage."""

    def score(self, query: str, documents: Sequence[str]) -> list[float]: ...


class RetrievalEvidenceCandidate(SourceEvidenceChunk):
    """One normalized pre-fusion evidence candidate from any retrieval channel."""

    retrieval_channel: Literal[
        "internal_vector",
        "internal_bm25",
        "live_web",
        "authoritative_web",
        "broad_web",
    ]
    rank: int = Field(ge=1)
    cosine_distance: float | None = Field(default=None, ge=0)
    bm25_score: float | None = None
    embedding: list[float] | None = Field(default=None, exclude=True)


class FusedRetrievalCandidate(BaseModel):
    """One canonical candidate with rank contributions from every retrieval channel."""

    canonical_candidate_id: str
    rrf_score: float = Field(gt=0)
    representative: RetrievalEvidenceCandidate
    contributions: list[RetrievalEvidenceCandidate] = Field(min_length=1)


class ReciprocalRankFusionTrace(BaseModel):
    """Inspectable deterministic record of rank-based candidate fusion."""

    rank_constant: int = Field(gt=0)
    input_candidate_count: int = Field(ge=0)
    canonical_candidate_count: int = Field(ge=0)
    candidates: list[FusedRetrievalCandidate]


class RerankedRetrievalCandidate(BaseModel):
    """One fused candidate with its cross-encoder relevance score and rank."""

    rank: int = Field(ge=1)
    cross_encoder_score: float
    candidate: FusedRetrievalCandidate


class CrossEncoderRerankingTrace(BaseModel):
    """Inspectable deterministic record of one query-document reranking call."""

    query: str = Field(min_length=1)
    input_candidate_count: int = Field(ge=0)
    candidates: list[RerankedRetrievalCandidate]


class MmrSelectedCandidate(BaseModel):
    """One final candidate selected for relevance and non-redundancy."""

    selection_order: int = Field(ge=1)
    mmr_score: float
    normalized_relevance: float = Field(ge=0, le=1)
    maximum_similarity_to_selected: float | None = None
    candidate: RerankedRetrievalCandidate


class MmrSelectionTrace(BaseModel):
    """Inspectable deterministic record of final evidence diversity selection."""

    requested_limit: int = Field(ge=1)
    relevance_weight: float = Field(ge=0, le=1)
    input_candidate_count: int = Field(ge=0)
    selected_candidates: list[MmrSelectedCandidate]


class EvidenceSelectionTrace(BaseModel):
    """Shared post-retrieval evidence selection trajectory for any candidate channels."""

    fusion: ReciprocalRankFusionTrace
    reranking: CrossEncoderRerankingTrace
    diversity_selection: MmrSelectionTrace
