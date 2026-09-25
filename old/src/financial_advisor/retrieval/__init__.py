"""Public retrieval building blocks used by application composition and tests."""

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
from financial_advisor.retrieval.knowledge_base.retrievers import (
    KeywordRetrievalTrace,
    LanceKeywordRetriever,
    LanceVectorRetriever,
    VectorRetrievalTrace,
)
from financial_advisor.retrieval.reranker import FastEmbedCrossEncoderReranker
from financial_advisor.retrieval.selection import (
    DEFAULT_FINAL_EVIDENCE_LIMIT,
    DEFAULT_MMR_RELEVANCE_WEIGHT,
    DEFAULT_RRF_RANK_CONSTANT,
    fuse_retrieval_candidates,
    rerank_fused_candidates,
    select_evidence_candidates,
    select_mmr_candidates,
)

__all__ = [
    "CrossEncoderRerankingTrace",
    "DEFAULT_FINAL_EVIDENCE_LIMIT",
    "DEFAULT_MMR_RELEVANCE_WEIGHT",
    "DEFAULT_RRF_RANK_CONSTANT",
    "EvidenceSelectionTrace",
    "FastEmbedCrossEncoderReranker",
    "FusedRetrievalCandidate",
    "KeywordRetrievalTrace",
    "LanceKeywordRetriever",
    "LanceVectorRetriever",
    "MmrSelectedCandidate",
    "MmrSelectionTrace",
    "QueryDocumentReranker",
    "QueryEmbedder",
    "ReciprocalRankFusionTrace",
    "RerankedRetrievalCandidate",
    "RetrievalEvidenceCandidate",
    "VectorRetrievalTrace",
    "fuse_retrieval_candidates",
    "rerank_fused_candidates",
    "select_evidence_candidates",
    "select_mmr_candidates",
]
