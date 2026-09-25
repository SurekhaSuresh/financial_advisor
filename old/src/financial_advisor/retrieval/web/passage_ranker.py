"""In-memory vector retrieval over freshly fetched live-web chunks."""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, Field

from financial_advisor.config import get_settings
from financial_advisor.retrieval.contracts import QueryEmbedder, RetrievalEvidenceCandidate
from financial_advisor.retrieval.similarity import cosine_similarity
from financial_advisor.retrieval.web.page_processing import WebEvidenceChunk

DEFAULT_WEB_VECTOR_CANDIDATE_LIMIT = get_settings().retrieval.web_passage_candidate_limit


class WebVectorRetrievalTrace(BaseModel):
    """Inspectable query-to-live-web chunk ranking result for one research task."""

    query: str = Field(min_length=1)
    fetched_chunk_count: int = Field(ge=0)
    candidate_limit: int = Field(ge=1)
    candidates: list[RetrievalEvidenceCandidate]


class InMemoryWebVectorRetriever:
    """Rank request-scoped web chunks with the same BGE vector semantics as local retrieval."""

    def __init__(self, embedder: QueryEmbedder) -> None:
        self._embedder = embedder

    def search(
        self,
        query: str,
        chunks: Sequence[WebEvidenceChunk],
        *,
        candidate_limit: int = DEFAULT_WEB_VECTOR_CANDIDATE_LIMIT,
    ) -> WebVectorRetrievalTrace:
        """Embed a query and page chunks once, then return top exact-cosine candidates."""

        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("A web vector retrieval query must not be blank.")
        if candidate_limit <= 0:
            raise ValueError("candidate_limit must be positive.")
        if not chunks:
            return WebVectorRetrievalTrace(
                query=normalized_query,
                fetched_chunk_count=0,
                candidate_limit=candidate_limit,
                candidates=[],
            )

        vectors = self._embedder.embed(
            [normalized_query, *(chunk.embedding_text for chunk in chunks)]
        )
        if len(vectors) != len(chunks) + 1:
            raise ValueError("Embedder must return one vector for the query and every web chunk.")
        query_vector, document_vectors = vectors[0], vectors[1:]
        best_by_canonical_id: dict[str, tuple[WebEvidenceChunk, list[float], float]] = {}
        for chunk, document_vector in zip(chunks, document_vectors, strict=True):
            similarity = cosine_similarity(query_vector, document_vector)
            existing = best_by_canonical_id.get(chunk.canonical_candidate_id)
            if existing is None or similarity > existing[2]:
                best_by_canonical_id[chunk.canonical_candidate_id] = (
                    chunk,
                    document_vector,
                    similarity,
                )

        ranked = sorted(
            best_by_canonical_id.values(),
            key=lambda item: (-item[2], item[0].canonical_candidate_id),
        )[:candidate_limit]
        candidates = [
            RetrievalEvidenceCandidate(
                **chunk.model_dump(),
                retrieval_channel="live_web",
                rank=rank,
                cosine_distance=max(0.0, 1 - similarity),
                embedding=document_vector,
            )
            for rank, (chunk, document_vector, similarity) in enumerate(ranked, start=1)
        ]
        return WebVectorRetrievalTrace(
            query=normalized_query,
            fetched_chunk_count=len(chunks),
            candidate_limit=candidate_limit,
            candidates=candidates,
        )
