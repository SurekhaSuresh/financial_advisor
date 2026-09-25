"""LanceDB vector and BM25 candidate retrievers for the curated knowledge base."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import lancedb
from lancedb.index import FTS
from pydantic import BaseModel, Field, HttpUrl

from financial_advisor.config import get_settings
from financial_advisor.retrieval.contracts import QueryEmbedder, RetrievalEvidenceCandidate
from financial_advisor.retrieval.knowledge_base.ingestion import KNOWLEDGE_TABLE

_POLICY = get_settings().retrieval
DEFAULT_VECTOR_CANDIDATE_LIMIT = _POLICY.local_vector_candidate_limit
DEFAULT_KEYWORD_CANDIDATE_LIMIT = _POLICY.local_keyword_candidate_limit
KNOWLEDGE_TEXT_FTS_INDEX = "knowledge_text_fts"


class VectorRetrievalTrace(BaseModel):
    """Inspectable deterministic record of one local vector-search call."""

    query: str = Field(min_length=1)
    candidate_limit: int = Field(ge=1)
    candidates: list[RetrievalEvidenceCandidate]


class KeywordRetrievalTrace(BaseModel):
    """Inspectable deterministic record of one local BM25-search call."""

    query: str = Field(min_length=1)
    candidate_limit: int = Field(ge=1)
    candidates: list[RetrievalEvidenceCandidate]


class LanceVectorRetriever:
    """Embed one query and retrieve its nearest local knowledge chunks."""

    def __init__(
        self,
        database_path: Path,
        embedder: QueryEmbedder,
        *,
        table_name: str = KNOWLEDGE_TABLE,
    ) -> None:
        self._database_path = database_path
        self._embedder = embedder
        self._table_name = table_name

    def search(
        self,
        query: str,
        *,
        candidate_limit: int = DEFAULT_VECTOR_CANDIDATE_LIMIT,
    ) -> VectorRetrievalTrace:
        """Run exact cosine search and preserve every returned candidate's provenance."""

        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("A vector retrieval query must not be blank.")
        if candidate_limit <= 0:
            raise ValueError("candidate_limit must be positive.")
        if not self._database_path.exists():
            raise FileNotFoundError(f"LanceDB directory does not exist: {self._database_path}")

        query_embeddings = self._embedder.embed([normalized_query])
        if len(query_embeddings) != 1:
            raise ValueError("Query embedder must return exactly one vector for one query.")

        table = lancedb.connect(str(self._database_path)).open_table(self._table_name)
        rows = (
            table.search(query_embeddings[0], vector_column_name="vector")
            .metric(  # type: ignore[attr-defined]
                "cosine"
            )
            .limit(candidate_limit)
            .to_list()
        )
        candidates = [
            RetrievalEvidenceCandidate(
                canonical_candidate_id=cast(
                    str, row.get("canonical_candidate_id", row["chunk_id"])
                ),
                retrieval_channel="internal_vector",
                rank=rank,
                chunk_id=cast(str, row["chunk_id"]),
                source_id=cast(str, row["source_id"]),
                source_title=cast(str, row["source_title"]),
                publisher=cast(str, row["publisher"]),
                source_url=cast(HttpUrl, row["source_url"]),
                topics=cast(list[str], row["topics"]),
                heading=cast(str, row["heading"]),
                heading_path=cast(list[str], row["heading_path"]),
                section_position=cast(int, row["section_position"]),
                chunk_position=cast(int, row["chunk_position"]),
                token_count=cast(int, row["token_count"]),
                text=cast(str, row["text"]),
                cosine_distance=cast(float, row["_distance"]),
                embedding=cast(list[float], row["vector"]),
            )
            for rank, row in enumerate(rows, start=1)
        ]
        return VectorRetrievalTrace(
            query=normalized_query,
            candidate_limit=candidate_limit,
            candidates=candidates,
        )


class LanceKeywordRetriever:
    """Create and query the local BM25 full-text index over chunk bodies."""

    def __init__(
        self,
        database_path: Path,
        *,
        table_name: str = KNOWLEDGE_TABLE,
        index_name: str = KNOWLEDGE_TEXT_FTS_INDEX,
    ) -> None:
        self._database_path = database_path
        self._table_name = table_name
        self._index_name = index_name

    def search(
        self,
        query: str,
        *,
        candidate_limit: int = DEFAULT_KEYWORD_CANDIDATE_LIMIT,
    ) -> KeywordRetrievalTrace:
        """Run BM25 search over chunk text and preserve each result's provenance."""

        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("A keyword retrieval query must not be blank.")
        if candidate_limit <= 0:
            raise ValueError("candidate_limit must be positive.")

        table = self._open_table()
        self._ensure_text_index(table)
        rows = table.search(normalized_query, query_type="fts").limit(candidate_limit).to_list()
        candidates = [
            RetrievalEvidenceCandidate(
                canonical_candidate_id=cast(
                    str, row.get("canonical_candidate_id", row["chunk_id"])
                ),
                retrieval_channel="internal_bm25",
                rank=rank,
                chunk_id=cast(str, row["chunk_id"]),
                source_id=cast(str, row["source_id"]),
                source_title=cast(str, row["source_title"]),
                publisher=cast(str, row["publisher"]),
                source_url=cast(HttpUrl, row["source_url"]),
                topics=cast(list[str], row["topics"]),
                heading=cast(str, row["heading"]),
                heading_path=cast(list[str], row["heading_path"]),
                section_position=cast(int, row["section_position"]),
                chunk_position=cast(int, row["chunk_position"]),
                token_count=cast(int, row["token_count"]),
                text=cast(str, row["text"]),
                bm25_score=cast(float, row["_score"]),
                embedding=cast(list[float], row["vector"]),
            )
            for rank, row in enumerate(rows, start=1)
        ]
        return KeywordRetrievalTrace(
            query=normalized_query,
            candidate_limit=candidate_limit,
            candidates=candidates,
        )

    def _open_table(self) -> lancedb.table.Table:
        if not self._database_path.exists():
            raise FileNotFoundError(f"LanceDB directory does not exist: {self._database_path}")
        return lancedb.connect(str(self._database_path)).open_table(self._table_name)

    def _ensure_text_index(self, table: lancedb.table.Table) -> None:
        existing_index_names = {index.name for index in table.list_indices()}
        if self._index_name not in existing_index_names:
            table.create_index(
                "text",
                config=FTS(stem=True, remove_stop_words=True),
                name=self._index_name,
            )
