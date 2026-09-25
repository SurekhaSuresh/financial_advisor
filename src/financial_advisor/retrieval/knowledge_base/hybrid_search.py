"""Hybrid vector and BM25 search over the curated LanceDB store."""

from pathlib import Path

import lancedb

from financial_advisor.config import (
    DEFAULT_KNOWLEDGE_CANDIDATE_LIMIT,
    KNOWLEDGE_TABLE_NAME,
)
from financial_advisor.contracts import Embed, RetrievalChannel, RetrievedEvidenceCandidate


class LocalHybridRetriever:
    """Retrieve evidence candidates from the curated local knowledge store."""

    def __init__(
        self,
        database_path: Path,
        embed: Embed,
        *,
        table_name: str = KNOWLEDGE_TABLE_NAME,
        candidate_limit: int = DEFAULT_KNOWLEDGE_CANDIDATE_LIMIT,
    ) -> None:
        if candidate_limit <= 0:
            raise ValueError("Knowledge candidate limit must be positive.")
        self.database_path = database_path
        self.embed = embed
        self.table_name = table_name
        self.candidate_limit = candidate_limit

    def search(self, query: str) -> list[RetrievedEvidenceCandidate]:
        """Search the same chunks by meaning and keywords."""

        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("Knowledge query must not be blank.")
        if not self.database_path.exists():
            raise FileNotFoundError(f"Knowledge store does not exist: {self.database_path}")

        knowledge_table = lancedb.connect(str(self.database_path)).open_table(self.table_name)
        query_vectors = self.embed([normalized_query])
        if len(query_vectors) != 1:
            raise ValueError("Embedder must return exactly one query vector.")
        query_vector = query_vectors[0]

        vector_search_rows = (
            knowledge_table.search(query_vector, vector_column_name="vector")
            .metric("cosine")  # type: ignore[attr-defined]
            .limit(self.candidate_limit)
            .to_list()
        )
        keyword_search_rows = (
            knowledge_table.search(normalized_query, query_type="fts")
            .limit(self.candidate_limit)
            .to_list()
        )

        vector_candidates = [
            self._create_knowledge_candidate(row, RetrievalChannel.VECTOR, rank)
            for rank, row in enumerate(vector_search_rows, start=1)
        ]
        keyword_candidates = [
            self._create_knowledge_candidate(row, RetrievalChannel.KEYWORD, rank)
            for rank, row in enumerate(keyword_search_rows, start=1)
        ]
        return [*vector_candidates, *keyword_candidates]

    @staticmethod
    def _create_knowledge_candidate(
        knowledge_row: dict[str, object],
        retrieval_channel: RetrievalChannel,
        rank: int,
    ) -> RetrievedEvidenceCandidate:
        """Create a retrieval candidate from one LanceDB row."""

        candidate_fields = dict(knowledge_row)
        candidate_fields["retrieval_channel"] = retrieval_channel
        candidate_fields["rank"] = rank
        candidate_fields["cosine_distance"] = candidate_fields.pop("_distance", None)
        candidate_fields["bm25_score"] = candidate_fields.pop("_score", None)
        return RetrievedEvidenceCandidate.model_validate(candidate_fields)
