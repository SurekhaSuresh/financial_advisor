"""Hybrid vector and BM25 search over the curated LanceDB store."""

from pathlib import Path

import lancedb

from financial_advisor.config import (
    DEFAULT_KNOWLEDGE_CANDIDATE_LIMIT,
    KNOWLEDGE_TABLE_NAME,
    Embed,
)
from financial_advisor.contracts import RetrievalChannel, RetrievedEvidenceCandidate


class LocalHybridRetriever:
    """Retrieve evidence candidates from the curated local knowledge store."""

    def __init__(self, database_path: Path, embed: Embed) -> None:
        self.database_path = database_path
        self.embed = embed

    def search(self, query: str) -> list[RetrievedEvidenceCandidate]:
        """Search the same chunks by meaning and keywords."""

        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("Knowledge query must not be blank.")
        if not self.database_path.exists():
            raise FileNotFoundError(f"Knowledge store does not exist: {self.database_path}")

        knowledge_table = lancedb.connect(str(self.database_path)).open_table(KNOWLEDGE_TABLE_NAME)
        [query_vector] = self.embed([normalized_query])

        vector_search_rows = (
            knowledge_table.search(query_vector, vector_column_name="vector")
            .metric("cosine")  # type: ignore[attr-defined]
            .limit(DEFAULT_KNOWLEDGE_CANDIDATE_LIMIT)
            .to_list()
        )
        keyword_search_rows = (
            knowledge_table.search(normalized_query, query_type="fts")
            .limit(DEFAULT_KNOWLEDGE_CANDIDATE_LIMIT)
            .to_list()
        )

        candidates = []
        for retrieval_channel, search_rows in (
            (RetrievalChannel.VECTOR, vector_search_rows),
            (RetrievalChannel.KEYWORD, keyword_search_rows),
        ):
            for rank, knowledge_row in enumerate(search_rows, start=1):
                knowledge_row["retrieval_channel"] = retrieval_channel
                knowledge_row["rank"] = rank
                knowledge_row["cosine_distance"] = knowledge_row.pop("_distance", None)
                knowledge_row["bm25_score"] = knowledge_row.pop("_score", None)
                candidates.append(RetrievedEvidenceCandidate.model_validate(knowledge_row))

        return candidates
