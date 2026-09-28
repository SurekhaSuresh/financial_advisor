"""Read-only inspection of local application data."""

from pathlib import Path

import lancedb
from pydantic import BaseModel

from financial_advisor.config import KNOWLEDGE_TABLE_NAME
from financial_advisor.contracts import DocumentChunk


class KnowledgeStoreInspection(BaseModel):
    publishers: set[str]
    topics: set[str]
    total_matching_chunks: int
    offset: int
    chunks: list[DocumentChunk]


def inspect_knowledge_chunks(
    database_path: Path,
    *,
    publisher: str | None = None,
    topic: str | None = None,
    metadata_field: str | None = None,
    metadata_value: str | None = None,
    min_token_count: int | None = None,
    max_token_count: int | None = None,
    offset: int = 0,
    limit: int = 50,
) -> KnowledgeStoreInspection:
    if min_token_count and max_token_count and min_token_count > max_token_count:
        raise ValueError("Minimum token count must not exceed maximum token count.")
    if not database_path.exists():
        raise FileNotFoundError(f"Knowledge store does not exist: {database_path}")

    database = lancedb.connect(str(database_path))
    if KNOWLEDGE_TABLE_NAME not in database.list_tables().tables:
        raise FileNotFoundError(f"Knowledge table does not exist: {KNOWLEDGE_TABLE_NAME}")
    knowledge_rows = database.open_table(KNOWLEDGE_TABLE_NAME).to_arrow().to_pylist()
    chunks = [DocumentChunk.model_validate(row) for row in knowledge_rows]

    matching_chunks = []
    for chunk in chunks:
        if publisher and chunk.publisher.lower() != publisher.lower():
            continue
        if topic and topic.lower() not in (value.lower() for value in chunk.topics):
            continue
        if min_token_count and chunk.token_count < min_token_count:
            continue
        if max_token_count and chunk.token_count > max_token_count:
            continue
        if (
            metadata_field
            and metadata_value
            and metadata_value.lower() not in str(getattr(chunk, metadata_field, "")).lower()
        ):
            continue
        matching_chunks.append(chunk)

    return KnowledgeStoreInspection(
        publishers={chunk.publisher for chunk in chunks},
        topics={topic for chunk in chunks for topic in chunk.topics},
        total_matching_chunks=len(matching_chunks),
        offset=offset,
        chunks=matching_chunks[offset : offset + limit],
    )
