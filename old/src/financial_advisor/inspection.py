"""Read-only SQLite and LanceDB inspection models for operational observability."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import cast

import lancedb
from pydantic import BaseModel, Field, HttpUrl

from financial_advisor.config import get_settings
from financial_advisor.retrieval.knowledge_base.ingestion import KNOWLEDGE_TABLE

_METADATA_FIELDS = frozenset(
    {
        "chunk_id",
        "canonical_candidate_id",
        "source_id",
        "source_title",
        "publisher",
        "source_url",
        "topics",
        "heading",
        "heading_path",
        "section_position",
        "chunk_position",
        "token_count",
        "body_start_token",
        "body_end_token_exclusive",
        "source_content_hash",
        "retrieved_at",
        "embedding_dimension",
    }
)


class KnowledgeStoreUnavailable(RuntimeError):
    """Raised when a local LanceDB corpus cannot be inspected."""


class KnowledgeStoreSummary(BaseModel):
    """Small aggregate view of the curated local knowledge corpus."""

    table_name: str
    chunk_count: int = Field(ge=0)
    publishers: dict[str, int]
    topics: dict[str, int]
    embedding_model_name: str
    embedding_dimension: int | None = Field(default=None, ge=1)


class KnowledgeChunkInspection(BaseModel):
    """Chunk text plus complete human-readable provenance and embedding summary."""

    chunk_id: str
    canonical_candidate_id: str
    source_id: str
    source_title: str
    publisher: str
    source_url: HttpUrl
    topics: list[str]
    heading: str
    heading_path: list[str]
    section_position: int
    chunk_position: int
    token_count: int
    body_start_token: int | None = None
    body_end_token_exclusive: int | None = None
    source_content_hash: str | None = None
    retrieved_at: str | None = None
    embedding_dimension: int = Field(ge=1)
    text: str


class KnowledgeChunkPage(BaseModel):
    """Filtered bounded page of inspectable knowledge chunks."""

    total_matching_chunks: int = Field(ge=0)
    offset: int = Field(ge=0)
    limit: int = Field(ge=1, le=100)
    chunks: list[KnowledgeChunkInspection]


class KnowledgeStoreInspector:
    """Read local LanceDB records without exposing retrieval vectors to the UI."""

    def __init__(self, database_path: Path, *, table_name: str = KNOWLEDGE_TABLE) -> None:
        self._database_path = database_path
        self._table_name = table_name

    def summary(self) -> KnowledgeStoreSummary:
        """Count chunks by publisher and topic for the inspection overview."""

        records = self._records()
        publishers = Counter(cast(str, record["publisher"]) for record in records)
        topics = Counter(
            topic
            for record in records
            for topic in cast(list[str], record["topics"])
        )
        return KnowledgeStoreSummary(
            table_name=self._table_name,
            chunk_count=len(records),
            publishers=dict(sorted(publishers.items())),
            topics=dict(sorted(topics.items())),
            embedding_model_name=get_settings().model_runtime.embedding_model_name,
            embedding_dimension=(
                len(cast(list[float], records[0]["vector"])) if records else None
            ),
        )

    def chunks(
        self,
        *,
        publisher: str | None = None,
        topic: str | None = None,
        source_id: str | None = None,
        chunk_id: str | None = None,
        metadata_field: str | None = None,
        metadata_value: str | None = None,
        min_token_count: int | None = None,
        max_token_count: int | None = None,
        offset: int = 0,
        limit: int = 50,
    ) -> KnowledgeChunkPage:
        """Filter deterministic metadata locally and return readable chunk details."""

        if not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100.")
        if offset < 0:
            raise ValueError("offset must be non-negative.")
        if min_token_count is not None and min_token_count < 1:
            raise ValueError("min_token_count must be at least 1.")
        if max_token_count is not None and max_token_count < 1:
            raise ValueError("max_token_count must be at least 1.")
        if (
            min_token_count is not None
            and max_token_count is not None
            and min_token_count > max_token_count
        ):
            raise ValueError("min_token_count must not exceed max_token_count.")
        normalized_publisher = publisher.strip().casefold() if publisher else None
        normalized_topic = topic.strip().casefold() if topic else None
        normalized_source_id = source_id.strip() if source_id else None
        normalized_chunk_id = chunk_id.strip() if chunk_id else None
        normalized_metadata_field = metadata_field.strip() if metadata_field else None
        normalized_metadata_value = metadata_value.strip().casefold() if metadata_value else None
        if normalized_metadata_field is None and normalized_metadata_value is not None:
            raise ValueError("A metadata filter value requires a metadata field.")
        if (
            normalized_metadata_field is not None
            and normalized_metadata_field not in _METADATA_FIELDS
        ):
            raise ValueError("Metadata field is not available for chunk inspection.")
        matching_records = [
            record
            for record in self._records()
            if (
                normalized_publisher is None
                or cast(str, record["publisher"]).casefold() == normalized_publisher
            )
            and (
                normalized_topic is None
                or normalized_topic
                in (item.casefold() for item in cast(list[str], record["topics"]))
            )
            and (
                normalized_source_id is None or record["source_id"] == normalized_source_id
            )
            and (normalized_chunk_id is None or record["chunk_id"] == normalized_chunk_id)
            and (
                min_token_count is None
                or cast(int, record["token_count"]) >= min_token_count
            )
            and (
                max_token_count is None
                or cast(int, record["token_count"]) <= max_token_count
            )
            and _matches_metadata(
                record,
                field=normalized_metadata_field,
                value=normalized_metadata_value,
            )
        ]
        chunks = [
            KnowledgeChunkInspection.model_validate(
                {
                    **record,
                    "canonical_candidate_id": record.get(
                        "canonical_candidate_id", record["chunk_id"]
                    ),
                    "embedding_dimension": len(cast(list[float], record["vector"])),
                    "retrieved_at": (
                        str(record["retrieved_at"])
                        if record.get("retrieved_at") is not None
                        else None
                    ),
                }
            )
            for record in matching_records[offset : offset + limit]
        ]
        return KnowledgeChunkPage(
            total_matching_chunks=len(matching_records),
            offset=offset,
            limit=limit,
            chunks=chunks,
        )

    def _records(self) -> list[dict[str, object]]:
        """Read the configured knowledge corpus and normalize Arrow records for inspection."""

        if not self._database_path.exists():
            raise KnowledgeStoreUnavailable(
                f"LanceDB corpus was not found at {self._database_path}."
            )
        database = lancedb.connect(str(self._database_path))
        table_names = set(database.list_tables().tables)
        if self._table_name not in table_names:
            raise KnowledgeStoreUnavailable(
                f"LanceDB table {self._table_name!r} was not found in the local corpus."
            )
        records = database.open_table(self._table_name).to_arrow().to_pylist()
        return [dict(record) for record in records]


def _matches_metadata(
    record: dict[str, object], *, field: str | None, value: str | None
) -> bool:
    """Match one allowlisted metadata value without exposing arbitrary database queries."""

    if field is None or value is None:
        return True
    if field == "canonical_candidate_id":
        candidate_value: object = record.get("canonical_candidate_id", record["chunk_id"])
    elif field == "heading_path":
        candidate_value = " > ".join(cast(list[str], record["heading_path"]))
    elif field == "embedding_dimension":
        candidate_value = len(cast(list[float], record["vector"]))
    else:
        candidate_value = record.get(field)
    if isinstance(candidate_value, list):
        return any(value in str(item).casefold() for item in candidate_value)
    return candidate_value is not None and value in str(candidate_value).casefold()
