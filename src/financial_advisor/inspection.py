"""Read-only views of ADK SQLite sessions and local knowledge chunks."""

import sqlite3
from collections import Counter
from pathlib import Path
from typing import cast

import lancedb
from pydantic import BaseModel, Field, HttpUrl

from financial_advisor.config import EMBEDDING_MODEL_NAME, KNOWLEDGE_TABLE_NAME

_ADK_SQLITE_TABLES = frozenset(
    {"adk_internal_metadata", "sessions", "events", "app_states", "user_states"}
)
_KNOWLEDGE_METADATA_FIELDS = frozenset(
    {
        "chunk_id",
        "canonical_candidate_id",
        "source_id",
        "source_title",
        "publisher",
        "source_url",
        "topics",
        "heading_path",
        "section_position",
        "chunk_position",
        "token_count",
    }
)


class SqliteTableSummary(BaseModel):
    table_name: str
    row_count: int = Field(ge=0)


class SqliteTableRows(BaseModel):
    table_name: str
    columns: list[str]
    rows: list[dict[str, object]]


class KnowledgeStoreSummary(BaseModel):
    table_name: str
    chunk_count: int = Field(ge=0)
    publishers: dict[str, int]
    topics: dict[str, int]
    embedding_model_name: str
    embedding_dimension: int | None = Field(default=None, ge=1)


class KnowledgeChunkInspection(BaseModel):
    chunk_id: str
    canonical_candidate_id: str
    source_id: str
    source_title: str
    publisher: str
    source_url: HttpUrl
    topics: list[str]
    heading_path: list[str]
    section_position: int
    chunk_position: int
    token_count: int
    embedding_dimension: int = Field(ge=1)
    text: str


class KnowledgeChunkPage(BaseModel):
    total_matching_chunks: int = Field(ge=0)
    offset: int = Field(ge=0)
    limit: int = Field(ge=1, le=100)
    chunks: list[KnowledgeChunkInspection]


class SqliteInspector:
    """Inspect only the ADK tables used by this application."""

    def __init__(self, database_url: str) -> None:
        url_prefix = "sqlite+aiosqlite:///"
        if not database_url.startswith(url_prefix):
            raise ValueError("SQLite inspection requires a sqlite+aiosqlite database URL.")
        self.database_path = Path(database_url.removeprefix(url_prefix))

    def tables(self) -> list[SqliteTableSummary]:
        if not self.database_path.exists():
            return []
        with self._connect() as connection:
            existing_tables = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            }
            return [
                SqliteTableSummary(
                    table_name=table_name,
                    row_count=connection.execute(
                        f'SELECT COUNT(*) FROM "{table_name}"'
                    ).fetchone()[0],
                )
                for table_name in sorted(_ADK_SQLITE_TABLES & existing_tables)
            ]

    def rows(self, table_name: str, *, limit: int = 50) -> SqliteTableRows:
        if table_name not in _ADK_SQLITE_TABLES:
            raise ValueError("Table is not available for inspection.")
        if not 1 <= limit <= 100:
            raise ValueError("Row limit must be between 1 and 100.")
        if not self.database_path.exists():
            raise ValueError("The ADK session database is not available.")

        with self._connect() as connection:
            cursor = connection.execute(
                f'SELECT * FROM "{table_name}" LIMIT ?',
                (limit,),
            )
            columns = [column[0] for column in cursor.description or []]
            rows = [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]
        return SqliteTableRows(table_name=table_name, columns=columns, rows=rows)

    def _connect(self) -> sqlite3.Connection:
        database_uri = f"file:{self.database_path.resolve()}?mode=ro"
        return sqlite3.connect(database_uri, uri=True)


class KnowledgeStoreInspector:
    """Inspect local knowledge metadata without exposing stored vectors."""

    def __init__(self, database_path: Path, table_name: str = KNOWLEDGE_TABLE_NAME) -> None:
        self.database_path = database_path
        self.table_name = table_name

    def summary(self) -> KnowledgeStoreSummary:
        records = self._records()
        publishers = Counter(cast(str, record["publisher"]) for record in records)
        topics = Counter(
            topic
            for record in records
            for topic in cast(list[str], record["topics"])
        )
        return KnowledgeStoreSummary(
            table_name=self.table_name,
            chunk_count=len(records),
            publishers=dict(sorted(publishers.items())),
            topics=dict(sorted(topics.items())),
            embedding_model_name=EMBEDDING_MODEL_NAME,
            embedding_dimension=(
                len(cast(list[float], records[0]["vector"])) if records else None
            ),
        )

    def chunks(
        self,
        *,
        publisher: str | None = None,
        topic: str | None = None,
        metadata_field: str | None = None,
        metadata_value: str | None = None,
        min_token_count: int | None = None,
        max_token_count: int | None = None,
        offset: int = 0,
        limit: int = 50,
    ) -> KnowledgeChunkPage:
        if not 1 <= limit <= 100:
            raise ValueError("Chunk limit must be between 1 and 100.")
        if offset < 0:
            raise ValueError("Chunk offset must not be negative.")
        if min_token_count is not None and min_token_count < 1:
            raise ValueError("Minimum token count must be positive.")
        if max_token_count is not None and max_token_count < 1:
            raise ValueError("Maximum token count must be positive.")
        if (
            min_token_count is not None
            and max_token_count is not None
            and min_token_count > max_token_count
        ):
            raise ValueError("Minimum token count must not exceed maximum token count.")
        if metadata_field is not None and metadata_field not in _KNOWLEDGE_METADATA_FIELDS:
            raise ValueError("Metadata field is not available for inspection.")
        if metadata_value is not None and metadata_field is None:
            raise ValueError("A metadata value requires a metadata field.")

        normalized_publisher = publisher.casefold() if publisher else None
        normalized_topic = topic.casefold() if topic else None
        normalized_metadata_value = metadata_value.casefold() if metadata_value else None
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
                min_token_count is None
                or cast(int, record["token_count"]) >= min_token_count
            )
            and (
                max_token_count is None
                or cast(int, record["token_count"]) <= max_token_count
            )
            and self._matches_metadata(
                record,
                metadata_field,
                normalized_metadata_value,
            )
        ]
        chunks = [
            KnowledgeChunkInspection.model_validate(
                {
                    **record,
                    "embedding_dimension": len(cast(list[float], record["vector"])),
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
        if not self.database_path.exists():
            raise FileNotFoundError(f"Knowledge store does not exist: {self.database_path}")
        database = lancedb.connect(str(self.database_path))
        if self.table_name not in set(database.list_tables().tables):
            raise FileNotFoundError(f"Knowledge table does not exist: {self.table_name}")
        return [
            dict(record)
            for record in database.open_table(self.table_name).to_arrow().to_pylist()
        ]

    @staticmethod
    def _matches_metadata(
        record: dict[str, object],
        metadata_field: str | None,
        metadata_value: str | None,
    ) -> bool:
        if metadata_field is None or metadata_value is None:
            return True
        field_value = record.get(metadata_field)
        if isinstance(field_value, list):
            return any(metadata_value in str(item).casefold() for item in field_value)
        return field_value is not None and metadata_value in str(field_value).casefold()
