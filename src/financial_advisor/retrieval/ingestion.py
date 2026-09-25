"""Build the curated local knowledge store from approved snapshots."""

import json
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from time import sleep
from typing import Literal
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import lancedb
import yaml
from lancedb.index import FTS
from pydantic import BaseModel, Field, HttpUrl

from financial_advisor.config import (
    KNOWLEDGE_CHUNK_MAX_TOKENS,
    KNOWLEDGE_CHUNK_MIN_TOKENS,
    KNOWLEDGE_CHUNK_OVERLAP_TOKENS,
    KNOWLEDGE_FETCH_ATTEMPTS,
    KNOWLEDGE_FETCH_RETRY_DELAY_SECONDS,
    KNOWLEDGE_FETCH_TIMEOUT_SECONDS,
    KNOWLEDGE_FETCH_USER_AGENT,
    KNOWLEDGE_TABLE_NAME,
    KNOWLEDGE_TEXT_INDEX_NAME,
)
from financial_advisor.retrieval.documents import chunk_document, parse_document
from financial_advisor.retrieval.text_models import LocalRetrievalModels


class KnowledgeSource(BaseModel):
    id: str = Field(pattern=r"^[a-z0-9_]+$")
    publisher: str
    title: str
    url: HttpUrl
    content_type: Literal["html", "pdf"]
    topics: list[str] = Field(min_length=1)


def load_sources(manifest_path: Path) -> list[KnowledgeSource]:
    """Load the explicit allowlist of stable knowledge sources."""

    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    return [KnowledgeSource.model_validate(item) for item in manifest["sources"]]


def fetch_snapshots(
    manifest_path: Path,
    snapshot_directory: Path,
    *,
    source_ids: set[str] | None = None,
) -> list[Path]:
    """Download approved sources with bounded retries and integrity metadata."""

    sources = load_sources(manifest_path)
    known_ids = {source.id for source in sources}
    if source_ids and not source_ids <= known_ids:
        raise ValueError(f"Unknown source IDs: {sorted(source_ids - known_ids)}")

    snapshot_directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for source in sources:
        if source_ids and source.id not in source_ids:
            continue
        request = Request(
            str(source.url),
            headers={
                "User-Agent": KNOWLEDGE_FETCH_USER_AGENT,
                "Accept-Language": "en-US,en;q=0.9",
            },
        )
        for attempt in range(KNOWLEDGE_FETCH_ATTEMPTS):
            try:
                with urlopen(
                    request,
                    timeout=KNOWLEDGE_FETCH_TIMEOUT_SECONDS,
                ) as response:  # noqa: S310
                    content = response.read()
                break
            except (HTTPError, URLError) as error:
                retryable = not isinstance(error, HTTPError) or error.code in {
                    429,
                    500,
                    502,
                    503,
                    504,
                }
                if not retryable or attempt == KNOWLEDGE_FETCH_ATTEMPTS - 1:
                    raise
                sleep(KNOWLEDGE_FETCH_RETRY_DELAY_SECONDS * 2**attempt)

        extension = "pdf" if source.content_type == "pdf" else "html"
        snapshot = snapshot_directory / f"{source.id}.{extension}"
        snapshot.write_bytes(content)
        snapshot.with_suffix(f"{snapshot.suffix}.metadata.json").write_text(
            json.dumps(
                {
                    "source_id": source.id,
                    "source_url": str(source.url),
                    "fetched_at": datetime.now(UTC).isoformat(),
                    "sha256": sha256(content).hexdigest(),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        written.append(snapshot)
    return written


def ingest_snapshots(
    manifest_path: Path,
    snapshot_directory: Path,
    database_path: Path,
    models: LocalRetrievalModels,
) -> int:
    """Parse, chunk, embed, and replace the local LanceDB table."""

    chunks = []
    for source in load_sources(manifest_path):
        extension = "pdf" if source.content_type == "pdf" else "html"
        snapshot = snapshot_directory / f"{source.id}.{extension}"
        metadata_path = snapshot.with_suffix(f"{snapshot.suffix}.metadata.json")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        content = snapshot.read_bytes()
        if metadata.get("source_id") != source.id:
            raise ValueError(f"Snapshot metadata does not match {source.id}.")
        if metadata.get("sha256") != sha256(content).hexdigest():
            raise ValueError(f"Snapshot content hash does not match {source.id}.")
        content_type: Literal["text/html", "application/pdf"] = (
            "application/pdf" if source.content_type == "pdf" else "text/html"
        )
        chunks.extend(
            chunk_document(
                parse_document(content, content_type),
                source_id=source.id,
                source_title=source.title,
                publisher=source.publisher,
                source_url=str(source.url),
                topics=source.topics,
                content_hash=metadata["sha256"],
                encode=models.encode,
                decode=models.decode,
                max_tokens=KNOWLEDGE_CHUNK_MAX_TOKENS,
                overlap_tokens=KNOWLEDGE_CHUNK_OVERLAP_TOKENS,
                min_tokens=KNOWLEDGE_CHUNK_MIN_TOKENS,
            )
        )

    vectors = models.embed([chunk.embedding_text for chunk in chunks])
    if not chunks or len(vectors) != len(chunks):
        raise ValueError("Every knowledge chunk requires exactly one embedding.")
    records = [
        {
            "chunk_id": chunk.chunk_id,
            "canonical_candidate_id": chunk.canonical_candidate_id,
            "source_id": chunk.source_id,
            "source_title": chunk.source_title,
            "publisher": chunk.publisher,
            "source_url": chunk.source_url,
            "topics": list(chunk.topics),
            "heading_path": list(chunk.heading_path),
            "section_position": chunk.section_position,
            "chunk_position": chunk.chunk_position,
            "token_count": chunk.token_count,
            "text": chunk.text,
            "vector": vector,
        }
        for chunk, vector in zip(chunks, vectors, strict=True)
    ]
    database_path.mkdir(parents=True, exist_ok=True)
    knowledge_table = lancedb.connect(str(database_path)).create_table(
        KNOWLEDGE_TABLE_NAME,
        data=records,
        mode="overwrite",
    )
    knowledge_table.create_index(
        "text",
        config=FTS(stem=True, remove_stop_words=True),
        name=KNOWLEDGE_TEXT_INDEX_NAME,
    )
    return len(records)
