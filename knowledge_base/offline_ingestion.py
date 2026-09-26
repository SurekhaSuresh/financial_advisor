"""Build the curated local knowledge store from approved snapshots."""

import json
from datetime import UTC, datetime
from hashlib import sha256
from http import HTTPStatus
from pathlib import Path
from time import sleep
from typing import Literal

import httpx
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
from financial_advisor.contracts import DocumentChunk
from financial_advisor.retrieval.content_processing import chunk_document, parse_document
from financial_advisor.retrieval.text_models import LocalRetrievalModels


class KnowledgeSource(BaseModel):
    """One approved source in the local knowledge manifest."""

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
) -> list[Path]:
    """Download approved sources with bounded retries and integrity metadata."""

    snapshot_directory.mkdir(parents=True, exist_ok=True)
    written_snapshot_paths: list[Path] = []
    for source in load_sources(manifest_path):
        content = _download_source(source)
        extension = "pdf" if source.content_type == "pdf" else "html"
        snapshot_path = snapshot_directory / f"{source.id}.{extension}"
        metadata_path = snapshot_directory / f"{source.id}.{extension}.metadata.json"
        metadata = {
            "source_id": source.id,
            "source_url": str(source.url),
            "fetched_at": datetime.now(UTC).isoformat(),
            "sha256": sha256(content).hexdigest(),
        }

        snapshot_path.write_bytes(content)
        metadata_path.write_text(
            json.dumps(metadata, indent=2) + "\n",
            encoding="utf-8",
        )
        written_snapshot_paths.append(snapshot_path)

    return written_snapshot_paths


def _download_source(source: KnowledgeSource) -> bytes:
    """Download one approved source with bounded retries."""

    for attempt_index in range(KNOWLEDGE_FETCH_ATTEMPTS):
        is_final_attempt = attempt_index == KNOWLEDGE_FETCH_ATTEMPTS - 1
        try:
            response = httpx.get(
                str(source.url),
                headers={"User-Agent": KNOWLEDGE_FETCH_USER_AGENT},
                timeout=KNOWLEDGE_FETCH_TIMEOUT_SECONDS,
                follow_redirects=True,
            )
            response.raise_for_status()
            return response.content
        except httpx.HTTPStatusError as error:
            is_transient_error = (
                error.response.status_code == HTTPStatus.TOO_MANY_REQUESTS
                or error.response.status_code >= HTTPStatus.INTERNAL_SERVER_ERROR
            )
            if not is_transient_error or is_final_attempt:
                raise
        except httpx.RequestError:
            if is_final_attempt:
                raise

        sleep(KNOWLEDGE_FETCH_RETRY_DELAY_SECONDS * 2**attempt_index)

    raise RuntimeError("Knowledge fetch attempts must be positive.")


def ingest_snapshots(
    manifest_path: Path,
    snapshot_directory: Path,
    database_path: Path,
    models: LocalRetrievalModels,
) -> int:
    """Parse, chunk, embed, and replace the local LanceDB table."""

    chunks: list[DocumentChunk] = []
    for source in load_sources(manifest_path):
        extension = "pdf" if source.content_type == "pdf" else "html"
        snapshot_path = snapshot_directory / f"{source.id}.{extension}"
        metadata_path = snapshot_directory / f"{source.id}.{extension}.metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        content = snapshot_path.read_bytes()

        if metadata.get("source_id") != source.id:
            raise ValueError(f"Snapshot metadata does not match {source.id}.")
        content_hash = sha256(content).hexdigest()
        if metadata.get("sha256") != content_hash:
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
                content_hash=content_hash,
                encode=models.encode,
                decode=models.decode,
                max_tokens=KNOWLEDGE_CHUNK_MAX_TOKENS,
                overlap_tokens=KNOWLEDGE_CHUNK_OVERLAP_TOKENS,
                min_tokens=KNOWLEDGE_CHUNK_MIN_TOKENS,
            )
        )

    if not chunks:
        raise ValueError("Knowledge ingestion produced no chunks.")

    vectors = models.embed([chunk.embedding_text for chunk in chunks])
    if len(vectors) != len(chunks):
        raise ValueError("Every knowledge chunk requires exactly one embedding.")

    knowledge_records = [
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
        data=knowledge_records,
        mode="overwrite",
    )
    knowledge_table.create_index(
        "text",
        config=FTS(stem=True, remove_stop_words=True),
        name=KNOWLEDGE_TEXT_INDEX_NAME,
    )

    return len(knowledge_records)


def main() -> None:
    """Download approved sources and rebuild the local knowledge database."""

    knowledge_directory = Path(__file__).resolve().parent
    data_directory = knowledge_directory / "data"
    manifest_path = data_directory / "sources.yaml"
    snapshot_directory = data_directory / "snapshots"

    snapshots = fetch_snapshots(manifest_path, snapshot_directory)
    models = LocalRetrievalModels(knowledge_directory.parent / ".local" / "model_cache")
    chunk_count = ingest_snapshots(
        manifest_path,
        snapshot_directory,
        data_directory / "lancedb",
        models,
    )
    print(f"Downloaded {len(snapshots)} sources and ingested {chunk_count} chunks.")


if __name__ == "__main__":
    main()
