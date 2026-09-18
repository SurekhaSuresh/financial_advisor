"""Parse approved snapshots and build the generated local LanceDB knowledge table."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from financial_advisor.config import get_settings
from financial_advisor.retrieval.knowledge_base.ingestion import (
    BgeEmbedder,
    BgeTokenWindowCodec,
    chunk_sections,
    load_source_manifest,
    parse_snapshot,
    write_lancedb,
)

_SETTINGS = get_settings()


def load_snapshot_provenance(snapshot: Path, source_id: str) -> tuple[str, datetime]:
    """Load the immutable fetch hash and timestamp stored beside one snapshot."""

    metadata_path = snapshot.with_suffix(f"{snapshot.suffix}.metadata.json")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing snapshot metadata for {source_id}: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("source_id") != source_id:
        raise ValueError(f"Snapshot metadata source ID does not match {source_id}.")
    content_hash = metadata.get("sha256")
    fetched_at = metadata.get("fetched_at")
    if not isinstance(content_hash, str) or not isinstance(fetched_at, str):
        raise ValueError(f"Snapshot metadata is incomplete for {source_id}.")
    return content_hash, datetime.fromisoformat(fetched_at)


def main() -> None:
    """Create the local vector table from all fetched manifest sources."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=_SETTINGS.storage.knowledge_manifest_path)
    parser.add_argument(
        "--snapshots",
        type=Path,
        default=_SETTINGS.storage.source_snapshot_directory,
    )
    parser.add_argument("--database", type=Path, default=_SETTINGS.storage.lancedb_database_path)
    parser.add_argument(
        "--model-cache",
        type=Path,
        default=_SETTINGS.model_runtime.model_cache_directory,
    )
    arguments = parser.parse_args()

    manifest = load_source_manifest(arguments.manifest)
    codec = BgeTokenWindowCodec(cache_directory=arguments.model_cache)
    chunks = []
    for source in manifest.sources:
        extension = "pdf" if source.content_type == "pdf" else "html"
        snapshot = arguments.snapshots / f"{source.id}.{extension}"
        if not snapshot.exists():
            raise FileNotFoundError(f"Missing snapshot for {source.id}: {snapshot}")
        source_content_hash, retrieved_at = load_snapshot_provenance(snapshot, source.id)
        chunks.extend(
            chunk_sections(
                source,
                parse_snapshot(source, snapshot),
                codec,
                source_content_hash=source_content_hash,
                retrieved_at=retrieved_at,
            )
        )

    embeddings = BgeEmbedder(cache_directory=arguments.model_cache).embed(
        [chunk.embedding_text for chunk in chunks]
    )
    write_lancedb(arguments.database, chunks, embeddings)
    print(f"Ingested {len(chunks)} chunks into {arguments.database}")


if __name__ == "__main__":
    main()
