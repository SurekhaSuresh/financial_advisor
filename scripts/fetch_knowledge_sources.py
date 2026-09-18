"""Fetch approved public source snapshots for offline local ingestion."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from financial_advisor.config import get_settings
from financial_advisor.retrieval.knowledge_base.ingestion import (
    KnowledgeSource,
    load_source_manifest,
)

_SETTINGS = get_settings()
_POLICY = _SETTINGS.knowledge_ingestion


def fetch_source(
    source: KnowledgeSource,
    output_directory: Path,
    *,
    max_attempts: int = _POLICY.source_fetch_max_attempts,
) -> Path:
    """Download one source and write its bytes plus provenance metadata locally."""

    extension = "pdf" if source.content_type == "pdf" else "html"
    destination = output_directory / f"{source.id}.{extension}"
    request = Request(
        str(source.url),
        headers={
            "User-Agent": _POLICY.source_fetch_user_agent,
            "Accept-Language": _POLICY.source_fetch_accept_language,
        },
    )
    for attempt in range(1, max_attempts + 1):
        try:
            with urlopen(request, timeout=_POLICY.source_fetch_timeout_seconds) as response:
                content = response.read()
            break
        except (HTTPError, URLError) as error:
            retryable = not isinstance(error, HTTPError) or error.code in {429, 500, 502, 503, 504}
            if not retryable or attempt == max_attempts:
                raise
            time.sleep(_POLICY.source_fetch_retry_backoff_base_seconds * 2 ** (attempt - 1))
    else:  # pragma: no cover - loop either breaks or raises
        raise RuntimeError(f"Could not fetch {source.id}")
    destination.write_bytes(content)
    destination.with_suffix(destination.suffix + ".metadata.json").write_text(
        json.dumps(
            {
                "source_id": source.id,
                "source_url": str(source.url),
                "fetched_at": datetime.now(UTC).isoformat(),
                "sha256": hashlib.sha256(content).hexdigest(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return destination


def main() -> None:
    """Fetch every source in the versioned project manifest."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=_SETTINGS.storage.knowledge_manifest_path)
    parser.add_argument("--output", type=Path, default=_SETTINGS.storage.source_snapshot_directory)
    parser.add_argument(
        "--source-id",
        action="append",
        dest="source_ids",
        help="Fetch only this manifest source ID; repeat to fetch a batch.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Report failed sources and continue fetching the remaining selected sources.",
    )
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    manifest = load_source_manifest(arguments.manifest)
    selected_source_ids = set(arguments.source_ids or [])
    if selected_source_ids:
        known_source_ids = {source.id for source in manifest.sources}
        unknown_source_ids = selected_source_ids - known_source_ids
        if unknown_source_ids:
            raise ValueError(f"Unknown source IDs: {sorted(unknown_source_ids)}")
    failures: list[str] = []
    for source in manifest.sources:
        if selected_source_ids and source.id not in selected_source_ids:
            continue
        try:
            print(fetch_source(source, arguments.output))
        except (HTTPError, URLError) as error:
            if not arguments.continue_on_error:
                raise
            failures.append(f"{source.id}: {error}")
            print(f"FAILED {source.id}: {error}")
    if failures:
        raise RuntimeError("One or more source snapshots failed:\n" + "\n".join(failures))


if __name__ == "__main__":
    main()
