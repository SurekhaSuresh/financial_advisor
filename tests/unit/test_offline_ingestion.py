import json
from hashlib import sha256
from http import HTTPStatus
from io import BytesIO
from pathlib import Path
from typing import Any, cast
from urllib.error import HTTPError

import lancedb
import pytest

from financial_advisor.config import KNOWLEDGE_TEXT_INDEX_NAME
from financial_advisor.retrieval.knowledge_base import offline_ingestion
from financial_advisor.retrieval.knowledge_base.offline_ingestion import (
    fetch_snapshots,
    ingest_snapshots,
)
from financial_advisor.retrieval.text_models import LocalRetrievalModels


class Models:
    def encode(self, text: str) -> list[int]:
        return [ord(character) for character in text]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(token) for token in tokens)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, float(index)] for index, _ in enumerate(texts)]


def write_manifest(manifest: Path) -> None:
    manifest.write_text(
        """sources:
  - id: test_source
    publisher: Investor.gov
    title: Test guide
    url: https://www.investor.gov/test
    content_type: html
    topics: [saving]
""",
        encoding="utf-8",
    )


def test_fetch_snapshots_retries_transient_http_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = tmp_path / "sources.yaml"
    write_manifest(manifest)
    responses: list[HTTPError | BytesIO] = [
        HTTPError(
            "https://www.investor.gov/test",
            HTTPStatus.SERVICE_UNAVAILABLE,
            "unavailable",
            hdrs=None,
            fp=None,
        ),
        BytesIO(b"snapshot"),
    ]
    retry_delays: list[float] = []

    def fake_urlopen(_request: object, *, timeout: float) -> BytesIO:
        assert timeout > 0
        response = responses.pop(0)
        if isinstance(response, HTTPError):
            raise response
        return response

    monkeypatch.setattr(offline_ingestion, "urlopen", fake_urlopen)
    monkeypatch.setattr(offline_ingestion, "sleep", retry_delays.append)

    written_paths = fetch_snapshots(manifest, tmp_path / "snapshots")

    assert written_paths[0].read_bytes() == b"snapshot"
    assert len(retry_delays) == 1


def test_fetch_snapshots_does_not_retry_permanent_http_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = tmp_path / "sources.yaml"
    write_manifest(manifest)
    request_count = 0

    def reject_request(_request: object, *, timeout: float) -> BytesIO:
        nonlocal request_count
        assert timeout > 0
        request_count += 1
        raise HTTPError(
            "https://www.investor.gov/test",
            HTTPStatus.NOT_FOUND,
            "not found",
            hdrs=None,
            fp=None,
        )

    monkeypatch.setattr(offline_ingestion, "urlopen", reject_request)

    with pytest.raises(HTTPError):
        fetch_snapshots(manifest, tmp_path / "snapshots")

    assert request_count == 1


def test_ingestion_builds_the_knowledge_table(tmp_path: Path) -> None:
    manifest = tmp_path / "sources.yaml"
    write_manifest(manifest)
    snapshots = tmp_path / "snapshots"
    snapshots.mkdir()
    snapshot = snapshots / "test_source.html"
    content = b"<main><h1>Saving</h1><p>Keep emergency savings available.</p></main>"
    snapshot.write_bytes(content)
    snapshot.with_suffix(".html.metadata.json").write_text(
        json.dumps({"source_id": "test_source", "sha256": sha256(content).hexdigest()}),
        encoding="utf-8",
    )
    database = tmp_path / "database"

    count = ingest_snapshots(
        manifest,
        snapshots,
        database,
        cast(LocalRetrievalModels, cast(Any, Models())),
    )

    table = lancedb.connect(str(database)).open_table("knowledge_chunks")
    assert count == 1
    assert table.count_rows() == 1
    assert KNOWLEDGE_TEXT_INDEX_NAME in {index.name for index in table.list_indices()}
    assert set(table.head(1).to_pylist()[0]) == {
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
        "text",
        "vector",
    }


def test_ingestion_rejects_modified_snapshot_content(tmp_path: Path) -> None:
    manifest = tmp_path / "sources.yaml"
    write_manifest(manifest)
    snapshots = tmp_path / "snapshots"
    snapshots.mkdir()
    snapshot = snapshots / "test_source.html"
    snapshot.write_bytes(b"modified")
    snapshot.with_suffix(".html.metadata.json").write_text(
        json.dumps({"source_id": "test_source", "sha256": "original-hash"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="content hash"):
        ingest_snapshots(
            manifest,
            snapshots,
            tmp_path / "database",
            cast(LocalRetrievalModels, cast(Any, Models())),
        )
