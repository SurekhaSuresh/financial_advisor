import json
from hashlib import sha256
from http import HTTPStatus
from pathlib import Path
from typing import Any, cast

import httpx
import lancedb
import pytest

from financial_advisor.config import KNOWLEDGE_TEXT_INDEX_NAME
from financial_advisor.retrieval.text_models import LocalRetrievalModels
from knowledge_base import offline_ingestion
from knowledge_base.offline_ingestion import (
    fetch_snapshots,
    ingest_snapshots,
)


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
    request = httpx.Request("GET", "https://www.investor.gov/test")
    responses = [
        httpx.Response(
            HTTPStatus.SERVICE_UNAVAILABLE,
            request=request,
        ),
        httpx.Response(HTTPStatus.OK, content=b"snapshot", request=request),
    ]
    retry_delays: list[float] = []

    def fake_get(
        _url: str,
        *,
        headers: dict[str, str],
        timeout: float,
        follow_redirects: bool,
    ) -> httpx.Response:
        assert headers
        assert timeout > 0
        assert follow_redirects
        return responses.pop(0)

    monkeypatch.setattr(offline_ingestion.httpx, "get", fake_get)
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

    def reject_request(
        url: str,
        *,
        headers: dict[str, str],
        timeout: float,
        follow_redirects: bool,
    ) -> httpx.Response:
        nonlocal request_count
        assert headers
        assert timeout > 0
        assert follow_redirects
        request_count += 1
        return httpx.Response(
            HTTPStatus.NOT_FOUND,
            request=httpx.Request("GET", url),
        )

    monkeypatch.setattr(offline_ingestion.httpx, "get", reject_request)

    with pytest.raises(httpx.HTTPStatusError):
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
