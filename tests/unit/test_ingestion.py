import json
from hashlib import sha256
from pathlib import Path
from typing import Any, cast

import lancedb

from financial_advisor.config import KNOWLEDGE_TEXT_INDEX_NAME
from financial_advisor.retrieval.ingestion import ingest_snapshots
from financial_advisor.retrieval.text_models import LocalRetrievalModels


class Models:
    def encode(self, text: str) -> list[int]:
        return [ord(character) for character in text]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(token) for token in tokens)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, float(index)] for index, _ in enumerate(texts)]


def test_ingestion_builds_the_knowledge_table(tmp_path: Path) -> None:
    manifest = tmp_path / "sources.yaml"
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
