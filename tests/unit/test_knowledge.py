"""Tests for deterministic knowledge parsing, chunking, and local vector storage."""

from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import lancedb
import pytest

from financial_advisor.retrieval.content_processing.chunking import canonical_candidate_id
from financial_advisor.retrieval.knowledge_base.ingestion import (
    KnowledgeSource,
    SourceSection,
    TokenWindowCodec,
    chunk_sections,
    parse_snapshot,
    write_lancedb,
)


class WhitespaceCodec(TokenWindowCodec):
    """Small deterministic tokenizer substitute that keeps unit tests offline."""

    def encode(self, text: str) -> list[int]:
        return list(range(len(text.split())))

    def decode(self, token_ids: Sequence[int]) -> str:
        return " ".join(f"token-{token_id}" for token_id in token_ids)


def make_source() -> KnowledgeSource:
    """Return a valid source shared by chunking tests."""

    return KnowledgeSource(
        id="test_source",
        publisher="Test publisher",
        title="Test source",
        url="https://example.com/source",
        content_type="html",
        topics=["testing"],
    )


def test_canonical_candidate_id_matches_same_document_version_across_storage() -> None:
    """Storage IDs may differ while one exact source passage remains one evidence item."""

    local_id = canonical_candidate_id(
        source_url="https://www.investor.gov/guide",
        source_content_hash="same-document-version",
        text="Shorter horizons often call for less volatile investments.",
    )
    web_id = canonical_candidate_id(
        source_url="https://www.investor.gov/guide/",
        source_content_hash="same-document-version",
        text="  Shorter horizons often call for less volatile investments.  ",
    )
    changed_version_id = canonical_candidate_id(
        source_url="https://www.investor.gov/guide",
        source_content_hash="changed-document-version",
        text="Shorter horizons often call for less volatile investments.",
    )

    assert local_id == web_id
    assert local_id != changed_version_id


def test_chunking_preserves_section_boundaries_and_applies_same_section_overlap() -> None:
    """Long sections overlap; a following section never receives prior-section tokens."""

    chunks = chunk_sections(
        make_source(),
        [
            SourceSection("First", "one two three four five six seven", 1),
            SourceSection("Second", "eight nine", 2),
        ],
        WhitespaceCodec(),
        max_tokens=8,
        min_tokens=0,
        overlap_tokens=1,
    )

    # token_count includes the source/heading prefix provided to the embedder.
    assert all(chunk.token_count <= 8 for chunk in chunks)
    assert [chunk.heading for chunk in chunks] == ["First", "First", "First", "Second"]
    assert chunks[1].text.startswith("token-2")
    assert chunks[-1].text == "token-0 token-1"


def test_chunking_merges_an_undersized_tail_only_within_its_section() -> None:
    """A short trailing passage joins its predecessor, never a different topic."""

    chunks = chunk_sections(
        make_source(),
        [
            SourceSection("First", "one two three four five six", 1),
            SourceSection("Second", "seven", 2),
        ],
        WhitespaceCodec(),
        max_tokens=8,
        min_tokens=8,
        overlap_tokens=1,
    )

    # The First-section tail would have contained only two body tokens. It is
    # merged into its preceding window, which modestly exceeds the nominal max.
    assert [chunk.heading for chunk in chunks] == ["First", "First", "Second"]
    assert chunks[1].body_start_token == 2
    assert chunks[1].body_end_token_exclusive == 6
    assert chunks[1].token_count == 9
    assert chunks[1].text == "token-2 token-3 token-4 token-5"

    # The tiny Second section remains separate rather than crossing a topic boundary.
    assert chunks[2].token_count < 8
    assert chunks[2].text == "token-0"


def test_html_parser_groups_text_under_the_nearest_heading(tmp_path: Path) -> None:
    """HTML headings retain their hierarchy as retrieval metadata."""

    snapshot = tmp_path / "source.html"
    snapshot.write_text(
        "<main><h1>Investing basics</h1><p>First paragraph.</p>"
        "<h2>Risk</h2><p>Second paragraph.</p>"
        "<h3>Time horizon</h3><p>Third paragraph.</p></main>",
        encoding="utf-8",
    )

    sections = parse_snapshot(make_source(), snapshot)

    assert [(section.heading, section.text, section.heading_path) for section in sections] == [
        ("Investing basics", "First paragraph.", ("Investing basics",)),
        ("Investing basics > Risk", "Second paragraph.", ("Investing basics", "Risk")),
        (
            "Investing basics > Risk > Time horizon",
            "Third paragraph.",
            ("Investing basics", "Risk", "Time horizon"),
        ),
    ]


def test_chunking_carries_the_full_heading_path() -> None:
    """A retrieved chunk can identify both its topic and its parent section."""

    chunks = chunk_sections(
        make_source(),
        [
            SourceSection(
                "Investing basics > Risk",
                "one two",
                1,
                ("Investing basics", "Risk"),
            )
        ],
        WhitespaceCodec(),
        max_tokens=12,
        min_tokens=0,
        overlap_tokens=1,
    )

    assert chunks[0].heading_path == ["Investing basics", "Risk"]


def test_lancedb_writer_stores_chunk_metadata_and_vectors(tmp_path: Path) -> None:
    """Generated chunk records can be opened from a local LanceDB table."""

    chunks = chunk_sections(
        make_source(),
        [SourceSection("Testing", "one two", 1)],
        WhitespaceCodec(),
        max_tokens=8,
        min_tokens=0,
        overlap_tokens=1,
        source_content_hash="test-snapshot-hash",
        retrieved_at=datetime(2026, 9, 19, tzinfo=UTC),
    )
    database_path = tmp_path / "lancedb"

    write_lancedb(database_path, chunks, [[0.1, 0.2]])

    table = lancedb.connect(str(database_path)).open_table("knowledge_chunks")
    records = table.to_arrow().to_pylist()
    assert records[0]["chunk_id"] == chunks[0].chunk_id
    assert records[0]["canonical_candidate_id"] == chunks[0].canonical_candidate_id
    assert records[0]["heading_path"] == ["Testing"]
    assert records[0]["body_start_token"] == 0
    assert records[0]["source_content_hash"] == "test-snapshot-hash"
    assert records[0]["retrieved_at"] == "2026-09-19T00:00:00+00:00"
    assert records[0]["vector"] == pytest.approx([0.1, 0.2])
