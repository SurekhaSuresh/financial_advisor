import pytest

import financial_advisor.retrieval.web.content_processing as content_processing
from financial_advisor.retrieval.web.content_processing import chunk_document


class Models:
    def encode(self, text: str) -> list[int]:
        return [ord(character) for character in text]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(token) for token in tokens)


def chunks(content: bytes):
    return chunk_document(
        content,
        "text/html",
        source_title="Guide",
        publisher="Publisher",
        source_url="https://example.com/guide",
        topics=["saving"],
        content_hash="document-hash",
        models=Models(),  # type: ignore[arg-type]
    )


def test_html_chunking_preserves_headings_and_ignores_scripts() -> None:
    result = chunks(
        b"<main><h1>Saving</h1><p>Keep cash available.</p>"
        b"<script>ignore me</script><h2>Investing</h2><p>Diversify.</p></main>"
    )

    assert [chunk.text for chunk in result] == [
        "Saving Keep cash available.",
        "Investing Diversify.",
    ]


def test_chunking_overlaps_within_a_section_and_uses_stable_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(content_processing, "DOCUMENT_CHUNK_MAX_TOKENS", 15)
    monkeypatch.setattr(content_processing, "DOCUMENT_CHUNK_OVERLAP_TOKENS", 3)
    content = b"<main><h1>Heading</h1><p>abcdefghijklmnopqrstuvwxyz</p></main>"
    first = chunks(content)
    second = chunks(content)

    assert len(first) > 1
    assert first[0].text[-3:] == first[1].text[:3]
    assert [chunk.canonical_candidate_id for chunk in first] == [
        chunk.canonical_candidate_id for chunk in second
    ]
    assert first[0].token_count <= 15


def test_chunking_does_not_overlap_sections(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(content_processing, "DOCUMENT_CHUNK_MAX_TOKENS", 42)
    monkeypatch.setattr(content_processing, "DOCUMENT_CHUNK_OVERLAP_TOKENS", 2)
    result = chunks(b"<main><h1>First</h1><p>abcdefghij</p><h2>Second</h2><p>klmnopqrst</p></main>")

    assert [chunk.text for chunk in result] == [
        "First abcdefghij",
        "Second klmnopqrst",
    ]
