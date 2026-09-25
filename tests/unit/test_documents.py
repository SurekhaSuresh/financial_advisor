from financial_advisor.retrieval.documents import Section, chunk_document, parse_document


def encode(text: str) -> list[int]:
    return [ord(character) for character in text]


def decode(tokens: list[int]) -> str:
    return "".join(chr(token) for token in tokens)


def test_html_parsing_preserves_headings_and_ignores_scripts() -> None:
    sections = parse_document(
        b"<main><h1>Saving</h1><p>Keep cash available.</p>"
        b"<script>ignore me</script><h2>Investing</h2><p>Diversify.</p></main>",
        "text/html",
    )

    assert sections == [
        Section(("Saving",), "Keep cash available."),
        Section(("Saving", "Investing"), "Diversify."),
    ]


def test_chunking_overlaps_within_a_section_and_uses_stable_ids() -> None:
    sections = [Section(("Heading",), "abcdefghijklmnopqrstuvwxyz")]
    arguments = {
        "source_id": "guide",
        "source_title": "Guide",
        "publisher": "Publisher",
        "source_url": "https://example.com/guide",
        "topics": ["saving"],
        "content_hash": "document-hash",
        "encode": encode,
        "decode": decode,
        "max_tokens": 45,
        "overlap_tokens": 3,
        "min_tokens": 0,
    }

    first = chunk_document(sections, **arguments)  # type: ignore[arg-type]
    second = chunk_document(sections, **arguments)  # type: ignore[arg-type]

    assert len(first) > 1
    assert first[0].text[-3:] == first[1].text[:3]
    assert [chunk.canonical_candidate_id for chunk in first] == [
        chunk.canonical_candidate_id for chunk in second
    ]
    assert first[0].chunk_id == "guide:1:1"
    assert first[0].token_count <= arguments["max_tokens"]
