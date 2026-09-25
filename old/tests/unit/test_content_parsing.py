from financial_advisor.retrieval.content_processing.parsing import (
    SourceSection,
    parse_document,
    parse_html_sections,
)
from financial_advisor.retrieval.web.fetch import FetchedWebDocument
from financial_advisor.retrieval.web.page_processing import (
    chunk_fetched_web_document,
    parse_fetched_web_document,
)


def test_html_parser_preserves_heading_context_and_ignores_scripts() -> None:
    sections = parse_html_sections(
        """
        <html><body><main>
          <h1>Investing basics</h1>
          <p>Build an emergency fund.</p>
          <script>do_not_retain()</script>
          <h2>Diversification</h2>
          <p>Spread investments across assets.</p>
          <li>Consider costs.</li>
        </main></body></html>
        """
    )

    assert [(section.heading_path, section.text) for section in sections] == [
        (("Investing basics",), "Build an emergency fund."),
        (
            ("Investing basics", "Diversification"),
            "Spread investments across assets. Consider costs.",
        ),
    ]


def test_web_document_parser_uses_the_validated_document_content_type() -> None:
    document = FetchedWebDocument(
        title="Investor guidance",
        discovered_url="https://www.investor.gov/example",
        final_url="https://www.investor.gov/example",
        content_type="text/html",
        content=b"<html><body><h1>Risk</h1><p>Risk can vary.</p></body></html>",
    )

    sections = parse_fetched_web_document(document)

    assert len(sections) == 1
    assert sections[0].heading_path == ("Risk",)
    assert sections[0].text == "Risk can vary."


def test_document_parser_replaces_invalid_html_bytes_instead_of_crashing() -> None:
    sections = parse_document(b"<html><body><p>Saving \xff first.</p></body></html>", "text/html")

    assert sections[0].text == "Saving \ufffd first."


class CharacterCodec:
    """Tiny deterministic tokenizer used to prove token-window behavior."""

    def encode(self, text: str) -> list[int]:
        return [ord(character) for character in text]

    def decode(self, token_ids: list[int]) -> str:
        return "".join(chr(token_id) for token_id in token_ids)


def test_web_chunker_keeps_page_provenance_and_section_local_overlap() -> None:
    document = FetchedWebDocument(
        title="Investor guidance",
        discovered_url="https://www.investor.gov/example",
        final_url="https://www.investor.gov/example",
        content_type="text/html",
        content=b"<html><body>unused because sections are explicit</body></html>",
    )
    sections = [
        SourceSection(
            heading="Diversification",
            heading_path=("Investing", "Diversification"),
            position=1,
            text="abcdefghijklmnopqrstuvwxyz" * 3,
        ),
        SourceSection(heading="Costs", heading_path=("Investing", "Costs"), position=2, text="xy"),
    ]

    chunks = chunk_fetched_web_document(
        document,
        sections,
        CharacterCodec(),
        max_tokens=100,
        overlap_tokens=2,
    )

    assert len(chunks) == 4
    assert chunks[0].text[-2:] == chunks[1].text[:2]
    assert chunks[0].body_start_token == 0
    assert chunks[0].body_end_token_exclusive - chunks[0].body_start_token == len(chunks[0].text)
    assert chunks[1].body_start_token == chunks[0].body_end_token_exclusive - 2
    assert chunks[-1].text == "xy"
    assert all(chunk.source_url.host == "www.investor.gov" for chunk in chunks)
    assert all(chunk.source_content_hash for chunk in chunks)
    assert chunks[0].heading_path == ["Investing", "Diversification"]
