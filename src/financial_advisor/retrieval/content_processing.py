"""Parse and chunk local snapshots and fetched web documents."""

from collections.abc import Callable, Sequence
from hashlib import sha256
from io import BytesIO
from typing import Literal
from unicodedata import normalize

from bs4 import BeautifulSoup
from pypdf import PdfReader

from financial_advisor.contracts import DocumentChunk, ParsedDocumentSection

TokenEncoder = Callable[[str], list[int]]
TokenDecoder = Callable[[Sequence[int]], str]


def parse_document(
    content: bytes,
    content_type: Literal["text/html", "application/xhtml+xml", "application/pdf"],
) -> list[ParsedDocumentSection]:
    """Extract ordered sections without summarizing source content."""

    # PDF Parsing
    if content_type == "application/pdf":
        pdf_sections: list[ParsedDocumentSection] = []
        for page_number, page in enumerate(PdfReader(BytesIO(content)).pages, start=1):
            page_text = " ".join((page.extract_text() or "").split())
            if page_text:
                pdf_sections.append(
                    ParsedDocumentSection(
                        heading_path=(f"Page {page_number}",),
                        text=page_text,
                    )
                )
        return pdf_sections

    # HTML Parsing
    soup = BeautifulSoup(content.decode("utf-8", errors="replace"), "html.parser")
    for ignored in soup(["script", "style", "noscript", "template"]):
        ignored.decompose()

    sections: list[ParsedDocumentSection] = []
    heading_path = ["Document introduction"]
    section_text_parts: list[str] = []

    def append_current_section() -> None:
        if section_text_parts:
            sections.append(
                ParsedDocumentSection(
                    heading_path=tuple(heading_path),
                    text=" ".join(section_text_parts),
                )
            )
            section_text_parts.clear()

    root = soup.find("main") or soup.body or soup
    for element in root.find_all(["h1", "h2", "h3", "h4", "p", "li"]):
        text = " ".join(element.get_text(" ", strip=True).split())
        if not text:
            continue
        if element.name and element.name.startswith("h"):
            append_current_section()
            heading_level = int(element.name[1])
            if heading_path == ["Document introduction"]:
                heading_path = []
            heading_path = [*heading_path[: heading_level - 1], text]
        else:
            section_text_parts.append(text)

    append_current_section()
    return sections


def chunk_document(
    sections: Sequence[ParsedDocumentSection],
    *,
    source_id: str,
    source_title: str,
    publisher: str,
    source_url: str,
    topics: Sequence[str],
    content_hash: str,
    encode: TokenEncoder,
    decode: TokenDecoder,
    max_tokens: int,
    overlap_tokens: int,
    min_tokens: int,
) -> list[DocumentChunk]:
    """Create overlapping token windows without crossing section boundaries."""

    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive.")
    if not 0 <= overlap_tokens < max_tokens:
        raise ValueError("overlap_tokens must be between zero and max_tokens.")
    if min_tokens < 0:
        raise ValueError("min_tokens must not be negative.")

    chunks: list[DocumentChunk] = []
    chunk_position = 0
    for section_position, section in enumerate(sections, start=1):
        prefix_token_count = len(
            encode(
                f"Source: {source_title}\n"
                f"Section: {' > '.join(section.heading_path)}\n\n"
            )
        )
        window_size = max_tokens - prefix_token_count
        if window_size <= 0:
            raise ValueError("Source title and heading leave no room for chunk content.")

        body_token_ids = encode(section.text)
        window_overlap = min(overlap_tokens, window_size - 1)
        window_step = window_size - window_overlap
        token_windows: list[range] = []
        window_start = 0
        while window_start < len(body_token_ids):
            token_window = range(
                window_start,
                min(window_start + window_size, len(body_token_ids)),
            )
            token_windows.append(token_window)
            if token_window.stop == len(body_token_ids):
                break
            window_start += window_step

        # Merge a very short final window into its preceding section-local window.
        if (
            len(token_windows) > 1
            and prefix_token_count + len(token_windows[-1]) < min_tokens
        ):
            token_windows[-2:] = [
                range(token_windows[-2].start, token_windows[-1].stop)
            ]

        for token_window in token_windows:
            chunk_text = decode(
                body_token_ids[token_window.start : token_window.stop]
            ).strip()
            if not chunk_text:
                continue

            chunk_position += 1
            normalized_chunk_text = " ".join(normalize("NFKC", chunk_text).casefold().split())
            canonical_candidate_id = sha256(
                f"{content_hash}\n{normalized_chunk_text}".encode()
            ).hexdigest()
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{source_id}:{section_position}:{chunk_position}",
                    canonical_candidate_id=canonical_candidate_id,
                    source_id=source_id,
                    source_title=source_title,
                    publisher=publisher,
                    source_url=source_url,
                    topics=tuple(topics),
                    heading_path=section.heading_path,
                    section_position=section_position,
                    chunk_position=chunk_position,
                    token_count=prefix_token_count + len(token_window),
                    text=chunk_text,
                )
            )
    return chunks
