"""Parse and chunk source documents used by web and offline ingestion."""

from io import BytesIO

from bs4 import BeautifulSoup
from pypdf import PdfReader

from financial_advisor.config import (
    DOCUMENT_CHUNK_MAX_TOKENS,
    DOCUMENT_CHUNK_OVERLAP_TOKENS,
)
from financial_advisor.contracts import DocumentChunk
from financial_advisor.retrieval.text_models import LocalRetrievalModels


def _extract_document_sections(
    content: bytes,
    content_type: str,
) -> list[str]:
    """Group PDF pages or HTML paragraphs into sections for chunking."""

    document_sections: list[str] = []
    if content_type == "application/pdf":
        for page in PdfReader(BytesIO(content)).pages:
            page_text = page.extract_text()
            if page_text:
                document_sections.append(page_text)
        return document_sections

    soup = BeautifulSoup(content, "html.parser")
    section_parts: list[str] = []
    for element in soup.find_all(["h1", "h2", "h3", "h4", "p", "li"]):
        text = element.get_text(" ", strip=True)
        if not text:
            continue
        if element.name.startswith("h") and section_parts:
            document_sections.append(" ".join(section_parts))
            section_parts = []
        section_parts.append(text)
    if section_parts:
        document_sections.append(" ".join(section_parts))
    return document_sections


def chunk_document(
    content: bytes,
    content_type: str,
    *,
    source_title: str,
    publisher: str,
    source_url: str,
    topics: tuple[str, ...],
    content_hash: str,
    models: LocalRetrievalModels,
) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    window_step = DOCUMENT_CHUNK_MAX_TOKENS - DOCUMENT_CHUNK_OVERLAP_TOKENS
    for section_position, section_text in enumerate(
        _extract_document_sections(content, content_type),
        start=1,
    ):
        section_tokens = models.encode(section_text)
        chunk_position = 1
        for window_start in range(0, len(section_tokens), window_step):
            chunk_tokens = section_tokens[window_start : window_start + DOCUMENT_CHUNK_MAX_TOKENS]
            chunk_text = models.decode(chunk_tokens).strip()
            chunks.append(
                DocumentChunk(
                    canonical_candidate_id=f"{content_hash}:{section_position}:{chunk_position}",
                    source_title=source_title,
                    publisher=publisher,
                    source_url=source_url,
                    topics=topics,
                    section_position=section_position,
                    chunk_position=chunk_position,
                    token_count=len(chunk_tokens),
                    text=chunk_text,
                )
            )
            chunk_position += 1
    return chunks
