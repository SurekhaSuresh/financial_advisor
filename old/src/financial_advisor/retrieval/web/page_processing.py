"""Parse and token-window original web documents into source-attributed chunks."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from urllib.parse import urlsplit

from financial_advisor.config import get_settings
from financial_advisor.retrieval.content_processing.chunking import (
    SourceEvidenceChunk,
    TokenWindowCodec,
    canonical_candidate_id,
    create_section_token_windows,
)
from financial_advisor.retrieval.content_processing.parsing import SourceSection, parse_document
from financial_advisor.retrieval.web.fetch import FetchedWebDocument

_POLICY = get_settings().web_research
WEB_MAX_CHUNK_TOKENS = _POLICY.web_chunk_max_tokens
WEB_MIN_CHUNK_TOKENS = _POLICY.web_chunk_min_tokens
WEB_CHUNK_OVERLAP_TOKENS = _POLICY.web_chunk_overlap_tokens


WebEvidenceChunk = SourceEvidenceChunk


def parse_fetched_web_document(document: FetchedWebDocument) -> list[SourceSection]:
    """Parse one validated fetched document without inventing or summarizing content."""

    return parse_document(_live_content(document), document.content_type)


def chunk_fetched_web_document(
    document: FetchedWebDocument,
    sections: Sequence[SourceSection],
    codec: TokenWindowCodec,
    *,
    max_tokens: int = WEB_MAX_CHUNK_TOKENS,
    min_tokens: int = WEB_MIN_CHUNK_TOKENS,
    overlap_tokens: int = WEB_CHUNK_OVERLAP_TOKENS,
) -> list[WebEvidenceChunk]:
    """Create same-section-overlapping live-web chunks with full page provenance."""

    content_hash = hashlib.sha256(_live_content(document)).hexdigest()
    source_url = document.final_url
    publisher = urlsplit(str(source_url)).hostname or "unknown publisher"
    chunks: list[WebEvidenceChunk] = []
    chunk_position = 0

    def prefix_for_section(section: SourceSection) -> str:
        heading_path = section.heading_path or (section.heading,)
        return f"Source: {document.title}\nSection: {' > '.join(heading_path)}\n\n"

    for window in create_section_token_windows(
        sections,
        codec,
        prefix_for_section,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
        min_tokens=min_tokens,
    ):
        heading_path = window.section.heading_path or (window.section.heading,)
        chunk_position += 1
        chunk_id = f"web:{content_hash}:{window.section.position}:{chunk_position}"
        chunks.append(
            WebEvidenceChunk(
                chunk_id=chunk_id,
                canonical_candidate_id=canonical_candidate_id(
                    source_url=str(source_url),
                    source_content_hash=content_hash,
                    text=window.text,
                ),
                source_id=f"web:{content_hash}",
                source_content_hash=content_hash,
                source_title=document.title,
                publisher=publisher,
                source_url=source_url,
                topics=["live_web"],
                published_at=document.published_at,
                retrieved_at=document.retrieved_at,
                heading=" > ".join(heading_path),
                heading_path=list(heading_path),
                section_position=window.section.position,
                chunk_position=chunk_position,
                token_count=window.prefix_token_count + window.body_token_count,
                body_start_token=window.body_start_token,
                body_end_token_exclusive=window.body_end_token_exclusive,
                text=window.text,
            )
        )
    return chunks


def _live_content(document: FetchedWebDocument) -> bytes:
    """Require transient page bytes only while a live page is being processed."""

    if document.content is None:
        raise ValueError("Fetched page content is unavailable outside its live retrieval run.")
    return document.content
