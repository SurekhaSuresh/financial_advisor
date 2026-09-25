"""Reusable BGE-token windowing for parsed stable and live-web source sections."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from hashlib import sha256
from typing import Protocol
from unicodedata import normalize
from urllib.parse import urlsplit, urlunsplit

from pydantic import BaseModel, Field, HttpUrl

from financial_advisor.retrieval.content_processing.parsing import SourceSection


class TokenWindowCodec(Protocol):
    """Encodes and decodes text using the same tokenizer as the embedding model."""

    def encode(self, text: str) -> list[int]: ...

    def decode(self, token_ids: Sequence[int]) -> str: ...


@dataclass(frozen=True)
class SectionTokenWindow:
    """One body window plus the source hierarchy tokens that accompany it."""

    section: SourceSection
    text: str
    prefix_token_count: int
    body_start_token: int
    body_end_token_exclusive: int
    body_token_count: int


class SourceEvidenceChunk(BaseModel):
    """Shared source content and provenance before any retrieval channel ranks it."""

    chunk_id: str
    canonical_candidate_id: str
    source_id: str
    source_title: str
    publisher: str
    source_url: HttpUrl
    topics: list[str]
    heading: str
    heading_path: list[str]
    section_position: int = Field(ge=1)
    chunk_position: int = Field(ge=1)
    token_count: int = Field(gt=0)
    body_start_token: int | None = Field(default=None, ge=0)
    body_end_token_exclusive: int | None = Field(default=None, gt=0)
    text: str = Field(min_length=1)
    source_content_hash: str | None = None
    published_at: date | None = None
    retrieved_at: datetime | None = None

    @property
    def embedding_text(self) -> str:
        """Give all evidence sources the same embedding representation."""

        return (
            f"Source: {self.source_title}\nSection: {' > '.join(self.heading_path)}\n\n{self.text}"
        )


def canonical_candidate_id(
    *, source_url: str, source_content_hash: str | None, text: str
) -> str:
    """Create a storage-independent identity for one exact source passage.

    A source-content hash identifies the exact document version. It is preferred
    over the URL so an unchanged document reached through a redirect can still
    match its curated snapshot. The canonical URL is a conservative fallback
    for callers that do not have a document hash.
    """

    normalized_text = " ".join(normalize("NFKC", text).split()).casefold()
    document_identity = source_content_hash or _canonical_source_url(source_url)
    digest = sha256(f"{document_identity}\n{normalized_text}".encode()).hexdigest()
    return f"evidence:{digest}"


def _canonical_source_url(source_url: str) -> str:
    """Normalize only URL details that do not identify different source content."""

    parsed = urlsplit(source_url)
    hostname = (parsed.hostname or "").casefold()
    port = parsed.port
    netloc = hostname if port in {None, 80, 443} else f"{hostname}:{port}"
    path = parsed.path.rstrip("/") or "/"
    return urlunsplit((parsed.scheme.casefold(), netloc, path, parsed.query, ""))


def create_section_token_windows(
    sections: Sequence[SourceSection],
    codec: TokenWindowCodec,
    prefix_for_section: Callable[[SourceSection], str],
    *,
    max_tokens: int,
    overlap_tokens: int,
    min_tokens: int = 0,
) -> list[SectionTokenWindow]:
    """Split sections into same-section-overlapping windows and merge a short tail.

    A nominal maximum bounds ordinary windows. The final undersized window of a
    section may merge into its immediate predecessor, even when that produces a
    modest overage, because one coherent passage is stronger than a tiny tail.
    Windows from distinct sections are never merged.
    """

    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive.")
    if not 0 <= overlap_tokens < max_tokens:
        raise ValueError("overlap_tokens must be non-negative and smaller than max_tokens.")
    if min_tokens < 0:
        raise ValueError("min_tokens must be non-negative.")
    # Callers may lower max_tokens for a focused test or a constrained source.
    # A configured floor above that limit cannot be meaningful, so disable the
    # merge rather than turning every final window into an over-limit chunk.
    effective_min_tokens = min_tokens if min_tokens <= max_tokens else 0

    windows: list[SectionTokenWindow] = []
    for section in sections:
        prefix_token_count = len(codec.encode(prefix_for_section(section)))
        body_token_budget = max_tokens - prefix_token_count
        if body_token_budget <= 0:
            raise ValueError(
                "Source title and heading path leave no room in the configured chunk size."
            )
        window_overlap = min(overlap_tokens, body_token_budget - 1)
        token_ids = codec.encode(section.text)
        section_windows: list[SectionTokenWindow] = []
        start = 0
        while start < len(token_ids):
            token_window = token_ids[start : start + body_token_budget]
            text = codec.decode(token_window)
            if text:
                section_windows.append(
                    SectionTokenWindow(
                        section=section,
                        text=text,
                        prefix_token_count=prefix_token_count,
                        body_start_token=start,
                        body_end_token_exclusive=start + len(token_window),
                        body_token_count=len(token_window),
                    )
                )
            if start + body_token_budget >= len(token_ids):
                break
            start += body_token_budget - window_overlap
        if (
            effective_min_tokens
            and len(section_windows) >= 2
            and (
                section_windows[-1].prefix_token_count + section_windows[-1].body_token_count
                < effective_min_tokens
            )
        ):
            previous_window = section_windows[-2]
            short_tail = section_windows[-1]
            merged_token_ids = token_ids[
                previous_window.body_start_token : short_tail.body_end_token_exclusive
            ]
            merged_text = codec.decode(merged_token_ids)
            if merged_text:
                section_windows[-2] = SectionTokenWindow(
                    section=section,
                    text=merged_text,
                    prefix_token_count=prefix_token_count,
                    body_start_token=previous_window.body_start_token,
                    body_end_token_exclusive=short_tail.body_end_token_exclusive,
                    body_token_count=len(merged_token_ids),
                )
                section_windows.pop()
        windows.extend(section_windows)
    return windows
