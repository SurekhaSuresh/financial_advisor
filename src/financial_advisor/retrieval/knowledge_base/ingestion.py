"""Offline ingestion primitives for the curated Financial Advisor knowledge base."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Literal, cast

import lancedb
import yaml
from fastembed import TextEmbedding
from pydantic import BaseModel, Field, HttpUrl
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from financial_advisor.config import get_settings
from financial_advisor.retrieval.content_processing.chunking import (
    SourceEvidenceChunk,
    TokenWindowCodec,
    canonical_candidate_id,
    create_section_token_windows,
)
from financial_advisor.retrieval.content_processing.parsing import (
    SourceSection,
    parse_html_sections,
    parse_pdf_sections,
)

_MODEL_RUNTIME = get_settings().model_runtime
BGE_MODEL_NAME = _MODEL_RUNTIME.embedding_model_name
DEFAULT_MODEL_CACHE = _MODEL_RUNTIME.model_cache_directory
KNOWLEDGE_TABLE = "knowledge_chunks"
_POLICY = get_settings().knowledge_ingestion
MAX_CHUNK_TOKENS = _POLICY.max_chunk_tokens
MIN_CHUNK_TOKENS = _POLICY.min_chunk_tokens
CHUNK_OVERLAP_TOKENS = _POLICY.chunk_overlap_tokens


class KnowledgeSource(BaseModel):
    """One approved stable public source declared in the source manifest."""

    id: str = Field(pattern=r"^[a-z0-9_]+$")
    publisher: str = Field(min_length=1)
    title: str = Field(min_length=1)
    url: HttpUrl
    content_type: Literal["html", "pdf"]
    topics: list[str] = Field(min_length=1)


class KnowledgeSourceManifest(BaseModel):
    """The versioned set of sources eligible for local ingestion."""

    schema_version: int
    sources: list[KnowledgeSource] = Field(min_length=1)


KnowledgeChunk = SourceEvidenceChunk


class BgeTokenWindowCodec:
    """Uses BGE's Hugging Face tokenizer for accurate chunk-size boundaries."""

    def __init__(
        self, model_name: str = BGE_MODEL_NAME, cache_directory: Path = DEFAULT_MODEL_CACHE
    ) -> None:
        cache_directory.mkdir(parents=True, exist_ok=True)
        try:
            tokenizer = AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call]
                model_name,
                cache_dir=str(cache_directory),
                local_files_only=True,
            )
        except OSError:
            # First-time setup downloads the model; normal repeat ingestions use
            # the local cache and do not depend on the network.
            tokenizer = AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call]
                model_name, cache_dir=str(cache_directory)
            )
        self._tokenizer = cast(PreTrainedTokenizerBase, tokenizer)
        # Sections may be longer than the model limit before this class splits them.
        self._tokenizer.model_max_length = 1_000_000

    def encode(self, text: str) -> list[int]:
        return cast(list[int], self._tokenizer.encode(text, add_special_tokens=False))

    def decode(self, token_ids: Sequence[int]) -> str:
        return self._tokenizer.decode(token_ids, skip_special_tokens=True).strip()


class BgeEmbedder:
    """Creates local BGE embeddings without a paid embedding API."""

    def __init__(
        self, model_name: str = BGE_MODEL_NAME, cache_directory: Path = DEFAULT_MODEL_CACHE
    ) -> None:
        cache_directory.mkdir(parents=True, exist_ok=True)
        self._model = TextEmbedding(model_name=model_name, cache_dir=str(cache_directory))

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self._model.embed(texts)]


def load_source_manifest(path: Path) -> KnowledgeSourceManifest:
    """Load and validate the project's explicit local-knowledge source allowlist."""

    raw_manifest = yaml.safe_load(path.read_text(encoding="utf-8"))
    return KnowledgeSourceManifest.model_validate(raw_manifest)


def parse_snapshot(source: KnowledgeSource, snapshot_path: Path) -> list[SourceSection]:
    """Extract heading-aware HTML sections or page-aware PDF sections from a snapshot."""

    if source.content_type == "html":
        return parse_html_sections(snapshot_path.read_text(encoding="utf-8"))
    return parse_pdf_sections(snapshot_path.read_bytes())


def chunk_sections(
    source: KnowledgeSource,
    sections: Sequence[SourceSection],
    codec: TokenWindowCodec,
    *,
    max_tokens: int = MAX_CHUNK_TOKENS,
    min_tokens: int = MIN_CHUNK_TOKENS,
    overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
    source_content_hash: str | None = None,
    retrieved_at: datetime | None = None,
) -> list[KnowledgeChunk]:
    """Create bounded chunks, merging an undersized same-section tail when configured."""

    chunks: list[KnowledgeChunk] = []
    chunk_position = 0

    def prefix_for_section(section: SourceSection) -> str:
        heading_path = section.heading_path or (section.heading,)
        return f"Source: {source.title}\nSection: {' > '.join(heading_path)}\n\n"

    for window in create_section_token_windows(
        sections,
        codec,
        prefix_for_section,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
        min_tokens=min_tokens,
    ):
        section_heading_path = window.section.heading_path or (window.section.heading,)
        chunk_position += 1
        chunk_id = f"{source.id}:{window.section.position}:{chunk_position}"
        chunks.append(
            KnowledgeChunk(
                chunk_id=chunk_id,
                canonical_candidate_id=canonical_candidate_id(
                    source_url=str(source.url),
                    source_content_hash=source_content_hash,
                    text=window.text,
                ),
                source_id=source.id,
                source_title=source.title,
                publisher=source.publisher,
                source_url=source.url,
                topics=source.topics,
                heading=window.section.heading,
                heading_path=list(section_heading_path),
                section_position=window.section.position,
                chunk_position=chunk_position,
                token_count=window.prefix_token_count + window.body_token_count,
                body_start_token=window.body_start_token,
                body_end_token_exclusive=window.body_end_token_exclusive,
                text=window.text,
                source_content_hash=source_content_hash,
                retrieved_at=retrieved_at,
            )
        )
    return chunks


def write_lancedb(
    database_path: Path,
    chunks: Sequence[KnowledgeChunk],
    embeddings: Sequence[Sequence[float]],
    *,
    table_name: str = KNOWLEDGE_TABLE,
) -> None:
    """Replace one generated knowledge table with source-attributed chunk vectors."""

    if len(chunks) != len(embeddings):
        raise ValueError("Each knowledge chunk must have exactly one embedding.")
    if not chunks:
        raise ValueError("At least one knowledge chunk is required for LanceDB ingestion.")

    database_path.mkdir(parents=True, exist_ok=True)
    records = [
        {
            "chunk_id": chunk.chunk_id,
            "canonical_candidate_id": chunk.canonical_candidate_id,
            "source_id": chunk.source_id,
            "source_title": chunk.source_title,
            "publisher": chunk.publisher,
            "source_url": str(chunk.source_url),
            "topics": chunk.topics,
            "heading": chunk.heading,
            "heading_path": chunk.heading_path,
            "section_position": chunk.section_position,
            "chunk_position": chunk.chunk_position,
            "token_count": chunk.token_count,
            "body_start_token": chunk.body_start_token,
            "body_end_token_exclusive": chunk.body_end_token_exclusive,
            "text": chunk.text,
            "source_content_hash": chunk.source_content_hash,
            "retrieved_at": chunk.retrieved_at.isoformat() if chunk.retrieved_at else None,
            "vector": list(embedding),
        }
        for chunk, embedding in zip(chunks, embeddings, strict=True)
    ]
    database = lancedb.connect(str(database_path))
    database.create_table(table_name, data=records, mode="overwrite")
