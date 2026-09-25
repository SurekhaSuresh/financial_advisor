"""Parse and chunk local snapshots and fetched web documents."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from typing import Literal
from unicodedata import normalize
from urllib.parse import urlsplit, urlunsplit

from bs4 import BeautifulSoup
from pypdf import PdfReader

Encode = Callable[[str], list[int]]
Decode = Callable[[Sequence[int]], str]


@dataclass(frozen=True)
class Section:
    heading_path: tuple[str, ...]
    text: str


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    canonical_candidate_id: str
    source_id: str
    source_title: str
    publisher: str
    source_url: str
    topics: tuple[str, ...]
    heading_path: tuple[str, ...]
    section_position: int
    chunk_position: int
    token_count: int
    text: str

    @property
    def embedding_text(self) -> str:
        heading = " > ".join(self.heading_path)
        return f"Source: {self.source_title}\nSection: {heading}\n\n{self.text}"


def parse_document(
    content: bytes,
    content_type: Literal["text/html", "application/xhtml+xml", "application/pdf"],
) -> list[Section]:
    """Extract ordered sections without summarizing source content."""

    if content_type == "application/pdf":
        return [
            Section((f"Page {number}",), text)
            for number, page in enumerate(PdfReader(BytesIO(content)).pages, start=1)
            if (text := " ".join((page.extract_text() or "").split()))
        ]

    soup = BeautifulSoup(content.decode("utf-8", errors="replace"), "html.parser")
    for ignored in soup(["script", "style", "noscript", "template"]):
        ignored.decompose()

    sections: list[Section] = []
    headings = ["Document introduction"]
    paragraphs: list[str] = []

    def flush() -> None:
        if paragraphs:
            sections.append(Section(tuple(headings), " ".join(paragraphs)))
            paragraphs.clear()

    root = soup.find("main") or soup.body or soup
    for element in root.find_all(["h1", "h2", "h3", "h4", "p", "li"]):
        text = " ".join(element.get_text(" ", strip=True).split())
        if not text:
            continue
        if element.name and element.name.startswith("h"):
            flush()
            level = int(element.name[1])
            if headings == ["Document introduction"]:
                headings = []
            headings = [*headings[: level - 1], text]
        else:
            paragraphs.append(text)
    flush()
    return sections


def chunk_document(
    sections: Sequence[Section],
    *,
    source_id: str,
    source_title: str,
    publisher: str,
    source_url: str,
    topics: Sequence[str],
    content_hash: str,
    encode: Encode,
    decode: Decode,
    max_tokens: int,
    overlap_tokens: int,
    min_tokens: int,
) -> list[Chunk]:
    """Create overlapping token windows without crossing section boundaries."""

    if max_tokens <= 0 or not 0 <= overlap_tokens < max_tokens or min_tokens < 0:
        raise ValueError("Invalid chunk-size configuration.")

    chunks: list[Chunk] = []
    chunk_position = 0
    for section_position, section in enumerate(sections, start=1):
        prefix = f"Source: {source_title}\nSection: {' > '.join(section.heading_path)}\n\n"
        prefix_token_count = len(encode(prefix))
        body_limit = max_tokens - prefix_token_count
        if body_limit <= 0:
            raise ValueError("Source title and heading leave no room for chunk content.")

        token_ids = encode(section.text)
        overlap = min(overlap_tokens, body_limit - 1)
        windows: list[tuple[int, int]] = []
        start = 0
        while start < len(token_ids):
            end = min(start + body_limit, len(token_ids))
            windows.append((start, end))
            if end == len(token_ids):
                break
            start += body_limit - overlap

        if len(windows) > 1 and min_tokens:
            tail_start, tail_end = windows[-1]
            if prefix_token_count + tail_end - tail_start < min_tokens:
                windows[-2:] = [(windows[-2][0], tail_end)]

        for start, end in windows:
            text = decode(token_ids[start:end]).strip()
            if text:
                chunk_position += 1
                chunks.append(
                    Chunk(
                        chunk_id=f"{source_id}:{section_position}:{chunk_position}",
                        canonical_candidate_id=_canonical_candidate_id(
                            source_url,
                            content_hash,
                            text,
                        ),
                        source_id=source_id,
                        source_title=source_title,
                        publisher=publisher,
                        source_url=source_url,
                        topics=tuple(topics),
                        heading_path=section.heading_path,
                        section_position=section_position,
                        chunk_position=chunk_position,
                        token_count=prefix_token_count + end - start,
                        text=text,
                    )
                )
    return chunks


def _canonical_candidate_id(url: str, content_hash: str, text: str) -> str:
    normalized_text = " ".join(normalize("NFKC", text).split()).casefold()
    document_id = content_hash or _canonical_url(url)
    return sha256(f"{document_id}\n{normalized_text}".encode()).hexdigest()


def _canonical_url(url: str) -> str:
    parsed = urlsplit(url)
    host = (parsed.hostname or "").casefold()
    port = parsed.port
    netloc = host if port in {None, 80, 443} else f"{host}:{port}"
    return urlunsplit(
        (parsed.scheme.casefold(), netloc, parsed.path.rstrip("/") or "/", parsed.query, "")
    )
