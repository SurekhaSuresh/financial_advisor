"""Structure-preserving HTML and PDF parsing shared by knowledge and live web evidence."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Literal

from bs4 import BeautifulSoup
from pypdf import PdfReader


@dataclass(frozen=True)
class SourceSection:
    """One parser-preserved document section before token-window chunking."""

    heading: str
    text: str
    position: int
    heading_path: tuple[str, ...] = ()


def parse_document(
    content: bytes, content_type: Literal["text/html", "application/xhtml+xml", "application/pdf"]
) -> list[SourceSection]:
    """Parse a validated document into ordered, structure-aware text sections."""

    if content_type in {"text/html", "application/xhtml+xml"}:
        return parse_html_sections(content.decode("utf-8", errors="replace"))
    return parse_pdf_sections(content)


def parse_html_sections(html: str) -> list[SourceSection]:
    """Group visible HTML paragraphs and lists under their heading hierarchy."""

    soup = BeautifulSoup(html, "html.parser")
    for ignored in soup(["script", "style", "noscript", "template"]):
        ignored.decompose()
    root = soup.find("main") or soup.body or soup
    heading_path = ["Document introduction"]
    section_parts: list[str] = []
    sections: list[SourceSection] = []

    def flush() -> None:
        if section_parts:
            current_path = tuple(heading_path)
            sections.append(
                SourceSection(
                    heading=" > ".join(current_path),
                    text=" ".join(section_parts),
                    position=len(sections) + 1,
                    heading_path=current_path,
                )
            )
            section_parts.clear()

    for element in root.find_all(["h1", "h2", "h3", "h4", "p", "li"]):
        text = " ".join(element.get_text(" ", strip=True).split())
        if not text:
            continue
        if element.name.startswith("h"):
            flush()
            level = int(element.name[1])
            if heading_path == ["Document introduction"]:
                heading_path = []
            heading_path = heading_path[: level - 1]
            heading_path.append(text)
        else:
            section_parts.append(text)
    flush()
    return sections


def parse_pdf_sections(content: bytes) -> list[SourceSection]:
    """Extract each text-bearing PDF page as one position-preserving section."""

    sections: list[SourceSection] = []
    for page_number, page in enumerate(PdfReader(BytesIO(content)).pages, start=1):
        text = " ".join((page.extract_text() or "").split())
        if text:
            sections.append(
                SourceSection(
                    heading=f"Page {page_number}",
                    text=text,
                    position=page_number,
                )
            )
    return sections
