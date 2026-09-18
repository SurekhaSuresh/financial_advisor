"""Compose live-web discovery, page processing, and in-memory evidence retrieval."""

from __future__ import annotations

from enum import StrEnum
from typing import Protocol

from pydantic import BaseModel, Field, HttpUrl

from financial_advisor.config import get_settings
from financial_advisor.retrieval.content_processing.chunking import TokenWindowCodec
from financial_advisor.retrieval.web.discovery import (
    MAX_WEB_DISCOVERY_RESULT_LIMIT,
    WebResearchUnavailable,
    WebSearchRequest,
    WebSearchResponse,
)
from financial_advisor.retrieval.web.fetch import (
    WebFetchStatus,
    WebPageFetcher,
    WebPageFetchOutcome,
)
from financial_advisor.retrieval.web.page_processing import (
    chunk_fetched_web_document,
    parse_fetched_web_document,
)
from financial_advisor.retrieval.web.passage_ranker import (
    InMemoryWebVectorRetriever,
    WebVectorRetrievalTrace,
)

DEFAULT_WEB_USABLE_PAGE_TARGET = get_settings().web_research.usable_page_target


class WebDiscoveryService(Protocol):
    """The provider-failover discovery behavior required by the composed pipeline."""

    def search(self, request: WebSearchRequest) -> WebSearchResponse | WebResearchUnavailable:
        """Discover bounded page URLs or return a structured unavailable result."""


class WebContentProcessingStatus(StrEnum):
    """Whether one successfully fetched page produced usable chunk candidates."""

    PROCESSED = "processed"
    FAILED = "failed"


class WebContentProcessingOutcome(BaseModel):
    """Trace-safe parse and chunk result for one fetched original page."""

    source_url: HttpUrl
    status: WebContentProcessingStatus
    section_count: int = Field(ge=0)
    chunk_count: int = Field(ge=0)
    error_message: str | None = Field(default=None, max_length=500)


class LiveWebResearchTrace(BaseModel):
    """Replayable trajectory from live URL discovery through ranked web candidates."""

    discovery: WebSearchResponse | WebResearchUnavailable
    discovered_result_count: int = Field(ge=0)
    usable_page_target: int = Field(ge=1)
    attempted_page_count: int = Field(ge=0)
    usable_page_count: int = Field(ge=0)
    page_fetches: list[WebPageFetchOutcome]
    content_processing: list[WebContentProcessingOutcome]
    web_vector_retrieval: WebVectorRetrievalTrace


class LiveWebResearchPipeline:
    """Run bounded live-web research without persisting time-sensitive pages."""

    def __init__(
        self,
        discovery: WebDiscoveryService,
        page_fetcher: WebPageFetcher,
        codec: TokenWindowCodec,
        web_retriever: InMemoryWebVectorRetriever,
        *,
        usable_page_target: int = DEFAULT_WEB_USABLE_PAGE_TARGET,
    ) -> None:
        if not 1 <= usable_page_target <= MAX_WEB_DISCOVERY_RESULT_LIMIT:
            raise ValueError("usable_page_target must be within the discovery-result limit.")
        self._discovery = discovery
        self._page_fetcher = page_fetcher
        self._codec = codec
        self._web_retriever = web_retriever
        self._usable_page_target = usable_page_target

    def research(self, request: WebSearchRequest) -> LiveWebResearchTrace:
        """Discover, fetch, parse, chunk, and rank usable live-web evidence."""

        discovery = self._discovery.search(request)
        if isinstance(discovery, WebResearchUnavailable):
            return LiveWebResearchTrace(
                discovery=discovery,
                discovered_result_count=0,
                usable_page_target=self._usable_page_target,
                attempted_page_count=0,
                usable_page_count=0,
                page_fetches=[],
                content_processing=[],
                web_vector_retrieval=self._web_retriever.search(request.query, []),
            )

        page_fetches: list[WebPageFetchOutcome] = []
        processing: list[WebContentProcessingOutcome] = []
        chunks = []
        usable_page_count = 0
        for result in discovery.results:
            if usable_page_count >= self._usable_page_target:
                break
            fetch_outcome = self._page_fetcher.fetch(
                result, allowed_domains=request.allowed_domains
            )
            page_fetches.append(fetch_outcome)
            if fetch_outcome.status is not WebFetchStatus.FETCHED:
                continue
            assert fetch_outcome.document is not None
            try:
                sections = parse_fetched_web_document(fetch_outcome.document)
                page_chunks = chunk_fetched_web_document(
                    fetch_outcome.document, sections, self._codec
                )
            except Exception:
                processing.append(
                    WebContentProcessingOutcome(
                        source_url=fetch_outcome.document.final_url,
                        status=WebContentProcessingStatus.FAILED,
                        section_count=0,
                        chunk_count=0,
                        error_message="Fetched page could not be parsed into usable evidence.",
                    )
                )
                continue
            chunks.extend(page_chunks)
            if page_chunks:
                usable_page_count += 1
            processing.append(
                WebContentProcessingOutcome(
                    source_url=fetch_outcome.document.final_url,
                    status=WebContentProcessingStatus.PROCESSED,
                    section_count=len(sections),
                    chunk_count=len(page_chunks),
                )
            )
        return LiveWebResearchTrace(
            discovery=discovery,
            discovered_result_count=len(discovery.results),
            usable_page_target=self._usable_page_target,
            attempted_page_count=len(page_fetches),
            usable_page_count=usable_page_count,
            page_fetches=page_fetches,
            content_processing=processing,
            web_vector_retrieval=self._web_retriever.search(request.query, chunks),
        )
