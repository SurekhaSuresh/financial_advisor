"""Bounded, policy-aware fetching of original pages discovered by web search."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime
from enum import StrEnum
from time import sleep as time_sleep
from typing import Literal, Protocol, cast
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pydantic import BaseModel, Field, HttpUrl, TypeAdapter, ValidationError

from financial_advisor.config import get_settings
from financial_advisor.retrieval.web.discovery import WebSearchResult
from financial_advisor.retrieval.web.url_policy import validate_http_url_policy

_HTTP_URL_ADAPTER = TypeAdapter(HttpUrl)
_ALLOWED_CONTENT_TYPES = {"text/html", "application/xhtml+xml", "application/pdf"}
_POLICY = get_settings().web_research


class WebFetchStatus(StrEnum):
    FETCHED = "fetched"
    FAILED = "failed"


class WebPageFetchError(Exception):
    """Classified failure raised by the document transport or fetch policy."""

    def __init__(self, message: str, *, retryable: bool) -> None:
        super().__init__(message)
        self.retryable = retryable


@dataclass(frozen=True)
class RawDocumentResponse:
    final_url: str
    content_type: str
    body: bytes


class DocumentHttpTransport(Protocol):
    def fetch(
        self, url: str, headers: Mapping[str, str], *, timeout_seconds: float, max_bytes: int
    ) -> RawDocumentResponse: ...


class UrllibDocumentTransport:
    """Standard-library transport with timeout and response-size protection."""

    def fetch(
        self, url: str, headers: Mapping[str, str], *, timeout_seconds: float, max_bytes: int
    ) -> RawDocumentResponse:
        request = Request(url, headers=dict(headers), method="GET")
        try:
            with urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310
                content_type = response.headers.get_content_type().lower()
                content_length = response.headers.get("Content-Length")
                if content_length and content_length.isdigit() and int(content_length) > max_bytes:
                    raise WebPageFetchError(
                        "Page exceeded the response-size limit.", retryable=False
                    )
                body = response.read(max_bytes + 1)
                final_url = response.geturl()
        except HTTPError as error:
            raise WebPageFetchError(
                f"Page returned HTTP {error.code}.",
                retryable=error.code == 429 or error.code >= 500,
            ) from error
        except (TimeoutError, URLError, OSError) as error:
            raise WebPageFetchError("Page connection failed.", retryable=True) from error
        if len(body) > max_bytes:
            raise WebPageFetchError("Page exceeded the response-size limit.", retryable=False)
        return RawDocumentResponse(final_url=final_url, content_type=content_type, body=body)


class FetchedWebDocument(BaseModel):
    title: str = Field(min_length=1, max_length=500)
    discovered_url: HttpUrl
    final_url: HttpUrl
    content_type: Literal["text/html", "application/xhtml+xml", "application/pdf"]
    published_at: date | None = None
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    # Raw bytes are required while a page is processed, but intentionally
    # excluded from persisted trace JSON. Replayed traces rehydrate this as None.
    content: bytes | None = Field(default=None, exclude=True)


class WebPageFetchOutcome(BaseModel):
    discovered_url: HttpUrl
    status: WebFetchStatus
    attempt_count: int = Field(ge=1, le=2)
    retry_delays_seconds: list[float] = Field(default_factory=list)
    error_message: str | None = Field(default=None, max_length=500)
    document: FetchedWebDocument | None = None


class WebPageFetcher:
    """Fetches original pages with URL policy, size limits, and bounded retries."""

    def __init__(
        self,
        *,
        transport: DocumentHttpTransport | None = None,
        timeout_seconds: float = _POLICY.page_timeout_seconds,
        max_bytes: int = _POLICY.page_max_bytes,
        max_attempts: int = _POLICY.page_max_attempts,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        if timeout_seconds <= 0 or max_bytes <= 0 or not 1 <= max_attempts <= 2:
            raise ValueError("Invalid page fetcher limits.")
        self._transport = transport or UrllibDocumentTransport()
        self._timeout_seconds = timeout_seconds
        self._max_bytes = max_bytes
        self._max_attempts = max_attempts
        self._sleep = sleep or time_sleep

    def fetch(
        self, result: WebSearchResult, *, allowed_domains: tuple[str, ...] = ()
    ) -> WebPageFetchOutcome:
        """Return a validated original document or a safe, structured failure."""

        delays: list[float] = []
        for attempt in range(1, self._max_attempts + 1):
            try:
                _validate_fetch_url(str(result.url), allowed_domains)
                raw = self._transport.fetch(
                    str(result.url),
                    {"User-Agent": _POLICY.page_fetch_user_agent},
                    timeout_seconds=self._timeout_seconds,
                    max_bytes=self._max_bytes,
                )
                _validate_fetch_url(raw.final_url, allowed_domains)
                if raw.content_type not in _ALLOWED_CONTENT_TYPES:
                    raise WebPageFetchError("Page content type is not supported.", retryable=False)
                supported_content_type = cast(
                    Literal["text/html", "application/xhtml+xml", "application/pdf"],
                    raw.content_type,
                )
                document = FetchedWebDocument(
                    title=result.title,
                    discovered_url=result.url,
                    final_url=_HTTP_URL_ADAPTER.validate_python(raw.final_url),
                    content_type=supported_content_type,
                    published_at=result.published_at,
                    content=raw.body,
                )
            except (WebPageFetchError, ValidationError) as error:
                retryable = isinstance(error, WebPageFetchError) and error.retryable
                if not retryable or attempt == self._max_attempts:
                    return WebPageFetchOutcome(
                        discovered_url=result.url,
                        status=WebFetchStatus.FAILED,
                        attempt_count=attempt,
                        retry_delays_seconds=delays,
                        error_message=(
                            str(error)
                            if isinstance(error, WebPageFetchError)
                            else "Invalid page metadata."
                        ),
                    )
                delays.append(_POLICY.page_retry_delay_seconds)
                self._sleep(_POLICY.page_retry_delay_seconds)
            else:
                return WebPageFetchOutcome(
                    discovered_url=result.url,
                    status=WebFetchStatus.FETCHED,
                    attempt_count=attempt,
                    retry_delays_seconds=delays,
                    document=document,
                )
        raise AssertionError("Fetch retry loop must return after final attempt.")


def _validate_fetch_url(url: str, allowed_domains: tuple[str, ...]) -> None:
    """Convert shared URL-policy errors into fetch-domain errors."""

    try:
        validate_http_url_policy(url, allowed_domains)
    except ValueError as error:
        raise WebPageFetchError(str(error).replace("URL", "Page URL"), retryable=False) from error
