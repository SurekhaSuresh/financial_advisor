"""Bounded and policy-checked fetching of discovered web pages."""

from dataclasses import dataclass
from http import HTTPStatus
from time import sleep
from typing import Literal, cast
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from financial_advisor.config import (
    DEFAULT_WEB_MAX_PAGE_BYTES,
    DEFAULT_WEB_TIMEOUT_SECONDS,
    WEB_FETCH_ATTEMPTS,
    WEB_FETCH_RETRY_DELAY_SECONDS,
    WEB_FETCH_USER_AGENT,
)
from financial_advisor.retrieval.providers import WebResult, validate_url

WebContentType = Literal["text/html", "application/xhtml+xml", "application/pdf"]
SUPPORTED_CONTENT_TYPES = frozenset({"text/html", "application/xhtml+xml", "application/pdf"})


@dataclass(frozen=True)
class WebDocument:
    title: str
    url: str
    content_type: WebContentType
    content: bytes


class WebPageFetcher:
    """Fetch original pages with URL, content-type, timeout, and size limits."""

    def __init__(
        self,
        *,
        timeout_seconds: float = DEFAULT_WEB_TIMEOUT_SECONDS,
        max_page_bytes: int = DEFAULT_WEB_MAX_PAGE_BYTES,
    ) -> None:
        if timeout_seconds <= 0 or max_page_bytes <= 0:
            raise ValueError("Web page limits must be positive.")
        self.timeout_seconds = timeout_seconds
        self.max_page_bytes = max_page_bytes

    def fetch(self, result: WebResult) -> WebDocument | None:
        """Return a safe supported document, or None when the page is unusable."""

        for attempt_number in range(1, WEB_FETCH_ATTEMPTS + 1):
            is_final_attempt = attempt_number == WEB_FETCH_ATTEMPTS
            try:
                validate_url(result.url)
                request = Request(
                    result.url,
                    headers={"User-Agent": WEB_FETCH_USER_AGENT},
                )
                with urlopen(request, timeout=self.timeout_seconds) as response:  # noqa: S310
                    content_type = response.headers.get_content_type().lower()
                    content_length = response.headers.get("Content-Length")
                    final_url = response.geturl()
                    validate_url(final_url)
                    if content_type not in SUPPORTED_CONTENT_TYPES:
                        return None
                    # Avoid downloading a body already declared larger than the limit.
                    if (
                        content_length
                        and content_length.isdigit()
                        and int(content_length) > self.max_page_bytes
                    ):
                        return None
                    # Reading one extra byte detects an oversized response without
                    # loading the rest of it into memory.
                    content = response.read(self.max_page_bytes + 1)
                if len(content) > self.max_page_bytes:
                    return None
                return WebDocument(
                    title=result.title,
                    url=final_url,
                    content_type=cast(WebContentType, content_type),
                    content=content,
                )
            except HTTPError as error:
                retryable_status = (
                    error.code == HTTPStatus.TOO_MANY_REQUESTS
                    or error.code >= HTTPStatus.INTERNAL_SERVER_ERROR
                )
                # Rate limits and server failures may be temporary. Other HTTP
                # failures are treated as permanent for this retrieval run.
                if not retryable_status or is_final_attempt:
                    return None
                sleep(WEB_FETCH_RETRY_DELAY_SECONDS)
            except ValueError:
                # URL-policy failures will not become valid on a retry.
                return None
            except (TimeoutError, URLError, OSError):
                # Retry a transient connection failure within the fixed attempt limit.
                if is_final_attempt:
                    return None
                sleep(WEB_FETCH_RETRY_DELAY_SECONDS)
        return None
