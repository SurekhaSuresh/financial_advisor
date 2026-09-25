"""Bounded and policy-checked fetching of discovered web pages."""

from dataclasses import dataclass
from http import HTTPStatus
from time import sleep
from typing import Literal, cast
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from financial_advisor.config import (
    DEFAULT_WEB_MAX_PAGE_BYTES,
    DEFAULT_WEB_TIMEOUT_SECONDS,
    WEB_FETCH_ATTEMPTS,
    WEB_FETCH_RETRY_DELAY_SECONDS,
    WEB_FETCH_USER_AGENT,
)
from financial_advisor.retrieval.web.providers import WebResult, validate_url

SupportedWebContentType = Literal["text/html", "application/xhtml+xml", "application/pdf"]
SUPPORTED_WEB_CONTENT_TYPES = frozenset({"text/html", "application/xhtml+xml", "application/pdf"})


@dataclass(frozen=True)
class FetchedWebDocument:
    title: str
    url: str
    content_type: SupportedWebContentType
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

    def fetch(self, discovered_result: WebResult) -> FetchedWebDocument | None:
        """Return a safe supported document, or None when the page is unusable."""

        try:
            validate_url(discovered_result.url)
            request = Request(
                discovered_result.url,
                headers={"User-Agent": WEB_FETCH_USER_AGENT},
            )
        except ValueError:
            return None

        for attempt_index in range(WEB_FETCH_ATTEMPTS):
            is_final_attempt = attempt_index == WEB_FETCH_ATTEMPTS - 1
            try:
                # URL policy restricts the request to HTTP/S before this call.
                with urlopen(request, timeout=self.timeout_seconds) as response:  # noqa: S310
                    response_content_type = response.headers.get_content_type().lower()
                    declared_content_length = response.headers.get("Content-Length")
                    response_url = response.geturl()
                    validate_url(response_url)
                    if response_content_type not in SUPPORTED_WEB_CONTENT_TYPES:
                        return None
                    # Avoid downloading a body already declared larger than the limit.
                    if (
                        declared_content_length
                        and declared_content_length.isdigit()
                        and int(declared_content_length) > self.max_page_bytes
                    ):
                        return None
                    # Reading one extra byte detects an oversized response without
                    # loading the rest of it into memory.
                    page_content = response.read(self.max_page_bytes + 1)
                if len(page_content) > self.max_page_bytes:
                    return None
                return FetchedWebDocument(
                    title=discovered_result.title,
                    url=response_url,
                    content_type=cast(SupportedWebContentType, response_content_type),
                    content=page_content,
                )
            except HTTPError as error:
                retryable_status = (
                    error.code == HTTPStatus.TOO_MANY_REQUESTS
                    or error.code >= HTTPStatus.INTERNAL_SERVER_ERROR
                )
                # Rate limits and server failures may be temporary. Other HTTP
                # failures are treated as permanent for this retrieval run.
                if not retryable_status:
                    return None
            except ValueError:
                # URL-policy failures will not become valid on a retry.
                return None
            except OSError:
                # Connection failures use the bounded retry path below.
                pass

            if is_final_attempt:
                return None
            sleep(WEB_FETCH_RETRY_DELAY_SECONDS)
        return None
