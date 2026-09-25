from collections.abc import Mapping
from datetime import date

from financial_advisor.retrieval.web.discovery import WebSearchResult
from financial_advisor.retrieval.web.fetch import (
    RawDocumentResponse,
    WebFetchStatus,
    WebPageFetcher,
    WebPageFetchError,
)


class FakeDocumentTransport:
    """Deterministic document transport used to test fetch policy and retries."""

    def __init__(self, outcomes: list[RawDocumentResponse | Exception]) -> None:
        self._outcomes = outcomes
        self.call_count = 0

    def fetch(
        self,
        url: str,
        headers: Mapping[str, str],
        *,
        timeout_seconds: float,
        max_bytes: int,
    ) -> RawDocumentResponse:
        del url, headers, timeout_seconds, max_bytes
        outcome = self._outcomes[self.call_count]
        self.call_count += 1
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def make_search_result() -> WebSearchResult:
    return WebSearchResult(
        title="Introduction to investing",
        url="https://www.investor.gov/introduction-investing",
        snippet="Discovery preview only.",
        published_at=date(2026, 9, 1),
    )


def test_fetcher_returns_a_validated_html_document() -> None:
    transport = FakeDocumentTransport(
        [
            RawDocumentResponse(
                final_url="https://www.investor.gov/introduction-investing",
                content_type="text/html",
                body=b"<html><body><h1>Investing</h1></body></html>",
            )
        ]
    )

    outcome = WebPageFetcher(transport=transport).fetch(
        make_search_result(), allowed_domains=("investor.gov",)
    )

    assert outcome.status is WebFetchStatus.FETCHED
    assert outcome.attempt_count == 1
    assert outcome.document is not None
    assert outcome.document.final_url.host == "www.investor.gov"
    assert outcome.document.content.startswith(b"<html")
    assert outcome.document.published_at == date(2026, 9, 1)


def test_fetcher_retries_a_transient_page_failure_once() -> None:
    transport = FakeDocumentTransport(
        [
            WebPageFetchError("Page connection failed.", retryable=True),
            RawDocumentResponse(
                final_url="https://www.investor.gov/introduction-investing",
                content_type="text/html",
                body=b"<html><body>Recovered page</body></html>",
            ),
        ]
    )
    delays: list[float] = []

    outcome = WebPageFetcher(transport=transport, sleep=delays.append).fetch(
        make_search_result(), allowed_domains=("investor.gov",)
    )

    assert outcome.status is WebFetchStatus.FETCHED
    assert outcome.attempt_count == 2
    assert outcome.retry_delays_seconds == [1.0]
    assert delays == [1.0]


def test_fetcher_rejects_a_redirect_to_an_unapproved_domain() -> None:
    transport = FakeDocumentTransport(
        [
            RawDocumentResponse(
                final_url="https://unapproved.example/investing",
                content_type="text/html",
                body=b"<html><body>Do not retain this page.</body></html>",
            )
        ]
    )

    outcome = WebPageFetcher(transport=transport).fetch(
        make_search_result(), allowed_domains=("investor.gov",)
    )

    assert outcome.status is WebFetchStatus.FAILED
    assert outcome.attempt_count == 1
    assert outcome.document is None
    assert outcome.error_message == "Page URL is outside the approved domain policy."
