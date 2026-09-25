from email.message import Message
from io import BytesIO
from urllib.error import HTTPError, URLError

import pytest

import financial_advisor.retrieval.web.page_fetch as page_fetch_module
from financial_advisor.retrieval.web.page_fetch import FetchedWebDocument, WebPageFetcher
from financial_advisor.retrieval.web.providers import WebResult


class WebResponse(BytesIO):
    def __init__(
        self,
        content: bytes,
        *,
        url: str = "https://www.example.com/article",
        content_type: str = "text/html",
        content_length: int | None = None,
    ) -> None:
        super().__init__(content)
        self.url = url
        self.headers = Message()
        self.headers["Content-Type"] = content_type
        if content_length is not None:
            self.headers["Content-Length"] = str(content_length)

    def geturl(self) -> str:
        return self.url


def test_fetch_returns_supported_bounded_web_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = WebResponse(b"<main>Useful content</main>")
    monkeypatch.setattr(page_fetch_module, "urlopen", lambda *_args, **_kwargs: response)

    document = WebPageFetcher().fetch(
        WebResult("Useful article", "https://www.example.com/article")
    )

    assert document == FetchedWebDocument(
        title="Useful article",
        url="https://www.example.com/article",
        content_type="text/html",
        content=b"<main>Useful content</main>",
    )


def test_fetch_retries_a_transient_connection_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outcomes: list[object] = [
        URLError("temporary failure"),
        WebResponse(b"available"),
    ]
    retry_delays: list[float] = []

    def open_next(*_args: object, **_kwargs: object) -> object:
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(page_fetch_module, "urlopen", open_next)
    monkeypatch.setattr(page_fetch_module, "sleep", retry_delays.append)

    document = WebPageFetcher().fetch(WebResult("Article", "https://www.example.com/article"))

    assert document is not None
    assert retry_delays == [page_fetch_module.WEB_FETCH_RETRY_DELAY_SECONDS]


def test_fetch_does_not_retry_a_permanent_http_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def fail(*_args: object, **_kwargs: object) -> object:
        nonlocal calls
        calls += 1
        raise HTTPError(
            "https://www.example.com/missing",
            404,
            "Not Found",
            {},
            None,
        )

    monkeypatch.setattr(page_fetch_module, "urlopen", fail)

    document = WebPageFetcher().fetch(WebResult("Missing", "https://www.example.com/missing"))

    assert document is None
    assert calls == 1


def test_fetch_rejects_a_page_larger_than_the_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = WebResponse(b"unused", content_length=11)
    monkeypatch.setattr(page_fetch_module, "urlopen", lambda *_args, **_kwargs: response)

    document = WebPageFetcher(max_page_bytes=10).fetch(
        WebResult("Large page", "https://www.example.com/large")
    )

    assert document is None
