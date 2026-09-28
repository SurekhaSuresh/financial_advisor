from email.message import Message
from io import BytesIO
from urllib.error import HTTPError, URLError

import pytest

import financial_advisor.retrieval.web.page_fetch as page_fetch_module
from financial_advisor.retrieval.web.page_fetch import fetch_page


class WebResponse(BytesIO):
    def __init__(
        self,
        content: bytes,
        *,
        url: str = "https://www.example.com/article",
        content_type: str = "text/html",
    ) -> None:
        super().__init__(content)
        self.url = url
        self.headers = Message()
        self.headers["Content-Type"] = content_type

    def geturl(self) -> str:
        return self.url


def test_fetch_rejects_unsupported_urls_and_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_request(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("Unsafe URLs must not be requested.")

    monkeypatch.setattr(page_fetch_module, "urlopen", unexpected_request)

    assert fetch_page("file:///tmp/private") is None
    assert fetch_page("https://user:password@example.com") is None


def test_fetch_returns_supported_web_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = WebResponse(b"<main>Useful content</main>")
    monkeypatch.setattr(page_fetch_module, "urlopen", lambda *_args, **_kwargs: response)

    document = fetch_page("https://www.example.com/article")

    assert document == (
        "https://www.example.com/article",
        "text/html",
        b"<main>Useful content</main>",
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

    document = fetch_page("https://www.example.com/article")

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

    document = fetch_page("https://www.example.com/missing")

    assert document is None
    assert calls == 1
