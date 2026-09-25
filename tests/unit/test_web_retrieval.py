import pytest

import financial_advisor.retrieval.web as web_module
from financial_advisor.contracts import RetrievalChannel
from financial_advisor.retrieval.providers import (
    WebError,
    WebResult,
    validate_url,
)
from financial_advisor.retrieval.web import WebCandidateRetriever
from financial_advisor.retrieval.web_fetch import WebDocument


def encode(text: str) -> list[int]:
    return [ord(character) for character in text]


def decode(tokens: list[int]) -> str:
    return "".join(chr(token) for token in tokens)


def test_url_policy_rejects_private_hosts_and_credentials() -> None:
    validate_url("https://www.example.com/article")

    with pytest.raises(ValueError, match="Private network"):
        validate_url("http://127.0.0.1/internal")
    with pytest.raises(ValueError, match="credentials"):
        validate_url("https://user:password@investor.gov/article")


def test_web_candidate_retriever_fails_over_and_ranks_fetched_page_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    searched_queries: list[str] = []

    def unavailable(_query: str, _limit: int) -> list[WebResult]:
        raise WebError("unavailable", retryable=False)

    def available(query: str, _limit: int) -> list[WebResult]:
        searched_queries.append(query)
        return [WebResult("Current guidance", "https://www.investor.gov/current")]

    retriever = WebCandidateRetriever(
        [("first", unavailable), ("second", available)],
        encode,
        decode,
        embed=lambda texts: [[1.0, float(index + 1)] for index, _ in enumerate(texts)],
        usable_page_target=1,
    )
    monkeypatch.setattr(
        retriever.page_fetcher,
        "fetch",
        lambda _result: WebDocument(
            "Current guidance",
            "https://www.investor.gov/current",
            "text/html",
            b"<main><h1>Rates</h1><p>Current rate guidance for savers.</p></main>",
        ),
    )

    candidates, limitations = retriever.search("current rates")

    assert candidates
    assert candidates[0].retrieval_channel is RetrievalChannel.WEB
    assert "Current rate guidance" in candidates[0].text
    assert candidates[0].cosine_distance is not None
    assert candidates[0].token_count > 0
    assert limitations == []
    assert searched_queries == ["current rates"]


def test_web_candidate_retriever_skips_invalid_page_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retriever = _retriever_with_fetched_page(monkeypatch)

    def reject_invalid_content(_content: bytes, _content_type: str) -> None:
        raise ValueError("invalid page")

    monkeypatch.setattr(web_module, "parse_document", reject_invalid_content)

    web_passages = retriever._fetch_and_chunk_pages(
        [WebResult("Invalid page", "https://www.example.com/invalid")]
    )

    assert web_passages == []


def test_web_candidate_retriever_propagates_unexpected_processing_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retriever = _retriever_with_fetched_page(monkeypatch)

    def fail_unexpectedly(_content: bytes, _content_type: str) -> None:
        raise RuntimeError("unexpected failure")

    monkeypatch.setattr(web_module, "parse_document", fail_unexpectedly)

    with pytest.raises(RuntimeError, match="unexpected failure"):
        retriever._fetch_and_chunk_pages(
            [WebResult("Broken page", "https://www.example.com/broken")]
        )


def _retriever_with_fetched_page(
    monkeypatch: pytest.MonkeyPatch,
) -> WebCandidateRetriever:
    retriever = WebCandidateRetriever(
        [],
        encode,
        decode,
        embed=lambda texts: [[1.0, float(index + 1)] for index, _ in enumerate(texts)],
        usable_page_target=1,
    )
    monkeypatch.setattr(
        retriever.page_fetcher,
        "fetch",
        lambda result: WebDocument(
            result.title,
            result.url,
            "text/html",
            b"<main><p>Fetched content.</p></main>",
        ),
    )
    return retriever
