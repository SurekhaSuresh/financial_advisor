import pytest

import financial_advisor.retrieval.web.web_search as web_search_module
from financial_advisor.retrieval.web.discovery import WebSourceDiscovery
from financial_advisor.retrieval.web.web_search import WebCandidateRetriever


class Models:
    def encode(self, text: str) -> list[int]:
        return [ord(character) for character in text]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(token) for token in tokens)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, float(index + 1)] for index, _ in enumerate(texts)]


def test_web_candidate_retriever_fails_over_and_ranks_fetched_page_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    searched_queries: list[str] = []

    def request_source_urls(
        provider_name: str,
        _api_key: str,
        query: str,
    ) -> list[tuple[str, str]]:
        if provider_name == "exa":
            raise ValueError("unavailable")
        searched_queries.append(query)
        return [("Current guidance", "https://www.investor.gov/current")]

    retriever = WebCandidateRetriever(
        Models(),  # type: ignore[arg-type]
        WebSourceDiscovery("exa-key", "brave-key"),
    )
    monkeypatch.setattr(retriever.source_discovery, "_request_source_urls", request_source_urls)
    monkeypatch.setattr(
        web_search_module,
        "fetch_page",
        lambda _url: (
            "https://www.investor.gov/current",
            "text/html",
            b"<main><h1>Rates</h1><p>Current rate guidance for savers.</p></main>",
        ),
    )

    candidates = retriever.search("current rates")

    assert candidates
    assert candidates[0].retrieval_channel == "web"
    assert "Current rate guidance" in candidates[0].text
    assert candidates[0].cosine_distance is not None
    assert candidates[0].token_count > 0
    assert searched_queries == ["current rates"]


def test_web_candidate_retriever_skips_invalid_page_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retriever = _retriever_with_fetched_page(monkeypatch)
    monkeypatch.setattr(
        retriever.source_discovery,
        "discover_source_urls",
        lambda _query: [("Invalid page", "https://www.example.com/invalid")],
    )

    def reject_invalid_content(*_args: object, **_kwargs: object) -> None:
        raise ValueError("invalid page")

    monkeypatch.setattr(web_search_module, "chunk_document", reject_invalid_content)

    web_passages = retriever.search("question")

    assert web_passages == []


def test_web_candidate_retriever_propagates_unexpected_processing_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retriever = _retriever_with_fetched_page(monkeypatch)
    monkeypatch.setattr(
        retriever.source_discovery,
        "discover_source_urls",
        lambda _query: [("Broken page", "https://www.example.com/broken")],
    )

    def fail_unexpectedly(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("unexpected failure")

    monkeypatch.setattr(web_search_module, "chunk_document", fail_unexpectedly)

    with pytest.raises(RuntimeError, match="unexpected failure"):
        retriever.search("question")


def _retriever_with_fetched_page(
    monkeypatch: pytest.MonkeyPatch,
) -> WebCandidateRetriever:
    retriever = WebCandidateRetriever(
        Models(),  # type: ignore[arg-type]
        WebSourceDiscovery(None, None),
    )
    monkeypatch.setattr(
        web_search_module,
        "fetch_page",
        lambda url: (url, "text/html", b"<main><p>Fetched content.</p></main>"),
    )
    return retriever
