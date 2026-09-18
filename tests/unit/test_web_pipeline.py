from collections.abc import Mapping, Sequence

from financial_advisor.domain import SearchMode
from financial_advisor.retrieval.web.discovery import (
    SearchProviderName,
    WebSearchRequest,
    WebSearchResponse,
    WebSearchResult,
)
from financial_advisor.retrieval.web.fetch import (
    RawDocumentResponse,
    WebPageFetcher,
    WebPageFetchError,
)
from financial_advisor.retrieval.web.passage_ranker import InMemoryWebVectorRetriever
from financial_advisor.retrieval.web.pipeline import (
    LiveWebResearchPipeline,
    WebContentProcessingStatus,
)


class FakeDiscovery:
    def __init__(self, response: WebSearchResponse) -> None:
        self._response = response

    def search(self, request: WebSearchRequest) -> WebSearchResponse:
        del request
        return self._response


class FakeDocumentTransport:
    def __init__(self, responses: list[RawDocumentResponse | Exception]) -> None:
        self._responses = responses

    def fetch(
        self,
        url: str,
        headers: Mapping[str, str],
        *,
        timeout_seconds: float,
        max_bytes: int,
    ) -> RawDocumentResponse:
        del url, headers, timeout_seconds, max_bytes
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class CharacterCodec:
    def encode(self, text: str) -> list[int]:
        return [ord(character) for character in text]

    def decode(self, token_ids: Sequence[int]) -> str:
        return "".join(chr(token_id) for token_id in token_ids)


class StaticEmbedder:
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        assert len(texts) == 2  # one query and one chunk for this fixture
        return [[1.0, 0.0], [0.9, 0.1]]


def test_live_pipeline_returns_ranked_web_candidate_and_processing_trace() -> None:
    request = WebSearchRequest(
        query="How should I diversify?",
        search_mode=SearchMode.AUTHORITATIVE_DOMAIN,
        allowed_domains=("investor.gov",),
    )
    discovery = FakeDiscovery(
        WebSearchResponse(
            provider=SearchProviderName.EXA,
            results=[
                WebSearchResult(
                    title="Investor guidance",
                    url="https://www.investor.gov/example",
                    snippet="Discovery only.",
                )
            ],
        )
    )
    fetcher = WebPageFetcher(
        transport=FakeDocumentTransport(
            [
                RawDocumentResponse(
                    final_url="https://www.investor.gov/example",
                    content_type="text/html",
                    body=(
                        b"<html><body><h1>Diversification</h1>"
                        b"<p>Spread investments.</p></body></html>"
                    ),
                )
            ]
        )
    )
    pipeline = LiveWebResearchPipeline(
        discovery,
        fetcher,
        CharacterCodec(),
        InMemoryWebVectorRetriever(StaticEmbedder()),
    )

    trace = pipeline.research(request)

    assert trace.discovered_result_count == 1
    assert trace.attempted_page_count == 1
    assert trace.usable_page_count == 1
    assert trace.content_processing[0].status is WebContentProcessingStatus.PROCESSED
    assert trace.content_processing[0].section_count == 1
    assert trace.content_processing[0].chunk_count == 1
    assert len(trace.web_vector_retrieval.candidates) == 1
    assert trace.web_vector_retrieval.candidates[0].retrieval_channel == "live_web"


def test_live_pipeline_tries_the_fifth_url_after_the_first_four_fetches_fail() -> None:
    request = WebSearchRequest(query="How should I diversify?", search_mode=SearchMode.BROAD_WEB)
    discovery = FakeDiscovery(
        WebSearchResponse(
            provider=SearchProviderName.EXA,
            results=[
                WebSearchResult(title=f"Result {number}", url=f"https://example.com/{number}")
                for number in range(1, 6)
            ],
        )
    )
    fetcher = WebPageFetcher(
        transport=FakeDocumentTransport(
            [
                WebPageFetchError("Page was unavailable.", retryable=False),
                WebPageFetchError("Page was unavailable.", retryable=False),
                WebPageFetchError("Page was unavailable.", retryable=False),
                WebPageFetchError("Page was unavailable.", retryable=False),
                RawDocumentResponse(
                    final_url="https://example.com/5",
                    content_type="text/html",
                    body=(
                        b"<html><body><h1>Diversification</h1>"
                        b"<p>Spread investments.</p></body></html>"
                    ),
                ),
            ]
        ),
        max_attempts=1,
    )
    pipeline = LiveWebResearchPipeline(
        discovery,
        fetcher,
        CharacterCodec(),
        InMemoryWebVectorRetriever(StaticEmbedder()),
    )

    trace = pipeline.research(request)

    assert trace.discovered_result_count == 5
    assert trace.attempted_page_count == 5
    assert trace.usable_page_count == 1
    assert len(trace.page_fetches) == 5
    assert trace.page_fetches[-1].document is not None
