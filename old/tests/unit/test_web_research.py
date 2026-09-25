from datetime import date

from pydantic import SecretStr

from financial_advisor.domain import SearchMode
from financial_advisor.retrieval.web.discovery import (
    CircuitBreaker,
    CircuitDecisionReason,
    CircuitState,
    ProviderSearchStatus,
    SearchProviderName,
    WebResearchError,
    WebResearchService,
    WebResearchUnavailable,
    WebSearchRequest,
    WebSearchResponse,
    WebSearchResult,
)
from financial_advisor.retrieval.web.providers import (
    BraveWebResearchProvider,
    ExaWebResearchProvider,
)


class FakeClock:
    def __init__(self) -> None:
        self.current = 0.0

    def __call__(self) -> float:
        return self.current

    def advance(self, seconds: float) -> None:
        self.current += seconds


class FakeProvider:
    def __init__(
        self, name: SearchProviderName, outcomes: list[WebSearchResponse | Exception]
    ) -> None:
        self._name = name
        self._outcomes = outcomes
        self.call_count = 0

    @property
    def name(self) -> SearchProviderName:
        return self._name

    def search(self, request: WebSearchRequest) -> WebSearchResponse:
        del request
        outcome = self._outcomes[self.call_count]
        self.call_count += 1
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FakeJsonTransport:
    def __init__(self, responses: list[dict[str, object]]) -> None:
        self._responses = responses
        self.calls: list[dict[str, object]] = []

    def request_json(
        self,
        method: str,
        url: str,
        headers: dict[str, str],
        *,
        timeout_seconds: float,
        params: dict[str, str] | None = None,
        json_body: dict[str, object] | None = None,
    ) -> dict[str, object]:
        self.calls.append(
            {
                "method": method,
                "url": url,
                "headers": headers,
                "timeout_seconds": timeout_seconds,
                "params": params,
                "json_body": json_body,
            }
        )
        return self._responses.pop(0)


def make_response(provider: SearchProviderName) -> WebSearchResponse:
    return WebSearchResponse(
        provider=provider,
        results=[
            WebSearchResult(
                title="Official source",
                url="https://www.investor.gov/example",
                snippet="Example result.",
            )
        ],
    )


def make_request() -> WebSearchRequest:
    return WebSearchRequest(
        query="How should a moderate investor diversify?",
        search_mode=SearchMode.AUTHORITATIVE_DOMAIN,
        allowed_domains=("investor.gov",),
    )


def test_breaker_opens_after_two_retry_exhausted_logical_failures() -> None:
    clock = FakeClock()
    breaker = CircuitBreaker(failure_threshold=2, cooldown_seconds=30, clock=clock)

    assert breaker.before_call().allowed is True
    assert breaker.record_failure() is CircuitState.CLOSED
    assert breaker.before_call().allowed is True
    assert breaker.record_failure() is CircuitState.OPEN

    decision = breaker.before_call()
    assert decision.allowed is False
    assert decision.state is CircuitState.OPEN
    assert decision.reason is CircuitDecisionReason.CIRCUIT_OPEN
    assert decision.retry_after_seconds == 30


def test_breaker_allows_one_half_open_probe_after_cooldown() -> None:
    clock = FakeClock()
    breaker = CircuitBreaker(failure_threshold=1, cooldown_seconds=30, clock=clock)
    breaker.record_failure()
    clock.advance(30)

    probe = breaker.before_call()
    assert probe.allowed is True
    assert probe.state is CircuitState.HALF_OPEN
    assert probe.is_recovery_probe is True

    breaker.record_success()
    assert breaker.state is CircuitState.CLOSED
    assert breaker.consecutive_failures == 0


def test_failed_half_open_probe_reopens_breaker() -> None:
    clock = FakeClock()
    breaker = CircuitBreaker(failure_threshold=1, cooldown_seconds=30, clock=clock)
    breaker.record_failure()
    clock.advance(30)
    assert breaker.before_call().is_recovery_probe is True

    assert breaker.record_failure() is CircuitState.OPEN
    assert breaker.before_call().allowed is False


def test_service_retries_a_transient_exa_failure_then_returns_exa_results() -> None:
    exa = FakeProvider(
        SearchProviderName.EXA,
        [
            WebResearchError("Exa timed out.", retryable=True),
            WebResearchError("Exa timed out.", retryable=True),
            make_response(SearchProviderName.EXA),
        ],
    )
    brave = FakeProvider(SearchProviderName.BRAVE, [make_response(SearchProviderName.BRAVE)])
    delays: list[float] = []
    service = WebResearchService(
        [exa, brave],
        {
            SearchProviderName.EXA: CircuitBreaker(),
            SearchProviderName.BRAVE: CircuitBreaker(),
        },
        sleep=delays.append,
    )

    result = service.search(make_request())

    assert isinstance(result, WebSearchResponse)
    assert result.provider is SearchProviderName.EXA
    assert exa.call_count == 3
    assert brave.call_count == 0
    assert delays == [1.0, 2.0]


def test_service_fails_over_to_brave_after_retry_exhausted_exa_failure() -> None:
    exa = FakeProvider(
        SearchProviderName.EXA,
        [WebResearchError("Exa unavailable.", retryable=True)] * 3,
    )
    brave = FakeProvider(SearchProviderName.BRAVE, [make_response(SearchProviderName.BRAVE)])
    service = WebResearchService(
        [exa, brave],
        {
            SearchProviderName.EXA: CircuitBreaker(),
            SearchProviderName.BRAVE: CircuitBreaker(),
        },
        sleep=lambda _: None,
    )

    result = service.search(make_request())

    assert isinstance(result, WebSearchResponse)
    assert result.provider is SearchProviderName.BRAVE
    assert exa.call_count == 3
    assert brave.call_count == 1


def test_service_skips_an_open_exa_circuit_and_uses_brave() -> None:
    clock = FakeClock()
    exa_breaker = CircuitBreaker(failure_threshold=1, clock=clock)
    exa_breaker.record_failure()
    exa = FakeProvider(SearchProviderName.EXA, [make_response(SearchProviderName.EXA)])
    brave = FakeProvider(SearchProviderName.BRAVE, [make_response(SearchProviderName.BRAVE)])
    service = WebResearchService(
        [exa, brave],
        {SearchProviderName.EXA: exa_breaker, SearchProviderName.BRAVE: CircuitBreaker()},
    )

    result = service.search(make_request())

    assert isinstance(result, WebSearchResponse)
    assert result.provider is SearchProviderName.BRAVE
    assert exa.call_count == 0
    assert brave.call_count == 1


def test_service_returns_structured_unavailable_when_both_providers_fail() -> None:
    exa = FakeProvider(
        SearchProviderName.EXA,
        [WebResearchError("Invalid Exa credential.", retryable=False)],
    )
    brave = FakeProvider(
        SearchProviderName.BRAVE,
        [WebResearchError("Invalid Brave credential.", retryable=False)],
    )
    service = WebResearchService(
        [exa, brave],
        {
            SearchProviderName.EXA: CircuitBreaker(),
            SearchProviderName.BRAVE: CircuitBreaker(),
        },
    )

    result = service.search(make_request())

    assert isinstance(result, WebResearchUnavailable)
    assert [outcome.status for outcome in result.provider_outcomes] == [
        ProviderSearchStatus.FAILED,
        ProviderSearchStatus.FAILED,
    ]
    assert [outcome.attempt_count for outcome in result.provider_outcomes] == [1, 1]


def test_exa_adapter_uses_domain_filter_and_normalizes_results() -> None:
    transport = FakeJsonTransport(
        [
            {
                "results": [
                    {
                        "title": "Investor guidance",
                        "url": "https://www.investor.gov/introduction-investing",
                        "summary": "Official investing guidance.",
                        "publishedDate": "2026-09-18T12:00:00Z",
                    }
                ]
            }
        ]
    )
    provider = ExaWebResearchProvider(SecretStr("exa-test-key"), 5, transport)

    response = provider.search(make_request())

    assert response.provider is SearchProviderName.EXA
    assert response.results[0].published_at == date(2026, 9, 18)
    assert transport.calls[0]["json_body"] == {
        "query": "How should a moderate investor diversify?",
        "type": "fast",
        "numResults": 8,
        "includeDomains": ["investor.gov"],
    }


def test_brave_adapter_uses_site_query_and_filters_unapproved_urls() -> None:
    transport = FakeJsonTransport(
        [
            {
                "web": {
                    "results": [
                        {
                            "title": "Investor guidance",
                            "url": "https://www.investor.gov/introduction-investing",
                            "description": "Official investing guidance.",
                        },
                        {
                            "title": "Unapproved source",
                            "url": "https://example.com/advice",
                            "description": "Do not retain this result.",
                        },
                    ]
                }
            }
        ]
    )
    provider = BraveWebResearchProvider(SecretStr("brave-test-key"), 5, transport)

    response = provider.search(make_request())

    assert response.provider is SearchProviderName.BRAVE
    assert [str(result.url) for result in response.results] == [
        "https://www.investor.gov/introduction-investing"
    ]
    assert transport.calls[0]["params"] == {
        "q": "(site:investor.gov) How should a moderate investor diversify?",
        "count": "8",
        "safesearch": "moderate",
    }
