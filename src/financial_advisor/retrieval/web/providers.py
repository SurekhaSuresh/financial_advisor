"""Exa and Brave adapters behind the provider-neutral web-research contract."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import date, datetime
from json import JSONDecodeError
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from pydantic import HttpUrl, SecretStr, TypeAdapter, ValidationError

from financial_advisor.config import get_settings
from financial_advisor.retrieval.web.discovery import (
    SearchProviderName,
    WebResearchError,
    WebSearchRequest,
    WebSearchResponse,
    WebSearchResult,
)
from financial_advisor.retrieval.web.url_policy import url_matches_allowed_domain

_EXA_SEARCH_PATH = "/search"
_BRAVE_SEARCH_PATH = "/res/v1/web/search"
_HTTP_URL_ADAPTER = TypeAdapter(HttpUrl)


class JsonHttpTransport(Protocol):
    """Minimal JSON-over-HTTP behavior required by provider adapters."""

    def request_json(
        self,
        method: str,
        url: str,
        headers: Mapping[str, str],
        *,
        timeout_seconds: float,
        params: Mapping[str, str] | None = None,
        json_body: Mapping[str, object] | None = None,
    ) -> Mapping[str, object]:
        """Make one JSON request or raise a classified WebResearchError."""


class UrllibJsonTransport:
    """Small standard-library HTTP transport for the synchronous application path."""

    def request_json(
        self,
        method: str,
        url: str,
        headers: Mapping[str, str],
        *,
        timeout_seconds: float,
        params: Mapping[str, str] | None = None,
        json_body: Mapping[str, object] | None = None,
    ) -> Mapping[str, object]:
        """Make one bounded request and normalize transport failures."""

        request_url = url
        if params:
            request_url = f"{url}?{urlencode(params)}"
        body = json.dumps(json_body).encode("utf-8") if json_body is not None else None
        request = Request(request_url, data=body, headers=dict(headers), method=method)
        try:
            with urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310
                decoded = response.read().decode("utf-8")
        except HTTPError as error:
            retryable = error.code == 429 or error.code >= 500
            raise WebResearchError(
                f"Search provider returned HTTP {error.code}.", retryable=retryable
            ) from error
        except (TimeoutError, URLError, OSError) as error:
            raise WebResearchError("Search provider connection failed.", retryable=True) from error

        try:
            payload = json.loads(decoded)
        except JSONDecodeError as error:
            raise WebResearchError(
                "Search provider returned invalid JSON.", retryable=False
            ) from error
        if not isinstance(payload, dict):
            raise WebResearchError(
                "Search provider returned an unexpected response shape.", retryable=False
            )
        return payload


class ExaWebResearchProvider:
    """Maps Exa's search API into Financial Advisor discovery results."""

    def __init__(
        self,
        api_key: SecretStr,
        timeout_seconds: float,
        transport: JsonHttpTransport | None = None,
        base_url: HttpUrl | None = None,
    ) -> None:
        self._api_key = api_key
        self._timeout_seconds = timeout_seconds
        self._transport = transport or UrllibJsonTransport()
        self._base_url = base_url or get_settings().provider_endpoints.exa_base_url

    @property
    def name(self) -> SearchProviderName:
        """Return the stable provider identity used by the service."""

        return SearchProviderName.EXA

    def search(self, request: WebSearchRequest) -> WebSearchResponse:
        """Discover URLs through Exa without treating snippets as final evidence."""

        payload: dict[str, object] = {
            "query": request.query,
            "type": "fast",
            "numResults": request.max_results,
        }
        if request.allowed_domains:
            payload["includeDomains"] = list(request.allowed_domains)
        response = self._transport.request_json(
            "POST",
            _provider_url(self._base_url, _EXA_SEARCH_PATH),
            {
                "Content-Type": "application/json",
                "x-api-key": self._api_key.get_secret_value(),
            },
            timeout_seconds=self._timeout_seconds,
            json_body=payload,
        )
        raw_results = response.get("results")
        if not isinstance(raw_results, list):
            raise WebResearchError("Exa response did not include a results list.", retryable=False)
        return WebSearchResponse(
            provider=self.name,
            results=_normalize_results(raw_results, request.allowed_domains),
        )


class BraveWebResearchProvider:
    """Maps Brave's web-search API into Financial Advisor discovery results."""

    def __init__(
        self,
        api_key: SecretStr,
        timeout_seconds: float,
        transport: JsonHttpTransport | None = None,
        base_url: HttpUrl | None = None,
    ) -> None:
        self._api_key = api_key
        self._timeout_seconds = timeout_seconds
        self._transport = transport or UrllibJsonTransport()
        self._base_url = base_url or get_settings().provider_endpoints.brave_base_url

    @property
    def name(self) -> SearchProviderName:
        """Return the stable provider identity used by the service."""

        return SearchProviderName.BRAVE

    def search(self, request: WebSearchRequest) -> WebSearchResponse:
        """Discover URLs through Brave and enforce the allowed-domain policy locally."""

        response = self._transport.request_json(
            "GET",
            _provider_url(self._base_url, _BRAVE_SEARCH_PATH),
            {"X-Subscription-Token": self._api_key.get_secret_value()},
            timeout_seconds=self._timeout_seconds,
            params={
                "q": _brave_query(request),
                "count": str(request.max_results),
                "safesearch": "moderate",
            },
        )
        web = response.get("web")
        if web is None:
            raw_results: list[object] = []
        elif isinstance(web, Mapping):
            results = web.get("results", [])
            if not isinstance(results, list):
                raise WebResearchError(
                    "Brave response did not include a valid results list.", retryable=False
                )
            raw_results = results
        else:
            raise WebResearchError(
                "Brave response did not include a valid web result object.", retryable=False
            )
        return WebSearchResponse(
            provider=self.name,
            results=_normalize_results(raw_results, request.allowed_domains),
        )


def _brave_query(request: WebSearchRequest) -> str:
    """Add documented site operators when authoritative-domain mode is requested."""

    if not request.allowed_domains:
        return request.query
    domains = " OR ".join(f"site:{domain}" for domain in request.allowed_domains)
    return f"({domains}) {request.query}"


def _provider_url(base_url: HttpUrl, path: str) -> str:
    """Join a configured provider base URL with an adapter-owned API path."""

    return f"{str(base_url).rstrip('/')}{path}"


def _normalize_results(
    raw_results: list[object], allowed_domains: tuple[str, ...]
) -> list[WebSearchResult]:
    """Validate provider fields and keep only URLs allowed by the research policy."""

    normalized: list[WebSearchResult] = []
    for raw_result in raw_results:
        if not isinstance(raw_result, Mapping):
            continue
        url = raw_result.get("url")
        title = raw_result.get("title")
        if not isinstance(url, str) or not isinstance(title, str) or not title.strip():
            continue
        if allowed_domains and not url_matches_allowed_domain(url, allowed_domains):
            continue
        try:
            normalized.append(
                WebSearchResult(
                    title=title.strip(),
                    url=_HTTP_URL_ADAPTER.validate_python(url),
                    snippet=_result_snippet(raw_result),
                    published_at=_published_date(raw_result),
                )
            )
        except ValidationError:
            continue
    return normalized


def _result_snippet(raw_result: Mapping[object, object]) -> str:
    """Choose a short provider excerpt without treating it as fetched-page evidence."""

    for field_name in ("description", "summary", "text"):
        value = raw_result.get(field_name)
        if isinstance(value, str) and value.strip():
            return value.strip()[:2_000]
    highlights = raw_result.get("highlights")
    if isinstance(highlights, list):
        for highlight in highlights:
            if isinstance(highlight, str) and highlight.strip():
                return highlight.strip()[:2_000]
    return ""


def _published_date(raw_result: Mapping[object, object]) -> date | None:
    """Best-effort parse of optional provider publication metadata."""

    raw_date = raw_result.get("publishedDate") or raw_result.get("page_age")
    if not isinstance(raw_date, str):
        return None
    try:
        return datetime.fromisoformat(raw_date.replace("Z", "+00:00")).date()
    except ValueError:
        return None
