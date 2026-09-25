"""Exa and Brave URL-discovery adapters plus URL policy."""

import ipaddress
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from json import JSONDecodeError
from time import monotonic, sleep
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen

from financial_advisor.config import (
    BRAVE_BASE_URL,
    DEFAULT_WEB_RESULT_LIMIT,
    DEFAULT_WEB_TIMEOUT_SECONDS,
    EXA_BASE_URL,
    WEB_CIRCUIT_COOLDOWN_SECONDS,
    WEB_CIRCUIT_FAILURE_THRESHOLD,
    WEB_DISCOVERY_ATTEMPTS,
    WEB_DISCOVERY_RETRY_DELAY_SECONDS,
)


@dataclass(frozen=True)
class WebResult:
    title: str
    url: str


class WebError(RuntimeError):
    def __init__(self, message: str, *, retryable: bool) -> None:
        super().__init__(message)
        self.retryable = retryable


SearchProvider = Callable[[str, int], list[WebResult]]


class WebDiscovery:
    """Run bounded provider retries, failover, and circuit breaking."""

    def __init__(
        self,
        providers: Sequence[tuple[str, SearchProvider]],
        *,
        result_limit: int = DEFAULT_WEB_RESULT_LIMIT,
    ) -> None:
        if result_limit <= 0:
            raise ValueError("Web result limit must be positive.")
        if len({name for name, _provider in providers}) != len(providers):
            raise ValueError("Web provider names must be unique.")
        self.providers = list(providers)
        self.result_limit = result_limit
        self._failures = {name: 0 for name, _provider in providers}
        self._opened_at: dict[str, float] = {}

    def search(self, query: str) -> list[WebResult]:
        """Return results from the first available provider."""

        for name, provider in self.providers:
            opened_at = self._opened_at.get(name)
            if opened_at is not None and monotonic() - opened_at < WEB_CIRCUIT_COOLDOWN_SECONDS:
                continue

            for attempt_number in range(1, WEB_DISCOVERY_ATTEMPTS + 1):
                try:
                    results = provider(query, self.result_limit)
                except WebError as error:
                    if error.retryable and attempt_number < WEB_DISCOVERY_ATTEMPTS:
                        sleep(WEB_DISCOVERY_RETRY_DELAY_SECONDS * 2 ** (attempt_number - 1))
                        continue
                    break
                else:
                    self._failures[name] = 0
                    self._opened_at.pop(name, None)
                    return results

            self._failures[name] += 1
            if self._failures[name] >= WEB_CIRCUIT_FAILURE_THRESHOLD:
                self._opened_at[name] = monotonic()

        raise WebError("No web search provider is available.", retryable=True)


def exa_provider(
    api_key: str,
    *,
    base_url: str = EXA_BASE_URL,
    timeout_seconds: float = DEFAULT_WEB_TIMEOUT_SECONDS,
) -> tuple[str, SearchProvider]:
    """Create an Exa discovery provider."""

    def search_exa(query: str, limit: int) -> list[WebResult]:
        body: dict[str, object] = {"query": query, "type": "fast", "numResults": limit}
        payload = _request_json(
            "POST",
            f"{base_url.rstrip('/')}/search",
            {"Content-Type": "application/json", "x-api-key": api_key},
            timeout_seconds,
            body=body,
        )
        results = payload.get("results")
        if not isinstance(results, list):
            raise WebError("Exa returned an invalid result list.", retryable=False)
        return _normalize_results(results)

    return "exa", search_exa


def brave_provider(
    api_key: str,
    *,
    base_url: str = BRAVE_BASE_URL,
    timeout_seconds: float = DEFAULT_WEB_TIMEOUT_SECONDS,
) -> tuple[str, SearchProvider]:
    """Create a Brave discovery provider."""

    def search_brave(query: str, limit: int) -> list[WebResult]:
        payload = _request_json(
            "GET",
            f"{base_url.rstrip('/')}/res/v1/web/search",
            {"X-Subscription-Token": api_key},
            timeout_seconds,
            params={"q": query, "count": str(limit), "safesearch": "moderate"},
        )
        web = payload.get("web", {})
        results = web.get("results", []) if isinstance(web, Mapping) else []
        if not isinstance(results, list):
            raise WebError("Brave returned an invalid result list.", retryable=False)
        return _normalize_results(results)

    return "brave", search_brave


def validate_url(url: str) -> None:
    """Allow only public HTTP/S URLs without embedded credentials."""

    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or parsed.hostname is None:
        raise ValueError("URL must use HTTP or HTTPS.")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("URLs containing credentials are not allowed.")
    host = parsed.hostname.casefold().rstrip(".")
    try:
        address = ipaddress.ip_address(host)
    except ValueError as error:
        if host == "localhost":
            raise ValueError("Local URLs are not allowed.") from error
    else:
        if not address.is_global:
            raise ValueError("Private network URLs are not allowed.")


def _request_json(
    method: str,
    url: str,
    headers: Mapping[str, str],
    timeout: float,
    *,
    params: Mapping[str, str] | None = None,
    body: Mapping[str, object] | None = None,
) -> Mapping[str, object]:
    request_url = f"{url}?{urlencode(params)}" if params else url
    request = Request(
        request_url,
        data=json.dumps(body).encode() if body is not None else None,
        headers=dict(headers),
        method=method,
    )
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310
            decoded = response.read().decode()
    except HTTPError as error:
        raise WebError(
            f"Search provider returned HTTP {error.code}.",
            retryable=error.code == 429 or error.code >= 500,
        ) from error
    except (TimeoutError, URLError, OSError) as error:
        raise WebError("Search provider connection failed.", retryable=True) from error
    try:
        payload = json.loads(decoded)
    except JSONDecodeError as error:
        raise WebError("Search provider returned invalid JSON.", retryable=False) from error
    if not isinstance(payload, Mapping):
        raise WebError("Search provider returned an invalid response.", retryable=False)
    return payload


def _normalize_results(raw_results: Sequence[object]) -> list[WebResult]:
    results: list[WebResult] = []
    for raw in raw_results:
        if not isinstance(raw, Mapping):
            continue
        title, url = raw.get("title"), raw.get("url")
        if not isinstance(title, str) or not title.strip() or not isinstance(url, str):
            continue
        try:
            validate_url(url)
        except ValueError:
            continue
        results.append(WebResult(title.strip(), url))
    return results
