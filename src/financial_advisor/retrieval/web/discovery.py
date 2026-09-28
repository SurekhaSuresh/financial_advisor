"""Discover web source URLs with Exa and Brave."""

# ruff: noqa: S310 -- Requests use fixed provider URLs from application configuration.

import json
from http import HTTPStatus
from time import sleep, time
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from financial_advisor.config import (
    BRAVE_PROVIDER,
    BRAVE_SEARCH_URL,
    DEFAULT_WEB_TIMEOUT_SECONDS,
    EXA_PROVIDER,
    EXA_SEARCH_URL,
    WEB_CIRCUIT_COOLDOWN_SECONDS,
    WEB_DISCOVERY_ATTEMPTS,
    WEB_DISCOVERY_RETRY_DELAY_SECONDS,
    WEB_RESULT_LIMIT,
)


class WebSourceDiscovery:
    """Search Exa then Brave with bounded retries and circuit breaking."""

    def __init__(self, exa_api_key: str | None, brave_api_key: str | None) -> None:
        self.api_keys = {EXA_PROVIDER: exa_api_key, BRAVE_PROVIDER: brave_api_key}
        self._provider_cooldowns: dict[str, float] = {}

    def discover_source_urls(self, query: str) -> list[tuple[str, str]]:
        """Return results from the first available provider."""

        for provider_name, api_key in self.api_keys.items():
            if api_key is None or time() < self._provider_cooldowns.get(provider_name, 0):
                continue
            self._provider_cooldowns.pop(provider_name, None)

            for attempt_index in range(WEB_DISCOVERY_ATTEMPTS):
                try:
                    return self._request_source_urls(provider_name, api_key, query)
                except ConnectionError:
                    if attempt_index < WEB_DISCOVERY_ATTEMPTS - 1:  # Final attempt, so don't sleep.
                        sleep(WEB_DISCOVERY_RETRY_DELAY_SECONDS * 2**attempt_index)
                except ValueError:
                    break

            self._provider_cooldowns[provider_name] = time() + WEB_CIRCUIT_COOLDOWN_SECONDS

        return []

    def _request_source_urls(
        self,
        provider_name: str,
        api_key: str,
        query: str,
    ) -> list[tuple[str, str]]:
        if provider_name == EXA_PROVIDER:
            request = Request(
                url=EXA_SEARCH_URL,
                data=json.dumps({"query": query, "numResults": WEB_RESULT_LIMIT}).encode(),
                headers={"Content-Type": "application/json", "x-api-key": api_key},
            )
        else:
            request = Request(
                f"{BRAVE_SEARCH_URL}?{urlencode({'q': query, 'count': WEB_RESULT_LIMIT})}",
                headers={"X-Subscription-Token": api_key},
            )

        try:
            with urlopen(request, timeout=DEFAULT_WEB_TIMEOUT_SECONDS) as response:
                provider_response = json.load(response)
        except OSError as error:
            if (
                isinstance(error, HTTPError)
                and error.code != HTTPStatus.TOO_MANY_REQUESTS
                and error.code < HTTPStatus.INTERNAL_SERVER_ERROR
            ):
                raise ValueError(f"Search provider returned HTTP {error.code}.") from error
            raise ConnectionError("Search provider request failed.") from error

        if provider_name == BRAVE_PROVIDER and isinstance(provider_response, dict):
            provider_response = provider_response.get("web")
        provider_results = (
            provider_response.get("results") if isinstance(provider_response, dict) else None
        )
        if not isinstance(provider_results, list):
            raise ValueError("Search provider returned invalid results.")

        results = []
        for result in provider_results:
            if not isinstance(result, dict):
                continue
            title = result.get("title")
            url = result.get("url")
            if isinstance(title, str) and title.strip() and isinstance(url, str):
                results.append((title.strip(), url))
        return results
