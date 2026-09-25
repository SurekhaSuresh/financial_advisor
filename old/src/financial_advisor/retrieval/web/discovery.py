"""Typed boundaries and resilience primitives for live-web research.

Provider-specific HTTP clients intentionally live behind ``WebResearchProvider``.
The circuit breaker is provider-local and is called only after a complete logical
provider operation (including that operation's bounded retries) has succeeded or
failed.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import UTC, date, datetime
from enum import StrEnum
from time import monotonic
from time import sleep as time_sleep
from typing import Protocol

from pydantic import BaseModel, Field, HttpUrl, model_validator

from financial_advisor.config import get_settings
from financial_advisor.domain import SearchMode

_POLICY = get_settings().web_research
MAX_WEB_DISCOVERY_RESULT_LIMIT = _POLICY.discovery_result_limit


class SearchProviderName(StrEnum):
    """The independent web-search providers available to the application."""

    EXA = "exa"
    BRAVE = "brave"


class CircuitState(StrEnum):
    """Health state of one external provider dependency."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitDecisionReason(StrEnum):
    """The reason a provider operation was permitted or skipped."""

    CLOSED = "closed"
    CIRCUIT_OPEN = "circuit_open"
    RECOVERY_PROBE = "recovery_probe"


class WebSearchRequest(BaseModel):
    """A bounded discovery request issued to one web-search provider."""

    query: str = Field(min_length=1, max_length=2_000)
    search_mode: SearchMode
    max_results: int = Field(
        default=MAX_WEB_DISCOVERY_RESULT_LIMIT,
        ge=1,
        le=MAX_WEB_DISCOVERY_RESULT_LIMIT,
    )
    allowed_domains: tuple[str, ...] = ()

    @model_validator(mode="after")
    def require_domains_for_authoritative_search(self) -> WebSearchRequest:
        """Prevent an authoritative-domain task from silently searching the broad web."""

        if self.search_mode is SearchMode.AUTHORITATIVE_DOMAIN and not self.allowed_domains:
            raise ValueError("Authoritative-domain search requires at least one allowed domain.")
        return self


class WebSearchResult(BaseModel):
    """Normalized URL-discovery result; it is not yet evidence."""

    title: str = Field(min_length=1, max_length=500)
    url: HttpUrl
    snippet: str = Field(default="", max_length=2_000)
    published_at: date | None = None


class WebSearchResponse(BaseModel):
    """Validated discovery results returned by a named provider."""

    provider: SearchProviderName
    results: list[WebSearchResult] = Field(max_length=8)
    searched_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class WebResearchProvider(Protocol):
    """Provider behavior used by the service, independent of HTTP vendor SDKs."""

    @property
    def name(self) -> SearchProviderName:
        """Return the provider identity used in traces and breaker state."""

    def search(self, request: WebSearchRequest) -> WebSearchResponse:
        """Discover URLs for one bounded research request."""


class WebResearchError(Exception):
    """A classified provider or page-research failure safe for trace handling."""

    def __init__(self, message: str, *, retryable: bool) -> None:
        super().__init__(message)
        self.retryable = retryable


class ProviderSearchStatus(StrEnum):
    """The final outcome for one provider during one research request."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"


class ProviderSearchOutcome(BaseModel):
    """Safe, trace-ready summary of one provider's work for a request."""

    provider: SearchProviderName
    status: ProviderSearchStatus
    attempt_count: int = Field(ge=0, le=5)
    reason: CircuitDecisionReason | None = None
    error_message: str | None = Field(default=None, max_length=500)
    retry_delays_seconds: list[float] = Field(default_factory=list)


class WebResearchUnavailable(BaseModel):
    """Structured result when no configured provider can perform web discovery."""

    message: str = "Live web research is unavailable right now."
    provider_outcomes: list[ProviderSearchOutcome] = Field(min_length=1)


class CircuitDecision(BaseModel):
    """Whether a provider call may proceed at the current monotonic time."""

    allowed: bool
    state: CircuitState
    reason: CircuitDecisionReason
    is_recovery_probe: bool = False
    retry_after_seconds: float | None = Field(default=None, ge=0)


class CircuitBreaker:
    """A small in-process circuit breaker for one external dependency.

    ``before_call`` is invoked once before a logical provider operation.
    ``record_success`` or ``record_failure`` must be invoked exactly once after
    an allowed operation, after its internal retries are complete.
    """

    def __init__(
        self,
        *,
        failure_threshold: int = _POLICY.circuit_failure_threshold,
        cooldown_seconds: float = _POLICY.circuit_cooldown_seconds,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be at least 1.")
        if cooldown_seconds <= 0:
            raise ValueError("cooldown_seconds must be greater than zero.")
        self._failure_threshold = failure_threshold
        self._cooldown_seconds = cooldown_seconds
        self._clock = clock or monotonic
        self._state = CircuitState.CLOSED
        self._consecutive_failures = 0
        self._opened_at: float | None = None

    @property
    def state(self) -> CircuitState:
        """Return the current stored state without advancing the cooldown."""

        return self._state

    @property
    def consecutive_failures(self) -> int:
        """Return completed logical failures since the last success."""

        return self._consecutive_failures

    def before_call(self) -> CircuitDecision:
        """Allow a normal call, skip an open circuit, or issue one recovery probe."""

        if self._state is CircuitState.CLOSED:
            return CircuitDecision(
                allowed=True,
                state=CircuitState.CLOSED,
                reason=CircuitDecisionReason.CLOSED,
            )

        if self._state is CircuitState.OPEN:
            assert self._opened_at is not None
            elapsed = self._clock() - self._opened_at
            if elapsed < self._cooldown_seconds:
                return CircuitDecision(
                    allowed=False,
                    state=CircuitState.OPEN,
                    reason=CircuitDecisionReason.CIRCUIT_OPEN,
                    retry_after_seconds=self._cooldown_seconds - elapsed,
                )
            self._state = CircuitState.HALF_OPEN

        return CircuitDecision(
            allowed=True,
            state=CircuitState.HALF_OPEN,
            reason=CircuitDecisionReason.RECOVERY_PROBE,
            is_recovery_probe=True,
        )

    def record_success(self) -> None:
        """Close the circuit and clear failures after one successful operation."""

        self._state = CircuitState.CLOSED
        self._consecutive_failures = 0
        self._opened_at = None

    def record_failure(self) -> CircuitState:
        """Record one retry-exhausted logical failure and return the new state."""

        self._consecutive_failures += 1
        if self._state is CircuitState.HALF_OPEN or (
            self._consecutive_failures >= self._failure_threshold
        ):
            self._state = CircuitState.OPEN
            self._opened_at = self._clock()
        return self._state


class WebResearchService:
    """Coordinates sequential provider retries, breaker decisions, and failover."""

    def __init__(
        self,
        providers: Sequence[WebResearchProvider],
        breakers: dict[SearchProviderName, CircuitBreaker],
        max_attempts: int = _POLICY.provider_max_attempts,
        retry_backoff_base_seconds: float = _POLICY.provider_retry_backoff_base_seconds,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        provider_names = [provider.name for provider in providers]
        if len(provider_names) != len(set(provider_names)):
            raise ValueError("Web research providers must have unique names.")
        if set(provider_names) != set(breakers):
            raise ValueError("Every configured provider must have one circuit breaker.")
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1.")
        if retry_backoff_base_seconds <= 0:
            raise ValueError("retry_backoff_base_seconds must be greater than zero.")
        self._providers = tuple(providers)
        self._breakers = breakers
        self._max_attempts = max_attempts
        self._retry_backoff_base_seconds = retry_backoff_base_seconds
        self._sleep = sleep or time_sleep

    def search(self, request: WebSearchRequest) -> WebSearchResponse | WebResearchUnavailable:
        """Search providers in configured order and return the first valid response."""

        outcomes: list[ProviderSearchOutcome] = []
        for provider in self._providers:
            breaker = self._breakers[provider.name]
            decision = breaker.before_call()
            if not decision.allowed:
                outcomes.append(
                    ProviderSearchOutcome(
                        provider=provider.name,
                        status=ProviderSearchStatus.SKIPPED,
                        attempt_count=0,
                        reason=decision.reason,
                    )
                )
                continue

            response, outcome = self._search_provider_with_retries(provider, request)
            if response is not None:
                breaker.record_success()
                outcomes.append(outcome)
                return response

            breaker.record_failure()
            outcomes.append(outcome)

        return WebResearchUnavailable(provider_outcomes=outcomes)

    def _search_provider_with_retries(
        self, provider: WebResearchProvider, request: WebSearchRequest
    ) -> tuple[WebSearchResponse | None, ProviderSearchOutcome]:
        """Run one provider operation with retries only for classified transient errors."""

        retry_delays_seconds: list[float] = []
        for attempt_number in range(1, self._max_attempts + 1):
            try:
                response = provider.search(request)
            except WebResearchError as error:
                if not error.retryable or attempt_number == self._max_attempts:
                    return None, ProviderSearchOutcome(
                        provider=provider.name,
                        status=ProviderSearchStatus.FAILED,
                        attempt_count=attempt_number,
                        error_message=str(error),
                        retry_delays_seconds=retry_delays_seconds,
                    )
                delay = self._retry_backoff_delay(attempt_number)
                retry_delays_seconds.append(delay)
                self._sleep(delay)
            except Exception:
                return None, ProviderSearchOutcome(
                    provider=provider.name,
                    status=ProviderSearchStatus.FAILED,
                    attempt_count=attempt_number,
                    error_message="Provider returned an unexpected application error.",
                    retry_delays_seconds=retry_delays_seconds,
                )
            else:
                return response, ProviderSearchOutcome(
                    provider=provider.name,
                    status=ProviderSearchStatus.SUCCEEDED,
                    attempt_count=attempt_number,
                    retry_delays_seconds=retry_delays_seconds,
                )

        raise AssertionError("Retry loop must return after the final provider attempt.")

    def _retry_backoff_delay(self, failed_attempt_number: int) -> float:
        """Return deterministic exponential delay after a retryable failed attempt."""

        multiplier = float(2 ** (failed_attempt_number - 1))
        return self._retry_backoff_base_seconds * multiplier
