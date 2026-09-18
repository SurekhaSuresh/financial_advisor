"""Tests for local runtime configuration boundaries."""

from financial_advisor.config import Settings
from financial_advisor.runtime import _configure_google_adk_credentials


def test_runtime_exposes_validated_gemini_key_to_adk_environment(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """ADK receives the server-side key after Pydantic has read the local .env file."""

    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    settings = Settings(google_api_key="test-google-key")

    _configure_google_adk_credentials(settings)

    assert __import__("os").environ["GOOGLE_API_KEY"] == "test-google-key"
