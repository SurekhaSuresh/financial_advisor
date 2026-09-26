"""Application settings and static configuration."""

from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class ApplicationSettings(BaseSettings):
    """Environment-backed values needed to assemble the application."""

    google_api_key: SecretStr | None = None
    exa_api_key: SecretStr | None = None
    brave_search_api_key: SecretStr | None = None
    advisor_analyst_model_name: str = "gemini-3.5-flash"
    client_model_name: str = "gemini-3.5-flash-lite"
    session_database_url: str = "sqlite+aiosqlite:///data/adk_sessions.db"
    knowledge_database_path: Path | None = None
    model_cache_directory: Path = Path(".local/model_cache")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_ignore_empty=True,
        extra="ignore",
    )

# Agent policy
MAX_CLIENT_FOLLOW_UPS = 2
MAX_RESEARCH_ATTEMPTS = 2
MAX_ADVISOR_LLM_CALLS = 30
MODEL_REQUEST_ATTEMPTS = 2
MODEL_REQUEST_TIMEOUT_MILLISECONDS = 120_000
MODEL_RETRY_DELAY_SECONDS = 1.0
CONVERSATION_TIMEOUT_SECONDS = 360.0

# Local model defaults
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL_NAME = "Xenova/ms-marco-MiniLM-L-6-v2"
TOKENIZER_MODEL_MAX_LENGTH = 1_000_000

# Final evidence selection
MAX_EVIDENCE_TEXT_LENGTH = 2_000
DEFAULT_EVIDENCE_LIMIT = 5
DEFAULT_RRF_RANK_CONSTANT = 60
DEFAULT_MMR_RELEVANCE_WEIGHT = 0.7

# Curated knowledge ingestion and retrieval
KNOWLEDGE_TABLE_NAME = "knowledge_chunks"
KNOWLEDGE_TEXT_INDEX_NAME = "knowledge_text_fts"
DEFAULT_KNOWLEDGE_CANDIDATE_LIMIT = 20
KNOWLEDGE_CHUNK_MAX_TOKENS = 500
KNOWLEDGE_CHUNK_OVERLAP_TOKENS = 60
KNOWLEDGE_CHUNK_MIN_TOKENS = 150
KNOWLEDGE_FETCH_TIMEOUT_SECONDS = 30
KNOWLEDGE_FETCH_ATTEMPTS = 3
KNOWLEDGE_FETCH_RETRY_DELAY_SECONDS = 1.0
KNOWLEDGE_FETCH_USER_AGENT = "FinancialAdvisorKnowledge/1.0"

# Web discovery, fetching, and passage selection
DEFAULT_WEB_RESULT_LIMIT = 8
DEFAULT_WEB_USABLE_PAGE_TARGET = 4
DEFAULT_WEB_SELECTED_PASSAGE_LIMIT = 10
DEFAULT_WEB_TIMEOUT_SECONDS = 10.0
DEFAULT_WEB_MAX_PAGE_BYTES = 2_000_000
WEB_DISCOVERY_ATTEMPTS = 3
WEB_CIRCUIT_FAILURE_THRESHOLD = 2
WEB_CIRCUIT_COOLDOWN_SECONDS = 30.0
WEB_DISCOVERY_RETRY_DELAY_SECONDS = 1.0
WEB_FETCH_ATTEMPTS = 2
WEB_FETCH_RETRY_DELAY_SECONDS = 1.0
WEB_FETCH_USER_AGENT = "FinancialAdvisorResearch/1.0"
WEB_CHUNK_MAX_TOKENS = 350
WEB_CHUNK_OVERLAP_TOKENS = 40
WEB_CHUNK_MIN_TOKENS = 120
EXA_BASE_URL = "https://api.exa.ai"
BRAVE_BASE_URL = "https://api.search.brave.com"
