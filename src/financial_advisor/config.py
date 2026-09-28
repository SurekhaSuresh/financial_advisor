"""Application settings and static configuration."""

import os
from collections.abc import Callable, Sequence
from pathlib import Path

from google.genai import types
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

SESSION_DATABASE_PATH = Path("data/adk_sessions.db")
SESSION_DATABASE_URL = f"sqlite+aiosqlite:///{SESSION_DATABASE_PATH}"
Embed = Callable[[Sequence[str]], list[list[float]]]
Rerank = Callable[[str, Sequence[str]], list[float]]
MAX_EVIDENCE_TEXT_LENGTH = 2_000
LOCAL_HYBRID_RETRIEVAL_PATH = "local_hybrid"
WEB_RETRIEVAL_PATH = "web"


class ApplicationSettings(BaseSettings):
    """Environment-backed values needed to assemble the application."""

    google_api_key: SecretStr = Field(
        default_factory=lambda: SecretStr(os.environ["GOOGLE_API_KEY"])
    )
    exa_api_key: SecretStr | None = None
    brave_search_api_key: SecretStr | None = None
    advisor_analyst_model_name: str = "gemini-3.5-flash"
    client_model_name: str = "gemini-3.5-flash-lite"
    session_database_path: Path = SESSION_DATABASE_PATH
    knowledge_database_path: Path = Path("data/knowledge_base/lancedb")
    model_cache_directory: Path = Path(".local/model_cache")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_ignore_empty=True,
        extra="ignore",
    )

# Agent policy
MAX_CLIENT_FOLLOW_UPS = 2
MAX_RESEARCH_ATTEMPTS = 2
MAX_RECOMMENDATION_REDRAFTS = 1
MAX_CONVERSATION_LLM_CALLS = 30
MODEL_REQUEST_ATTEMPTS = 2
MODEL_REQUEST_TIMEOUT_MILLISECONDS = 120_000
MODEL_RETRY_DELAY_SECONDS = 1.0
CONVERSATION_TIMEOUT_SECONDS = 360.0
MODEL_REQUEST_TIMEOUT = types.GenerateContentConfig(
    http_options=types.HttpOptions(timeout=MODEL_REQUEST_TIMEOUT_MILLISECONDS)
)
MODEL_RETRY_OPTIONS = types.HttpRetryOptions(
    attempts=MODEL_REQUEST_ATTEMPTS,
    initial_delay=MODEL_RETRY_DELAY_SECONDS,
)

# Agent and session names
CLIENT_AGENT = "client_agent"
ANALYST_AGENT = "analyst_agent"
ANALYST_DRAFT_STATE_KEY = "temp:analyst_draft"
FINALIZE_RECOMMENDATION = "finalize_recommendation"
CONVERSATION_STATUS_STATE_KEY = "conversation_status"
CLIENT_FOLLOW_UP_COUNT_STATE_KEY = "client_follow_up_count"
CURRENT_CLIENT_QUESTION_STATE_KEY = "current_client_question"
RESEARCH_ATTEMPT_COUNT_STATE_KEY = "research_attempt_count"
RECOMMENDATION_REDRAFT_COUNT_STATE_KEY = "recommendation_redraft_count"
RETRIEVED_EVIDENCE_STATE_KEY = "temp:retrieved_evidence"
TRUSTED_RESEARCH_BRIEF_STATE_KEY = "trusted_research_brief"
FINAL_RECOMMENDATION_STATE_KEY = "final_recommendation"
RECOMMENDATION_REDRAFT_MESSAGE = (
    "Redraft the recommendation using only evidence IDs from the research brief."
)
ESCALATION_MESSAGE = (
    "I cannot provide a safe, evidence-grounded recommendation for this request. "
    "Please consult a qualified financial professional."
)
# Local model defaults
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL_NAME = "Xenova/ms-marco-MiniLM-L-6-v2"

# Final evidence selection
DEFAULT_EVIDENCE_LIMIT = 5
DEFAULT_RRF_RANK_CONSTANT = 60
DEFAULT_MMR_RELEVANCE_WEIGHT = 0.7

# Curated knowledge ingestion and retrieval
KNOWLEDGE_TABLE_NAME = "knowledge_chunks"
KNOWLEDGE_TEXT_INDEX_NAME = "knowledge_text_fts"
DEFAULT_KNOWLEDGE_CANDIDATE_LIMIT = 20
KNOWLEDGE_FETCH_TIMEOUT_SECONDS = 30
KNOWLEDGE_FETCH_ATTEMPTS = 3
KNOWLEDGE_FETCH_RETRY_DELAY_SECONDS = 1.0
KNOWLEDGE_FETCH_USER_AGENT = "FinancialAdvisorKnowledge/1.0"

# Web discovery, fetching, and passage selection
WEB_RESULT_LIMIT = 8
DEFAULT_WEB_USABLE_PAGE_TARGET = 4
DEFAULT_WEB_SELECTED_PASSAGE_LIMIT = 10
DEFAULT_WEB_TIMEOUT_SECONDS = 10.0
WEB_DISCOVERY_ATTEMPTS = 3
WEB_CIRCUIT_COOLDOWN_SECONDS = 30.0
WEB_DISCOVERY_RETRY_DELAY_SECONDS = 1.0
WEB_FETCH_ATTEMPTS = 2
WEB_FETCH_RETRY_DELAY_SECONDS = 1.0
WEB_FETCH_USER_AGENT = "FinancialAdvisorResearch/1.0"
EXA_SEARCH_URL = "https://api.exa.ai/search"
BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"
EXA_PROVIDER = "exa"
BRAVE_PROVIDER = "brave"

# Shared document chunking
DOCUMENT_CHUNK_MAX_TOKENS = 350
DOCUMENT_CHUNK_OVERLAP_TOKENS = 40
