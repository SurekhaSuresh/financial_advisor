"""Environment settings and validated operating policies for Financial Advisor."""

from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from financial_advisor.domain import ApprovedDomainId


def _default_approved_domain_registry() -> dict[ApprovedDomainId, tuple[str, ...]]:
    """Return the reviewed domain policy used to resolve Advisor selections."""

    return {
        ApprovedDomainId.INVESTOR_GOV: ("investor.gov",),
        ApprovedDomainId.SEC_GOV: ("sec.gov",),
        ApprovedDomainId.FINRA_ORG: ("finra.org",),
        ApprovedDomainId.CONSUMERFINANCE_GOV: ("consumerfinance.gov",),
        ApprovedDomainId.FDIC_GOV: ("fdic.gov",),
        ApprovedDomainId.IRS_GOV: ("irs.gov",),
        ApprovedDomainId.TREASURY_GOV: ("treasury.gov",),
    }


class KnowledgeIngestionPolicy(BaseModel):
    """Bounded chunking policy for the stable curated knowledge base."""

    model_config = ConfigDict(frozen=True)

    max_chunk_tokens: int = Field(default=500, ge=100, le=1_000)
    min_chunk_tokens: int = Field(default=150, ge=1, le=1_000)
    chunk_overlap_tokens: int = Field(default=60, ge=0, le=250)
    source_fetch_max_attempts: int = Field(default=3, ge=1, le=5)
    source_fetch_timeout_seconds: float = Field(default=30.0, gt=0, le=120)
    source_fetch_retry_backoff_base_seconds: float = Field(default=2.0, gt=0, le=30)
    source_fetch_user_agent: str = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 Chrome/131 Safari/537.36"
    )
    source_fetch_accept_language: str = "en-US,en;q=0.9"

    @model_validator(mode="after")
    def validate_chunk_window_policy(self) -> "KnowledgeIngestionPolicy":
        """Keep the configured minimum, overlap, and nominal maximum coherent."""

        if self.min_chunk_tokens > self.max_chunk_tokens:
            raise ValueError("min_chunk_tokens must not exceed max_chunk_tokens.")
        if self.chunk_overlap_tokens >= self.max_chunk_tokens:
            raise ValueError("chunk_overlap_tokens must be smaller than max_chunk_tokens.")
        return self


class RetrievalPolicy(BaseModel):
    """Candidate-selection limits shared by all retrieval channels."""

    model_config = ConfigDict(frozen=True)

    local_vector_candidate_limit: int = Field(default=20, ge=1, le=100)
    local_keyword_candidate_limit: int = Field(default=20, ge=1, le=100)
    web_passage_candidate_limit: int = Field(default=10, ge=1, le=50)
    rrf_rank_constant: int = Field(default=60, ge=1, le=200)
    final_evidence_limit: int = Field(default=5, ge=1, le=15)
    mmr_relevance_weight: float = Field(default=0.7, ge=0, le=1)


class WebResearchPolicy(BaseModel):
    """Bounded live-web discovery, fetching, and resilience behavior."""

    model_config = ConfigDict(frozen=True)

    discovery_result_limit: int = Field(default=8, ge=1, le=20)
    usable_page_target: int = Field(default=4, ge=1, le=10)
    approved_domain_registry: dict[ApprovedDomainId, tuple[str, ...]] = Field(
        default_factory=_default_approved_domain_registry
    )
    provider_max_attempts: int = Field(default=3, ge=1, le=5)
    provider_timeout_seconds: float = Field(default=10.0, gt=0, le=60)
    provider_retry_backoff_base_seconds: float = Field(default=1.0, gt=0, le=10)
    page_timeout_seconds: float = Field(default=10.0, gt=0, le=60)
    page_max_bytes: int = Field(default=2_000_000, ge=10_000, le=10_000_000)
    page_max_attempts: int = Field(default=2, ge=1, le=2)
    page_retry_delay_seconds: float = Field(default=1.0, gt=0, le=10)
    page_fetch_user_agent: str = "FinancialAdvisorResearch/0.1"
    web_chunk_max_tokens: int = Field(default=350, ge=100, le=1_000)
    web_chunk_min_tokens: int = Field(default=120, ge=1, le=1_000)
    web_chunk_overlap_tokens: int = Field(default=40, ge=0, le=250)
    circuit_failure_threshold: int = Field(default=2, ge=1, le=10)
    circuit_cooldown_seconds: float = Field(default=30.0, gt=0, le=300)

    @model_validator(mode="after")
    def validate_web_chunk_window_policy(self) -> "WebResearchPolicy":
        """Keep live-web chunking bounds coherent before the runtime starts."""

        if self.web_chunk_min_tokens > self.web_chunk_max_tokens:
            raise ValueError("web_chunk_min_tokens must not exceed web_chunk_max_tokens.")
        if self.web_chunk_overlap_tokens >= self.web_chunk_max_tokens:
            raise ValueError("web_chunk_overlap_tokens must be smaller than web_chunk_max_tokens.")
        return self

class WorkflowPolicy(BaseModel):
    """Business limits enforced independently of agent output."""

    model_config = ConfigDict(frozen=True)

    max_client_follow_ups: int = Field(default=2, ge=0, le=10)
    max_research_plans_per_client_question: int = Field(default=2, ge=1, le=2)


class ModelRuntimeSettings(BaseModel):
    """Local model identities and artifact location used by retrieval runtime."""

    model_config = ConfigDict(frozen=True)

    embedding_model_name: str = "BAAI/bge-small-en-v1.5"
    cross_encoder_model_name: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    model_cache_directory: Path = Path(".local/model_cache")


class GeminiSettings(BaseModel):
    """Pinned Gemini model IDs and bounded generation settings."""

    model_config = ConfigDict(frozen=True)

    client_model: str = Field(default="gemini-3.5-flash-lite", min_length=1)
    advisor_model: str = Field(default="gemini-3.5-flash", min_length=1)
    analyst_model: str = Field(default="gemini-3.5-flash", min_length=1)
    fallback_model: str = Field(default="gemini-3.5-flash-lite", min_length=1)
    temperature: float = Field(default=0.1, ge=0, le=1)
    client_temperature: float = Field(default=0.55, ge=0, le=1)
    client_max_output_tokens: int = Field(default=750, ge=128, le=4_096)
    advisor_max_output_tokens: int = Field(default=3_000, ge=256, le=8_192)
    analyst_max_output_tokens: int = Field(default=4_096, ge=256, le=8_192)
    client_primary_attempts: int = Field(default=2, ge=1, le=3)
    advisor_primary_attempts: int = Field(default=2, ge=1, le=3)
    analyst_primary_attempts: int = Field(default=2, ge=1, le=3)
    client_turn_timeout_seconds: float = Field(default=150.0, gt=0, le=300)
    advisor_turn_timeout_seconds: float = Field(default=200.0, gt=0, le=300)
    analyst_task_timeout_seconds: float = Field(default=250.0, gt=0, le=300)


class StorageSettings(BaseModel):
    """Default local artifact locations; CLI callers may override these paths."""

    model_config = ConfigDict(frozen=True)

    knowledge_manifest_path: Path = Path("data/knowledge_sources.yaml")
    source_snapshot_directory: Path = Path("data/source_snapshots")
    lancedb_database_path: Path = Path("data/lancedb")
    sqlite_database_path: Path = Path("data/financial_advisor.db")


class ProviderEndpointSettings(BaseModel):
    """Configurable provider base URLs; adapter paths and payloads stay in code."""

    model_config = ConfigDict(frozen=True)

    exa_base_url: HttpUrl = HttpUrl("https://api.exa.ai")
    brave_base_url: HttpUrl = HttpUrl("https://api.search.brave.com")


class Settings(BaseSettings):
    """Runtime settings that are safe to expose within application code."""

    app_env: str = "development"
    log_level: str = "INFO"
    exa_api_key: SecretStr | None = None
    brave_search_api_key: SecretStr | None = None
    google_api_key: SecretStr | None = None
    knowledge_ingestion: KnowledgeIngestionPolicy = Field(default_factory=KnowledgeIngestionPolicy)
    retrieval: RetrievalPolicy = Field(default_factory=RetrievalPolicy)
    web_research: WebResearchPolicy = Field(default_factory=WebResearchPolicy)
    workflow: WorkflowPolicy = Field(default_factory=WorkflowPolicy)
    model_runtime: ModelRuntimeSettings = Field(default_factory=ModelRuntimeSettings)
    gemini: GeminiSettings = Field(default_factory=GeminiSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    provider_endpoints: ProviderEndpointSettings = Field(default_factory=ProviderEndpointSettings)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_nested_delimiter="__",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Return one settings instance for the current process."""

    return Settings()
