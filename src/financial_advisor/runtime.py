"""Local runtime composition for the Financial Advisor application."""

from __future__ import annotations

import os
from dataclasses import dataclass

from financial_advisor.agents.advisor.adk import AdkAdvisorService
from financial_advisor.agents.advisor.workflow import AdvisorWorkflowCoordinator
from financial_advisor.agents.analyst.adk import AdkAnalystService
from financial_advisor.agents.analyst.execution import AnalystResearchExecutor
from financial_advisor.agents.analyst.workflow import AnalystWorkflowCoordinator
from financial_advisor.agents.client.adk import AdkClientService
from financial_advisor.api import ScenarioExecutionManager
from financial_advisor.config import Settings
from financial_advisor.persistence import SessionRepository
from financial_advisor.retrieval.knowledge_base.ingestion import BgeEmbedder, BgeTokenWindowCodec
from financial_advisor.retrieval.knowledge_base.retrievers import (
    LanceKeywordRetriever,
    LanceVectorRetriever,
)
from financial_advisor.retrieval.pipeline import EvidenceRetrievalPipeline
from financial_advisor.retrieval.reranker import FastEmbedCrossEncoderReranker
from financial_advisor.retrieval.web.discovery import (
    CircuitBreaker,
    SearchProviderName,
    WebResearchService,
)
from financial_advisor.retrieval.web.fetch import WebPageFetcher
from financial_advisor.retrieval.web.passage_ranker import InMemoryWebVectorRetriever
from financial_advisor.retrieval.web.pipeline import LiveWebResearchPipeline
from financial_advisor.retrieval.web.providers import (
    BraveWebResearchProvider,
    ExaWebResearchProvider,
)
from financial_advisor.scenario import ScenarioRunner
from financial_advisor.workflow import WorkflowEngine


class RuntimeConfigurationError(RuntimeError):
    """Raised when a requested local live scenario cannot be configured safely."""


@dataclass(frozen=True)
class LocalRuntime:
    """The process-wide services used by FastAPI routes during one local run."""

    repository: SessionRepository
    execution_manager: ScenarioExecutionManager


def build_local_runtime(
    settings: Settings, *, repository: SessionRepository | None = None
) -> LocalRuntime:
    """Wire the real local agents and retrieval stack from validated settings."""

    _validate_live_runtime_requirements(settings)
    _configure_google_adk_credentials(settings)
    assert settings.google_api_key is not None
    assert settings.exa_api_key is not None
    assert settings.brave_search_api_key is not None

    repository = repository or SessionRepository(settings.storage.sqlite_database_path)
    workflow = WorkflowEngine(repository)
    embedder = BgeEmbedder(
        model_name=settings.model_runtime.embedding_model_name,
        cache_directory=settings.model_runtime.model_cache_directory,
    )
    codec = BgeTokenWindowCodec(
        model_name=settings.model_runtime.embedding_model_name,
        cache_directory=settings.model_runtime.model_cache_directory,
    )
    local_vector = LanceVectorRetriever(settings.storage.lancedb_database_path, embedder)
    local_bm25 = LanceKeywordRetriever(settings.storage.lancedb_database_path)
    discovery = WebResearchService(
        providers=(
            ExaWebResearchProvider(
                settings.exa_api_key,
                timeout_seconds=settings.web_research.provider_timeout_seconds,
                base_url=settings.provider_endpoints.exa_base_url,
            ),
            BraveWebResearchProvider(
                settings.brave_search_api_key,
                timeout_seconds=settings.web_research.provider_timeout_seconds,
                base_url=settings.provider_endpoints.brave_base_url,
            ),
        ),
        breakers={
            SearchProviderName.EXA: CircuitBreaker(
                failure_threshold=settings.web_research.circuit_failure_threshold,
                cooldown_seconds=settings.web_research.circuit_cooldown_seconds,
            ),
            SearchProviderName.BRAVE: CircuitBreaker(
                failure_threshold=settings.web_research.circuit_failure_threshold,
                cooldown_seconds=settings.web_research.circuit_cooldown_seconds,
            ),
        },
        max_attempts=settings.web_research.provider_max_attempts,
        retry_backoff_base_seconds=settings.web_research.provider_retry_backoff_base_seconds,
    )
    live_web = LiveWebResearchPipeline(
        discovery=discovery,
        page_fetcher=WebPageFetcher(),
        codec=codec,
        web_retriever=InMemoryWebVectorRetriever(embedder),
        usable_page_target=settings.web_research.usable_page_target,
    )
    evidence_retriever = EvidenceRetrievalPipeline(
        local_vector=local_vector,
        local_bm25=local_bm25,
        live_web=live_web,
        reranker=FastEmbedCrossEncoderReranker(settings.model_runtime.model_cache_directory),
        diversity_embedder=embedder,
        approved_domain_registry=settings.web_research.approved_domain_registry,
    )
    analyst = AnalystWorkflowCoordinator(
        workflow,
        AnalystResearchExecutor(evidence_retriever, AdkAnalystService(settings.gemini)),
    )
    advisor = AdvisorWorkflowCoordinator(workflow, AdkAdvisorService(settings.gemini))
    runner = ScenarioRunner(workflow, AdkClientService(settings.gemini), advisor, analyst)
    return LocalRuntime(
        repository=repository,
        execution_manager=ScenarioExecutionManager(workflow, runner),
    )


def _configure_google_adk_credentials(settings: Settings) -> None:
    """Expose the validated server-side Gemini key through ADK's required environment API."""

    assert settings.google_api_key is not None
    os.environ["GOOGLE_API_KEY"] = settings.google_api_key.get_secret_value()


def _validate_live_runtime_requirements(settings: Settings) -> None:
    """Fail startup clearly when a required runtime dependency is absent."""

    missing: list[str] = []
    if settings.google_api_key is None:
        missing.append("GOOGLE_API_KEY")
    if settings.exa_api_key is None:
        missing.append("EXA_API_KEY")
    if settings.brave_search_api_key is None:
        missing.append("BRAVE_SEARCH_API_KEY")
    if not settings.storage.lancedb_database_path.exists():
        missing.append(f"LanceDB corpus at {settings.storage.lancedb_database_path}")
    if missing:
        raise RuntimeConfigurationError(
            "Live scenario runtime requires " + ", ".join(missing) + "."
        )
