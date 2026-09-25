"""Run Advisor-authorized retrieval channels before one shared evidence selection pass."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from enum import StrEnum
from typing import Protocol
from uuid import UUID

from pydantic import BaseModel, Field, model_validator

from financial_advisor.config import get_settings
from financial_advisor.domain import (
    ApprovedDomainId,
    Evidence,
    WebResearchMode,
    WebResearchScope,
)
from financial_advisor.retrieval import (
    DEFAULT_FINAL_EVIDENCE_LIMIT,
    DEFAULT_MMR_RELEVANCE_WEIGHT,
    EvidenceSelectionTrace,
    QueryDocumentReranker,
    QueryEmbedder,
    RetrievalEvidenceCandidate,
    select_evidence_candidates,
)
from financial_advisor.retrieval.evidence import assemble_selected_evidence
from financial_advisor.retrieval.knowledge_base.retrievers import (
    DEFAULT_VECTOR_CANDIDATE_LIMIT,
    KeywordRetrievalTrace,
    VectorRetrievalTrace,
)
from financial_advisor.retrieval.web.discovery import WebResearchUnavailable, WebSearchRequest
from financial_advisor.retrieval.web.pipeline import LiveWebResearchTrace


class LocalVectorCandidateRetriever(Protocol):
    """Vector-candidate behavior needed by combined evidence retrieval."""

    def search(self, query: str, *, candidate_limit: int) -> VectorRetrievalTrace:
        """Return ranked local vector candidates."""


class LocalKeywordCandidateRetriever(Protocol):
    """Keyword-candidate behavior needed by combined evidence retrieval."""

    def search(self, query: str, *, candidate_limit: int) -> KeywordRetrievalTrace:
        """Return ranked local BM25 candidates."""


class LiveWebCandidateRetriever(Protocol):
    """Live-web research behavior needed by combined evidence retrieval."""

    def research(self, request: WebSearchRequest) -> LiveWebResearchTrace:
        """Return the complete live-web trajectory and its ranked candidates."""


class EvidenceRetrievalRequest(BaseModel):
    """Deterministic operations granted by an Advisor research plan."""

    query: str = Field(min_length=1, max_length=2_000)
    include_local_hybrid: bool = True
    web_scopes: tuple[WebResearchScope, ...] = Field(default=(), max_length=2)

    @model_validator(mode="after")
    def validate_operations(self) -> EvidenceRetrievalRequest:
        """Require work and keep the two possible web scopes unambiguous."""

        if not self.include_local_hybrid and not self.web_scopes:
            raise ValueError("Evidence retrieval requires local retrieval or a web scope.")
        modes = [scope.mode for scope in self.web_scopes]
        if len(set(modes)) != len(modes):
            raise ValueError("A research plan may contain at most one scope of each web mode.")
        return self


class ScopedWebResearchTrace(BaseModel):
    """One sequential web-scope execution and its independent replayable trace."""

    scope: WebResearchScope
    trace: LiveWebResearchTrace


class EvidenceRetrievalStatus(StrEnum):
    """Whether the plan yielded evidence safe enough for Analyst synthesis."""

    COMPLETED = "completed"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class RetrievalSourceName(StrEnum):
    """One selected evidence source whose availability is shown to downstream agents."""

    LOCAL_HYBRID = "local_hybrid"
    AUTHORITATIVE_WEB = "authoritative_web"
    BROAD_WEB = "broad_web"


class RetrievalSourceStatus(StrEnum):
    """Whether one selected source contributed usable evidence in this run."""

    EVIDENCE_AVAILABLE = "evidence_available"
    NO_USABLE_EVIDENCE = "no_usable_evidence"
    UNAVAILABLE = "unavailable"


class RetrievalSourceOutcome(BaseModel):
    """Trace-safe availability result for one selected retrieval source."""

    source: RetrievalSourceName
    status: RetrievalSourceStatus
    detail: str | None = Field(default=None, max_length=500)


class CombinedEvidenceRetrievalTrace(BaseModel):
    """Replayable all-channel evidence trajectory for one Analyst research task."""

    query: str = Field(min_length=1)
    local_vector: VectorRetrievalTrace | None = None
    local_bm25: KeywordRetrievalTrace | None = None
    web_researches: list[ScopedWebResearchTrace] = Field(default_factory=list, max_length=2)
    live_web: LiveWebResearchTrace | None = None
    status: EvidenceRetrievalStatus
    reason: str | None = Field(default=None, max_length=500)
    source_outcomes: list[RetrievalSourceOutcome] = Field(default_factory=list)
    selection: EvidenceSelectionTrace | None = None
    evidence: list[Evidence] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_reason(cls, value: object) -> object:
        """Read pre-scope persisted traces during the incremental contract migration."""

        if isinstance(value, dict) and "reason" not in value:
            legacy_reason = value.get("insufficient_evidence_reason")
            if legacy_reason is not None:
                return {**value, "reason": legacy_reason}
        return value

    @property
    def insufficient_evidence_reason(self) -> str | None:
        """Provide a read-only compatibility view for pre-scope callers."""

        return self.reason

    @property
    def evidence_limitations(self) -> list[str]:
        """Return deterministic limitations for the Analyst and Advisor to disclose."""

        return [
            outcome.detail
            for outcome in self.source_outcomes
            if outcome.status is not RetrievalSourceStatus.EVIDENCE_AVAILABLE
            and outcome.detail is not None
        ]

    @model_validator(mode="after")
    def validate_status_and_selection(self) -> CombinedEvidenceRetrievalTrace:
        """Prevent a trace from claiming selected evidence when none is safe to use."""

        has_local_pair = self.local_vector is not None and self.local_bm25 is not None
        if (self.local_vector is None) != (self.local_bm25 is None):
            raise ValueError("Local vector and BM25 traces must be present together.")
        if not has_local_pair and not self.web_researches and self.live_web is None:
            raise ValueError("A retrieval trace must include at least one executed channel.")
        if self.status is EvidenceRetrievalStatus.COMPLETED:
            if self.selection is None:
                raise ValueError("Completed retrieval requires an evidence selection.")
            if not self.evidence:
                raise ValueError("Completed retrieval requires assembled evidence.")
            if self.reason is not None:
                raise ValueError("Completed retrieval cannot include an unavailable reason.")
        else:
            if self.selection is not None:
                raise ValueError("Unavailable retrieval cannot include a selection.")
            if self.evidence:
                raise ValueError("Unavailable retrieval cannot include evidence.")
            if self.reason is None:
                raise ValueError("Unavailable retrieval requires a safe reason.")
        return self


class TaskEvidenceRetrievalTrace(BaseModel):
    """One complete retrieval trajectory recorded for one Analyst task."""

    task_id: UUID
    retrieval: CombinedEvidenceRetrievalTrace
    recorded_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class EvidenceRetrievalPipeline:
    """Execute only granted channels, then select their evidence jointly and transparently."""

    def __init__(
        self,
        local_vector: LocalVectorCandidateRetriever,
        local_bm25: LocalKeywordCandidateRetriever,
        live_web: LiveWebCandidateRetriever,
        reranker: QueryDocumentReranker,
        diversity_embedder: QueryEmbedder,
        *,
        approved_domain_registry: Mapping[ApprovedDomainId, tuple[str, ...]] | None = None,
    ) -> None:
        self._local_vector = local_vector
        self._local_bm25 = local_bm25
        self._live_web = live_web
        self._reranker = reranker
        self._diversity_embedder = diversity_embedder
        self._approved_domain_registry = (
            dict(approved_domain_registry)
            if approved_domain_registry is not None
            else get_settings().web_research.approved_domain_registry
        )

    def retrieve(
        self,
        request: EvidenceRetrievalRequest,
        *,
        local_candidate_limit: int = DEFAULT_VECTOR_CANDIDATE_LIMIT,
        selection_limit: int = DEFAULT_FINAL_EVIDENCE_LIMIT,
        relevance_weight: float = DEFAULT_MMR_RELEVANCE_WEIGHT,
    ) -> CombinedEvidenceRetrievalTrace:
        """Run enabled channels in plan order before one RRF, reranking, and MMR sequence."""

        query = request.query.strip()
        if not query:
            raise ValueError("An evidence-retrieval query must not be blank.")

        local_vector: VectorRetrievalTrace | None = None
        local_bm25: KeywordRetrievalTrace | None = None
        candidates: list[RetrievalEvidenceCandidate] = []
        source_outcomes: list[RetrievalSourceOutcome] = []
        if request.include_local_hybrid:
            local_vector = self._local_vector.search(query, candidate_limit=local_candidate_limit)
            local_bm25 = self._local_bm25.search(query, candidate_limit=local_candidate_limit)
            candidates.extend(local_vector.candidates)
            candidates.extend(local_bm25.candidates)
            local_has_candidates = bool(local_vector.candidates or local_bm25.candidates)
            source_outcomes.append(
                self._source_outcome(
                    source=RetrievalSourceName.LOCAL_HYBRID,
                    has_evidence=local_has_candidates,
                    no_evidence_detail="Local knowledge retrieval returned no usable evidence.",
                )
            )

        web_researches: list[ScopedWebResearchTrace] = []
        for scope in request.web_scopes:
            trace = self._live_web.research(self._to_web_search_request(query, scope))
            scoped_trace = ScopedWebResearchTrace(scope=scope, trace=trace)
            web_researches.append(scoped_trace)
            scoped_candidates = self._scoped_candidates(scoped_trace)
            candidates.extend(scoped_candidates)
            source = self._web_source_name(scope)
            source_outcomes.append(
                self._source_outcome(
                    source=source,
                    has_evidence=bool(scoped_candidates),
                    unavailable=isinstance(trace.discovery, WebResearchUnavailable),
                    no_evidence_detail=f"{source.value} research returned no usable evidence.",
                )
            )
        if not candidates:
            return CombinedEvidenceRetrievalTrace(
                query=query,
                local_vector=local_vector,
                local_bm25=local_bm25,
                web_researches=web_researches,
                status=EvidenceRetrievalStatus.INSUFFICIENT_EVIDENCE,
                reason="No selected retrieval source returned usable evidence candidates.",
                source_outcomes=source_outcomes,
            )
        selection = select_evidence_candidates(
            query,
            candidates,
            self._reranker,
            self._diversity_embedder,
            selection_limit=selection_limit,
            relevance_weight=relevance_weight,
        )
        evidence = assemble_selected_evidence(selection.diversity_selection.selected_candidates)
        return CombinedEvidenceRetrievalTrace(
            query=query,
            local_vector=local_vector,
            local_bm25=local_bm25,
            web_researches=web_researches,
            status=EvidenceRetrievalStatus.COMPLETED,
            source_outcomes=source_outcomes,
            selection=selection,
            evidence=evidence,
        )

    def _to_web_search_request(
        self, query: str, scope: WebResearchScope
    ) -> WebSearchRequest:
        """Resolve reviewed IDs before creating the lower-level provider request."""

        allowed_domains: tuple[str, ...] = ()
        if scope.mode is WebResearchMode.AUTHORITATIVE_DOMAIN:
            allowed_domains = self._resolve_domains(scope.approved_domain_ids)
        return WebSearchRequest(
            query=query,
            search_mode=scope.mode,
            allowed_domains=allowed_domains,
        )

    def _resolve_domains(self, domain_ids: tuple[ApprovedDomainId, ...]) -> tuple[str, ...]:
        """Resolve policy IDs without accepting raw domains from an Advisor plan."""

        resolved: list[str] = []
        for domain_id in domain_ids:
            try:
                domains = self._approved_domain_registry[domain_id]
            except KeyError as error:
                message = f"No domain policy is configured for {domain_id.value!r}."
                raise ValueError(message) from error
            for domain in domains:
                if domain not in resolved:
                    resolved.append(domain)
        return tuple(resolved)

    @staticmethod
    def _scoped_candidates(
        scoped_trace: ScopedWebResearchTrace,
    ) -> list[RetrievalEvidenceCandidate]:
        """Preserve whether a selected web passage came from broad or approved research."""

        channel = (
            "authoritative_web"
            if scoped_trace.scope.mode is WebResearchMode.AUTHORITATIVE_DOMAIN
            else "broad_web"
        )
        return [
            candidate.model_copy(update={"retrieval_channel": channel})
            for candidate in scoped_trace.trace.web_vector_retrieval.candidates
        ]

    @staticmethod
    def _web_source_name(scope: WebResearchScope) -> RetrievalSourceName:
        """Map a search-policy mode to its evidence-provenance source name."""

        return (
            RetrievalSourceName.AUTHORITATIVE_WEB
            if scope.mode is WebResearchMode.AUTHORITATIVE_DOMAIN
            else RetrievalSourceName.BROAD_WEB
        )

    @staticmethod
    def _source_outcome(
        *,
        source: RetrievalSourceName,
        has_evidence: bool,
        no_evidence_detail: str,
        unavailable: bool = False,
    ) -> RetrievalSourceOutcome:
        """Create one consistent source result from availability and evidence facts."""

        if unavailable:
            return RetrievalSourceOutcome(
                source=source,
                status=RetrievalSourceStatus.UNAVAILABLE,
                detail=f"{source.value} research was unavailable for this run.",
            )
        if has_evidence:
            return RetrievalSourceOutcome(
                source=source,
                status=RetrievalSourceStatus.EVIDENCE_AVAILABLE,
            )
        return RetrievalSourceOutcome(
            source=source,
            status=RetrievalSourceStatus.NO_USABLE_EVIDENCE,
            detail=no_evidence_detail,
        )
