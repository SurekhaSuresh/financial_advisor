from collections.abc import Sequence

from pydantic import HttpUrl

from financial_advisor.domain import (
    ApprovedDomainId,
    Evidence,
    WebResearchMode,
    WebResearchScope,
)
from financial_advisor.retrieval import (
    KeywordRetrievalTrace,
    RetrievalEvidenceCandidate,
    VectorRetrievalTrace,
)
from financial_advisor.retrieval.pipeline import (
    EvidenceRetrievalPipeline,
    EvidenceRetrievalRequest,
    EvidenceRetrievalStatus,
)
from financial_advisor.retrieval.web.discovery import (
    ProviderSearchOutcome,
    ProviderSearchStatus,
    SearchProviderName,
    WebResearchUnavailable,
    WebSearchRequest,
    WebSearchResponse,
)
from financial_advisor.retrieval.web.passage_ranker import WebVectorRetrievalTrace
from financial_advisor.retrieval.web.pipeline import LiveWebResearchTrace


class FakeVectorRetriever:
    def search(self, query: str, *, candidate_limit: int) -> VectorRetrievalTrace:
        return VectorRetrievalTrace(
            query=query,
            candidate_limit=candidate_limit,
            candidates=[make_candidate("local-vector", "internal_vector", [1.0, 0.0])],
        )


class FakeKeywordRetriever:
    def search(self, query: str, *, candidate_limit: int) -> KeywordRetrievalTrace:
        return KeywordRetrievalTrace(
            query=query,
            candidate_limit=candidate_limit,
            candidates=[make_candidate("local-bm25", "internal_bm25", [0.0, 1.0])],
        )


class EmptyVectorRetriever:
    def search(self, query: str, *, candidate_limit: int) -> VectorRetrievalTrace:
        return VectorRetrievalTrace(query=query, candidate_limit=candidate_limit, candidates=[])


class EmptyKeywordRetriever:
    def search(self, query: str, *, candidate_limit: int) -> KeywordRetrievalTrace:
        return KeywordRetrievalTrace(query=query, candidate_limit=candidate_limit, candidates=[])


class FakeLiveWebRetriever:
    def research(self, request: WebSearchRequest) -> LiveWebResearchTrace:
        return LiveWebResearchTrace(
            discovery=WebSearchResponse(provider=SearchProviderName.EXA, results=[]),
            discovered_result_count=0,
            usable_page_target=4,
            attempted_page_count=0,
            usable_page_count=0,
            page_fetches=[],
            content_processing=[],
            web_vector_retrieval=WebVectorRetrievalTrace(
                query=request.query,
                fetched_chunk_count=1,
                candidate_limit=10,
                candidates=[make_candidate("live-web", "live_web", [0.7, 0.7])],
            ),
        )


class EmptyLiveWebRetriever:
    def research(self, request: WebSearchRequest) -> LiveWebResearchTrace:
        return LiveWebResearchTrace(
            discovery=WebSearchResponse(provider=SearchProviderName.EXA, results=[]),
            discovered_result_count=0,
            usable_page_target=4,
            attempted_page_count=0,
            usable_page_count=0,
            page_fetches=[],
            content_processing=[],
            web_vector_retrieval=WebVectorRetrievalTrace(
                query=request.query,
                fetched_chunk_count=0,
                candidate_limit=10,
                candidates=[],
            ),
        )


class UnavailableLiveWebRetriever:
    def research(self, request: WebSearchRequest) -> LiveWebResearchTrace:
        return LiveWebResearchTrace(
            discovery=WebResearchUnavailable(
                provider_outcomes=[
                    ProviderSearchOutcome(
                        provider=SearchProviderName.EXA,
                        status=ProviderSearchStatus.FAILED,
                        attempt_count=3,
                        error_message="Provider connection failed.",
                    )
                ]
            ),
            discovered_result_count=0,
            usable_page_target=4,
            attempted_page_count=0,
            usable_page_count=0,
            page_fetches=[],
            content_processing=[],
            web_vector_retrieval=WebVectorRetrievalTrace(
                query=request.query,
                fetched_chunk_count=0,
                candidate_limit=10,
                candidates=[],
            ),
        )


class StaticReranker:
    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        del query
        return [float(index) for index, _ in enumerate(documents, start=1)]


class FailingEmbedder:
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        del texts
        raise AssertionError("MMR should reuse the candidates' stored embeddings.")


def make_candidate(
    candidate_id: str, channel: str, embedding: list[float]
) -> RetrievalEvidenceCandidate:
    return RetrievalEvidenceCandidate(
        chunk_id=candidate_id,
        canonical_candidate_id=candidate_id,
        source_id=f"source:{candidate_id}",
        source_title=candidate_id,
        publisher="Test publisher",
        source_url=HttpUrl("https://example.com/source"),
        topics=["testing"],
        heading="Test heading",
        heading_path=["Test heading"],
        section_position=1,
        chunk_position=1,
        token_count=5,
        text=f"Evidence from {candidate_id}.",
        retrieval_channel=channel,  # type: ignore[arg-type]
        rank=1,
        embedding=embedding,
    )


def broad_web_request() -> EvidenceRetrievalRequest:
    return EvidenceRetrievalRequest(
        query="How should I diversify?",
        web_scopes=(WebResearchScope(mode=WebResearchMode.BROAD_WEB),),
    )


def test_combined_pipeline_fuses_all_three_candidate_channels_before_selection() -> None:
    request = broad_web_request()
    pipeline = EvidenceRetrievalPipeline(
        FakeVectorRetriever(),
        FakeKeywordRetriever(),
        FakeLiveWebRetriever(),
        StaticReranker(),
        FailingEmbedder(),
    )

    trace = pipeline.retrieve(request, selection_limit=3)

    assert trace.status is EvidenceRetrievalStatus.COMPLETED
    assert trace.selection is not None
    assert trace.selection.fusion.input_candidate_count == 3
    assert {
        candidate.representative.retrieval_channel
        for candidate in trace.selection.fusion.candidates
    } == {"internal_vector", "internal_bm25", "broad_web"}
    assert len(trace.selection.diversity_selection.selected_candidates) == 3
    assert len(trace.evidence) == 3
    assert "cross_encoder_score" not in Evidence.model_fields


def test_combined_pipeline_returns_insufficient_evidence_without_selection() -> None:
    request = broad_web_request()
    pipeline = EvidenceRetrievalPipeline(
        EmptyVectorRetriever(),
        EmptyKeywordRetriever(),
        EmptyLiveWebRetriever(),
        StaticReranker(),
        FailingEmbedder(),
    )

    trace = pipeline.retrieve(request)

    assert trace.status is EvidenceRetrievalStatus.INSUFFICIENT_EVIDENCE
    assert trace.selection is None
    assert trace.reason is not None
    assert trace.evidence == []


def test_empty_web_scope_preserves_a_limitation_and_uses_available_local_evidence() -> None:
    pipeline = EvidenceRetrievalPipeline(
        FakeVectorRetriever(),
        FakeKeywordRetriever(),
        EmptyLiveWebRetriever(),
        StaticReranker(),
        FailingEmbedder(),
    )

    trace = pipeline.retrieve(broad_web_request())

    assert trace.status is EvidenceRetrievalStatus.COMPLETED
    assert trace.evidence
    assert trace.evidence_limitations == [
        "broad_web research returned no usable evidence."
    ]


def test_assembled_live_web_evidence_keeps_source_dates_without_retrieval_scores() -> None:
    request = broad_web_request()
    pipeline = EvidenceRetrievalPipeline(
        FakeVectorRetriever(),
        FakeKeywordRetriever(),
        FakeLiveWebRetriever(),
        StaticReranker(),
        FailingEmbedder(),
    )

    trace = pipeline.retrieve(request, selection_limit=3)
    live_evidence = next(item for item in trace.evidence if item.source_type == "web")

    assert live_evidence.published_at is None
    assert live_evidence.retrieved_at.tzinfo is not None
    assert live_evidence.excerpt == "Evidence from live-web."


class RecordingLiveWebRetriever:
    def __init__(self) -> None:
        self.requests: list[WebSearchRequest] = []

    def research(self, request: WebSearchRequest) -> LiveWebResearchTrace:
        self.requests.append(request)
        candidate_id = f"web-{len(self.requests)}"
        return LiveWebResearchTrace(
            discovery=WebSearchResponse(provider=SearchProviderName.EXA, results=[]),
            discovered_result_count=0,
            usable_page_target=4,
            attempted_page_count=0,
            usable_page_count=0,
            page_fetches=[],
            content_processing=[],
            web_vector_retrieval=WebVectorRetrievalTrace(
                query=request.query,
                fetched_chunk_count=1,
                candidate_limit=10,
                candidates=[make_candidate(candidate_id, "live_web", [0.7, 0.7])],
            ),
        )


def test_pipeline_runs_authoritative_then_broad_scope_sequentially() -> None:
    live_web = RecordingLiveWebRetriever()
    pipeline = EvidenceRetrievalPipeline(
        EmptyVectorRetriever(),
        EmptyKeywordRetriever(),
        live_web,
        StaticReranker(),
        FailingEmbedder(),
        approved_domain_registry={ApprovedDomainId.IRS_GOV: ("irs.gov",)},
    )
    request = EvidenceRetrievalRequest(
        query="How do capital gains taxes affect a sale?",
        include_local_hybrid=False,
        web_scopes=(
            WebResearchScope(
                mode=WebResearchMode.AUTHORITATIVE_DOMAIN,
                approved_domain_ids=(ApprovedDomainId.IRS_GOV,),
            ),
            WebResearchScope(mode=WebResearchMode.BROAD_WEB),
        ),
    )

    trace = pipeline.retrieve(request, selection_limit=2)

    assert [item.search_mode for item in live_web.requests] == [
        WebResearchMode.AUTHORITATIVE_DOMAIN,
        WebResearchMode.BROAD_WEB,
    ]
    assert live_web.requests[0].allowed_domains == ("irs.gov",)
    assert live_web.requests[1].allowed_domains == ()
    assert [item.scope.mode for item in trace.web_researches] == [
        WebResearchMode.AUTHORITATIVE_DOMAIN,
        WebResearchMode.BROAD_WEB,
    ]
    assert trace.selection is not None
    assert {
        candidate.representative.retrieval_channel
        for candidate in trace.selection.fusion.candidates
    } == {"authoritative_web", "broad_web"}


def test_empty_local_scope_preserves_a_limitation_and_uses_available_web_evidence() -> None:
    pipeline = EvidenceRetrievalPipeline(
        EmptyVectorRetriever(),
        EmptyKeywordRetriever(),
        FakeLiveWebRetriever(),
        StaticReranker(),
        FailingEmbedder(),
    )

    trace = pipeline.retrieve(broad_web_request())

    assert trace.status is EvidenceRetrievalStatus.COMPLETED
    assert trace.evidence
    assert trace.evidence_limitations == [
        "Local knowledge retrieval returned no usable evidence."
    ]


def test_mixed_no_evidence_and_web_unavailable_is_insufficient_evidence() -> None:
    pipeline = EvidenceRetrievalPipeline(
        EmptyVectorRetriever(),
        EmptyKeywordRetriever(),
        UnavailableLiveWebRetriever(),
        StaticReranker(),
        FailingEmbedder(),
    )

    trace = pipeline.retrieve(broad_web_request())

    assert trace.status is EvidenceRetrievalStatus.INSUFFICIENT_EVIDENCE
    assert trace.evidence_limitations == [
        "Local knowledge retrieval returned no usable evidence.",
        "broad_web research was unavailable for this run.",
    ]


def test_web_only_provider_outage_is_insufficient_evidence_with_web_outcome() -> None:
    pipeline = EvidenceRetrievalPipeline(
        EmptyVectorRetriever(),
        EmptyKeywordRetriever(),
        UnavailableLiveWebRetriever(),
        StaticReranker(),
        FailingEmbedder(),
    )
    request = EvidenceRetrievalRequest(
        query="How should I diversify?",
        include_local_hybrid=False,
        web_scopes=(WebResearchScope(mode=WebResearchMode.BROAD_WEB),),
    )

    trace = pipeline.retrieve(request)

    assert trace.status is EvidenceRetrievalStatus.INSUFFICIENT_EVIDENCE
    assert trace.source_outcomes[0].source.value == "broad_web"
    assert trace.source_outcomes[0].status.value == "unavailable"
