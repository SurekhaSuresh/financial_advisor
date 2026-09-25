"""Tests for the first deterministic vector-retrieval layer."""

from collections.abc import Sequence
from pathlib import Path

import pytest
from pydantic import HttpUrl

from financial_advisor.retrieval import (
    LanceKeywordRetriever,
    LanceVectorRetriever,
    RetrievalEvidenceCandidate,
    fuse_retrieval_candidates,
    rerank_fused_candidates,
    select_evidence_candidates,
    select_mmr_candidates,
)
from financial_advisor.retrieval.content_processing.chunking import canonical_candidate_id
from financial_advisor.retrieval.knowledge_base.ingestion import (
    KnowledgeSource,
    SourceSection,
    chunk_sections,
    write_lancedb,
)
from financial_advisor.retrieval.web.page_processing import WebEvidenceChunk
from financial_advisor.retrieval.web.passage_ranker import InMemoryWebVectorRetriever


class StaticEmbedder:
    """Offline test embedder returning a preselected vector for every input."""

    def __init__(self, vector: list[float]) -> None:
        self._vector = vector

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector for _ in texts]


class StaticReranker:
    """Offline test reranker returning a configured score for each document."""

    def __init__(self, scores: list[float]) -> None:
        self._scores = scores
        self.queries: list[str] = []
        self.documents: list[list[str]] = []

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        self.queries.append(query)
        self.documents.append(list(documents))
        return self._scores


class StaticDocumentEmbedder:
    """Offline test embedder returning vectors in supplied-document order."""

    def __init__(self, vectors: list[list[float]]) -> None:
        self._vectors = vectors

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return self._vectors


class FailingEmbedder:
    """Proves MMR does not re-embed candidates that already carry vectors."""

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        raise AssertionError("MMR should reuse the stored vector without embedding.")


class WhitespaceCodec:
    """Small local codec for deterministic retrieval-fixture chunk creation."""

    def encode(self, text: str) -> list[int]:
        return list(range(len(text.split())))

    def decode(self, token_ids: Sequence[int]) -> str:
        return " ".join(f"token-{token_id}" for token_id in token_ids)


def _candidate(
    canonical_candidate_id: str,
    channel: str,
    rank: int,
    embedding: list[float] | None = None,
) -> RetrievalEvidenceCandidate:
    """Create a minimal valid candidate for deterministic fusion tests."""

    return RetrievalEvidenceCandidate(
        canonical_candidate_id=canonical_candidate_id,
        retrieval_channel=channel,  # type: ignore[arg-type]
        rank=rank,
        chunk_id=canonical_candidate_id,
        source_id="test_source",
        source_title="Test source",
        publisher="Test publisher",
        source_url=HttpUrl("https://example.com/source"),
        topics=["testing"],
        heading="Test heading",
        heading_path=["Test heading"],
        section_position=1,
        chunk_position=1,
        token_count=10,
        text="Test evidence text.",
        embedding=embedding,
    )


def _web_chunk(
    chunk_id: str, text: str, *, canonical_candidate_id: str | None = None
) -> WebEvidenceChunk:
    return WebEvidenceChunk(
        chunk_id=chunk_id,
        canonical_candidate_id=canonical_candidate_id or chunk_id,
        source_id="web:test-page",
        source_title="Live investor guidance",
        publisher="www.investor.gov",
        source_url=HttpUrl("https://www.investor.gov/example"),
        topics=["live_web"],
        heading="Diversification",
        heading_path=["Diversification"],
        section_position=1,
        chunk_position=1,
        token_count=10,
        body_start_token=0,
        body_end_token_exclusive=5,
        text=text,
        source_content_hash="test-page-hash",
    )


def test_web_vector_retriever_returns_ranked_normalized_live_web_candidates() -> None:
    retriever = InMemoryWebVectorRetriever(
        StaticDocumentEmbedder([[1.0, 0.0], [0.1, 0.9], [0.9, 0.1]])
    )

    trace = retriever.search(
        "How should I diversify?",
        [_web_chunk("web-a", "Unrelated text."), _web_chunk("web-b", "Diversify holdings.")],
    )

    assert trace.fetched_chunk_count == 2
    assert [candidate.chunk_id for candidate in trace.candidates] == ["web-b", "web-a"]
    assert [candidate.rank for candidate in trace.candidates] == [1, 2]
    assert all(candidate.retrieval_channel == "live_web" for candidate in trace.candidates)
    assert trace.candidates[0].embedding == [0.9, 0.1]


def test_web_vector_retriever_keeps_the_best_exact_duplicate() -> None:
    retriever = InMemoryWebVectorRetriever(
        StaticDocumentEmbedder([[1.0, 0.0], [0.1, 0.9], [0.9, 0.1]])
    )

    trace = retriever.search(
        "How should I diversify?",
        [
            _web_chunk("web-first", "First copy.", canonical_candidate_id="web:shared"),
            _web_chunk("web-best", "Best copy.", canonical_candidate_id="web:shared"),
        ],
    )

    assert len(trace.candidates) == 1
    assert trace.candidates[0].chunk_id == "web-best"
    assert trace.candidates[0].canonical_candidate_id == "web:shared"


def test_vector_search_returns_ranked_provenance_preserving_candidates(tmp_path: Path) -> None:
    """The retriever exposes ordered candidates, distance, and citation fields."""

    source = KnowledgeSource(
        id="retrieval_test",
        publisher="Test publisher",
        title="Retrieval test source",
        url=HttpUrl("https://example.com/retrieval"),
        content_type="html",
        topics=["testing"],
    )
    chunks = chunk_sections(
        source,
        [SourceSection("Diversification", "one two", 1)],
        codec=WhitespaceCodec(),
        max_tokens=10,
        overlap_tokens=1,
    )
    database_path = tmp_path / "lancedb"
    write_lancedb(database_path, chunks, [[1.0, 0.0]])

    trace = LanceVectorRetriever(database_path, StaticEmbedder([1.0, 0.0])).search(
        "diversify", candidate_limit=5
    )

    assert trace.query == "diversify"
    assert trace.candidate_limit == 5
    assert trace.candidates[0].rank == 1
    assert trace.candidates[0].chunk_id == chunks[0].chunk_id
    assert trace.candidates[0].source_url == source.url
    assert trace.candidates[0].cosine_distance == pytest.approx(0.0)
    assert trace.candidates[0].embedding == [1.0, 0.0]
    assert "embedding" not in trace.candidates[0].model_dump()


def test_vector_search_rejects_blank_query_and_missing_database(tmp_path: Path) -> None:
    """Bad input and missing generated data fail clearly before a database query."""

    retriever = LanceVectorRetriever(tmp_path / "missing", StaticEmbedder([1.0, 0.0]))

    with pytest.raises(ValueError, match="must not be blank"):
        retriever.search("   ")
    with pytest.raises(FileNotFoundError, match="does not exist"):
        retriever.search("valid query")


def test_keyword_search_creates_local_index_and_returns_provenance(tmp_path: Path) -> None:
    """BM25 index creation and keyword candidates remain local and inspectable."""

    source = KnowledgeSource(
        id="keyword_test",
        publisher="Test publisher",
        title="Keyword test source",
        url=HttpUrl("https://example.com/keyword"),
        content_type="html",
        topics=["testing"],
    )
    chunks = chunk_sections(
        source,
        [SourceSection("Expense ratios", "expense ratios affect fund returns", 1)],
        codec=WhitespaceCodec(),
        max_tokens=12,
        overlap_tokens=1,
    )
    chunks[0].text = "expense ratios affect fund returns"
    database_path = tmp_path / "lancedb"
    write_lancedb(database_path, chunks, [[1.0, 0.0]])

    trace = LanceKeywordRetriever(database_path).search("expense ratios", candidate_limit=3)

    assert trace.query == "expense ratios"
    assert trace.candidates[0].rank == 1
    assert trace.candidates[0].chunk_id == chunks[0].chunk_id
    assert trace.candidates[0].bm25_score is not None
    assert trace.candidates[0].bm25_score > 0


def test_rrf_combines_channel_ranks_and_preserves_provenance() -> None:
    """One chunk found by two channels outranks a one-channel alternative."""

    vector_a = _candidate("chunk-a", "internal_vector", 1)
    keyword_a = _candidate("chunk-a", "internal_bm25", 3)
    vector_b = _candidate("chunk-b", "internal_vector", 1)

    trace = fuse_retrieval_candidates([vector_a, keyword_a, vector_b])

    assert trace.input_candidate_count == 3
    assert trace.canonical_candidate_count == 2
    assert trace.candidates[0].canonical_candidate_id == "chunk-a"
    assert trace.candidates[0].rrf_score == pytest.approx(1 / 61 + 1 / 63)
    assert [item.retrieval_channel for item in trace.candidates[0].contributions] == [
        "internal_vector",
        "internal_bm25",
    ]
    assert trace.candidates[0].representative == vector_a


def test_rrf_counts_an_accidental_same_channel_duplicate_once() -> None:
    """A retried or paginated duplicate cannot inflate one channel's RRF vote."""

    rank_three = _candidate("chunk-a", "internal_vector", 3)
    rank_one = _candidate("chunk-a", "internal_vector", 1)

    trace = fuse_retrieval_candidates([rank_three, rank_one])

    assert trace.canonical_candidate_count == 1
    assert trace.candidates[0].rrf_score == pytest.approx(1 / 61)
    assert trace.candidates[0].contributions == [rank_one]


def test_rrf_rejects_a_non_positive_rank_constant() -> None:
    """The RRF denominator must remain positive and meaningful."""

    with pytest.raises(ValueError, match="rank_constant must be positive"):
        fuse_retrieval_candidates([], rank_constant=0)


def test_cross_encoder_reranking_scores_query_document_pairs() -> None:
    """Reranking changes RRF order using query-aware pair scores."""

    fused = fuse_retrieval_candidates(
        [
            _candidate("chunk-a", "internal_vector", 1),
            _candidate("chunk-b", "internal_vector", 2),
        ]
    ).candidates
    reranker = StaticReranker([0.1, 0.9])

    trace = rerank_fused_candidates("  expense ratios  ", fused, reranker)

    assert reranker.queries == ["expense ratios"]
    assert reranker.documents == [[candidate.representative.text for candidate in fused]]
    assert trace.input_candidate_count == 2
    assert trace.candidates[0].rank == 1
    assert trace.candidates[0].cross_encoder_score == 0.9
    assert trace.candidates[0].candidate.canonical_candidate_id == "chunk-b"


def test_cross_encoder_reranking_rejects_bad_input_or_misaligned_scores() -> None:
    """A failed reranker cannot silently associate scores with wrong chunks."""

    candidate = fuse_retrieval_candidates([_candidate("chunk-a", "internal_vector", 1)]).candidates

    with pytest.raises(ValueError, match="must not be blank"):
        rerank_fused_candidates(" ", candidate, StaticReranker([0.1]))
    with pytest.raises(ValueError, match="one score"):
        rerank_fused_candidates("valid", candidate, StaticReranker([]))


def test_mmr_prefers_a_diverse_candidate_after_the_best_relevance_result() -> None:
    """MMR avoids choosing a near duplicate when an alternative is available."""

    fused = fuse_retrieval_candidates(
        [
            _candidate("chunk-a", "internal_vector", 1),
            _candidate("chunk-b", "internal_vector", 2),
            _candidate("chunk-c", "internal_vector", 3),
        ]
    ).candidates
    reranked = rerank_fused_candidates("risk", fused, StaticReranker([0.9, 0.75, 0.74]))

    trace = select_mmr_candidates(
        reranked.candidates,
        StaticDocumentEmbedder([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        selection_limit=2,
        relevance_weight=0.7,
    )

    assert [
        item.candidate.candidate.canonical_candidate_id for item in trace.selected_candidates
    ] == ["chunk-a", "chunk-c"]
    assert trace.selected_candidates[0].maximum_similarity_to_selected is None
    assert trace.selected_candidates[1].maximum_similarity_to_selected == pytest.approx(0.0)


def test_mmr_rejects_invalid_configuration_or_vector_alignment() -> None:
    """MMR fails safely instead of silently selecting with broken inputs."""

    reranked = rerank_fused_candidates(
        "risk",
        fuse_retrieval_candidates([_candidate("chunk-a", "internal_vector", 1)]).candidates,
        StaticReranker([0.9]),
    ).candidates

    with pytest.raises(ValueError, match="selection_limit must be positive"):
        select_mmr_candidates(reranked, StaticDocumentEmbedder([[1.0, 0.0]]), selection_limit=0)
    with pytest.raises(ValueError, match="between 0 and 1"):
        select_mmr_candidates(reranked, StaticDocumentEmbedder([[1.0, 0.0]]), relevance_weight=2)
    with pytest.raises(ValueError, match="one vector"):
        select_mmr_candidates(reranked, StaticDocumentEmbedder([]))


def test_mmr_reuses_private_stored_vectors_without_reembedding() -> None:
    """Internal candidates use their LanceDB vectors; only missing vectors need BGE."""

    reranked = rerank_fused_candidates(
        "risk",
        fuse_retrieval_candidates(
            [
                _candidate("chunk-a", "internal_vector", 1, [1.0, 0.0]),
                _candidate("chunk-b", "internal_vector", 2, [0.0, 1.0]),
            ]
        ).candidates,
        StaticReranker([0.9, 0.8]),
    ).candidates

    trace = select_mmr_candidates(reranked, FailingEmbedder(), selection_limit=2)

    assert len(trace.selected_candidates) == 2


def test_local_retrievers_feed_the_shared_evidence_selection_helper(tmp_path: Path) -> None:
    """Local candidate retrievers use the same selection helper as the full pipeline."""

    source = KnowledgeSource(
        id="hybrid_test",
        publisher="Test publisher",
        title="Hybrid test source",
        url=HttpUrl("https://example.com/hybrid"),
        content_type="html",
        topics=["testing"],
    )
    chunks = chunk_sections(
        source,
        [
            SourceSection("Expense ratios", "one two", 1),
            SourceSection("Emergency savings", "three four", 2),
        ],
        codec=WhitespaceCodec(),
        max_tokens=10,
        overlap_tokens=1,
    )
    chunks[0].text = "expense ratios affect fund returns"
    chunks[1].text = "emergency savings cover unexpected expenses"
    for chunk in chunks:
        chunk.canonical_candidate_id = canonical_candidate_id(
            source_url=str(chunk.source_url),
            source_content_hash="test-document-version",
            text=chunk.text,
        )
    database_path = tmp_path / "lancedb"
    write_lancedb(database_path, chunks, [[1.0, 0.0], [0.0, 1.0]])

    vector_search = LanceVectorRetriever(database_path, StaticEmbedder([1.0, 0.0])).search(
        "expense ratios", candidate_limit=2
    )
    keyword_search = LanceKeywordRetriever(database_path).search(
        "expense ratios", candidate_limit=2
    )
    selection = select_evidence_candidates(
        "expense ratios",
        [*vector_search.candidates, *keyword_search.candidates],
        StaticReranker([0.9, 0.1]),
        FailingEmbedder(),
        selection_limit=2,
    )

    assert len(vector_search.candidates) == 2
    assert keyword_search.candidates[0].chunk_id == chunks[0].chunk_id
    assert selection.fusion.canonical_candidate_count == 2
    assert (
        selection.reranking.candidates[0].candidate.canonical_candidate_id
        == chunks[0].canonical_candidate_id
    )
    assert len(selection.diversity_selection.selected_candidates) == 2
