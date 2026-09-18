"""Tests for the boundary between retrieval traces and Analyst citations."""

from datetime import UTC, date, datetime

import pytest
from pydantic import HttpUrl

from financial_advisor.retrieval import (
    FusedRetrievalCandidate,
    MmrSelectedCandidate,
    RerankedRetrievalCandidate,
    RetrievalEvidenceCandidate,
)
from financial_advisor.retrieval.evidence import assemble_selected_evidence


def make_selected_candidate(*, channel: str = "live_web") -> MmrSelectedCandidate:
    candidate = RetrievalEvidenceCandidate(
        chunk_id="web:content:1:1",
        canonical_candidate_id="web:content:1:1",
        source_id="web:content",
        source_title="Diversification guidance",
        publisher="Investor.gov",
        source_url=HttpUrl("https://www.investor.gov/diversification"),
        topics=["live_web"],
        heading="Diversification",
        heading_path=["Investing", "Diversification"],
        section_position=1,
        chunk_position=1,
        token_count=8,
        text="Spreading investments can reduce concentration risk.",
        published_at=date(2026, 9, 1),
        retrieved_at=datetime(2026, 9, 18, 12, 0, tzinfo=UTC),
        retrieval_channel=channel,  # type: ignore[arg-type]
        rank=2,
        embedding=[0.5, 0.5],
    )
    fused = FusedRetrievalCandidate(
        canonical_candidate_id=candidate.canonical_candidate_id,
        rrf_score=0.1,
        representative=candidate,
        contributions=[candidate],
    )
    reranked = RerankedRetrievalCandidate(rank=1, cross_encoder_score=0.9, candidate=fused)
    return MmrSelectedCandidate(
        selection_order=1,
        mmr_score=0.8,
        normalized_relevance=1.0,
        candidate=reranked,
    )


def test_assembler_keeps_provenance_but_not_retrieval_internals() -> None:
    evidence = assemble_selected_evidence([make_selected_candidate()])

    assert len(evidence) == 1
    assert evidence[0].source_type == "web"
    assert evidence[0].published_at == date(2026, 9, 1)
    assert evidence[0].retrieved_at == datetime(2026, 9, 18, 12, 0, tzinfo=UTC)
    assert evidence[0].excerpt == "Spreading investments can reduce concentration risk."
    assert set(evidence[0].model_dump()) == {
        "evidence_id",
        "title",
        "publisher",
        "url",
        "published_at",
        "retrieved_at",
        "excerpt",
        "source_type",
    }


def test_assembler_creates_stable_server_owned_evidence_ids() -> None:
    first = assemble_selected_evidence([make_selected_candidate()])
    second = assemble_selected_evidence([make_selected_candidate()])

    assert first[0].evidence_id == second[0].evidence_id


def test_assembler_rejects_an_empty_selection() -> None:
    with pytest.raises(ValueError, match="At least one selected candidate"):
        assemble_selected_evidence([])
