"""Convert selected retrieval candidates into the Analyst's citation contract."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Literal
from uuid import NAMESPACE_URL, UUID, uuid5

from financial_advisor.domain import Evidence
from financial_advisor.retrieval import MmrSelectedCandidate, RetrievalEvidenceCandidate


def assemble_selected_evidence(
    selected_candidates: Sequence[MmrSelectedCandidate],
    *,
    selected_at: datetime | None = None,
) -> list[Evidence]:
    """Create clean, stable citations from the final MMR-selected candidates.

    Retrieval scores, ranks, embeddings, and channel contributions intentionally
    remain in the retrieval trace. The Analyst receives only source provenance
    and excerpts it may cite by evidence ID.
    """

    if not selected_candidates:
        raise ValueError("At least one selected candidate is required to assemble evidence.")

    evidence_selected_at = selected_at or datetime.now(UTC)
    return [
        _assemble_evidence(item.candidate.candidate.representative, evidence_selected_at)
        for item in selected_candidates
    ]


def _assemble_evidence(candidate: RetrievalEvidenceCandidate, selected_at: datetime) -> Evidence:
    """Map one normalized candidate to the deliberately smaller Analyst contract."""

    source_type: Literal["vector_store", "web"] = (
        "web"
        if candidate.retrieval_channel in {"live_web", "authoritative_web", "broad_web"}
        else "vector_store"
    )
    return Evidence(
        evidence_id=_stable_evidence_id(candidate),
        title=candidate.source_title,
        publisher=candidate.publisher,
        url=candidate.source_url,
        published_at=candidate.published_at,
        # For live web, retain the original page-fetch time. Local chunks are
        # stable snapshots and do not yet carry snapshot timestamps, so this
        # records when the snapshot was selected for this research task.
        retrieved_at=candidate.retrieved_at or selected_at,
        excerpt=candidate.text[:1_000],
        source_type=source_type,
    )


def _stable_evidence_id(candidate: RetrievalEvidenceCandidate) -> UUID:
    """Derive a reproducible server-owned citation ID from source provenance."""

    return uuid5(
        NAMESPACE_URL,
        f"financial-advisor:{candidate.canonical_candidate_id}:{candidate.source_url}",
    )
