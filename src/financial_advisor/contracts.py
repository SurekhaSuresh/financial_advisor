"""Shared contracts ordered by the application's execution flow."""

import json
from collections.abc import Collection
from datetime import datetime
from enum import StrEnum
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field

from financial_advisor.config import MAX_EVIDENCE_TEXT_LENGTH


class RetrievalChannel(StrEnum):
    VECTOR = "vector"
    KEYWORD = "keyword"
    WEB = "web"


class ConversationStatus(StrEnum):
    ACTIVE = "active"
    RESOLVED = "resolved"
    ESCALATED = "escalated"


class ClientTask(BaseModel):
    """One opening or recommendation-review request sent by the Advisor."""

    advisor_response_json: str | None = None
    follow_up_count: int = Field(default=0, ge=0)


class ClientResult(BaseModel):
    """One Client question or a short acceptance response."""

    action: Literal["question", "accept"]
    message: str = Field(min_length=1, max_length=2_000)


class DocumentChunk(BaseModel):
    """One source passage with the metadata required by retrieval."""

    canonical_candidate_id: str
    source_title: str
    publisher: str
    source_url: str
    topics: tuple[str, ...]
    section_position: int
    chunk_position: int
    token_count: int
    text: str

    @property
    def embedding_text(self) -> str:
        return f"Source: {self.source_title}\n\n{self.text}"


class Evidence(BaseModel):
    """One server-selected source passage available to the Analyst."""

    evidence_id: UUID
    title: str = Field(min_length=1, max_length=500)
    publisher: str = Field(min_length=1, max_length=300)
    url: str | None = None
    text: str = Field(min_length=1, max_length=MAX_EVIDENCE_TEXT_LENGTH)
    source: RetrievalChannel


class RetrievedEvidenceCandidate(DocumentChunk):
    """One source passage awaiting deterministic evidence selection."""

    retrieval_channel: RetrievalChannel
    rank: int = Field(ge=1)
    cosine_distance: float | None = Field(default=None, ge=0)
    bm25_score: float | None = None
    vector: list[float] = Field(min_length=1, exclude=True)


class ResearchTask(BaseModel):
    """One complete research request created by the Advisor."""

    question: str = Field(min_length=1, max_length=2_000)
    retrieval_paths: list[Literal["local_hybrid", "web"]] = Field(
        min_length=1,
        max_length=2,
    )


class CitedText(BaseModel):
    """Text grounded in one or more selected evidence records."""

    text: str = Field(min_length=1, max_length=2_000)
    evidence_ids: list[UUID] = Field(min_length=1)

    def uses_available_evidence(self, available_evidence_ids: Collection[UUID]) -> bool:
        """Return whether every cited evidence ID is available."""

        return all(evidence_id in available_evidence_ids for evidence_id in self.evidence_ids)


class RecommendationDraft(BaseModel):
    """Advisor-written content awaiting deterministic evidence validation."""

    summary: CitedText
    options: dict[str, CitedText] = Field(min_length=1)
    assumptions: list[str] = Field(min_length=1)
    risks: list[CitedText] = Field(min_length=1)
    next_steps: list[str] = Field(min_length=1)


RECOMMENDATION_DRAFT_SCHEMA = json.dumps(RecommendationDraft.model_json_schema())


class ResearchBrief(BaseModel):
    """Analyst result assembled from supported claims and selected evidence."""

    findings: list[CitedText] = Field(default_factory=list)
    scenario_comparisons: list[CitedText] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class EvidenceCitation(BaseModel):
    """Server-created citation metadata shown in the UI."""

    evidence_id: UUID
    title: str = Field(min_length=1, max_length=500)
    publisher: str = Field(min_length=1, max_length=300)
    url: str | None = None


class Recommendation(RecommendationDraft):
    """Advisor response assembled from supported claims and trusted citations."""

    citations: list[EvidenceCitation] = Field(min_length=1)
    limitations: list[str] = Field(default_factory=list)


class SessionSummary(BaseModel):
    """Small persisted-session record used by conversation history."""

    session_id: str
    client_question: str | None
    updated_at: datetime
    status: ConversationStatus


class SessionSnapshot(BaseModel):
    """Persisted conversation state and progress used by the UI."""

    session_id: str
    status: ConversationStatus
    client_results: list[ClientResult]
    recommendations: list[Recommendation]
    progress_updates: list[str]
    state: dict[str, object]
    events: list[dict[str, object]]
