"""Shared contracts for agents and deterministic retrieval."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, HttpUrl, field_validator

from financial_advisor.config import MAX_EVIDENCE_TEXT_LENGTH

Embed = Callable[[Sequence[str]], list[list[float]]]
Rerank = Callable[[str, Sequence[str]], list[float]]


class ParsedDocumentSection(BaseModel):
    """One document section before token chunking."""

    heading_path: tuple[str, ...]
    text: str


class DocumentChunk(BaseModel):
    """One source passage with the metadata required by retrieval."""

    chunk_id: str
    canonical_candidate_id: str
    source_id: str
    source_title: str
    publisher: str
    source_url: str
    topics: tuple[str, ...]
    heading_path: tuple[str, ...]
    section_position: int
    chunk_position: int
    token_count: int
    text: str

    @property
    def embedding_text(self) -> str:
        section_heading = " > ".join(self.heading_path)
        return f"Source: {self.source_title}\nSection: {section_heading}\n\n{self.text}"


class ClientProfile(BaseModel):
    """Synthetic client facts used by the Client and Advisor."""

    name: str = Field(min_length=1, max_length=200)
    age: int = Field(ge=18, le=120)
    risk_tolerance: Literal["low", "moderate", "high"]
    emergency_fund_months: int = Field(ge=0, le=60)
    retirement_savings: Decimal = Field(ge=0)
    brokerage_savings: Decimal = Field(ge=0)
    student_loan_balance: Decimal = Field(ge=0)
    student_loan_rate_percent: Decimal = Field(ge=0, le=100)
    primary_goal: str = Field(min_length=1, max_length=500)
    goal_time_horizon_years: int = Field(ge=1, le=100)


class RetrievalChannel(StrEnum):
    """The source and method that produced an evidence candidate."""

    VECTOR = "vector"
    KEYWORD = "keyword"
    WEB = "web"


class Evidence(BaseModel):
    """One server-selected source passage available to the Analyst."""

    evidence_id: UUID
    title: str = Field(min_length=1, max_length=500)
    publisher: str = Field(min_length=1, max_length=300)
    url: HttpUrl | None = None
    text: str = Field(min_length=1, max_length=MAX_EVIDENCE_TEXT_LENGTH)
    source: RetrievalChannel


class RetrievedEvidenceCandidate(BaseModel):
    """One source passage awaiting deterministic evidence selection."""

    chunk_id: str
    canonical_candidate_id: str
    source_id: str
    source_title: str
    publisher: str
    source_url: HttpUrl
    topics: list[str]
    heading_path: list[str]
    section_position: int = Field(ge=1)
    chunk_position: int = Field(ge=1)
    token_count: int = Field(gt=0)
    text: str
    retrieval_channel: RetrievalChannel
    rank: int = Field(ge=1)
    cosine_distance: float | None = Field(default=None, ge=0)
    bm25_score: float | None = None
    vector: list[float] = Field(min_length=1, exclude=True)

    @property
    def embedding_text(self) -> str:
        return f"Source: {self.source_title}\n\n{self.text}"


class RetrievalResult(BaseModel):
    """Evidence and deterministic limitations returned to the Analyst."""

    evidence: list[Evidence] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class Finding(BaseModel):
    """An Analyst statement grounded in selected evidence."""

    text: str = Field(min_length=1, max_length=2_000)
    evidence_ids: list[UUID] = Field(min_length=1)


class ScenarioComparison(BaseModel):
    """One evidence-backed option comparison prepared for the Advisor."""

    scenario: str = Field(min_length=1, max_length=300)
    summary: str = Field(min_length=1, max_length=2_000)
    benefits: list[str] = Field(min_length=1)
    tradeoffs: list[str] = Field(min_length=1)
    evidence_ids: list[UUID] = Field(min_length=1)


class ResearchBrief(BaseModel):
    """Analyst result assembled from supported claims and selected evidence."""

    task_id: UUID = Field(default_factory=uuid4)
    findings: list[Finding] = Field(min_length=1)
    scenario_comparisons: list[ScenarioComparison] = Field(default_factory=list)
    evidence: list[Evidence] = Field(min_length=1)
    limitations: list[str] = Field(default_factory=list)


class ResearchResult(BaseModel):
    """The outcome returned by the Analyst AgentTool."""

    success: bool = False
    brief: ResearchBrief | None = None


class RetrievalPath(StrEnum):
    """A deterministic evidence source selected by the Advisor."""

    LOCAL_HYBRID = "local_hybrid"
    WEB = "web"


class ResearchTask(BaseModel):
    """One complete research request created by the Advisor."""

    question: str = Field(min_length=1, max_length=2_000)
    client_profile_json: str
    retrieval_paths: list[str] = Field(min_length=1, max_length=2)

    @field_validator("retrieval_paths")
    @classmethod
    def require_unique_retrieval_paths(
        cls,
        retrieval_paths: list[str],
    ) -> list[str]:
        """Require supported retrieval paths without duplicates."""

        supported_paths = {path.value for path in RetrievalPath}
        if not set(retrieval_paths).issubset(supported_paths):
            raise ValueError("A research task contains an unsupported retrieval path.")
        if len(retrieval_paths) != len(set(retrieval_paths)):
            raise ValueError("A research task may select each retrieval path only once.")
        return retrieval_paths


class CitedText(BaseModel):
    """Client-facing text with the evidence IDs supporting it."""

    text: str = Field(min_length=1, max_length=2_000)
    evidence_ids: list[UUID] = Field(min_length=1)


class EvidenceCitation(BaseModel):
    """Server-created citation metadata shown in the UI."""

    evidence_id: UUID
    title: str = Field(min_length=1, max_length=500)
    publisher: str = Field(min_length=1, max_length=300)
    url: HttpUrl | None = None


class RecommendationOption(BaseModel):
    """One client-facing option and its evidence-backed explanation."""

    title: str = Field(min_length=1, max_length=300)
    description: CitedText


class Recommendation(BaseModel):
    """Advisor response assembled from supported claims and trusted citations."""

    recommendation_id: UUID = Field(default_factory=uuid4)
    summary: CitedText
    options: list[RecommendationOption] = Field(min_length=1)
    assumptions: list[str] = Field(min_length=1)
    risks: list[CitedText] = Field(min_length=1)
    next_steps: list[str] = Field(min_length=1)
    citations: list[EvidenceCitation] = Field(min_length=1)
    limitations: list[str] = Field(default_factory=list)


class ConversationStatus(StrEnum):
    """The lifecycle state of a financial-advisor conversation."""

    ACTIVE = "active"
    RESOLVED = "resolved"
    ESCALATED = "escalated"


class ConversationResult(BaseModel):
    """The terminal result returned to the application."""

    session_id: str
    status: ConversationStatus
    recommendation: Recommendation | None = None


class ConversationStarted(BaseModel):
    """Identifier returned after a conversation is scheduled."""

    session_id: str
    status: ConversationStatus


class PersistedSessionEvent(BaseModel):
    """One ordered ADK event exposed for live progress and replay."""

    sequence: int = Field(ge=1)
    event_id: str
    timestamp: datetime
    author: str
    event: dict[str, object]


class SessionSummary(BaseModel):
    """Small persisted-session record used by conversation history."""

    session_id: str
    client_question: str | None
    updated_at: datetime
    status: ConversationStatus


class ClientTask(BaseModel):
    """One opening or recommendation-review request sent by the Advisor."""

    client_profile_json: str
    advisor_response_json: str | None = None
    follow_up_count: int = Field(default=0, ge=0)


class ClientAction(StrEnum):
    """The two results a simulated Client may return to the Advisor."""

    QUESTION = "question"
    ACCEPT = "accept"


class ClientResult(BaseModel):
    """One Client question or a short acceptance response."""

    action: ClientAction
    message: str = Field(min_length=1, max_length=2_000)


class SessionSnapshot(BaseModel):
    """Persisted conversation state and progress used by the UI."""

    session_id: str
    client_profile: ClientProfile
    status: ConversationStatus
    updated_at: datetime
    client_results: list[ClientResult]
    recommendations: list[Recommendation]
    progress_updates: list[str]
    state: dict[str, object]
    events: list[PersistedSessionEvent]
