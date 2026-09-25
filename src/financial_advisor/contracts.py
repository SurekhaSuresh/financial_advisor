"""Shared contracts for agents and deterministic retrieval."""

from __future__ import annotations

from collections.abc import Iterable
from decimal import Decimal
from enum import StrEnum
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, HttpUrl, model_validator

from financial_advisor.config import MAX_EVIDENCE_TEXT_LENGTH


def _uses_available_evidence(referenced_ids: Iterable[UUID], available_ids: set[UUID]) -> bool:
    """Return whether every referenced evidence ID is available."""

    return set(referenced_ids).issubset(available_ids)


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
    vector: list[float] | None = Field(default=None, exclude=True)

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

    task_id: UUID
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
    """A bounded initial or refinement task created by the Advisor."""

    task_id: UUID = Field(default_factory=uuid4)
    question: str = Field(min_length=1, max_length=2_000)
    client_profile: ClientProfile
    retrieval_paths: list[RetrievalPath] = Field(min_length=1, max_length=2)
    previous_brief: ResearchBrief | None = None
    material_gaps: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_refinement(self) -> ResearchTask:
        """Keep initial research and refinement inputs unambiguous."""

        if self.previous_brief is None and self.material_gaps:
            raise ValueError("Material gaps require a previous research brief.")
        if self.previous_brief is not None and not self.material_gaps:
            raise ValueError("A refinement requires at least one material gap.")
        if len(self.retrieval_paths) != len(set(self.retrieval_paths)):
            raise ValueError("A research task may select each retrieval path only once.")
        return self


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


class ClientTask(BaseModel):
    """One opening or recommendation-review request sent by the Advisor."""

    profile: ClientProfile
    advisor_response: Recommendation | None = None
    follow_up_count: int = Field(default=0, ge=0)


class ClientAction(StrEnum):
    """The two results a simulated Client may return to the Advisor."""

    QUESTION = "question"
    ACCEPT = "accept"


class ClientResult(BaseModel):
    """One Client question or a short acceptance response."""

    action: ClientAction
    message: str = Field(min_length=1, max_length=2_000)


def create_research_brief(
    task: ResearchTask,
    findings: list[Finding],
    scenario_comparisons: list[ScenarioComparison],
    evidence: list[Evidence],
    limitations: list[str],
) -> ResearchBrief:
    """Keep supported Analyst claims and assemble one trusted brief."""

    available_ids = {item.evidence_id for item in evidence}
    supported_findings = [
        finding
        for finding in findings
        if _uses_available_evidence(finding.evidence_ids, available_ids)
    ]
    if not supported_findings:
        raise ValueError("Research produced no supported findings.")

    supported_comparisons = [
        comparison
        for comparison in scenario_comparisons
        if _uses_available_evidence(comparison.evidence_ids, available_ids)
    ]
    return ResearchBrief(
        task_id=task.task_id,
        findings=supported_findings,
        scenario_comparisons=supported_comparisons,
        evidence=evidence,
        limitations=limitations,
    )


def create_recommendation(
    brief: ResearchBrief,
    summary: CitedText,
    options: list[RecommendationOption],
    assumptions: list[str],
    risks: list[CitedText],
    next_steps: list[str],
) -> Recommendation:
    """Keep supported Advisor claims and attach server-created citations."""

    evidence_by_id = {item.evidence_id: item for item in brief.evidence}
    available_ids = set(evidence_by_id)
    if not _uses_available_evidence(summary.evidence_ids, available_ids):
        raise ValueError("Recommendation summary is not supported by the research brief.")

    supported_options = [
        option
        for option in options
        if _uses_available_evidence(option.description.evidence_ids, available_ids)
    ]
    supported_risks = [
        risk for risk in risks if _uses_available_evidence(risk.evidence_ids, available_ids)
    ]
    if not supported_options or not supported_risks:
        raise ValueError("Recommendation has no supported options or risks.")

    claims = [summary, *(option.description for option in supported_options), *supported_risks]
    cited_ids = list(
        dict.fromkeys(evidence_id for claim in claims for evidence_id in claim.evidence_ids)
    )
    return Recommendation(
        summary=summary,
        options=supported_options,
        assumptions=assumptions,
        risks=supported_risks,
        next_steps=next_steps,
        citations=[
            EvidenceCitation(
                evidence_id=evidence_id,
                title=evidence_by_id[evidence_id].title,
                publisher=evidence_by_id[evidence_id].publisher,
                url=evidence_by_id[evidence_id].url,
            )
            for evidence_id in cited_ids
        ],
        limitations=brief.limitations,
    )
