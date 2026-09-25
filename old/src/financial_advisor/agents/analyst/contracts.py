"""Validated boundary between Analyst reasoning and the workflow engine."""

from pydantic import BaseModel, ConfigDict, Field, model_validator

from financial_advisor.domain import (
    AdvisorResearchPlan,
    AnalystTask,
    Calculation,
    ClaimIntegrityWarning,
    Evidence,
    Finding,
    Recommendation,
    ResearchBrief,
    ScenarioComparison,
)


class AnalystClaimIntegrityError(ValueError):
    """Raised when filtering leaves no evidence-backed Analyst findings."""


class AnalystConversationTurn(BaseModel):
    """A completed Client/Advisor exchange used only to resolve follow-up references."""

    model_config = ConfigDict(frozen=True)

    client_question: str = Field(min_length=1, max_length=2_000)
    advisor_recommendation: Recommendation


class PriorValidatedResearchTurn(BaseModel):
    """One successful prior research turn, excluding failed execution traces."""

    model_config = ConfigDict(frozen=True)

    client_question: str = Field(min_length=1, max_length=2_000)
    research_plan: AdvisorResearchPlan
    research_brief: ResearchBrief


class AnalystResearchRequest(BaseModel):
    """The task and server-selected evidence supplied to the Analyst."""

    model_config = ConfigDict(frozen=True)

    task: AnalystTask
    evidence: list[Evidence] = Field(min_length=1)
    evidence_limitations: list[str] = Field(default_factory=list)
    prior_chat_history: list[AnalystConversationTurn] = Field(default_factory=list, max_length=2)
    prior_validated_research_turns: list[PriorValidatedResearchTurn] = Field(
        default_factory=list, max_length=2
    )

    @model_validator(mode="after")
    def validate_unique_evidence(self) -> "AnalystResearchRequest":
        """Prevent an ambiguous Analyst evidence set before model invocation."""

        evidence_ids = [item.evidence_id for item in self.evidence]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("Analyst evidence IDs must be unique.")
        return self


class AnalystResearchDraft(BaseModel):
    """Structured reasoning returned by the Analyst before server-side assembly."""

    model_config = ConfigDict(frozen=True)

    findings: list[Finding] = Field(min_length=1, max_length=10)
    scenario_comparison: list[ScenarioComparison] = Field(default_factory=list)
    calculations: list[Calculation] = Field(default_factory=list)
    caveats: list[str] = Field(min_length=1)


def create_research_brief(
    request: AnalystResearchRequest, draft: AnalystResearchDraft
) -> ResearchBrief:
    """Attach server-selected evidence to validated Analyst reasoning.

    The model can cite evidence IDs but cannot invent, omit, or replace the
    evidence objects selected by the retrieval pipeline.
    """

    selected_evidence_ids = {evidence.evidence_id for evidence in request.evidence}
    valid_findings = [
        finding
        for finding in draft.findings
        if set(finding.evidence_ids).issubset(selected_evidence_ids)
    ]
    valid_comparisons = [
        comparison
        for comparison in draft.scenario_comparison
        if set(comparison.evidence_ids).issubset(selected_evidence_ids)
    ]
    warnings: list[ClaimIntegrityWarning] = []
    removed_finding_count = len(draft.findings) - len(valid_findings)
    if removed_finding_count:
        warnings.append(
            ClaimIntegrityWarning(
                actor="analyst",
                claim_type="finding",
                removed_count=removed_finding_count,
            )
        )
    removed_comparison_count = len(draft.scenario_comparison) - len(valid_comparisons)
    if removed_comparison_count:
        warnings.append(
            ClaimIntegrityWarning(
                actor="analyst",
                claim_type="scenario_comparison",
                removed_count=removed_comparison_count,
            )
        )
    if not valid_findings:
        raise AnalystClaimIntegrityError(
            "No evidence-backed Analyst findings remained after validation."
        )
    return ResearchBrief(
        task_id=request.task.task_id,
        findings=valid_findings,
        scenario_comparison=valid_comparisons,
        calculations=draft.calculations,
        evidence=request.evidence,
        caveats=[*request.evidence_limitations, *draft.caveats],
        integrity_warnings=warnings,
    )
