"""Typed contracts for Advisor planning and client-facing synthesis."""

import re
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from financial_advisor.domain import (
    AdvisorDecisionType,
    AdvisorResearchPlan,
    AnalystProfileContext,
    ApprovedDomainId,
    CitedText,
    ClaimIntegrityWarning,
    ClientMessage,
    EvidenceCitation,
    Finding,
    Recommendation,
    RecommendationOption,
    ResearchBrief,
    ResearchRefinement,
    ScenarioComparison,
    WebResearchMode,
    WebResearchScope,
)

_INTERNAL_EVIDENCE_ID_LIST = re.compile(
    r"\s*\[(?:[0-9a-fA-F]{8}-[0-9a-fA-F-]{27,})(?:\s*,\s*[0-9a-fA-F]{8}-[0-9a-fA-F-]{27,})*\]"
)


class AdvisorClaimIntegrityError(ValueError):
    """Raised when filtering leaves no safe complete Advisor response."""


class AdvisorResearchPlanDraft(BaseModel):
    """Structured Advisor decision before the server creates a trusted plan."""

    model_config = ConfigDict(frozen=True)

    decision_type: AdvisorDecisionType
    summary: str = Field(min_length=1, max_length=1_000)
    progress_summary: str = Field(
        default="I’m reviewing the information needed to guide the next step.",
        min_length=1,
        max_length=240,
    )
    escalation_summary: str | None = Field(default=None, min_length=1, max_length=600)
    research_question: str | None = Field(default=None, min_length=1, max_length=2_000)
    include_local_hybrid: bool = True
    web_scopes: tuple[WebResearchScope, ...] = Field(default=(), max_length=2)
    refinement: ResearchRefinement | None = None

    @model_validator(mode="after")
    def validate_research_decision(self) -> "AdvisorResearchPlanDraft":
        """Permit retrieval details only when the Advisor delegates research."""

        if self.decision_type is AdvisorDecisionType.DELEGATE_RESEARCH:
            if self.research_question is None:
                raise ValueError("Delegated research requires a research question.")
            if not self.include_local_hybrid and not self.web_scopes:
                raise ValueError("Delegated research requires at least one enabled source.")
            if self.escalation_summary is not None:
                raise ValueError("Only escalation may include an escalation summary.")
            return self
        if self.decision_type is AdvisorDecisionType.ESCALATE:
            if self.research_question is not None or self.web_scopes or self.refinement is not None:
                raise ValueError("Escalation may not include a research plan.")
            return self
        if (
            self.research_question is not None
            or self.web_scopes
            or self.refinement is not None
            or self.escalation_summary is not None
        ):
            raise ValueError("Only delegated research may include a research plan.")
        return self


class AvailableResearchOptions(BaseModel):
    """Server-owned retrieval capabilities the Advisor may authorize for one plan."""

    model_config = ConfigDict(frozen=True)

    local_hybrid: bool = True
    authoritative_web: bool = True
    broad_web: bool = True

    @model_validator(mode="after")
    def validate_at_least_one_option(self) -> "AvailableResearchOptions":
        """Avoid asking the Advisor to plan research when no operation is available."""

        if not any((self.local_hybrid, self.authoritative_web, self.broad_web)):
            raise ValueError("At least one research option must be available.")
        return self


class AdvisorPlanningRequest(BaseModel):
    """Trusted session facts supplied to the Advisor when it plans research."""

    model_config = ConfigDict(frozen=True)

    client_message: ClientMessage
    client_context: AnalystProfileContext
    chat_history: list["AdvisorChatHistoryTurn"] = Field(default_factory=list, max_length=4)
    available_state: "AdvisorStateContext | None" = None
    prior_plan: AdvisorResearchPlan | None = None
    prior_research_limitations: list[str] = Field(default_factory=list)
    prior_plan_attempt_number: int = Field(default=0, ge=0, le=2)
    max_research_plans_per_client_question: int = Field(default=2, ge=1, le=2)
    available_research: AvailableResearchOptions = Field(default_factory=AvailableResearchOptions)
    approved_authoritative_domains: dict[ApprovedDomainId, tuple[str, ...]] = Field(
        default_factory=dict
    )


class AdvisorChatHistoryTurn(BaseModel):
    """A bounded successful prior Client/Advisor exchange for conversation context."""

    model_config = ConfigDict(frozen=True)

    client_question: str = Field(min_length=1, max_length=2_000)
    advisor_recommendation: Recommendation


class AdvisorStateContext(BaseModel):
    """Validated research state the Advisor may consider for a direct answer."""

    model_config = ConfigDict(frozen=True)

    findings: list[Finding] = Field(default_factory=list)
    scenario_comparison: list[ScenarioComparison] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    evidence_ids: list[UUID] = Field(default_factory=list)


class AdvisorResponseRequest(BaseModel):
    """Validated brief and Client question supplied to the response-writing Advisor."""

    model_config = ConfigDict(frozen=True)

    research_brief: ResearchBrief
    client_message: ClientMessage
    client_context: AnalystProfileContext
    chat_history: list[AdvisorChatHistoryTurn] = Field(default_factory=list, max_length=2)


def create_research_plan(
    request: AdvisorPlanningRequest,
    draft: AdvisorResearchPlanDraft,
) -> AdvisorResearchPlan:
    """Create a server-owned plan from a validated delegated-research draft."""

    if draft.decision_type is not AdvisorDecisionType.DELEGATE_RESEARCH:
        raise ValueError("Only a delegated-research decision can create a research plan.")
    assert draft.research_question is not None
    if draft.include_local_hybrid and not request.available_research.local_hybrid:
        raise ValueError("Advisor selected unavailable local_hybrid research.")
    for scope in draft.web_scopes:
        if (
            scope.mode is WebResearchMode.AUTHORITATIVE_DOMAIN
            and not request.available_research.authoritative_web
        ):
            raise ValueError("Advisor selected unavailable authoritative_web research.")
        if scope.mode is WebResearchMode.AUTHORITATIVE_DOMAIN and any(
            domain_id not in request.approved_authoritative_domains
            for domain_id in scope.approved_domain_ids
        ):
            raise ValueError("Advisor selected an unapproved authoritative domain ID.")
        if scope.mode is WebResearchMode.BROAD_WEB and not request.available_research.broad_web:
            raise ValueError("Advisor selected unavailable broad_web research.")
    return AdvisorResearchPlan(
        session_id=request.client_message.session_id,
        client_message_id=request.client_message.message_id,
        attempt_number=request.prior_plan_attempt_number + 1,
        question=draft.research_question,
        include_local_hybrid=draft.include_local_hybrid,
        web_scopes=draft.web_scopes,
        rationale=draft.summary,
        refinement=draft.refinement,
    )


class AdvisorRecommendationDraft(BaseModel):
    """Client-facing Advisor wording before server evidence and limitations attach."""

    model_config = ConfigDict(frozen=True)

    summary: CitedText
    options: list[RecommendationOption] = Field(min_length=1, max_length=5)
    rationale: list[CitedText] = Field(min_length=1, max_length=5)
    assumptions: list[str] = Field(min_length=1, max_length=4)
    risks: list[CitedText] = Field(min_length=1, max_length=5)
    next_steps: list[str] = Field(min_length=1, max_length=4)

    @model_validator(mode="after")
    def validate_cited_point_budget(self) -> "AdvisorRecommendationDraft":
        """Keep a richer recommendation focused rather than repetitive."""

        cited_point_count = len(self.options) + len(self.rationale) + len(self.risks)
        if cited_point_count > 10:
            raise ValueError(
                "Advisor recommendations may contain at most 10 cited supporting points."
            )
        return self


def create_recommendation(
    session_id: UUID,
    brief: ResearchBrief,
    draft: AdvisorRecommendationDraft,
) -> Recommendation:
    """Attach trusted evidence IDs and deterministic source limitations to advice wording."""

    selected_evidence_ids = {evidence.evidence_id for evidence in brief.evidence}

    def is_valid(cited_text: CitedText) -> bool:
        return set(cited_text.evidence_ids).issubset(selected_evidence_ids)

    def clean_cited_text(cited_text: CitedText) -> CitedText:
        """Remove internal UUID citation markers; the UI renders validated citations itself."""

        cleaned_text = _INTERNAL_EVIDENCE_ID_LIST.sub("", cited_text.text).strip()
        if not cleaned_text:
            raise AdvisorClaimIntegrityError(
                "Advisor claim contained only internal evidence identifiers."
            )
        return cited_text.model_copy(update={"text": cleaned_text})

    if not is_valid(draft.summary):
        raise AdvisorClaimIntegrityError("Advisor summary cited evidence outside the selected set.")
    valid_options = [
        option.model_copy(update={"description": clean_cited_text(option.description)})
        for option in draft.options
        if is_valid(option.description)
    ]
    valid_rationale = [clean_cited_text(item) for item in draft.rationale if is_valid(item)]
    valid_risks = [clean_cited_text(item) for item in draft.risks if is_valid(item)]
    warnings: list[ClaimIntegrityWarning] = []
    for claim_type, original_count, valid_count in (
        ("option", len(draft.options), len(valid_options)),
        ("rationale", len(draft.rationale), len(valid_rationale)),
        ("risk", len(draft.risks), len(valid_risks)),
    ):
        if removed_count := original_count - valid_count:
            warnings.append(
                ClaimIntegrityWarning(
                    actor="advisor", claim_type=claim_type, removed_count=removed_count
                )
            )
    if not valid_options or not valid_rationale or not valid_risks:
        raise AdvisorClaimIntegrityError(
            "Advisor response lacked a required evidence-backed section after validation."
        )

    claim_evidence_ids = [*draft.summary.evidence_ids]
    claim_evidence_ids.extend(
        evidence_id for option in valid_options for evidence_id in option.description.evidence_ids
    )
    claim_evidence_ids.extend(
        evidence_id for rationale in valid_rationale for evidence_id in rationale.evidence_ids
    )
    claim_evidence_ids.extend(
        evidence_id for risk in valid_risks for evidence_id in risk.evidence_ids
    )
    evidence_by_id = {evidence.evidence_id: evidence for evidence in brief.evidence}
    unique_claim_ids = list(dict.fromkeys(claim_evidence_ids))
    citations = [
        EvidenceCitation(
            evidence_id=evidence_by_id[evidence_id].evidence_id,
            title=evidence_by_id[evidence_id].title,
            publisher=evidence_by_id[evidence_id].publisher,
            url=evidence_by_id[evidence_id].url,
        )
        for evidence_id in unique_claim_ids
    ]
    return Recommendation(
        session_id=session_id,
        summary=clean_cited_text(draft.summary),
        options=valid_options,
        rationale=valid_rationale,
        assumptions=draft.assumptions,
        risks=valid_risks,
        next_steps=draft.next_steps,
        evidence_ids=unique_claim_ids,
        citations=citations,
        educational_disclaimer=(
            "Educational information only; not individualized financial advice."
        ),
        limitations=brief.caveats,
        integrity_warnings=warnings,
    )
