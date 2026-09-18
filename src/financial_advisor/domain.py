"""Validated contracts shared by Financial Advisor components."""

from datetime import UTC, date, datetime
from decimal import Decimal
from enum import StrEnum
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator

type Actor = Literal["client", "advisor", "analyst", "workflow_engine", "system"]


class RiskTolerance(StrEnum):
    """Client-stated comfort with investment risk."""

    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


class SessionState(StrEnum):
    """States managed by the deterministic workflow engine."""

    NEW = "new"
    CLIENT_OPENS = "client_opens"
    ADVISOR_ASSESSES = "advisor_assesses"
    ADVISOR_DELEGATES = "advisor_delegates"
    ANALYST_RESEARCHES = "analyst_researches"
    ADVISOR_PROPOSES = "advisor_proposes"
    CLIENT_REVIEWS = "client_reviews"
    RESOLVED = "resolved"
    ESCALATED = "escalated"


class AdvisorDecisionType(StrEnum):
    """Actions the Advisor may request from the workflow engine."""

    ANSWER_FROM_STATE = "answer_from_state"
    DELEGATE_RESEARCH = "delegate_research"
    ESCALATE = "escalate"


class ResearchTool(StrEnum):
    """Deterministic evidence operations an Advisor may authorize."""

    LOCAL_HYBRID = "local_hybrid"
    LIVE_WEB = "live_web"


class WebResearchMode(StrEnum):
    """Scope policy for one live web research operation."""

    AUTHORITATIVE_DOMAIN = "authoritative_domain"
    BROAD_WEB = "broad_web"


# Name retained for the low-level web-provider request contract. Advisor and
# Analyst task contracts use WebResearchScope through AdvisorResearchPlan.
SearchMode = WebResearchMode


class ApprovedDomainId(StrEnum):
    """Stable server-owned identifiers for reviewed authoritative sources."""

    INVESTOR_GOV = "investor_gov"
    SEC_GOV = "sec_gov"
    FINRA_ORG = "finra_org"
    CONSUMERFINANCE_GOV = "consumerfinance_gov"
    FDIC_GOV = "fdic_gov"
    IRS_GOV = "irs_gov"
    TREASURY_GOV = "treasury_gov"


class WebResearchScope(BaseModel):
    """One Advisor-authorized live-web operation within a research plan."""

    mode: WebResearchMode
    approved_domain_ids: tuple[ApprovedDomainId, ...] = ()

    @model_validator(mode="after")
    def validate_domain_policy(self) -> "WebResearchScope":
        """Keep the authoritative and broad-web source policies disjoint."""

        if self.mode is WebResearchMode.AUTHORITATIVE_DOMAIN and not self.approved_domain_ids:
            raise ValueError("Authoritative-domain research requires approved domain IDs.")
        if self.mode is WebResearchMode.BROAD_WEB and self.approved_domain_ids:
            raise ValueError("Broad-web research cannot include approved domain IDs.")
        if len(set(self.approved_domain_ids)) != len(self.approved_domain_ids):
            raise ValueError("Approved domain IDs must be unique within a web scope.")
        return self


class AdvisorResearchPlan(BaseModel):
    """Trusted, persisted evidence plan created from a validated Advisor decision."""

    plan_id: UUID = Field(default_factory=uuid4)
    session_id: UUID
    client_message_id: UUID
    attempt_number: int = Field(ge=1, le=2)
    question: str = Field(min_length=1, max_length=2_000)
    include_local_hybrid: bool = True
    web_scopes: tuple[WebResearchScope, ...] = Field(default=(), max_length=2)
    rationale: str = Field(min_length=1, max_length=1_000)
    refinement: "ResearchRefinement | None" = None

    @model_validator(mode="after")
    def validate_enabled_tools(self) -> "AdvisorResearchPlan":
        """Require at least one authorized retrieval operation in a plan."""

        if not self.include_local_hybrid and not self.web_scopes:
            raise ValueError("An Advisor research plan must enable at least one research tool.")
        modes = [scope.mode for scope in self.web_scopes]
        if len(set(modes)) != len(modes):
            raise ValueError("An Advisor research plan may contain one scope of each web mode.")
        return self


class ResearchRefinement(BaseModel):
    """Persist why a bounded second plan was needed after a valid first brief."""

    model_config = ConfigDict(frozen=True)

    material_gap: str = Field(min_length=1, max_length=1_000)
    why_current_evidence_cannot_answer_gap: str = Field(min_length=1, max_length=1_000)
    authorized_retrieval_changes: str = Field(min_length=1, max_length=1_000)


class EventType(StrEnum):
    """Safe-to-display events recorded for a session."""

    SESSION_STARTED = "session_started"
    STATE_TRANSITIONED = "state_transitioned"
    CLIENT_MESSAGE_CREATED = "client_message_created"
    ADVISOR_DECISION_CREATED = "advisor_decision_created"
    ADVISOR_PLANNER_RATIONALE_CREATED = "advisor_planner_rationale_created"
    CLIENT_PROGRESS_CREATED = "client_progress_created"
    ANALYST_TASK_CREATED = "analyst_task_created"
    TOOL_STARTED = "tool_started"
    TOOL_COMPLETED = "tool_completed"
    TOOL_FAILED = "tool_failed"
    EVIDENCE_SELECTED = "evidence_selected"
    CLAIM_FILTERED = "claim_filtered"
    RECOMMENDATION_CREATED = "recommendation_created"
    CLIENT_REVIEW_CREATED = "client_review_created"
    SESSION_RESOLVED = "session_resolved"
    SESSION_ESCALATED = "session_escalated"


class ErrorCode(StrEnum):
    """Safe, client-displayable categories for expected application failures."""

    VALIDATION_FAILED = "validation_failed"
    INVALID_STATE_TRANSITION = "invalid_state_transition"
    FOLLOW_UP_LIMIT_REACHED = "follow_up_limit_reached"
    RESEARCH_UNAVAILABLE = "research_unavailable"
    MODEL_UNAVAILABLE = "model_unavailable"
    INTERNAL_ERROR = "internal_error"


class ClientProfile(BaseModel):
    """Synthetic client facts available to the Client and Advisor agents."""

    client_id: str = Field(min_length=1, max_length=100)
    name: str = Field(min_length=1, max_length=100)
    age: int = Field(ge=18, le=120)
    risk_tolerance: RiskTolerance
    emergency_fund_months: int = Field(ge=0, le=120)
    retirement_savings: Decimal = Field(ge=0, max_digits=14, decimal_places=2)
    brokerage_savings: Decimal = Field(ge=0, max_digits=14, decimal_places=2)
    student_loan_balance: Decimal = Field(ge=0, max_digits=14, decimal_places=2)
    student_loan_rate_percent: Decimal = Field(ge=0, le=100, max_digits=5, decimal_places=3)
    primary_goal: str = Field(min_length=1, max_length=300)
    goal_time_horizon_years: int = Field(ge=0, le=100)


class AnalystProfileContext(BaseModel):
    """Identity-free financial context approved for Analyst research."""

    model_config = ConfigDict(frozen=True)

    age: int = Field(ge=18, le=120)
    risk_tolerance: RiskTolerance
    emergency_fund_months: int = Field(ge=0, le=120)
    retirement_savings: Decimal = Field(ge=0, max_digits=14, decimal_places=2)
    brokerage_savings: Decimal = Field(ge=0, max_digits=14, decimal_places=2)
    student_loan_balance: Decimal = Field(ge=0, max_digits=14, decimal_places=2)
    student_loan_rate_percent: Decimal = Field(ge=0, le=100, max_digits=5, decimal_places=3)
    primary_goal: str = Field(min_length=1, max_length=300)
    goal_time_horizon_years: int = Field(ge=0, le=100)


def to_analyst_profile_context(profile: ClientProfile) -> AnalystProfileContext:
    """Create the approved Analyst context without client identity fields."""

    return AnalystProfileContext(
        age=profile.age,
        risk_tolerance=profile.risk_tolerance,
        emergency_fund_months=profile.emergency_fund_months,
        retirement_savings=profile.retirement_savings,
        brokerage_savings=profile.brokerage_savings,
        student_loan_balance=profile.student_loan_balance,
        student_loan_rate_percent=profile.student_loan_rate_percent,
        primary_goal=profile.primary_goal,
        goal_time_horizon_years=profile.goal_time_horizon_years,
    )


class ClientMessage(BaseModel):
    """A Client-to-Advisor message; profile context stays in workflow state."""

    session_id: UUID
    message_id: UUID = Field(default_factory=uuid4)
    text: str = Field(min_length=1, max_length=2_000)
    follow_up_number: int = Field(ge=0, le=2)


class AdvisorDecision(BaseModel):
    """The Advisor's requested next action, subject to workflow validation."""

    decision_type: AdvisorDecisionType
    summary: str = Field(min_length=1, max_length=1_000)
    progress_summary: str = Field(
        default="I’m reviewing the information needed to guide the next step.",
        min_length=1,
        max_length=240,
    )
    escalation_summary: str | None = Field(default=None, min_length=1, max_length=600)
    research_plan: AdvisorResearchPlan | None = None

    @model_validator(mode="after")
    def validate_research_fields(self) -> "AdvisorDecision":
        """Require research details only when the Advisor delegates research."""

        needs_research = self.decision_type is AdvisorDecisionType.DELEGATE_RESEARCH
        if needs_research and self.research_plan is None:
            raise ValueError("Delegated research requires a trusted research plan.")
        if not needs_research and self.research_plan is not None:
            raise ValueError("Only delegated research may include research details.")
        if (
            self.decision_type is not AdvisorDecisionType.ESCALATE
            and self.escalation_summary is not None
        ):
            raise ValueError("Only escalation may include an escalation summary.")
        return self


class AnalystTask(BaseModel):
    """A bounded research request created by the Advisor."""

    task_id: UUID = Field(default_factory=uuid4)
    session_id: UUID
    question: str = Field(min_length=1, max_length=2_000)
    research_plan: AdvisorResearchPlan
    client_context: AnalystProfileContext
    prohibited_actions: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_research_authorization(self) -> "AnalystTask":
        """Require a server-recorded plan rather than model-created tool authority."""

        if self.research_plan.session_id != self.session_id:
            raise ValueError("Analyst task plan session must match the task session.")
        if self.research_plan.question != self.question:
            raise ValueError("Analyst task question must match its research plan.")
        return self


class Evidence(BaseModel):
    """A cited chunk from the local knowledge store or live web research."""

    evidence_id: UUID = Field(default_factory=uuid4)
    title: str = Field(min_length=1, max_length=500)
    publisher: str = Field(min_length=1, max_length=300)
    url: HttpUrl | None = None
    published_at: date | None = None
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    excerpt: str = Field(min_length=1, max_length=1_000)
    source_type: Literal["vector_store", "web"]


class Finding(BaseModel):
    """An evidence-backed observation made by the Analyst."""

    statement: str = Field(min_length=1, max_length=1_000)
    evidence_ids: list[UUID] = Field(min_length=1)


class ClaimIntegrityWarning(BaseModel):
    """Safe audit metadata for a claim removed for invalid evidence references."""

    actor: Literal["analyst", "advisor"]
    claim_type: str = Field(min_length=1, max_length=100)
    removed_count: int = Field(ge=1)
    reason: Literal["evidence_outside_selected_set"] = "evidence_outside_selected_set"


class ScenarioComparison(BaseModel):
    """A comparison of one client-relevant option or scenario."""

    scenario: str = Field(min_length=1, max_length=300)
    summary: str = Field(min_length=1, max_length=1_000)
    benefits: list[str] = Field(min_length=1)
    tradeoffs: list[str] = Field(min_length=1)
    evidence_ids: list[UUID] = Field(min_length=1)


class Calculation(BaseModel):
    """A transparent deterministic calculation used in a research brief."""

    label: str = Field(min_length=1, max_length=300)
    formula: str = Field(min_length=1, max_length=500)
    result: Decimal
    unit: str = Field(min_length=1, max_length=100)


class ResearchBrief(BaseModel):
    """The Analyst's internal, evidence-backed result for the Advisor."""

    task_id: UUID
    findings: list[Finding] = Field(min_length=1)
    scenario_comparison: list[ScenarioComparison] = Field(default_factory=list)
    calculations: list[Calculation] = Field(default_factory=list)
    evidence: list[Evidence] = Field(min_length=1)
    caveats: list[str] = Field(min_length=1)
    integrity_warnings: list[ClaimIntegrityWarning] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_evidence_references(self) -> "ResearchBrief":
        """Ensure every finding and comparison cites evidence in this brief."""

        evidence_ids = [item.evidence_id for item in self.evidence]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("Evidence IDs must be unique within a research brief.")
        available_ids = set(evidence_ids)
        cited_ids = [
            evidence_id for finding in self.findings for evidence_id in finding.evidence_ids
        ] + [
            evidence_id
            for comparison in self.scenario_comparison
            for evidence_id in comparison.evidence_ids
        ]
        unknown_ids = set(cited_ids) - available_ids
        if unknown_ids:
            raise ValueError("Findings and comparisons may cite only evidence in the brief.")
        return self


class RecommendationOption(BaseModel):
    """One client-facing option evaluated by the Advisor."""

    title: str = Field(min_length=1, max_length=300)
    description: "CitedText"
    suitability: str = Field(min_length=1, max_length=500)


class CitedText(BaseModel):
    """Client-facing claim text associated with server-validated evidence IDs."""

    text: str = Field(min_length=1, max_length=2_000)
    evidence_ids: list[UUID] = Field(min_length=1)


class EvidenceCitation(BaseModel):
    """Server-owned citation metadata rendered by the client interface."""

    evidence_id: UUID
    title: str = Field(min_length=1, max_length=500)
    publisher: str = Field(min_length=1, max_length=300)
    url: HttpUrl | None = None


class Recommendation(BaseModel):
    """The only advice-shaped object sent from the Advisor to the Client and UI."""

    recommendation_id: UUID = Field(default_factory=uuid4)
    session_id: UUID
    summary: CitedText
    options: list[RecommendationOption] = Field(min_length=1)
    rationale: list[CitedText] = Field(min_length=1)
    assumptions: list[str] = Field(min_length=1)
    risks: list[CitedText] = Field(min_length=1)
    next_steps: list[str] = Field(min_length=1)
    evidence_ids: list[UUID] = Field(min_length=1)
    citations: list[EvidenceCitation] = Field(min_length=1)
    educational_disclaimer: str = Field(min_length=1, max_length=1_000)
    limitations: list[str] = Field(default_factory=list)
    integrity_warnings: list[ClaimIntegrityWarning] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_claim_citations(self) -> "Recommendation":
        """Require every advice claim to reference server-provided citation metadata."""

        citation_ids = [citation.evidence_id for citation in self.citations]
        if len(citation_ids) != len(set(citation_ids)):
            raise ValueError("Recommendation citation IDs must be unique.")
        available_ids = set(citation_ids)
        claim_ids = [*self.summary.evidence_ids]
        claim_ids.extend(
            evidence_id
            for option in self.options
            for evidence_id in option.description.evidence_ids
        )
        claim_ids.extend(
            evidence_id for rationale in self.rationale for evidence_id in rationale.evidence_ids
        )
        claim_ids.extend(evidence_id for risk in self.risks for evidence_id in risk.evidence_ids)
        if not set(claim_ids).issubset(available_ids):
            raise ValueError("Recommendation claims may cite only attached citation metadata.")
        if set(self.evidence_ids) != set(claim_ids):
            raise ValueError("Recommendation evidence IDs must match its claim citations.")
        return self


class ClientReview(BaseModel):
    """The Client's bounded response after receiving a recommendation."""

    session_id: UUID
    follow_up_number: int = Field(ge=0)
    accepted: bool
    message: str = Field(min_length=1, max_length=2_000)


class SessionEvent(BaseModel):
    """An immutable, safe-to-display event in the session trace."""

    model_config = ConfigDict(frozen=True)

    event_id: UUID = Field(default_factory=uuid4)
    session_id: UUID
    trace_id: UUID
    sequence: int = Field(ge=1)
    actor: Actor
    event_type: EventType
    summary: str = Field(min_length=1, max_length=2_000)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ApplicationError(BaseModel):
    """A structured, non-sensitive failure result for the API and session trace."""

    error_id: UUID = Field(default_factory=uuid4)
    code: ErrorCode
    message: str = Field(min_length=1, max_length=1_000)
    recoverable: bool
    session_id: UUID | None = None
