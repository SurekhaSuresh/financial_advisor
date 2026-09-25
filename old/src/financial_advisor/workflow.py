"""Deterministic workflow engine for Financial Advisor sessions."""

from dataclasses import dataclass, field
from uuid import UUID, uuid4

from financial_advisor.config import get_settings
from financial_advisor.domain import (
    Actor,
    AdvisorDecision,
    AdvisorDecisionType,
    AdvisorResearchPlan,
    AnalystTask,
    ApplicationError,
    ClaimIntegrityWarning,
    ClientMessage,
    ClientProfile,
    ClientReview,
    ErrorCode,
    EventType,
    Recommendation,
    ResearchBrief,
    SessionEvent,
    SessionState,
    to_analyst_profile_context,
)
from financial_advisor.persistence import SessionRepository
from financial_advisor.retrieval.pipeline import (
    CombinedEvidenceRetrievalTrace,
    EvidenceRetrievalStatus,
    TaskEvidenceRetrievalTrace,
)

MAX_FOLLOW_UPS = get_settings().workflow.max_client_follow_ups
MAX_RESEARCH_PLANS_PER_CLIENT_QUESTION = (
    get_settings().workflow.max_research_plans_per_client_question
)

GENERIC_ESCALATION_RESPONSE = (
    "We could not complete an evidence-backed analysis right now, so no investment "
    "recommendation was generated. Please try again shortly or consult a qualified "
    "financial professional for time-sensitive decisions."
)
SAFETY_ESCALATION_RESPONSE = (
    "I can’t recommend whether to buy or sell a particular investment or make an "
    "individualized financial decision. I can help explain general educational factors "
    "such as diversification, risk tolerance, and goal time horizon. For a specific "
    "investment decision, consult a qualified financial professional."
)

ALLOWED_TRANSITIONS: dict[SessionState, frozenset[SessionState]] = {
    SessionState.NEW: frozenset({SessionState.CLIENT_OPENS, SessionState.ESCALATED}),
    SessionState.CLIENT_OPENS: frozenset({SessionState.ADVISOR_ASSESSES, SessionState.ESCALATED}),
    SessionState.ADVISOR_ASSESSES: frozenset(
        {
            SessionState.ADVISOR_DELEGATES,
            SessionState.ADVISOR_PROPOSES,
            SessionState.ESCALATED,
        }
    ),
    SessionState.ADVISOR_DELEGATES: frozenset(
        {SessionState.ANALYST_RESEARCHES, SessionState.ESCALATED}
    ),
    SessionState.ANALYST_RESEARCHES: frozenset(
        {SessionState.ADVISOR_ASSESSES, SessionState.ADVISOR_PROPOSES, SessionState.ESCALATED}
    ),
    SessionState.ADVISOR_PROPOSES: frozenset(
        {SessionState.ADVISOR_ASSESSES, SessionState.CLIENT_REVIEWS, SessionState.ESCALATED}
    ),
    SessionState.CLIENT_REVIEWS: frozenset(
        {SessionState.ADVISOR_ASSESSES, SessionState.RESOLVED, SessionState.ESCALATED}
    ),
    SessionState.RESOLVED: frozenset(),
    SessionState.ESCALATED: frozenset(),
}


class WorkflowViolation(Exception):
    """Raised when an action conflicts with the deterministic workflow rules."""

    def __init__(self, error: ApplicationError) -> None:
        super().__init__(error.message)
        self.error = error


@dataclass
class WorkflowSession:
    """The authoritative working state for one client-advice conversation."""

    session_id: UUID
    trace_id: UUID
    client_profile: ClientProfile
    state: SessionState = SessionState.NEW
    follow_up_count: int = 0
    active_task_id: UUID | None = None
    terminal_response: str | None = None
    evidence_ids: set[UUID] = field(default_factory=set)
    events: list[SessionEvent] = field(default_factory=list)
    messages: list[ClientMessage] = field(default_factory=list)
    research_plans: list[AdvisorResearchPlan] = field(default_factory=list)
    analyst_tasks: list[AnalystTask] = field(default_factory=list)
    retrieval_traces: list[TaskEvidenceRetrievalTrace] = field(default_factory=list)
    research_briefs: list[ResearchBrief] = field(default_factory=list)
    recommendations: list[Recommendation] = field(default_factory=list)
    client_reviews: list[ClientReview] = field(default_factory=list)


class WorkflowEngine:
    """Applies validated agent outputs through the allowed session lifecycle."""

    def __init__(self, repository: SessionRepository | None = None) -> None:
        self._repository = repository
        if self._repository is not None:
            self._repository.initialize()

    def start_session(self, client_profile: ClientProfile) -> WorkflowSession:
        """Create a new session before the Client sends its opening message."""

        session = WorkflowSession(
            session_id=uuid4(), trace_id=uuid4(), client_profile=client_profile
        )
        self._record_event(
            session,
            actor="workflow_engine",
            event_type=EventType.SESSION_STARTED,
            summary="Client Agent is preparing a financial-planning question.",
        )
        self._persist(session)
        return session

    def submit_opening_message(self, session: WorkflowSession, message: ClientMessage) -> None:
        """Accept the single opening Client message and route it to the Advisor."""

        self._require_state(session, SessionState.NEW)
        if message.session_id != session.session_id:
            self._raise_violation(
                ErrorCode.VALIDATION_FAILED,
                "Client message session does not match the active workflow session.",
                session.session_id,
            )
        if message.follow_up_number != 0:
            self._raise_violation(
                ErrorCode.VALIDATION_FAILED,
                "The opening Client message must use follow-up number zero.",
                session.session_id,
            )

        self._record_event(
            session,
            actor="client",
            event_type=EventType.CLIENT_MESSAGE_CREATED,
            summary="Client submitted a financial-planning question. Advisor is assessing it.",
        )
        session.messages.append(message)
        self._transition(session, SessionState.CLIENT_OPENS)
        self._transition(session, SessionState.ADVISOR_ASSESSES)
        self._persist(session)

    def apply_advisor_decision(self, session: WorkflowSession, decision: AdvisorDecision) -> None:
        """Apply the Advisor's validated request for the next workflow action."""

        self._require_state(session, SessionState.ADVISOR_ASSESSES)
        self._record_event(
            session,
            actor="advisor",
            event_type=EventType.ADVISOR_PLANNER_RATIONALE_CREATED,
            # The Advisor Planner's validated, concise rationale—not hidden reasoning.
            summary=decision.summary,
        )
        self._record_event(
            session,
            actor="advisor",
            event_type=EventType.CLIENT_PROGRESS_CREATED,
            summary=decision.progress_summary,
        )
        if decision.decision_type is AdvisorDecisionType.ANSWER_FROM_STATE:
            self._transition(session, SessionState.ADVISOR_PROPOSES)
            self._persist(session)
            return
        if decision.decision_type is AdvisorDecisionType.DELEGATE_RESEARCH:
            if decision.research_plan is not None:
                self._record_research_plan(session, decision.research_plan)
            self._transition(session, SessionState.ADVISOR_DELEGATES)
            self._persist(session)
            return

        self._transition(session, SessionState.ESCALATED)
        session.terminal_response = decision.escalation_summary or SAFETY_ESCALATION_RESPONSE
        self._record_event(
            session,
            actor="workflow_engine",
            event_type=EventType.SESSION_ESCALATED,
            summary="Advisor escalated the session.",
        )
        self._persist(session)

    def submit_analyst_task(self, session: WorkflowSession, task: AnalystTask) -> None:
        """Validate the Advisor-created task and begin Analyst research."""

        self._require_state(session, SessionState.ADVISOR_DELEGATES)
        if task.session_id != session.session_id:
            self._raise_violation(
                ErrorCode.VALIDATION_FAILED,
                "Analyst task session does not match the active workflow session.",
                session.session_id,
            )
        if task.client_context != to_analyst_profile_context(session.client_profile):
            self._raise_violation(
                ErrorCode.VALIDATION_FAILED,
                "Analyst task client context does not match the approved profile context.",
                session.session_id,
            )
        if task.research_plan is not None and not any(
            plan.plan_id == task.research_plan.plan_id for plan in session.research_plans
        ):
            self._raise_violation(
                ErrorCode.VALIDATION_FAILED,
                "Analyst task must use a research plan recorded by the Advisor.",
                session.session_id,
            )

        session.active_task_id = task.task_id
        session.analyst_tasks.append(task)
        self._record_event(
            session,
            actor="advisor",
            event_type=EventType.ANALYST_TASK_CREATED,
            summary="Advisor delegated authorized research to the Analyst.",
        )
        self._transition(session, SessionState.ANALYST_RESEARCHES)
        self._persist(session)

    def submit_research_brief(self, session: WorkflowSession, brief: ResearchBrief) -> None:
        """Accept a valid Analyst brief and make it available to the Advisor."""

        self._require_state(session, SessionState.ANALYST_RESEARCHES)
        if brief.task_id != session.active_task_id:
            self._raise_violation(
                ErrorCode.VALIDATION_FAILED,
                "Research brief does not belong to the active Analyst task.",
                session.session_id,
            )

        retrieval_trace = next(
            (item for item in session.retrieval_traces if item.task_id == brief.task_id),
            None,
        )
        if retrieval_trace is not None:
            current_selected_evidence_ids = {
                item.evidence_id for item in retrieval_trace.retrieval.evidence
            }
            submitted_evidence_ids = {item.evidence_id for item in brief.evidence}
            permitted_evidence_ids = current_selected_evidence_ids | session.evidence_ids
            if not current_selected_evidence_ids.issubset(
                submitted_evidence_ids
            ) or not submitted_evidence_ids.issubset(permitted_evidence_ids):
                self._raise_violation(
                    ErrorCode.VALIDATION_FAILED,
                    (
                        "Research brief must preserve current selected evidence and may use only "
                        "previously validated selected evidence."
                    ),
                    session.session_id,
                )

        session.evidence_ids.update(item.evidence_id for item in brief.evidence)
        session.research_briefs.append(brief)
        self._record_event(
            session,
            actor="analyst",
            event_type=EventType.EVIDENCE_SELECTED,
            summary="Analyst completed the research brief. Advisor is preparing a cited response.",
        )
        self._record_claim_filter_events(session, brief.integrity_warnings)
        self._transition(session, SessionState.ADVISOR_PROPOSES)
        self._persist(session)

    def begin_evidence_retrieval(self, session: WorkflowSession, task_id: UUID) -> None:
        """Record the start of the active Analyst task's retrieval operation."""

        self._require_active_analyst_task(session, task_id)
        self._record_event(
            session,
            actor="analyst",
            event_type=EventType.TOOL_STARTED,
            summary="Analyst is retrieving authorized financial guidance.",
        )
        self._persist(session)

    def record_evidence_retrieval(
        self,
        session: WorkflowSession,
        task_id: UUID,
        retrieval: CombinedEvidenceRetrievalTrace,
    ) -> None:
        """Persist a complete retrieval trajectory before Analyst reasoning begins."""

        self._require_active_analyst_task(session, task_id)
        if any(item.task_id == task_id for item in session.retrieval_traces):
            self._raise_violation(
                ErrorCode.VALIDATION_FAILED,
                "An evidence retrieval trace is already recorded for the active task.",
                session.session_id,
            )
        session.retrieval_traces.append(
            TaskEvidenceRetrievalTrace(task_id=task_id, retrieval=retrieval)
        )
        summary = (
            "Analyst selected supporting evidence and is synthesizing a research brief."
            if retrieval.status is EvidenceRetrievalStatus.COMPLETED
            else "Analyst completed evidence retrieval without sufficient supporting evidence."
        )
        self._record_event(
            session,
            actor="analyst",
            event_type=EventType.TOOL_COMPLETED,
            summary=summary,
        )
        self._persist(session)

    def record_evidence_retrieval_failure(self, session: WorkflowSession, task_id: UUID) -> None:
        """Record an unexpected retrieval-operation failure without exposing internals."""

        self._require_active_analyst_task(session, task_id)
        self._record_event(
            session,
            actor="analyst",
            event_type=EventType.TOOL_FAILED,
            summary="Analyst evidence retrieval encountered an unexpected application failure.",
        )
        self._persist(session)

    def escalate_analyst_research(
        self, session: WorkflowSession, task_id: UUID, *, reason: str
    ) -> None:
        """End an Analyst task safely when sufficient evidence is unavailable."""

        self._require_active_analyst_task(session, task_id)
        self._transition(session, SessionState.ESCALATED)
        session.terminal_response = GENERIC_ESCALATION_RESPONSE
        self._record_event(
            session,
            actor="workflow_engine",
            event_type=EventType.SESSION_ESCALATED,
            summary=reason,
        )
        self._persist(session)

    def return_to_advisor_for_replan(self, session: WorkflowSession, task_id: UUID) -> bool:
        """Allow one Advisor-reviewed web-enabled replan after a local-only miss."""

        self._require_active_analyst_task(session, task_id)
        task = next(item for item in session.analyst_tasks if item.task_id == task_id)
        plan = task.research_plan
        if (
            plan is None
            or plan.attempt_number >= MAX_RESEARCH_PLANS_PER_CLIENT_QUESTION
            or not plan.include_local_hybrid
            or plan.web_scopes
        ):
            return False
        self._transition(session, SessionState.ADVISOR_ASSESSES)
        self._record_event(
            session,
            actor="workflow_engine",
            event_type=EventType.ADVISOR_DECISION_CREATED,
            summary=(
                "Local-only research had insufficient evidence; Advisor may create one "
                "web-enabled replan."
            ),
        )
        self._persist(session)
        return True

    def return_to_advisor_with_prior_brief(
        self, session: WorkflowSession, task_id: UUID, *, reason: str
    ) -> bool:
        """Retain earlier validated research if a bounded refinement cannot complete."""

        self._require_active_analyst_task(session, task_id)
        task = next(item for item in session.analyst_tasks if item.task_id == task_id)
        if task.research_plan.attempt_number <= 1 or not session.research_briefs:
            return False
        self._record_event(
            session,
            actor="analyst",
            event_type=EventType.TOOL_FAILED,
            summary=reason,
        )
        self._transition(session, SessionState.ADVISOR_PROPOSES)
        self._record_event(
            session,
            actor="workflow_engine",
            event_type=EventType.ADVISOR_DECISION_CREATED,
            summary=(
                "Refined research was unavailable; Advisor will use earlier validated "
                "research and disclose the limitation."
            ),
        )
        self._persist(session)
        return True

    def begin_research_brief_review(self, session: WorkflowSession, task_id: UUID) -> bool:
        """Allow one bounded Advisor adequacy review before client wording begins."""

        self._require_state(session, SessionState.ADVISOR_PROPOSES)
        task = next(item for item in session.analyst_tasks if item.task_id == task_id)
        if task.research_plan.attempt_number >= MAX_RESEARCH_PLANS_PER_CLIENT_QUESTION:
            return False
        self._transition(session, SessionState.ADVISOR_ASSESSES)
        self._record_event(
            session,
            actor="advisor",
            event_type=EventType.ADVISOR_DECISION_CREATED,
            summary="Advisor is reviewing whether the research brief fully answers the request.",
        )
        self._persist(session)
        return True

    def record_analyst_synthesis_failure(self, session: WorkflowSession, task_id: UUID) -> None:
        """Record model synthesis failure and end the active task safely."""

        self._require_active_analyst_task(session, task_id)
        self._record_event(
            session,
            actor="analyst",
            event_type=EventType.TOOL_FAILED,
            summary="Analyst model synthesis was unavailable after evidence retrieval completed.",
        )
        self._transition(session, SessionState.ESCALATED)
        session.terminal_response = GENERIC_ESCALATION_RESPONSE
        self._record_event(
            session,
            actor="workflow_engine",
            event_type=EventType.SESSION_ESCALATED,
            summary="Analyst synthesis was unavailable; session escalated safely.",
        )
        self._persist(session)

    def escalate_advisor_response(self, session: WorkflowSession, *, reason: str) -> None:
        """Safely end a response when the Advisor cannot retain required cited claims."""

        self._require_state(session, SessionState.ADVISOR_PROPOSES)
        self._transition(session, SessionState.ESCALATED)
        session.terminal_response = GENERIC_ESCALATION_RESPONSE
        self._record_event(
            session,
            actor="workflow_engine",
            event_type=EventType.SESSION_ESCALATED,
            summary=reason,
        )
        self._persist(session)

    def escalate_advisor_assessment(self, session: WorkflowSession, *, reason: str) -> None:
        """Safely end a session when the Advisor cannot assess the Client message."""

        self._require_state(session, SessionState.ADVISOR_ASSESSES)
        self._transition(session, SessionState.ESCALATED)
        session.terminal_response = GENERIC_ESCALATION_RESPONSE
        self._record_event(
            session,
            actor="workflow_engine",
            event_type=EventType.SESSION_ESCALATED,
            summary=reason,
        )
        self._persist(session)

    def escalate_client_interaction(self, session: WorkflowSession, *, reason: str) -> None:
        """Safely terminate when the simulated Client cannot create or review a message."""

        if session.state not in {SessionState.NEW, SessionState.CLIENT_REVIEWS}:
            self._raise_violation(
                ErrorCode.INVALID_STATE_TRANSITION,
                "Client interaction can be escalated only before opening or during review.",
                session.session_id,
            )
        self._transition(session, SessionState.ESCALATED)
        session.terminal_response = GENERIC_ESCALATION_RESPONSE
        self._record_event(
            session,
            actor="workflow_engine",
            event_type=EventType.SESSION_ESCALATED,
            summary=reason,
        )
        self._persist(session)

    def escalate_unexpected_runtime_failure(self, session: WorkflowSession) -> None:
        """Safely close any active session after an unexpected execution-boundary failure.

        Known Client, Advisor, Analyst, and retrieval failures use their more specific
        methods. This final boundary prevents an unhandled background-task exception
        from leaving an inspectable session indefinitely non-terminal.
        """

        if session.state in {SessionState.RESOLVED, SessionState.ESCALATED}:
            return
        self._transition(session, SessionState.ESCALATED)
        session.terminal_response = GENERIC_ESCALATION_RESPONSE
        self._record_event(
            session,
            actor="workflow_engine",
            event_type=EventType.SESSION_ESCALATED,
            summary="Scenario execution encountered an unexpected application failure.",
        )
        self._persist(session)

    def submit_recommendation(
        self, session: WorkflowSession, recommendation: Recommendation
    ) -> None:
        """Deliver an evidence-backed Advisor recommendation to the Client."""

        self._require_state(session, SessionState.ADVISOR_PROPOSES)
        if recommendation.session_id != session.session_id:
            self._raise_violation(
                ErrorCode.VALIDATION_FAILED,
                "Recommendation session does not match the active workflow session.",
                session.session_id,
            )
        if not set(recommendation.evidence_ids).issubset(session.evidence_ids):
            self._raise_violation(
                ErrorCode.VALIDATION_FAILED,
                "Recommendation may cite only evidence available in the session.",
                session.session_id,
            )

        self._record_event(
            session,
            actor="advisor",
            event_type=EventType.RECOMMENDATION_CREATED,
            summary=(
                "Advisor validated citations and delivered an evidence-backed recommendation. "
                "Client Agent is reviewing it."
            ),
        )
        session.recommendations.append(recommendation)
        self._record_claim_filter_events(session, recommendation.integrity_warnings)
        self._transition(session, SessionState.CLIENT_REVIEWS)
        self._persist(session)

    def submit_client_review(self, session: WorkflowSession, review: ClientReview) -> None:
        """Resolve a session or route a bounded Client follow-up to the Advisor."""

        self._require_state(session, SessionState.CLIENT_REVIEWS)
        if review.session_id != session.session_id:
            self._raise_violation(
                ErrorCode.VALIDATION_FAILED,
                "Client review session does not match the active workflow session.",
                session.session_id,
            )

        expected_follow_up = session.follow_up_count + 1
        if not review.accepted and review.follow_up_number != expected_follow_up:
            self._raise_violation(
                ErrorCode.VALIDATION_FAILED,
                "Client follow-up number is out of sequence.",
                session.session_id,
            )
        session.client_reviews.append(review)
        self._record_event(
            session,
            actor="client",
            event_type=EventType.CLIENT_REVIEW_CREATED,
            summary="Client Agent completed its review of the Advisor recommendation.",
        )
        if review.accepted:
            self._transition(session, SessionState.RESOLVED)
            self._record_event(
                session,
                actor="workflow_engine",
                event_type=EventType.SESSION_RESOLVED,
                summary="Client accepted the recommendation. Conversation resolved.",
            )
            self._persist(session)
            return

        if expected_follow_up > MAX_FOLLOW_UPS:
            self._transition(session, SessionState.ESCALATED)
            session.terminal_response = (
                "The simulated Client has reached the configured follow-up limit, so the "
                "conversation has ended."
            )
            self._record_event(
                session,
                actor="workflow_engine",
                event_type=EventType.SESSION_ESCALATED,
                summary="Client follow-up limit reached; session closed safely.",
            )
            self._persist(session)
            self._raise_violation(
                ErrorCode.FOLLOW_UP_LIMIT_REACHED,
                "The maximum number of Client follow-ups has been reached.",
                session.session_id,
            )

        session.follow_up_count = expected_follow_up
        session.messages.append(
            ClientMessage(
                session_id=session.session_id,
                text=review.message,
                follow_up_number=expected_follow_up,
            )
        )
        self._record_event(
            session,
            actor="client",
            event_type=EventType.CLIENT_MESSAGE_CREATED,
            summary="Client submitted a bounded follow-up question to the Advisor.",
        )
        self._transition(session, SessionState.ADVISOR_ASSESSES)
        self._persist(session)

    def _transition(self, session: WorkflowSession, next_state: SessionState) -> None:
        """Move a session through one explicitly allowed state transition."""

        previous_state = session.state
        if next_state not in ALLOWED_TRANSITIONS[previous_state]:
            self._raise_violation(
                ErrorCode.INVALID_STATE_TRANSITION,
                f"Cannot transition from {previous_state.value} to {next_state.value}.",
                session.session_id,
            )
        session.state = next_state
        self._record_event(
            session,
            actor="workflow_engine",
            event_type=EventType.STATE_TRANSITIONED,
            summary=f"Workflow transitioned from {previous_state.value} to {next_state.value}.",
        )

    def _record_research_plan(self, session: WorkflowSession, plan: AdvisorResearchPlan) -> None:
        """Validate and append the immutable Advisor plan before any Analyst task exists."""

        if plan.session_id != session.session_id:
            self._raise_violation(
                ErrorCode.VALIDATION_FAILED,
                "Advisor research plan session does not match the active workflow session.",
                session.session_id,
            )
        if not session.messages or plan.client_message_id != session.messages[-1].message_id:
            self._raise_violation(
                ErrorCode.VALIDATION_FAILED,
                "Advisor research plan must belong to the latest Client message.",
                session.session_id,
            )
        prior_attempts = [
            item
            for item in session.research_plans
            if item.client_message_id == plan.client_message_id
        ]
        if plan.attempt_number != len(prior_attempts) + 1:
            self._raise_violation(
                ErrorCode.VALIDATION_FAILED,
                "Advisor research plan attempt number is not sequential for this Client message.",
                session.session_id,
            )
        if plan.attempt_number > MAX_RESEARCH_PLANS_PER_CLIENT_QUESTION:
            self._raise_violation(
                ErrorCode.VALIDATION_FAILED,
                "Advisor research plan exceeds the configured per-question attempt limit.",
                session.session_id,
            )
        session.research_plans.append(plan)

    def _record_claim_filter_events(
        self, session: WorkflowSession, warnings: list[ClaimIntegrityWarning]
    ) -> None:
        """Persist safe audit events without retaining model-supplied invalid identifiers."""

        for warning in warnings:
            self._record_event(
                session,
                actor=warning.actor,
                event_type=EventType.CLAIM_FILTERED,
                summary=(
                    f"{warning.actor.capitalize()} removed {warning.removed_count} "
                    f"{warning.claim_type} claim(s) because they cited evidence outside "
                    "the selected evidence set."
                ),
            )

    def _require_state(self, session: WorkflowSession, expected_state: SessionState) -> None:
        """Reject an action attempted from the wrong session state."""

        if session.state is not expected_state:
            self._raise_violation(
                ErrorCode.INVALID_STATE_TRANSITION,
                f"Action requires {expected_state.value}; current state is {session.state.value}.",
                session.session_id,
            )

    def _require_active_analyst_task(self, session: WorkflowSession, task_id: UUID) -> None:
        """Require the active task while the workflow is in its Analyst state."""

        self._require_state(session, SessionState.ANALYST_RESEARCHES)
        if session.active_task_id != task_id:
            self._raise_violation(
                ErrorCode.VALIDATION_FAILED,
                "Evidence retrieval does not belong to the active Analyst task.",
                session.session_id,
            )

    def _record_event(
        self,
        session: WorkflowSession,
        *,
        actor: Actor,
        event_type: EventType,
        summary: str,
    ) -> None:
        """Append one ordered, safe-to-display event to a session."""

        session.events.append(
            SessionEvent(
                session_id=session.session_id,
                trace_id=session.trace_id,
                sequence=len(session.events) + 1,
                actor=actor,
                event_type=event_type,
                summary=summary,
            )
        )

    def _persist(self, session: WorkflowSession) -> None:
        """Write the completed result of one workflow action when persistence is enabled."""

        if self._repository is not None:
            self._repository.save(session)

    def _raise_violation(self, code: ErrorCode, message: str, session_id: UUID) -> None:
        """Raise a structured expected error for an invalid workflow action."""

        raise WorkflowViolation(
            ApplicationError(
                code=code,
                message=message,
                recoverable=False,
                session_id=session_id,
            )
        )
