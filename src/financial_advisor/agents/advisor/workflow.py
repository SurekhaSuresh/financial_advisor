"""Connect structured Advisor decisions to the deterministic workflow engine."""

from typing import Protocol

from financial_advisor.agents.advisor.contracts import (
    AdvisorChatHistoryTurn,
    AdvisorClaimIntegrityError,
    AdvisorPlanningRequest,
    AdvisorRecommendationDraft,
    AdvisorResearchPlanDraft,
    AdvisorResponseRequest,
    AdvisorStateContext,
    AvailableResearchOptions,
    create_recommendation,
    create_research_plan,
)
from financial_advisor.config import get_settings
from financial_advisor.domain import (
    AdvisorDecision,
    AdvisorDecisionType,
    AdvisorResearchPlan,
    AnalystTask,
    ClientMessage,
    Recommendation,
    ResearchBrief,
    to_analyst_profile_context,
)
from financial_advisor.workflow import WorkflowEngine, WorkflowSession


class AdvisorService(Protocol):
    """Structured Advisor operations used by the workflow coordinator."""

    async def plan(self, request: AdvisorPlanningRequest) -> AdvisorResearchPlanDraft:
        """Return one validated next-action draft."""

    async def recommend(
        self, request: AdvisorResponseRequest, *, repair: bool = False
    ) -> AdvisorRecommendationDraft:
        """Return one client-wording draft grounded in the Analyst brief."""


class AdvisorWorkflowCoordinator:
    """Create trusted plans and client recommendations without bypassing workflow rules."""

    def __init__(self, workflow: WorkflowEngine, advisor: AdvisorService) -> None:
        self._workflow = workflow
        self._advisor = advisor

    async def assess(self, session: WorkflowSession, message: ClientMessage) -> AnalystTask | None:
        """Ask Advisor to answer from state, escalate, or create the next Analyst task."""

        settings = get_settings()
        prior_plans = [
            plan for plan in session.research_plans if plan.client_message_id == message.message_id
        ]
        prior_limitations = (
            session.retrieval_traces[-1].retrieval.evidence_limitations
            if session.retrieval_traces
            else []
        )
        request = AdvisorPlanningRequest(
            client_message=message,
            client_context=to_analyst_profile_context(session.client_profile),
            chat_history=self._chat_history(session, message),
            available_state=self._available_state(session),
            prior_plan=prior_plans[-1] if prior_plans else None,
            prior_research_limitations=prior_limitations,
            prior_plan_attempt_number=len(prior_plans),
            max_research_plans_per_client_question=(
                settings.workflow.max_research_plans_per_client_question
            ),
            available_research=AvailableResearchOptions(
                authoritative_web=bool(settings.web_research.approved_domain_registry)
            ),
            approved_authoritative_domains=settings.web_research.approved_domain_registry,
        )
        try:
            draft = await self._advisor.plan(request)
        except Exception as error:
            self._workflow.escalate_advisor_assessment(
                session,
                reason="Advisor planning was unavailable; session escalated safely.",
            )
            raise AdvisorWorkflowUnavailable("Advisor planning was unavailable.") from error
        if draft.decision_type is not AdvisorDecisionType.DELEGATE_RESEARCH:
            self._workflow.apply_advisor_decision(
                session,
                AdvisorDecision(
                    decision_type=draft.decision_type,
                    summary=draft.summary,
                    progress_summary=draft.progress_summary,
                    escalation_summary=draft.escalation_summary,
                ),
            )
            return None

        plan = create_research_plan(request, draft)
        if session.research_briefs and self._duplicates_completed_research_scope(plan, prior_plans):
            self._workflow.apply_advisor_decision(
                session,
                AdvisorDecision(
                    decision_type=AdvisorDecisionType.ANSWER_FROM_STATE,
                    summary=(
                        "The proposed refinement repeats an already completed research scope, "
                        "so the Advisor will use the latest validated research brief instead."
                    ),
                    progress_summary=(
                        "I have enough validated information to prepare a clear response to your "
                        "question."
                    ),
                ),
            )
            return None
        self._workflow.apply_advisor_decision(
            session,
            AdvisorDecision(
                decision_type=AdvisorDecisionType.DELEGATE_RESEARCH,
                summary=draft.summary,
                progress_summary=draft.progress_summary,
                research_plan=plan,
            ),
        )
        task = AnalystTask(
            session_id=session.session_id,
            question=plan.question,
            research_plan=plan,
            client_context=to_analyst_profile_context(session.client_profile),
        )
        self._workflow.submit_analyst_task(session, task)
        return task

    @staticmethod
    def _duplicates_completed_research_scope(
        candidate: AdvisorResearchPlan, prior_plans: list[AdvisorResearchPlan]
    ) -> bool:
        """Prevent a refinement from spending budget on an identical completed scope."""

        candidate_scope = (
            " ".join(candidate.question.casefold().split()),
            candidate.include_local_hybrid,
            tuple(
                sorted(
                    (
                        scope.mode.value,
                        tuple(sorted(domain_id.value for domain_id in scope.approved_domain_ids)),
                    )
                    for scope in candidate.web_scopes
                )
            ),
        )
        return any(
            candidate_scope
            == (
                " ".join(prior.question.casefold().split()),
                prior.include_local_hybrid,
                tuple(
                    sorted(
                        (
                            scope.mode.value,
                            tuple(
                                sorted(domain_id.value for domain_id in scope.approved_domain_ids)
                            ),
                        )
                        for scope in prior.web_scopes
                    )
                ),
            )
            for prior in prior_plans
        )

    async def respond(
        self, session: WorkflowSession, brief: ResearchBrief, message: ClientMessage
    ) -> Recommendation:
        """Create and deliver the only client-facing Advisor response for a brief."""

        for repair in (False, True):
            try:
                draft = await self._advisor.recommend(
                    AdvisorResponseRequest(
                        research_brief=brief,
                        client_message=message,
                        client_context=to_analyst_profile_context(session.client_profile),
                        chat_history=self._chat_history(session, message)[-2:],
                    ),
                    repair=repair,
                )
            except Exception as error:
                self._workflow.escalate_advisor_response(
                    session,
                    reason="Advisor response generation was unavailable; session escalated safely.",
                )
                raise AdvisorWorkflowUnavailable(
                    "Advisor response generation was unavailable."
                ) from error
            try:
                recommendation = create_recommendation(session.session_id, brief, draft)
            except AdvisorClaimIntegrityError:
                if repair:
                    self._workflow.escalate_advisor_response(
                        session,
                        reason=(
                            "Advisor response could not retain required evidence-backed claims; "
                            "session escalated safely."
                        ),
                    )
                    raise
                continue
            break
        else:  # pragma: no cover - the loop always breaks or raises.
            raise RuntimeError("Advisor citation repair policy did not complete.")
        self._workflow.submit_recommendation(session, recommendation)
        return recommendation

    async def answer_from_state(
        self, session: WorkflowSession, message: ClientMessage
    ) -> Recommendation:
        """Answer a follow-up only from the most recent validated Analyst brief."""

        if not session.research_briefs:
            self._workflow.escalate_advisor_response(
                session,
                reason=(
                    "Advisor had no validated research state for a direct response; "
                    "session escalated safely."
                ),
            )
            raise AdvisorWorkflowUnavailable("No validated research state was available.")
        return await self.respond(session, session.research_briefs[-1], message)

    @staticmethod
    def _chat_history(
        session: WorkflowSession, current_message: ClientMessage
    ) -> list[AdvisorChatHistoryTurn]:
        """Return up to four completed question-and-answer pairs before this message."""

        previous_messages = [
            message
            for message in session.messages
            if message.message_id != current_message.message_id
        ]
        turns = [
            AdvisorChatHistoryTurn(
                client_question=message.text,
                advisor_recommendation=recommendation,
            )
            for message, recommendation in zip(
                previous_messages, session.recommendations, strict=False
            )
        ]
        return turns[-4:]

    @staticmethod
    def _available_state(session: WorkflowSession) -> AdvisorStateContext | None:
        """Expose only validated prior research findings for the state-answer decision."""

        if not session.research_briefs:
            return None
        brief = session.research_briefs[-1]
        return AdvisorStateContext(
            findings=brief.findings,
            scenario_comparison=brief.scenario_comparison,
            caveats=brief.caveats,
            evidence_ids=[evidence.evidence_id for evidence in brief.evidence],
        )


class AdvisorWorkflowUnavailable(RuntimeError):
    """Raised after the coordinator records a safe Advisor-side escalation."""
