"""Top-level composition of the bounded Client, Advisor, and Analyst agents."""

from typing import Protocol
from uuid import UUID

from financial_advisor.agents.advisor.workflow import (
    AdvisorWorkflowCoordinator,
    AdvisorWorkflowUnavailable,
)
from financial_advisor.agents.analyst.execution import (
    AnalystRefinementUnavailable,
    AnalystResearchUnavailable,
    AnalystSynthesisUnavailable,
)
from financial_advisor.agents.analyst.workflow import AnalystWorkflowCoordinator
from financial_advisor.agents.client.adk import ClientModelError
from financial_advisor.agents.client.contracts import (
    ClientConversationTurn,
    ClientReviewRequest,
)
from financial_advisor.config import get_settings
from financial_advisor.domain import (
    ClientMessage,
    ClientProfile,
    ClientReview,
    Recommendation,
    ResearchBrief,
    SessionState,
)
from financial_advisor.workflow import WorkflowEngine, WorkflowSession


class ClientConversationService(Protocol):
    """The Client-only conversation operations used by the scenario runner."""

    async def open_conversation(
        self, profile: ClientProfile, session_id: UUID
    ) -> ClientMessage: ...

    async def review_recommendation(self, request: ClientReviewRequest) -> ClientReview: ...


class ScenarioRunner:
    """Run a Client-led bounded conversation through the three-agent workflow."""

    def __init__(
        self,
        workflow: WorkflowEngine,
        client: ClientConversationService,
        advisor: AdvisorWorkflowCoordinator,
        analyst: AnalystWorkflowCoordinator,
    ) -> None:
        self._workflow = workflow
        self._client = client
        self._advisor = advisor
        self._analyst = analyst

    async def run(
        self, profile: ClientProfile, *, session: WorkflowSession | None = None
    ) -> WorkflowSession:
        """Run Client opening, up to two reviews, and a terminal resolution or escalation.

        An API execution manager may create and persist the session before scheduling this
        coroutine, so it can return a stable session and trace ID immediately to a caller.
        """

        if session is None:
            session = self._workflow.start_session(profile)
        elif session.client_profile != profile:
            raise ValueError("The supplied session does not belong to the supplied client profile.")
        try:
            message = await self._client.open_conversation(profile, session.session_id)
        except ClientModelError:
            self._workflow.escalate_client_interaction(
                session, reason="Client simulation was unavailable before the opening message."
            )
            return session
        self._workflow.submit_opening_message(session, message)

        while session.state is SessionState.ADVISOR_ASSESSES:
            try:
                task = await self._advisor.assess(session, message)
            except AdvisorWorkflowUnavailable:
                return session
            if task is None:
                if session.state.value != SessionState.ADVISOR_PROPOSES.value:
                    return session
                try:
                    recommendation = await self._advisor.answer_from_state(session, message)
                except AdvisorWorkflowUnavailable:
                    return session
            else:
                try:
                    execution = await self._analyst.execute(session, task)
                except (AnalystResearchUnavailable, AnalystSynthesisUnavailable):
                    if session.state is SessionState.ADVISOR_ASSESSES:
                        continue
                    return session
                brief = execution.research_brief
                if self._workflow.begin_research_brief_review(session, task.task_id):
                    try:
                        revised_task = await self._advisor.assess(session, message)
                    except AdvisorWorkflowUnavailable:
                        return session
                    if revised_task is not None:
                        try:
                            execution = await self._analyst.execute(session, revised_task)
                        except AnalystRefinementUnavailable:
                            brief = self._brief_with_refinement_caveat(session)
                        except (AnalystResearchUnavailable, AnalystSynthesisUnavailable):
                            return session
                        else:
                            brief = execution.research_brief
                    if not self._is_ready_for_recommendation(session):
                        return session
                try:
                    recommendation = await self._advisor.respond(session, brief, message)
                except AdvisorWorkflowUnavailable:
                    return session

            if not await self._review_with_client(session, profile, recommendation):
                return session
            if session.state is not SessionState.ADVISOR_ASSESSES:
                return session
            message = session.messages[-1]

        return session

    @staticmethod
    def _brief_with_refinement_caveat(session: WorkflowSession) -> ResearchBrief:
        """Create a response-only caveated view of the prior validated brief."""

        brief = session.research_briefs[-1]
        return brief.model_copy(
            update={
                "caveats": [
                    *brief.caveats,
                    (
                        "The attempted refined research was unavailable, so this response is "
                        "limited to the earlier validated evidence."
                    ),
                ]
            }
        )

    @staticmethod
    def _is_ready_for_recommendation(session: WorkflowSession) -> bool:
        """Keep terminal Advisor review outcomes from reaching response generation."""

        return session.state is SessionState.ADVISOR_PROPOSES

    async def _review_with_client(
        self, session: WorkflowSession, profile: ClientProfile, recommendation: Recommendation
    ) -> bool:
        """Ask the Client to accept or issue one budgeted follow-up through the Advisor."""

        max_follow_ups = get_settings().workflow.max_client_follow_ups
        if session.follow_up_count >= max_follow_ups:
            self._workflow.submit_client_review(
                session,
                ClientReview(
                    session_id=session.session_id,
                    follow_up_number=session.follow_up_count,
                    accepted=True,
                    message=(
                        "The follow-up limit is complete; the simulated Client accepts "
                        "the response."
                    ),
                ),
            )
            return True

        try:
            review = await self._client.review_recommendation(
                ClientReviewRequest(
                    profile=profile,
                    recommendation=recommendation,
                    completed_follow_up_count=session.follow_up_count,
                    max_follow_ups=max_follow_ups,
                    prior_chat_history=[
                        ClientConversationTurn(
                            client_question=message.text,
                            advisor_recommendation=prior_recommendation,
                        )
                        for message, prior_recommendation in zip(
                            session.messages[:-1], session.recommendations[:-1], strict=True
                        )
                    ][-2:],
                )
            )
        except ClientModelError:
            self._workflow.escalate_client_interaction(
                session,
                reason="Client simulation was unavailable while reviewing the recommendation.",
            )
            return False
        self._workflow.submit_client_review(session, review)
        return True
