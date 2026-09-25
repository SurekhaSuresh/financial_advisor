"""Connect Analyst execution results to the deterministic workflow engine."""

from financial_advisor.agents.analyst.contracts import (
    AnalystConversationTurn,
    PriorValidatedResearchTurn,
)
from financial_advisor.agents.analyst.execution import (
    AnalystRefinementUnavailable,
    AnalystResearchExecution,
    AnalystResearchExecutor,
    AnalystResearchUnavailable,
    AnalystSynthesisUnavailable,
)
from financial_advisor.domain import AnalystTask
from financial_advisor.workflow import WorkflowEngine, WorkflowSession


class AnalystWorkflowCoordinator:
    """Persist Analyst outcomes before advancing or escalating workflow state."""

    def __init__(self, workflow: WorkflowEngine, executor: AnalystResearchExecutor) -> None:
        self._workflow = workflow
        self._executor = executor

    async def execute(
        self, session: WorkflowSession, task: AnalystTask
    ) -> AnalystResearchExecution:
        """Run one active task and apply its complete outcome to the session."""

        self._workflow.begin_evidence_retrieval(session, task.task_id)
        try:
            execution = await self._executor.execute(
                task,
                prior_chat_history=self._prior_chat_history(session, task),
                prior_validated_research_turns=self._prior_validated_research_turns(session, task),
            )
        except AnalystResearchUnavailable as error:
            self._workflow.record_evidence_retrieval(session, task.task_id, error.retrieval)
            if self._workflow.return_to_advisor_for_replan(session, task.task_id):
                raise
            if self._workflow.return_to_advisor_with_prior_brief(
                session,
                task.task_id,
                reason=(
                    "Refined evidence retrieval was unavailable; earlier validated research "
                    "remains available."
                ),
            ):
                raise AnalystRefinementUnavailable() from error
            self._workflow.escalate_analyst_research(
                session,
                task.task_id,
                reason=(
                    "Analyst could not find sufficient supporting evidence; "
                    "session escalated safely."
                ),
            )
            raise
        except AnalystSynthesisUnavailable as error:
            self._workflow.record_evidence_retrieval(session, task.task_id, error.retrieval)
            if self._workflow.return_to_advisor_with_prior_brief(
                session,
                task.task_id,
                reason=(
                    "Refined Analyst synthesis was unavailable; earlier validated research "
                    "remains available."
                ),
            ):
                raise AnalystRefinementUnavailable() from error
            self._workflow.record_analyst_synthesis_failure(session, task.task_id)
            raise
        except Exception:
            self._workflow.record_evidence_retrieval_failure(session, task.task_id)
            self._workflow.escalate_analyst_research(
                session,
                task.task_id,
                reason=(
                    "Analyst research encountered an unexpected application failure; "
                    "session escalated safely."
                ),
            )
            raise

        self._workflow.record_evidence_retrieval(session, task.task_id, execution.retrieval)
        self._workflow.submit_research_brief(session, execution.research_brief)
        return execution

    @staticmethod
    def _prior_chat_history(
        session: WorkflowSession, task: AnalystTask
    ) -> list[AnalystConversationTurn]:
        """Return completed successful Client/Advisor exchanges before this task."""

        return [
            AnalystConversationTurn(
                client_question=message.text,
                advisor_recommendation=recommendation,
            )
            for message, recommendation in zip(
                session.messages, session.recommendations, strict=False
            )
            if message.message_id != task.research_plan.client_message_id
        ][-2:]

    @staticmethod
    def _prior_validated_research_turns(
        session: WorkflowSession, task: AnalystTask
    ) -> list[PriorValidatedResearchTurn]:
        """Return successful prior research only; failed traces never enter model context."""

        tasks_by_id = {item.task_id: item for item in session.analyst_tasks}
        messages_by_id = {message.message_id: message for message in session.messages}
        return [
            PriorValidatedResearchTurn(
                client_question=messages_by_id[prior_task.research_plan.client_message_id].text,
                research_plan=prior_task.research_plan,
                research_brief=brief,
            )
            for brief in session.research_briefs
            if brief.task_id in tasks_by_id
            and (prior_task := tasks_by_id[brief.task_id]).task_id != task.task_id
        ][-2:]
