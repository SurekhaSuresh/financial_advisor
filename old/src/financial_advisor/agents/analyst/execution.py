"""Deterministic evidence operation followed by bounded Analyst synthesis."""

from __future__ import annotations

import logging
from typing import Protocol

from pydantic import BaseModel

from financial_advisor.agents.analyst.contracts import (
    AnalystConversationTurn,
    AnalystResearchRequest,
    PriorValidatedResearchTurn,
)
from financial_advisor.domain import AnalystTask, Evidence, ResearchBrief
from financial_advisor.retrieval.pipeline import (
    CombinedEvidenceRetrievalTrace,
    EvidenceRetrievalRequest,
    EvidenceRetrievalStatus,
)

logger = logging.getLogger(__name__)


class EvidenceRetriever(Protocol):
    """The three-channel evidence operation used by the Analyst."""

    def retrieve(self, request: EvidenceRetrievalRequest) -> CombinedEvidenceRetrievalTrace:
        """Return the full persisted-ready evidence trajectory."""


class AnalystSynthesisService(Protocol):
    """The typed reasoning operation used after successful evidence retrieval."""

    async def research(self, request: AnalystResearchRequest) -> ResearchBrief:
        """Synthesize the approved evidence into one internal research brief."""


class AnalystResearchUnavailable(RuntimeError):
    """Raised when a task has no evidence safe enough for model synthesis."""

    def __init__(self, retrieval: CombinedEvidenceRetrievalTrace) -> None:
        super().__init__(retrieval.reason)
        self.retrieval = retrieval


class AnalystSynthesisUnavailable(RuntimeError):
    """Raised when evidence exists but the Analyst model cannot synthesize it."""

    def __init__(self, retrieval: CombinedEvidenceRetrievalTrace) -> None:
        super().__init__("Analyst synthesis is unavailable right now.")
        self.retrieval = retrieval


class AnalystRefinementUnavailable(RuntimeError):
    """Raised when a second-plan failure can safely fall back to a prior brief."""


class AnalystResearchExecution(BaseModel):
    """One end-to-end Analyst trajectory before the Advisor consumes its brief."""

    task: AnalystTask
    retrieval: CombinedEvidenceRetrievalTrace
    research_brief: ResearchBrief


class AnalystResearchExecutor:
    """Run selected evidence retrieval before model-based Analyst synthesis."""

    def __init__(
        self,
        evidence_retriever: EvidenceRetriever,
        synthesis_service: AnalystSynthesisService,
    ) -> None:
        self._evidence_retriever = evidence_retriever
        self._synthesis_service = synthesis_service

    async def execute(
        self,
        task: AnalystTask,
        *,
        prior_chat_history: list[AnalystConversationTurn] | None = None,
        prior_validated_research_turns: list[PriorValidatedResearchTurn] | None = None,
    ) -> AnalystResearchExecution:
        """Retrieve evidence first, then request one structured Analyst draft."""

        request = EvidenceRetrievalRequest(
            query=task.research_plan.question,
            include_local_hybrid=task.research_plan.include_local_hybrid,
            web_scopes=task.research_plan.web_scopes,
        )
        retrieval = self._evidence_retriever.retrieve(request)
        if retrieval.status is not EvidenceRetrievalStatus.COMPLETED:
            raise AnalystResearchUnavailable(retrieval)
        prior_chat_history = prior_chat_history or []
        prior_validated_research_turns = prior_validated_research_turns or []
        merged_evidence = _merge_selected_evidence(
            current_evidence=retrieval.evidence,
            prior_turns=prior_validated_research_turns,
        )
        try:
            brief = await self._synthesis_service.research(
                AnalystResearchRequest(
                    task=task,
                    evidence=merged_evidence,
                    evidence_limitations=retrieval.evidence_limitations,
                    prior_chat_history=prior_chat_history,
                    prior_validated_research_turns=prior_validated_research_turns,
                )
            )
        except Exception as error:
            logger.exception(
                "Analyst synthesis failed after completed evidence retrieval.",
                extra={"task_id": str(task.task_id)},
            )
            raise AnalystSynthesisUnavailable(retrieval) from error
        return AnalystResearchExecution(
            task=task,
            retrieval=retrieval,
            research_brief=brief,
        )


def _merge_selected_evidence(
    *,
    current_evidence: list[Evidence],
    prior_turns: list[PriorValidatedResearchTurn],
) -> list[Evidence]:
    """Exact-deduplicate historical and current selected evidence, preferring current."""

    merged_by_identity = {
        _evidence_identity(evidence): evidence
        for turn in prior_turns
        for evidence in turn.research_brief.evidence
    }
    for evidence in current_evidence:
        merged_by_identity[_evidence_identity(evidence)] = evidence
    return list(merged_by_identity.values())


def _evidence_identity(evidence: Evidence) -> tuple[str, str, str, str, str | None, str]:
    """Create a stable exact identity for selected-evidence deduplication."""

    return (
        evidence.source_type,
        evidence.publisher,
        evidence.title,
        str(evidence.url) if evidence.url is not None else "",
        evidence.published_at.isoformat() if evidence.published_at is not None else None,
        evidence.excerpt,
    )
