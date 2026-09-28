"""Evidence-grounded Analyst used as a tool by the Advisor."""

from functools import partial

from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import BaseLlm
from google.genai import types

from financial_advisor.agents.analyst.prompt import ANALYST_INSTRUCTION
from financial_advisor.config import (
    ANALYST_AGENT,
    ANALYST_DRAFT_STATE_KEY,
    MODEL_REQUEST_TIMEOUT,
    RETRIEVED_EVIDENCE_STATE_KEY,
)
from financial_advisor.contracts import (
    Evidence,
    ResearchBrief,
    ResearchTask,
)
from financial_advisor.retrieval.pipeline import RetrievalPipeline


def create_analyst_agent(
    model: str | BaseLlm,
    retrieval_pipeline: RetrievalPipeline,
) -> LlmAgent:
    """Create an Analyst that skips the model when retrieval finds no evidence."""

    return LlmAgent(
        name=ANALYST_AGENT,
        description="Researches a financial question for the Advisor using trusted sources.",
        model=model,
        instruction=ANALYST_INSTRUCTION,
        input_schema=ResearchTask,
        output_schema=ResearchBrief,
        output_key=ANALYST_DRAFT_STATE_KEY,
        before_agent_callback=partial(
            _retrieve_before_agent,
            retrieval_pipeline=retrieval_pipeline,
        ),
        after_agent_callback=_ground_after_agent,
        generate_content_config=MODEL_REQUEST_TIMEOUT,
    )


def _retrieve_before_agent(
    callback_context: CallbackContext,
    *,
    retrieval_pipeline: RetrievalPipeline,
) -> types.Content | None:
    """Retrieve evidence before the model runs, or return an empty brief."""

    if callback_context.user_content is None or callback_context.user_content.parts is None:
        raise ValueError("The Analyst requires a ResearchTask input.")
    task_json = "".join(part.text or "" for part in callback_context.user_content.parts)
    research_task = ResearchTask.model_validate_json(task_json)
    retrieved_evidence = retrieval_pipeline.retrieve(
        research_task.question,
        research_task.retrieval_paths,
    )

    if not retrieved_evidence:
        return types.Content(
            role="model",
            parts=[types.Part.from_text(text=ResearchBrief().model_dump_json())],
        )

    callback_context.state[RETRIEVED_EVIDENCE_STATE_KEY] = [
        evidence.model_dump(mode="json") for evidence in retrieved_evidence
    ]
    return None


def _ground_after_agent(
    callback_context: CallbackContext,
) -> types.Content:
    """Replace the model draft with a brief grounded in trusted evidence."""

    draft_brief = ResearchBrief.model_validate(callback_context.state[ANALYST_DRAFT_STATE_KEY])
    retrieved_evidence = [
        Evidence.model_validate(evidence)
        for evidence in callback_context.state[RETRIEVED_EVIDENCE_STATE_KEY]
    ]
    available_evidence_ids = {evidence.evidence_id for evidence in retrieved_evidence}
    supported_findings = [
        finding
        for finding in draft_brief.findings
        if finding.uses_available_evidence(available_evidence_ids)
    ]
    grounded_brief = ResearchBrief()
    if supported_findings:
        grounded_brief = draft_brief.model_copy(
            update={
                "findings": supported_findings,
                "evidence": retrieved_evidence,
                "scenario_comparisons": [
                    comparison
                    for comparison in draft_brief.scenario_comparisons
                    if comparison.uses_available_evidence(available_evidence_ids)
                ],
            }
        )
    return types.Content(
        role="model",
        parts=[types.Part.from_text(text=grounded_brief.model_dump_json())],
    )
