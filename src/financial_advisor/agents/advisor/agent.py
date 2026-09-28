"""Root Advisor that coordinates the Client and Analyst AgentTools."""

import json
from typing import Any

from google.adk.agents import LlmAgent
from google.adk.models import BaseLlm
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.tool_context import ToolContext

from financial_advisor.agents.advisor.prompt import ADVISOR_INSTRUCTION
from financial_advisor.config import (
    ANALYST_AGENT,
    CLIENT_AGENT,
    CLIENT_FOLLOW_UP_COUNT_STATE_KEY,
    CONVERSATION_STATUS_STATE_KEY,
    CURRENT_CLIENT_QUESTION_STATE_KEY,
    ESCALATION_MESSAGE,
    FINAL_RECOMMENDATION_STATE_KEY,
    MAX_CLIENT_FOLLOW_UPS,
    MAX_RECOMMENDATION_REDRAFTS,
    MAX_RESEARCH_ATTEMPTS,
    MODEL_REQUEST_TIMEOUT,
    RECOMMENDATION_REDRAFT_COUNT_STATE_KEY,
    RECOMMENDATION_REDRAFT_MESSAGE,
    RESEARCH_ATTEMPT_COUNT_STATE_KEY,
    TRUSTED_RESEARCH_BRIEF_STATE_KEY,
)
from financial_advisor.contracts import (
    ConversationStatus,
    EvidenceCitation,
    Recommendation,
    RecommendationDraft,
    ResearchBrief,
)


def create_advisor_agent(
    model: str | BaseLlm,
    client_agent: LlmAgent,
    analyst_agent: LlmAgent,
) -> LlmAgent:
    """Create the root Advisor with explicit Client and Analyst boundaries."""

    return LlmAgent(
        name="advisor_agent",
        description="Coordinates Client questions, Analyst research, and recommendations.",
        model=model,
        instruction=ADVISOR_INSTRUCTION,
        tools=[
            AgentTool(client_agent),
            AgentTool(analyst_agent),
            FunctionTool(finalize_recommendation),
            FunctionTool(escalate_conversation),
        ],
        before_tool_callback=_prepare_agent_tool_call,
        after_tool_callback=_record_agent_tool_result,
        generate_content_config=MODEL_REQUEST_TIMEOUT,
    )


def _prepare_agent_tool_call(
    tool: BaseTool,
    args: dict[str, Any],
    tool_context: ToolContext,
) -> dict[str, object] | None:
    """Supply trusted inputs and stop calls that exceed workflow limits."""

    state = tool_context.state
    if tool.name == CLIENT_AGENT:
        recommendation = state.get(FINAL_RECOMMENDATION_STATE_KEY)
        args.update(
            advisor_response_json=json.dumps(recommendation) if recommendation else None,
            follow_up_count=state.get(CLIENT_FOLLOW_UP_COUNT_STATE_KEY, 0),
        )

    if tool.name == ANALYST_AGENT:
        research_attempts = state.get(RESEARCH_ATTEMPT_COUNT_STATE_KEY, 0)
        if research_attempts >= MAX_RESEARCH_ATTEMPTS:
            previous_brief = state.get(TRUSTED_RESEARCH_BRIEF_STATE_KEY)
            return previous_brief or escalate_conversation(tool_context)

        state[RESEARCH_ATTEMPT_COUNT_STATE_KEY] = research_attempts + 1

    return None


def _record_agent_tool_result(
    tool: BaseTool,
    args: dict[str, Any],
    tool_context: ToolContext,
    tool_response: dict[str, Any],
) -> dict[str, Any] | None:
    """Validate AgentTool results and apply terminal state changes."""

    state = tool_context.state
    if tool.name == ANALYST_AGENT:
        if state.get(CONVERSATION_STATUS_STATE_KEY) == ConversationStatus.ESCALATED.value:
            return None
        brief = ResearchBrief.model_validate(tool_response)
        brief_data = brief.model_dump(mode="json")
        if brief.findings:
            state[TRUSTED_RESEARCH_BRIEF_STATE_KEY] = brief_data
        return brief_data

    if tool.name == CLIENT_AGENT:
        action = tool_response["action"]

        # Client opening question
        if args.get("advisor_response_json") is None:
            if action != "question":
                raise ValueError("The Client must open the conversation with a question.")
            state[CURRENT_CLIENT_QUESTION_STATE_KEY] = tool_response["message"]
        elif action == "accept":
            # Client accepted the recommendation.
            state[CONVERSATION_STATUS_STATE_KEY] = ConversationStatus.RESOLVED.value
            tool_context.actions.skip_summarization = True
        else:
            # Client requested a recommendation follow-up.
            state[CURRENT_CLIENT_QUESTION_STATE_KEY] = tool_response["message"]
            state[CLIENT_FOLLOW_UP_COUNT_STATE_KEY] = (
                state.get(CLIENT_FOLLOW_UP_COUNT_STATE_KEY, 0) + 1
            )

    return None


def finalize_recommendation(
    recommendation_draft_json: str,
    tool_context: ToolContext,
) -> dict[str, object]:
    """Validate a JSON RecommendationDraft and store the grounded recommendation."""

    state = tool_context.state
    draft = RecommendationDraft.model_validate_json(recommendation_draft_json)
    trusted_brief = ResearchBrief.model_validate(state.get(TRUSTED_RESEARCH_BRIEF_STATE_KEY))
    available_evidence_ids = {evidence.evidence_id for evidence in trusted_brief.evidence}

    supported_options = {
        title: claim
        for title, claim in draft.options.items()
        if claim.uses_available_evidence(available_evidence_ids)
    }
    supported_risks = [
        claim for claim in draft.risks if claim.uses_available_evidence(available_evidence_ids)
    ]

    # A usable recommendation needs a supported summary, option, and risk.
    if (
        not draft.summary.uses_available_evidence(available_evidence_ids)
        or not supported_options
        or not supported_risks
    ):
        redraft_count = state.get(RECOMMENDATION_REDRAFT_COUNT_STATE_KEY, 0)
        if redraft_count >= MAX_RECOMMENDATION_REDRAFTS:
            return escalate_conversation(tool_context)
        state[RECOMMENDATION_REDRAFT_COUNT_STATE_KEY] = redraft_count + 1
        return {"message": RECOMMENDATION_REDRAFT_MESSAGE}

    cited_evidence_ids = {
        evidence_id
        for claim in [draft.summary, *supported_options.values(), *supported_risks]
        for evidence_id in claim.evidence_ids
    }
    recommendation = Recommendation.model_validate(
        {
            **draft.model_dump(),
            "options": supported_options,
            "risks": supported_risks,
            "citations": [
                EvidenceCitation(
                    evidence_id=evidence.evidence_id,
                    title=evidence.title,
                    publisher=evidence.publisher,
                    url=evidence.url,
                )
                for evidence in trusted_brief.evidence
                if evidence.evidence_id in cited_evidence_ids
            ],
            "limitations": trusted_brief.limitations,
        }
    )
    recommendation_data = recommendation.model_dump(mode="json")
    state[RECOMMENDATION_REDRAFT_COUNT_STATE_KEY] = 0
    state[FINAL_RECOMMENDATION_STATE_KEY] = recommendation_data
    if state.get(CLIENT_FOLLOW_UP_COUNT_STATE_KEY, 0) >= MAX_CLIENT_FOLLOW_UPS:
        state[CONVERSATION_STATUS_STATE_KEY] = ConversationStatus.RESOLVED.value
        tool_context.actions.skip_summarization = True
    return recommendation_data


def escalate_conversation(tool_context: ToolContext, message: str = "") -> dict[str, object]:
    """End safely with the Advisor's summary or the static fallback message."""

    tool_context.state[CONVERSATION_STATUS_STATE_KEY] = ConversationStatus.ESCALATED.value
    tool_context.actions.skip_summarization = True
    return {"message": message.strip() or ESCALATION_MESSAGE}
