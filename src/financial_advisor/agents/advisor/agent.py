"""Root Advisor that coordinates the Client and Analyst AgentTools."""

import json
from typing import Any

from google.adk.agents import LlmAgent
from google.adk.models import BaseLlm
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types
from pydantic import BaseModel, Field

from financial_advisor.agents.advisor.prompt import ADVISOR_INSTRUCTION
from financial_advisor.agents.analyst.agent import TRUSTED_ANALYST_BRIEF_STATE_KEY
from financial_advisor.agents.client import LATEST_CLIENT_RESULT_STATE_KEY
from financial_advisor.config import (
    MAX_CLIENT_FOLLOW_UPS,
    MAX_RESEARCH_ATTEMPTS,
    MODEL_REQUEST_TIMEOUT_MILLISECONDS,
)
from financial_advisor.contracts import (
    CitedText,
    ClientAction,
    ClientProfile,
    ClientResult,
    ConversationStatus,
    EvidenceCitation,
    Recommendation,
    RecommendationOption,
    ResearchBrief,
    ResearchResult,
)

FINAL_ADVISOR_RECOMMENDATION_STATE_KEY = "final_advisor_recommendation"
ADVISOR_RECOMMENDATION_HISTORY_STATE_KEY = "advisor_recommendations"
CONVERSATION_STATUS_STATE_KEY = "conversation_status"
CLIENT_FOLLOW_UP_COUNT_STATE_KEY = "client_follow_up_count"
CLIENT_RESULT_HISTORY_STATE_KEY = "client_results"
RESEARCH_ATTEMPT_COUNT_STATE_KEY = "research_attempt_count"

_FOLLOW_UP_LIMIT_MESSAGE = "The available recommendation is sufficient for this exercise."


class RecommendationDraft(BaseModel):
    """Advisor-written content awaiting deterministic evidence validation."""

    summary: CitedText
    options: list[RecommendationOption] = Field(min_length=1)
    assumptions: list[str] = Field(min_length=1)
    risks: list[CitedText] = Field(min_length=1)
    next_steps: list[str] = Field(min_length=1)


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
        input_schema=ClientProfile,
        tools=[
            AgentTool(client_agent),
            AgentTool(analyst_agent),
            FunctionTool(finalize_recommendation),
        ],
        before_tool_callback=_prepare_agent_tool_call,
        after_tool_callback=_record_agent_tool_result,
        generate_content_config=types.GenerateContentConfig(
            http_options=types.HttpOptions(
                timeout=MODEL_REQUEST_TIMEOUT_MILLISECONDS,
            )
        ),
    )


def _prepare_agent_tool_call(
    tool: BaseTool,
    args: dict[str, Any],
    tool_context: ToolContext,
) -> dict[str, object] | None:
    """Supply trusted inputs and enforce the research-call limit."""

    if tool.name not in {"client_agent", "analyst_agent"}:
        return None
    if tool_context.user_content is None or tool_context.user_content.parts is None:
        raise ValueError("The Advisor requires a ClientProfile input.")

    profile_json = "".join(part.text or "" for part in tool_context.user_content.parts)
    trusted_profile = ClientProfile.model_validate_json(profile_json)
    args["client_profile_json"] = trusted_profile.model_dump_json()

    if tool.name == "client_agent":
        stored_recommendation = tool_context.state.get(
            FINAL_ADVISOR_RECOMMENDATION_STATE_KEY,
        )
        args["advisor_response_json"] = (
            json.dumps(stored_recommendation)
            if stored_recommendation is not None
            else None
        )
        args["follow_up_count"] = tool_context.state.get(
            CLIENT_FOLLOW_UP_COUNT_STATE_KEY,
            0,
        )
        return None

    research_attempt_count = tool_context.state.get(
        RESEARCH_ATTEMPT_COUNT_STATE_KEY,
        0,
    )
    if research_attempt_count >= MAX_RESEARCH_ATTEMPTS:
        return ResearchResult(success=False).model_dump(mode="json")
    tool_context.state[RESEARCH_ATTEMPT_COUNT_STATE_KEY] = research_attempt_count + 1
    return None


def _record_agent_tool_result(
    tool: BaseTool,
    args: dict[str, Any],
    tool_context: ToolContext,
    tool_response: dict[str, Any],
) -> dict[str, Any] | None:
    """Make Analyst output JSON-safe and record Client results."""

    if tool.name == "analyst_agent":
        return ResearchResult.model_validate(tool_response).model_dump(mode="json")

    if tool.name != "client_agent":
        return None

    client_result = ClientResult.model_validate(tool_response)
    client_result_data = client_result.model_dump(mode="json")
    tool_context.state[CLIENT_RESULT_HISTORY_STATE_KEY] = [
        *tool_context.state.get(CLIENT_RESULT_HISTORY_STATE_KEY, []),
        client_result_data,
    ]
    if args.get("advisor_response_json") is None:
        if client_result.action is not ClientAction.QUESTION:
            raise ValueError("The Client must open the conversation with a question.")
        return None

    if client_result.action is ClientAction.ACCEPT:
        tool_context.state[CONVERSATION_STATUS_STATE_KEY] = (
            ConversationStatus.RESOLVED.value
        )
        return None

    follow_up_count = tool_context.state.get(CLIENT_FOLLOW_UP_COUNT_STATE_KEY, 0)
    if follow_up_count < MAX_CLIENT_FOLLOW_UPS:
        tool_context.state[CLIENT_FOLLOW_UP_COUNT_STATE_KEY] = follow_up_count + 1
        return None

    accepted_result = ClientResult(
        action=ClientAction.ACCEPT,
        message=_FOLLOW_UP_LIMIT_MESSAGE,
    )
    accepted_result_data = accepted_result.model_dump(mode="json")
    client_result_history = tool_context.state[CLIENT_RESULT_HISTORY_STATE_KEY]
    client_result_history[-1] = accepted_result_data
    tool_context.state[CLIENT_RESULT_HISTORY_STATE_KEY] = client_result_history
    tool_context.state[LATEST_CLIENT_RESULT_STATE_KEY] = accepted_result_data
    tool_context.state[CONVERSATION_STATUS_STATE_KEY] = ConversationStatus.RESOLVED.value
    return accepted_result_data


def finalize_recommendation(
    draft_json: str,
    tool_context: ToolContext,
) -> dict[str, object]:
    """Build and store a recommendation grounded in the trusted Analyst brief."""

    draft_data = json.loads(draft_json)
    if isinstance(draft_data.get("risks"), dict):
        draft_data["risks"] = [draft_data["risks"]]
    draft = RecommendationDraft.model_validate(draft_data)
    stored_brief_data = tool_context.state.get(TRUSTED_ANALYST_BRIEF_STATE_KEY)
    if stored_brief_data is None:
        raise RuntimeError("The Advisor requires successful research before finalization.")

    trusted_brief = ResearchBrief.model_validate(stored_brief_data)
    available_evidences = {item.evidence_id: item for item in trusted_brief.evidence}
    if not set(draft.summary.evidence_ids).issubset(available_evidences):
        raise ValueError("Recommendation summary is not supported by the research brief.")

    supported_options = [
        option
        for option in draft.options
        if set(option.description.evidence_ids).issubset(available_evidences)
    ]
    supported_risks = [
        risk for risk in draft.risks if set(risk.evidence_ids).issubset(available_evidences)
    ]
    if not supported_options or not supported_risks:
        raise ValueError("Recommendation has no supported options or risks.")

    supported_claims = [
        draft.summary,
        *(option.description for option in supported_options),
        *supported_risks,
    ]
    cited_evidence_ids = list(
        dict.fromkeys(
            evidence_id
            for claim in supported_claims
            for evidence_id in claim.evidence_ids
        )
    )
    recommendation = Recommendation(
        summary=draft.summary,
        options=supported_options,
        assumptions=draft.assumptions,
        risks=supported_risks,
        next_steps=draft.next_steps,
        citations=[
            EvidenceCitation(
                evidence_id=evidence_id,
                title=available_evidences[evidence_id].title,
                publisher=available_evidences[evidence_id].publisher,
                url=available_evidences[evidence_id].url,
            )
            for evidence_id in cited_evidence_ids
        ],
        limitations=trusted_brief.limitations,
    )
    recommendation_data = recommendation.model_dump(mode="json")
    tool_context.state[FINAL_ADVISOR_RECOMMENDATION_STATE_KEY] = recommendation_data
    tool_context.state[ADVISOR_RECOMMENDATION_HISTORY_STATE_KEY] = [
        *tool_context.state.get(ADVISOR_RECOMMENDATION_HISTORY_STATE_KEY, []),
        recommendation_data,
    ]
    return recommendation_data
