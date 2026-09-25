"""Root Advisor that coordinates the Client and Analyst AgentTools."""

from typing import Any

from google.adk.agents import LlmAgent
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.tool_context import ToolContext
from pydantic import BaseModel, Field

from financial_advisor.agents.advisor.prompt import ADVISOR_INSTRUCTION
from financial_advisor.agents.analyst.agent import TRUSTED_ANALYST_BRIEF_STATE_KEY
from financial_advisor.contracts import (
    CitedText,
    ClientProfile,
    EvidenceCitation,
    Recommendation,
    RecommendationOption,
    ResearchBrief,
)

FINAL_ADVISOR_RECOMMENDATION_STATE_KEY = "final_advisor_recommendation"


class RecommendationDraft(BaseModel):
    """Advisor-written content awaiting deterministic evidence validation."""

    summary: CitedText
    options: list[RecommendationOption] = Field(min_length=1)
    assumptions: list[str] = Field(min_length=1)
    risks: list[CitedText] = Field(min_length=1)
    next_steps: list[str] = Field(min_length=1)


def create_advisor_agent(
    model: str,
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
    )


def _prepare_agent_tool_call(
    tool: BaseTool,
    args: dict[str, Any],
    tool_context: ToolContext,
) -> None:
    """Keep server-owned profile and recommendation data unchanged."""

    if tool.name not in {"client_agent", "analyst_agent"}:
        return
    if tool_context.user_content is None or tool_context.user_content.parts is None:
        raise ValueError("The Advisor requires a ClientProfile input.")

    profile_json = "".join(part.text or "" for part in tool_context.user_content.parts)
    trusted_profile = ClientProfile.model_validate_json(profile_json).model_dump(mode="json")
    args["client_profile"] = trusted_profile

    if tool.name == "client_agent":
        args["advisor_response"] = tool_context.state.get(
            FINAL_ADVISOR_RECOMMENDATION_STATE_KEY
        )


def finalize_recommendation(
    draft: RecommendationDraft,
    tool_context: ToolContext,
) -> dict[str, object]:
    """Build and store a recommendation grounded in the trusted Analyst brief."""

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
    return recommendation_data
