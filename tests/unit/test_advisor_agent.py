from decimal import Decimal
from types import SimpleNamespace
from typing import Any, cast
from uuid import uuid4

import pytest
from google.adk.tools import AgentTool, BaseTool, FunctionTool, ToolContext
from google.genai import types

from financial_advisor.agents.advisor.agent import (
    FINAL_ADVISOR_RECOMMENDATION_STATE_KEY,
    RecommendationDraft,
    _prepare_agent_tool_call,
    create_advisor_agent,
    finalize_recommendation,
)
from financial_advisor.agents.advisor.prompt import ADVISOR_INSTRUCTION
from financial_advisor.agents.analyst.agent import (
    TRUSTED_ANALYST_BRIEF_STATE_KEY,
    create_analyst_agent,
)
from financial_advisor.agents.client import create_client_agent
from financial_advisor.contracts import (
    CitedText,
    ClientProfile,
    Evidence,
    EvidenceCitation,
    Finding,
    Recommendation,
    RecommendationOption,
    ResearchBrief,
    RetrievalChannel,
)


def profile() -> ClientProfile:
    return ClientProfile(
        name="Maya Chen",
        age=38,
        risk_tolerance="moderate",
        emergency_fund_months=6,
        retirement_savings=Decimal("120000"),
        brokerage_savings=Decimal("35000"),
        student_loan_balance=Decimal("18000"),
        student_loan_rate_percent=Decimal("5.8"),
        primary_goal="Buy a home",
        goal_time_horizon_years=5,
    )


def research_brief() -> ResearchBrief:
    selected_evidence = Evidence(
        evidence_id=uuid4(),
        title="Liquidity guidance",
        publisher="Investor.gov",
        url="https://www.investor.gov/example",
        text="Shorter-term goals generally require attention to liquidity.",
        source=RetrievalChannel.VECTOR,
    )
    return ResearchBrief(
        task_id=uuid4(),
        findings=[
            Finding(
                text="A near-term home goal requires attention to liquidity.",
                evidence_ids=[selected_evidence.evidence_id],
            )
        ],
        evidence=[selected_evidence],
        limitations=["Tax information was not supplied."],
    )


def tool_context(state: dict[str, object]) -> ToolContext:
    return cast(
        ToolContext,
        SimpleNamespace(
            state=state,
            user_content=types.Content(
                role="user",
                parts=[types.Part.from_text(text=profile().model_dump_json())],
            ),
        ),
    )


def test_advisor_is_the_root_with_client_and_analyst_agent_tools() -> None:
    client_agent = create_client_agent("test-model")
    analyst_agent = create_analyst_agent(
        "test-model",
        retrieval_pipeline=cast(Any, SimpleNamespace()),
    )

    advisor = create_advisor_agent("test-model", client_agent, analyst_agent)

    assert advisor.name == "advisor_agent"
    assert advisor.model == "test-model"
    assert advisor.input_schema is ClientProfile
    assert advisor.output_schema is None
    assert [tool.name for tool in advisor.tools] == [
        "client_agent",
        "analyst_agent",
        "finalize_recommendation",
    ]
    assert isinstance(advisor.tools[0], AgentTool)
    assert isinstance(advisor.tools[1], AgentTool)
    assert isinstance(advisor.tools[2], FunctionTool)


def test_advisor_prompt_preserves_agent_and_safety_boundaries() -> None:
    assert "only agent permitted to interact with both" in ADVISOR_INSTRUCTION
    assert "Client and Analyst must never interact" in ADVISOR_INSTRUCTION
    assert "never create citations" in ADVISOR_INSTRUCTION
    assert "specific security or trade" in ADVISOR_INSTRUCTION


def test_analyst_receives_trusted_profile_without_changing_stored_research() -> None:
    stored_brief = research_brief()
    state: dict[str, object] = {
        TRUSTED_ANALYST_BRIEF_STATE_KEY: stored_brief.model_dump(mode="json")
    }
    tool_args: dict[str, Any] = {
        "client_profile": profile().model_copy(update={"age": 99}).model_dump(mode="json"),
    }

    _prepare_agent_tool_call(
        tool=cast(BaseTool, SimpleNamespace(name="analyst_agent")),
        args=tool_args,
        tool_context=tool_context(state),
    )

    assert ClientProfile.model_validate(tool_args["client_profile"]) == profile()
    assert ResearchBrief.model_validate(state[TRUSTED_ANALYST_BRIEF_STATE_KEY]) == stored_brief


def test_client_receives_only_the_stored_final_recommendation() -> None:
    brief = research_brief()
    evidence_id = brief.evidence[0].evidence_id
    cited_text = CitedText(text="Preserve liquidity.", evidence_ids=[evidence_id])
    stored_recommendation = Recommendation(
        summary=cited_text,
        options=[RecommendationOption(title="Preserve liquidity", description=cited_text)],
        assumptions=["The goal remains unchanged."],
        risks=[cited_text],
        next_steps=["Confirm the goal amount."],
        citations=[
            EvidenceCitation(
                evidence_id=evidence_id,
                title=brief.evidence[0].title,
                publisher=brief.evidence[0].publisher,
                url=brief.evidence[0].url,
            )
        ],
    ).model_dump(mode="json")
    tool_args: dict[str, Any] = {
        "client_profile": profile().model_dump(mode="json"),
        "advisor_response": stored_recommendation,
        "follow_up_count": 0,
    }

    _prepare_agent_tool_call(
        tool=cast(BaseTool, SimpleNamespace(name="client_agent")),
        args=tool_args,
        tool_context=tool_context({}),
    )

    assert tool_args["advisor_response"] is None

    _prepare_agent_tool_call(
        tool=cast(BaseTool, SimpleNamespace(name="client_agent")),
        args=tool_args,
        tool_context=tool_context(
            {FINAL_ADVISOR_RECOMMENDATION_STATE_KEY: stored_recommendation}
        ),
    )

    assert tool_args["advisor_response"] == stored_recommendation


def test_finalizer_filters_unsupported_claims_and_stores_recommendation() -> None:
    brief = research_brief()
    evidence_id = brief.evidence[0].evidence_id
    supported = CitedText(
        text="Preserve liquidity for the near-term goal.",
        evidence_ids=[evidence_id],
    )
    unsupported = CitedText(text="Unsupported claim.", evidence_ids=[uuid4()])
    draft = RecommendationDraft(
        summary=supported,
        options=[
            RecommendationOption(title="Preserve liquidity", description=supported),
            RecommendationOption(title="Unsupported", description=unsupported),
        ],
        assumptions=["The five-year goal remains unchanged."],
        risks=[supported, unsupported],
        next_steps=["Confirm the required down payment."],
    )
    state: dict[str, object] = {
        TRUSTED_ANALYST_BRIEF_STATE_KEY: brief.model_dump(mode="json")
    }

    result = Recommendation.model_validate(
        finalize_recommendation(draft, tool_context(state))
    )

    assert [option.title for option in result.options] == ["Preserve liquidity"]
    assert result.risks == [supported]
    assert [citation.evidence_id for citation in result.citations] == [evidence_id]
    assert result.limitations == brief.limitations
    assert (
        Recommendation.model_validate(state[FINAL_ADVISOR_RECOMMENDATION_STATE_KEY])
        == result
    )


def test_finalizer_requires_successful_research() -> None:
    supported = CitedText(text="Claim.", evidence_ids=[uuid4()])
    draft = RecommendationDraft(
        summary=supported,
        options=[RecommendationOption(title="Option", description=supported)],
        assumptions=["Assumption."],
        risks=[supported],
        next_steps=["Next step."],
    )

    with pytest.raises(RuntimeError, match="requires successful research"):
        finalize_recommendation(draft, tool_context({}))


def test_finalizer_rejects_an_unsupported_summary() -> None:
    brief = research_brief()
    supported = CitedText(
        text="Supported claim.",
        evidence_ids=[brief.evidence[0].evidence_id],
    )
    draft = RecommendationDraft(
        summary=CitedText(text="Unsupported summary.", evidence_ids=[uuid4()]),
        options=[RecommendationOption(title="Supported", description=supported)],
        assumptions=["The goal remains unchanged."],
        risks=[supported],
        next_steps=["Confirm the goal amount."],
    )

    with pytest.raises(ValueError, match="summary is not supported"):
        finalize_recommendation(
            draft,
            tool_context(
                {TRUSTED_ANALYST_BRIEF_STATE_KEY: brief.model_dump(mode="json")}
            ),
        )
