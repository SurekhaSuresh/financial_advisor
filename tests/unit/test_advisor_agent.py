import json
from decimal import Decimal
from types import SimpleNamespace
from typing import Any, cast
from uuid import uuid4

import pytest
from google.adk.tools import AgentTool, BaseTool, FunctionTool, ToolContext
from google.genai import types

from financial_advisor.agents.advisor.agent import (
    ADVISOR_RECOMMENDATION_HISTORY_STATE_KEY,
    CLIENT_FOLLOW_UP_COUNT_STATE_KEY,
    CLIENT_RESULT_HISTORY_STATE_KEY,
    CONVERSATION_STATUS_STATE_KEY,
    FINAL_ADVISOR_RECOMMENDATION_STATE_KEY,
    RESEARCH_ATTEMPT_COUNT_STATE_KEY,
    RecommendationDraft,
    _prepare_agent_tool_call,
    _record_agent_tool_result,
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
    ClientAction,
    ClientProfile,
    ClientResult,
    ConversationStatus,
    Evidence,
    EvidenceCitation,
    Finding,
    Recommendation,
    RecommendationOption,
    ResearchBrief,
    ResearchResult,
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
    tool_declarations = [tool._get_declaration() for tool in advisor.tools]
    assert [declaration.name for declaration in tool_declarations] == [
        "client_agent",
        "analyst_agent",
        "finalize_recommendation",
    ]
    for declaration in tool_declarations:
        assert "additional_properties" not in declaration.model_dump_json(
            exclude_none=True
        )
    assert advisor.before_tool_callback is _prepare_agent_tool_call
    assert advisor.after_tool_callback is _record_agent_tool_result


def test_advisor_prompt_preserves_agent_and_safety_boundaries() -> None:
    assert "only agent permitted to interact with both" in ADVISOR_INSTRUCTION
    assert "Client and Analyst must never interact" in ADVISOR_INSTRUCTION
    assert "progress summary as plain text" in ADVISOR_INSTRUCTION
    assert "never create citations" in ADVISOR_INSTRUCTION
    assert "specific security or trade" in ADVISOR_INSTRUCTION


def test_analyst_receives_trusted_profile_without_changing_stored_research() -> None:
    stored_brief = research_brief()
    state: dict[str, object] = {
        TRUSTED_ANALYST_BRIEF_STATE_KEY: stored_brief.model_dump(mode="json")
    }
    tool_args: dict[str, Any] = {
        "client_profile_json": profile().model_copy(update={"age": 99}).model_dump_json(),
    }

    _prepare_agent_tool_call(
        tool=cast(BaseTool, SimpleNamespace(name="analyst_agent")),
        args=tool_args,
        tool_context=tool_context(state),
    )

    assert ClientProfile.model_validate_json(tool_args["client_profile_json"]) == profile()
    assert ResearchBrief.model_validate(state[TRUSTED_ANALYST_BRIEF_STATE_KEY]) == stored_brief
    assert state[RESEARCH_ATTEMPT_COUNT_STATE_KEY] == 1


def test_advisor_limits_research_without_replacing_a_successful_brief() -> None:
    stored_brief = research_brief()
    state: dict[str, object] = {
        TRUSTED_ANALYST_BRIEF_STATE_KEY: stored_brief.model_dump(mode="json"),
        RESEARCH_ATTEMPT_COUNT_STATE_KEY: 2,
    }

    result = _prepare_agent_tool_call(
        tool=cast(BaseTool, SimpleNamespace(name="analyst_agent")),
        args={},
        tool_context=tool_context(state),
    )

    assert result == {"success": False, "brief": None}
    assert ResearchBrief.model_validate(state[TRUSTED_ANALYST_BRIEF_STATE_KEY]) == stored_brief
    assert state[RESEARCH_ATTEMPT_COUNT_STATE_KEY] == 2


def test_advisor_makes_analyst_result_json_serializable() -> None:
    result = _record_agent_tool_result(
        cast(BaseTool, SimpleNamespace(name="analyst_agent")),
        {},
        tool_context({}),
        ResearchResult(success=True, brief=research_brief()).model_dump(),
    )

    json.dumps(result)


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
        "client_profile_json": profile().model_dump_json(),
        "advisor_response_json": json.dumps(stored_recommendation),
        "follow_up_count": 0,
    }

    _prepare_agent_tool_call(
        tool=cast(BaseTool, SimpleNamespace(name="client_agent")),
        args=tool_args,
        tool_context=tool_context({}),
    )

    assert tool_args["advisor_response_json"] is None

    _prepare_agent_tool_call(
        tool=cast(BaseTool, SimpleNamespace(name="client_agent")),
        args=tool_args,
        tool_context=tool_context(
            {FINAL_ADVISOR_RECOMMENDATION_STATE_KEY: stored_recommendation}
        ),
    )

    assert Recommendation.model_validate_json(tool_args["advisor_response_json"]) == (
        Recommendation.model_validate(stored_recommendation)
    )


def test_advisor_tracks_client_follow_ups_and_resolves_after_the_limit() -> None:
    state: dict[str, object] = {}
    context = tool_context(state)
    tool = cast(BaseTool, SimpleNamespace(name="client_agent"))
    args: dict[str, Any] = {"advisor_response_json": "{}"}
    follow_up = ClientResult(
        action=ClientAction.QUESTION,
        message="How would this change if my timeline shortened?",
    ).model_dump(mode="json")

    assert _record_agent_tool_result(tool, args, context, follow_up) is None
    assert state[CLIENT_FOLLOW_UP_COUNT_STATE_KEY] == 1
    assert _record_agent_tool_result(tool, args, context, follow_up) is None
    assert state[CLIENT_FOLLOW_UP_COUNT_STATE_KEY] == 2

    result = _record_agent_tool_result(tool, args, context, follow_up)

    assert ClientResult.model_validate(result).action is ClientAction.ACCEPT
    assert state[CLIENT_FOLLOW_UP_COUNT_STATE_KEY] == 2
    assert state[CONVERSATION_STATUS_STATE_KEY] == ConversationStatus.RESOLVED
    client_result_history = cast(
        list[object],
        state[CLIENT_RESULT_HISTORY_STATE_KEY],
    )
    assert ClientResult.model_validate(
        client_result_history[-1]
    ).action is ClientAction.ACCEPT


def test_advisor_resolves_when_client_accepts_the_recommendation() -> None:
    state: dict[str, object] = {}

    result = _record_agent_tool_result(
        cast(BaseTool, SimpleNamespace(name="client_agent")),
        {"advisor_response_json": "{}"},
        tool_context(state),
        ClientResult(
            action=ClientAction.ACCEPT,
            message="This addresses my goal and trade-offs.",
        ).model_dump(mode="json"),
    )

    assert result is None
    assert state[CONVERSATION_STATUS_STATE_KEY] == ConversationStatus.RESOLVED
    assert len(cast(list[object], state[CLIENT_RESULT_HISTORY_STATE_KEY])) == 1


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
        finalize_recommendation(
            draft_json=draft.model_dump_json(),
            tool_context=tool_context(state),
        )
    )

    assert [option.title for option in result.options] == ["Preserve liquidity"]
    assert result.risks == [supported]
    assert [citation.evidence_id for citation in result.citations] == [evidence_id]
    assert result.limitations == brief.limitations
    assert (
        Recommendation.model_validate(state[FINAL_ADVISOR_RECOMMENDATION_STATE_KEY])
        == result
    )
    recommendation_history = cast(
        list[object],
        state[ADVISOR_RECOMMENDATION_HISTORY_STATE_KEY],
    )
    assert Recommendation.model_validate(recommendation_history[0]) == result


def test_finalizer_accepts_one_risk_object() -> None:
    brief = research_brief()
    evidence_id = brief.evidence[0].evidence_id
    cited_text = CitedText(text="Preserve liquidity.", evidence_ids=[evidence_id])
    draft = RecommendationDraft(
        summary=cited_text,
        options=[RecommendationOption(title="Preserve liquidity", description=cited_text)],
        assumptions=["The goal remains unchanged."],
        risks=[cited_text],
        next_steps=["Confirm the goal amount."],
    ).model_dump(mode="json")
    draft["risks"] = draft["risks"][0]

    result = Recommendation.model_validate(
        finalize_recommendation(
            draft_json=json.dumps(draft),
            tool_context=tool_context(
                {TRUSTED_ANALYST_BRIEF_STATE_KEY: brief.model_dump(mode="json")}
            ),
        )
    )

    assert result.risks == [cited_text]


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
        finalize_recommendation(
            draft_json=draft.model_dump_json(),
            tool_context=tool_context({}),
        )


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
            draft_json=draft.model_dump_json(),
            tool_context=tool_context(
                {TRUSTED_ANALYST_BRIEF_STATE_KEY: brief.model_dump(mode="json")}
            ),
        )
