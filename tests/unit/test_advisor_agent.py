import json
from types import SimpleNamespace
from typing import Any, cast
from uuid import uuid4

from google.adk.tools import AgentTool, BaseTool, FunctionTool, ToolContext

from financial_advisor.agents.advisor.agent import (
    _prepare_agent_tool_call,
    _record_agent_tool_result,
    create_advisor_agent,
    escalate_conversation,
    finalize_recommendation,
)
from financial_advisor.agents.advisor.prompt import ADVISOR_INSTRUCTION
from financial_advisor.agents.analyst.agent import create_analyst_agent
from financial_advisor.agents.client import create_client_agent
from financial_advisor.config import (
    CLIENT_FOLLOW_UP_COUNT_STATE_KEY,
    CONVERSATION_STATUS_STATE_KEY,
    CURRENT_CLIENT_QUESTION_STATE_KEY,
    ESCALATION_MESSAGE,
    FINAL_RECOMMENDATION_STATE_KEY,
    MAX_RESEARCH_ATTEMPTS,
    RECOMMENDATION_REDRAFT_COUNT_STATE_KEY,
    RECOMMENDATION_REDRAFT_MESSAGE,
    RESEARCH_ATTEMPT_COUNT_STATE_KEY,
    TRUSTED_RESEARCH_BRIEF_STATE_KEY,
)
from financial_advisor.contracts import (
    CitedText,
    ClientResult,
    Evidence,
    EvidenceCitation,
    Recommendation,
    RecommendationDraft,
    ResearchBrief,
)


def research_brief() -> ResearchBrief:
    selected_evidence = Evidence(
        evidence_id=uuid4(),
        title="Liquidity guidance",
        publisher="Investor.gov",
        url="https://www.investor.gov/example",
        text="Shorter-term goals generally require attention to liquidity.",
        source="vector",
    )
    return ResearchBrief(
        findings=[
            CitedText(
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
            actions=SimpleNamespace(skip_summarization=False),
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
    assert advisor.input_schema is None
    assert advisor.output_schema is None
    assert [tool.name for tool in advisor.tools] == [
        "client_agent",
        "analyst_agent",
        "finalize_recommendation",
        "escalate_conversation",
    ]
    assert isinstance(advisor.tools[0], AgentTool)
    assert isinstance(advisor.tools[1], AgentTool)
    assert isinstance(advisor.tools[2], FunctionTool)
    assert isinstance(advisor.tools[3], FunctionTool)
    tool_declarations = [tool._get_declaration() for tool in advisor.tools]
    assert [declaration.name for declaration in tool_declarations] == [
        "client_agent",
        "analyst_agent",
        "finalize_recommendation",
        "escalate_conversation",
    ]
    for declaration in tool_declarations:
        assert "additional_properties" not in declaration.model_dump_json(exclude_none=True)
    assert advisor.before_tool_callback is _prepare_agent_tool_call
    assert advisor.after_tool_callback is _record_agent_tool_result


def test_advisor_prompt_preserves_agent_and_safety_boundaries() -> None:
    assert "sole coordinator" in ADVISOR_INSTRUCTION
    assert "Client and Analyst must not interact directly" in ADVISOR_INSTRUCTION
    assert "progress summary as plain text" in ADVISOR_INSTRUCTION
    assert "do not invent citation metadata" in ADVISOR_INSTRUCTION.lower()
    assert "specific security or trade" in ADVISOR_INSTRUCTION
    assert "Never invent or assume Client facts" in ADVISOR_INSTRUCTION
    assert "Follow the workflow exactly in order" in ADVISOR_INSTRUCTION
    assert f"research_attempts_maximum: {MAX_RESEARCH_ATTEMPTS}" in ADVISOR_INSTRUCTION
    assert "{client_follow_up_count?}" in ADVISOR_INSTRUCTION
    assert "{research_attempt_count?}" in ADVISOR_INSTRUCTION
    assert json.dumps(RecommendationDraft.model_json_schema()) in ADVISOR_INSTRUCTION


def test_analyst_call_counts_attempt_without_changing_stored_research() -> None:
    stored_brief = research_brief()
    state: dict[str, object] = {
        TRUSTED_RESEARCH_BRIEF_STATE_KEY: stored_brief.model_dump(mode="json")
    }
    tool_args: dict[str, Any] = {}

    _prepare_agent_tool_call(
        tool=cast(BaseTool, SimpleNamespace(name="analyst_agent")),
        args=tool_args,
        tool_context=tool_context(state),
    )

    assert tool_args == {}
    assert ResearchBrief.model_validate(state[TRUSTED_RESEARCH_BRIEF_STATE_KEY]) == stored_brief
    assert state[RESEARCH_ATTEMPT_COUNT_STATE_KEY] == 1


def test_advisor_makes_analyst_result_json_serializable() -> None:
    state: dict[str, object] = {}
    result = _record_agent_tool_result(
        cast(BaseTool, SimpleNamespace(name="analyst_agent")),
        {},
        tool_context(state),
        research_brief().model_dump(),
    )

    json.dumps(result)
    assert ResearchBrief.model_validate(state[TRUSTED_RESEARCH_BRIEF_STATE_KEY])


def test_empty_research_keeps_the_previous_brief() -> None:
    previous_brief = research_brief()
    state: dict[str, object] = {
        TRUSTED_RESEARCH_BRIEF_STATE_KEY: previous_brief.model_dump(mode="json")
    }

    _record_agent_tool_result(
        cast(BaseTool, SimpleNamespace(name="analyst_agent")),
        {},
        tool_context(state),
        ResearchBrief().model_dump(),
    )

    assert ResearchBrief.model_validate(state[TRUSTED_RESEARCH_BRIEF_STATE_KEY]) == previous_brief


def test_client_receives_only_the_stored_final_recommendation() -> None:
    brief = research_brief()
    evidence_id = brief.evidence[0].evidence_id
    cited_text = CitedText(text="Preserve liquidity.", evidence_ids=[evidence_id])
    stored_recommendation = Recommendation(
        summary=cited_text,
        options={"Preserve liquidity": cited_text},
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
        tool_context=tool_context({FINAL_RECOMMENDATION_STATE_KEY: stored_recommendation}),
    )

    assert Recommendation.model_validate_json(tool_args["advisor_response_json"]) == (
        Recommendation.model_validate(stored_recommendation)
    )


def test_advisor_resolves_after_answering_the_last_client_follow_up() -> None:
    state: dict[str, object] = {}
    context = tool_context(state)
    tool = cast(BaseTool, SimpleNamespace(name="client_agent"))
    args: dict[str, Any] = {"advisor_response_json": "{}"}
    follow_up = ClientResult(
        action="question",
        message="How would this change if my timeline shortened?",
    ).model_dump(mode="json")

    assert _record_agent_tool_result(tool, args, context, follow_up) is None
    assert state[CLIENT_FOLLOW_UP_COUNT_STATE_KEY] == 1
    assert _record_agent_tool_result(tool, args, context, follow_up) is None
    assert state[CLIENT_FOLLOW_UP_COUNT_STATE_KEY] == 2
    assert state[CURRENT_CLIENT_QUESTION_STATE_KEY] == follow_up["message"]

    brief = research_brief()
    evidence_id = brief.evidence[0].evidence_id
    supported = CitedText(text="Preserve liquidity.", evidence_ids=[evidence_id])
    draft = RecommendationDraft(
        summary=supported,
        options={"Preserve liquidity": supported},
        assumptions=["The goal remains unchanged."],
        risks=[supported],
        next_steps=["Confirm the goal amount."],
    )
    state[TRUSTED_RESEARCH_BRIEF_STATE_KEY] = brief.model_dump(mode="json")

    Recommendation.model_validate(finalize_recommendation(draft.model_dump_json(), context))
    assert state[CLIENT_FOLLOW_UP_COUNT_STATE_KEY] == 2
    assert state[CONVERSATION_STATUS_STATE_KEY] == "resolved"
    assert context.actions.skip_summarization is True


def test_advisor_resolves_when_client_accepts_the_recommendation() -> None:
    state: dict[str, object] = {}

    result = _record_agent_tool_result(
        cast(BaseTool, SimpleNamespace(name="client_agent")),
        {"advisor_response_json": "{}"},
        tool_context(state),
        ClientResult(
            action="accept",
            message="This addresses my goal and trade-offs.",
        ).model_dump(mode="json"),
    )

    assert result is None
    assert state[CONVERSATION_STATUS_STATE_KEY] == "resolved"


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
        options={"Preserve liquidity": supported, "Unsupported": unsupported},
        assumptions=["The five-year goal remains unchanged."],
        risks=[supported, unsupported],
        next_steps=["Confirm the required down payment."],
    )
    state: dict[str, object] = {TRUSTED_RESEARCH_BRIEF_STATE_KEY: brief.model_dump(mode="json")}

    result = Recommendation.model_validate(
        finalize_recommendation(
            recommendation_draft_json=draft.model_dump_json(),
            tool_context=tool_context(state),
        )
    )

    assert list(result.options) == ["Preserve liquidity"]
    assert result.risks == [supported]
    assert [citation.evidence_id for citation in result.citations] == [evidence_id]
    assert result.limitations == brief.limitations
    assert Recommendation.model_validate(state[FINAL_RECOMMENDATION_STATE_KEY]) == result


def test_finalizer_requests_one_redraft_then_escalates() -> None:
    brief = research_brief()
    supported = CitedText(
        text="Supported claim.",
        evidence_ids=[brief.evidence[0].evidence_id],
    )
    draft = RecommendationDraft(
        summary=CitedText(text="Unsupported summary.", evidence_ids=[uuid4()]),
        options={"Supported": supported},
        assumptions=["The goal remains unchanged."],
        risks=[supported],
        next_steps=["Confirm the goal amount."],
    )

    state: dict[str, object] = {TRUSTED_RESEARCH_BRIEF_STATE_KEY: brief.model_dump(mode="json")}
    context = tool_context(state)

    assert finalize_recommendation(draft.model_dump_json(), context) == {
        "message": RECOMMENDATION_REDRAFT_MESSAGE
    }
    assert state[RECOMMENDATION_REDRAFT_COUNT_STATE_KEY] == 1
    assert finalize_recommendation(draft.model_dump_json(), context) == {
        "message": ESCALATION_MESSAGE
    }
    assert state[CONVERSATION_STATUS_STATE_KEY] == "escalated"
    assert context.actions.skip_summarization is True


def test_advisor_blocks_a_third_research_call() -> None:
    state: dict[str, object] = {
        RESEARCH_ATTEMPT_COUNT_STATE_KEY: MAX_RESEARCH_ATTEMPTS,
    }
    context = tool_context(state)

    result = _prepare_agent_tool_call(
        cast(BaseTool, SimpleNamespace(name="analyst_agent")),
        {},
        context,
    )

    assert result == {"message": ESCALATION_MESSAGE}
    assert state[CONVERSATION_STATUS_STATE_KEY] == "escalated"
    assert context.actions.skip_summarization is True


def test_advisor_does_not_replace_the_research_escalation_response() -> None:
    response = {"message": ESCALATION_MESSAGE}

    result = _record_agent_tool_result(
        cast(BaseTool, SimpleNamespace(name="analyst_agent")),
        {},
        tool_context({CONVERSATION_STATUS_STATE_KEY: "escalated"}),
        response,
    )

    assert result is None


def test_advisor_reuses_previous_brief_when_research_is_exhausted() -> None:
    previous_brief = research_brief().model_dump(mode="json")
    state: dict[str, object] = {
        RESEARCH_ATTEMPT_COUNT_STATE_KEY: MAX_RESEARCH_ATTEMPTS,
        TRUSTED_RESEARCH_BRIEF_STATE_KEY: previous_brief,
    }
    context = tool_context(state)

    result = _prepare_agent_tool_call(
        cast(BaseTool, SimpleNamespace(name="analyst_agent")),
        {},
        context,
    )

    assert result == previous_brief
    assert CONVERSATION_STATUS_STATE_KEY not in state
    assert context.actions.skip_summarization is False


def test_escalation_ends_the_conversation() -> None:
    state: dict[str, object] = {}
    context = tool_context(state)

    assert escalate_conversation(context) == {"message": ESCALATION_MESSAGE}
    assert state[CONVERSATION_STATUS_STATE_KEY] == "escalated"
    assert context.actions.skip_summarization is True


def test_escalation_uses_the_advisor_summary() -> None:
    context = tool_context({})

    assert escalate_conversation(context, "This request requires human advice.") == {
        "message": "This request requires human advice."
    }
