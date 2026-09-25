from collections.abc import Sequence
from decimal import Decimal
from types import SimpleNamespace
from typing import Any, cast
from uuid import uuid4

import pytest
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.tools import AgentTool
from google.genai import types
from pydantic import ValidationError

from financial_advisor.agents.analyst import (
    ANALYST_INSTRUCTION,
    _finalize_research,
    create_analyst_agent,
)
from financial_advisor.contracts import (
    ClientProfile,
    Evidence,
    Finding,
    ResearchBrief,
    ResearchResult,
    ResearchTask,
    RetrievalChannel,
    RetrievalPath,
    RetrievalResult,
)


class StubRetrievalPipeline:
    def __init__(
        self,
        result: RetrievalResult | None = None,
        error: Exception | None = None,
    ) -> None:
        self.result = result or RetrievalResult()
        self.error = error
        self.calls: list[tuple[str, list[RetrievalPath]]] = []

    def retrieve(
        self,
        query: str,
        paths: Sequence[RetrievalPath],
    ) -> RetrievalResult:
        self.calls.append((query, list(paths)))
        if self.error is not None:
            raise self.error
        return self.result


def research_task() -> ResearchTask:
    return ResearchTask(
        question="How should I balance liquidity and investing?",
        client_profile=ClientProfile(
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
        ),
        retrieval_paths=[RetrievalPath.LOCAL_HYBRID, RetrievalPath.WEB],
    )


def callback_context(task: ResearchTask, state: dict[str, object]) -> CallbackContext:
    return cast(
        CallbackContext,
        SimpleNamespace(
            state=state,
            user_content=types.Content(
                role="user",
                parts=[types.Part.from_text(text=task.model_dump_json(exclude_none=True))],
            ),
        ),
    )


def response_text(response: LlmResponse) -> str:
    assert response.content is not None and response.content.parts is not None
    return response.content.parts[0].text or ""


def test_analyst_has_no_model_controlled_retrieval_tool() -> None:
    agent = create_analyst_agent(
        "test-model",
        retrieval_pipeline=cast(Any, StubRetrievalPipeline()),
    )

    assert agent.name == "analyst_agent"
    assert agent.model == "test-model"
    assert agent.input_schema is ResearchTask
    assert agent.output_schema is ResearchResult
    assert agent.tools == []
    assert agent.output_key is None


def test_analyst_can_be_called_as_an_agent_tool() -> None:
    agent = create_analyst_agent(
        "test-model",
        retrieval_pipeline=cast(Any, StubRetrievalPipeline()),
    )

    tool = AgentTool(agent)

    assert tool.name == "analyst_agent"
    assert tool.agent is agent


def test_analyst_prompt_requires_retrieved_evidence_ids() -> None:
    assert "Never invent an evidence ID" in ANALYST_INSTRUCTION


def test_empty_retrieval_skips_the_analyst_model() -> None:
    task = research_task()
    state: dict[str, object] = {}
    pipeline = StubRetrievalPipeline(RetrievalResult(limitations=["Web research was unavailable."]))
    agent = create_analyst_agent(
        "test-model",
        retrieval_pipeline=cast(Any, pipeline),
    )

    prepare_research = cast(Any, agent.before_model_callback)
    response = prepare_research(
        callback_context=callback_context(task, state),
        llm_request=LlmRequest(),
    )

    assert response is not None
    result = ResearchResult.model_validate_json(response_text(response))
    assert result.success is False
    assert result.brief is None
    assert state == {}
    assert pipeline.calls == [(task.question, task.retrieval_paths)]


def test_retrieval_error_propagates_to_the_workflow() -> None:
    task = research_task()
    pipeline = StubRetrievalPipeline(error=ConnectionError("store unavailable"))
    agent = create_analyst_agent(
        "test-model",
        retrieval_pipeline=cast(Any, pipeline),
    )

    prepare_research = cast(Any, agent.before_model_callback)
    with pytest.raises(ConnectionError, match="store unavailable"):
        prepare_research(
            callback_context=callback_context(task, {}),
            llm_request=LlmRequest(),
        )


def test_invalid_advisor_input_remains_a_tool_failure() -> None:
    agent = create_analyst_agent(
        "test-model",
        retrieval_pipeline=cast(Any, StubRetrievalPipeline()),
    )
    context = cast(
        CallbackContext,
        SimpleNamespace(
            state={},
            user_content=types.Content(
                role="user",
                parts=[types.Part.from_text(text="not JSON")],
            ),
        ),
    )

    prepare_research = cast(Any, agent.before_model_callback)
    with pytest.raises(ValidationError):
        prepare_research(callback_context=context, llm_request=LlmRequest())


def test_invalid_model_output_propagates_to_the_workflow() -> None:
    with pytest.raises(ValidationError):
        _finalize_research(
            callback_context(research_task(), {}),
            LlmResponse(
                content=types.Content(
                    role="model",
                    parts=[types.Part.from_text(text="not JSON")],
                )
            ),
        )


def test_missing_retrieval_state_propagates_to_the_workflow() -> None:
    task = research_task()
    evidence_id = uuid4()
    brief = ResearchBrief(
        task_id=task.task_id,
        findings=[Finding(text="Supported finding.", evidence_ids=[evidence_id])],
        evidence=[
            Evidence(
                evidence_id=evidence_id,
                title="Source",
                publisher="Publisher",
                text="Evidence.",
                source=RetrievalChannel.WEB,
            )
        ],
    )

    with pytest.raises(RuntimeError, match="retrieval state is missing"):
        _finalize_research(
            callback_context(task, {}),
            LlmResponse(
                content=types.Content(
                    role="model",
                    parts=[
                        types.Part.from_text(text=ResearchResult(brief=brief).model_dump_json())
                    ],
                )
            ),
        )


def test_missing_research_brief_propagates_to_the_workflow() -> None:
    with pytest.raises(ValueError, match="did not return a research brief"):
        _finalize_research(
            callback_context(research_task(), {}),
            LlmResponse(
                content=types.Content(
                    role="model",
                    parts=[types.Part.from_text(text=ResearchResult().model_dump_json())],
                )
            ),
        )


def test_analyst_keeps_only_claims_supported_by_retrieval() -> None:
    task = research_task()
    retrieved = Evidence(
        evidence_id=uuid4(),
        title="Liquidity guidance",
        publisher="Investor.gov",
        url="https://www.investor.gov/example",
        text="Shorter-term goals generally require attention to liquidity.",
        source=RetrievalChannel.VECTOR,
    )
    fabricated = Evidence(
        evidence_id=uuid4(),
        title="Fabricated source",
        publisher="Unknown",
        text="Unsupported content.",
        source=RetrievalChannel.WEB,
    )
    draft_brief = ResearchBrief(
        task_id=uuid4(),
        findings=[
            Finding(text="Preserve near-term liquidity.", evidence_ids=[retrieved.evidence_id]),
            Finding(text="Unsupported claim.", evidence_ids=[fabricated.evidence_id]),
        ],
        evidence=[retrieved, fabricated],
    )
    pipeline = StubRetrievalPipeline(
        RetrievalResult(
            evidence=[retrieved],
            limitations=["Current web research was unavailable."],
        )
    )
    agent = create_analyst_agent(
        "test-model",
        retrieval_pipeline=cast(Any, pipeline),
    )
    state: dict[str, object] = {}
    context = callback_context(task, state)
    llm_request = LlmRequest()

    prepare_research = cast(Any, agent.before_model_callback)
    assert prepare_research(callback_context=context, llm_request=llm_request) is None
    assert len(llm_request.contents) == 1

    draft_response = LlmResponse(
        content=types.Content(
            role="model",
            parts=[
                types.Part.from_text(
                    text=ResearchResult(
                        brief=draft_brief,
                    ).model_dump_json()
                )
            ],
        )
    )
    finalized_response = _finalize_research(context, draft_response)

    assert finalized_response is not None
    result = ResearchResult.model_validate_json(response_text(finalized_response))
    assert result.success is True
    assert result.brief is not None
    assert result.brief.task_id == task.task_id
    assert [finding.text for finding in result.brief.findings] == ["Preserve near-term liquidity."]
    assert result.brief.evidence == [retrieved]
    assert result.brief.limitations == ["Current web research was unavailable."]
    assert pipeline.calls == [(task.question, task.retrieval_paths)]
