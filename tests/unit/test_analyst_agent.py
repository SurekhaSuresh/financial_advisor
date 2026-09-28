from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any, cast
from uuid import uuid4

import pytest
from google.adk.agents.callback_context import CallbackContext
from google.adk.tools import AgentTool
from google.genai import types
from pydantic import ValidationError

from financial_advisor.agents.analyst.agent import (
    _ground_after_agent,
    create_analyst_agent,
)
from financial_advisor.agents.analyst.prompt import ANALYST_INSTRUCTION
from financial_advisor.config import (
    ANALYST_DRAFT_STATE_KEY,
    RETRIEVED_EVIDENCE_STATE_KEY,
)
from financial_advisor.contracts import (
    CitedText,
    Evidence,
    ResearchBrief,
    ResearchTask,
)


class StubRetrievalPipeline:
    def __init__(
        self,
        evidence: list[Evidence] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.evidence = evidence or []
        self.error = error
        self.calls: list[tuple[str, list[str]]] = []

    def retrieve(
        self,
        query: str,
        paths: Sequence[str],
    ) -> list[Evidence]:
        self.calls.append((query, list(paths)))
        if self.error is not None:
            raise self.error
        return self.evidence


def research_task() -> ResearchTask:
    return ResearchTask(
        question="How should I balance liquidity and investing?",
        retrieval_paths=["local_hybrid", "web"],
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


def response_text(content: types.Content) -> str:
    assert content.parts is not None
    return content.parts[0].text or ""


def test_analyst_has_no_model_controlled_retrieval_tool() -> None:
    agent = create_analyst_agent(
        "test-model",
        retrieval_pipeline=cast(Any, StubRetrievalPipeline()),
    )

    assert agent.name == "analyst_agent"
    assert agent.model == "test-model"
    assert agent.input_schema is ResearchTask
    assert agent.output_schema is ResearchBrief
    assert agent.tools == []
    assert agent.output_key == ANALYST_DRAFT_STATE_KEY


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
    assert f"{{{RETRIEVED_EVIDENCE_STATE_KEY}}}" in ANALYST_INSTRUCTION


def test_unsupported_findings_return_an_empty_brief() -> None:
    evidence = Evidence(
        evidence_id=uuid4(),
        title="Source",
        publisher="Publisher",
        text="Evidence.",
        source="web",
    )
    context = callback_context(
        research_task(),
        {
            ANALYST_DRAFT_STATE_KEY: ResearchBrief(
                findings=[CitedText(text="Unsupported finding.", evidence_ids=[uuid4()])]
            ).model_dump(mode="json"),
            RETRIEVED_EVIDENCE_STATE_KEY: [evidence.model_dump(mode="json")],
        },
    )
    response = _ground_after_agent(context)

    assert ResearchBrief.model_validate_json(response_text(response)) == ResearchBrief()


def test_empty_retrieval_skips_the_analyst_model() -> None:
    task = research_task()
    state: dict[str, object] = {}
    pipeline = StubRetrievalPipeline()
    agent = create_analyst_agent(
        "test-model",
        retrieval_pipeline=cast(Any, pipeline),
    )

    retrieve_before_agent = cast(Any, agent.before_agent_callback)
    response = retrieve_before_agent(callback_context=callback_context(task, state))

    assert response is not None
    assert response.parts is not None
    result = ResearchBrief.model_validate_json(response.parts[0].text or "")
    assert result.findings == []
    assert result.evidence == []
    assert state == {}
    assert pipeline.calls == [(task.question, task.retrieval_paths)]


def test_retrieval_error_propagates_to_the_workflow() -> None:
    task = research_task()
    pipeline = StubRetrievalPipeline(error=ConnectionError("store unavailable"))
    agent = create_analyst_agent(
        "test-model",
        retrieval_pipeline=cast(Any, pipeline),
    )

    retrieve_before_agent = cast(Any, agent.before_agent_callback)
    with pytest.raises(ConnectionError, match="store unavailable"):
        retrieve_before_agent(callback_context=callback_context(task, {}))


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

    retrieve_before_agent = cast(Any, agent.before_agent_callback)
    with pytest.raises(ValidationError):
        retrieve_before_agent(callback_context=context)


def test_missing_model_output_propagates_to_the_workflow() -> None:
    with pytest.raises(KeyError, match=ANALYST_DRAFT_STATE_KEY):
        _ground_after_agent(callback_context(research_task(), {}))


def test_grounding_replaces_model_evidence() -> None:
    task = research_task()
    evidence = Evidence(
        evidence_id=uuid4(),
        title="Liquidity guidance",
        publisher="Investor.gov",
        text="Preserve liquidity for near-term goals.",
        source="vector",
    )
    pipeline = StubRetrievalPipeline([evidence])
    context = callback_context(task, {})
    agent = create_analyst_agent("test-model", cast(Any, pipeline))
    retrieve_before_agent = cast(Any, agent.before_agent_callback)
    assert retrieve_before_agent(context) is None

    model_output = ResearchBrief(
        findings=[
            CitedText(
                text="Preserve liquidity.",
                evidence_ids=[evidence.evidence_id],
            )
        ],
        evidence=[
            Evidence(
                evidence_id=uuid4(),
                title="Model source",
                publisher="Model publisher",
                text="Model evidence.",
                source="web",
            )
        ],
    ).model_dump(mode="json")
    context.state[ANALYST_DRAFT_STATE_KEY] = model_output

    response = _ground_after_agent(context)

    result = ResearchBrief.model_validate_json(response_text(response))
    assert result.evidence == [evidence]


def test_missing_retrieval_state_propagates_to_the_workflow() -> None:
    task = research_task()
    evidence_id = uuid4()
    brief = ResearchBrief(
        findings=[CitedText(text="Supported finding.", evidence_ids=[evidence_id])],
        evidence=[
            Evidence(
                evidence_id=evidence_id,
                title="Source",
                publisher="Publisher",
                text="Evidence.",
                source="web",
            )
        ],
    )

    with pytest.raises(KeyError, match=RETRIEVED_EVIDENCE_STATE_KEY):
        _ground_after_agent(
            callback_context(
                task,
                {ANALYST_DRAFT_STATE_KEY: brief.model_dump(mode="json")},
            )
        )


def test_analyst_keeps_only_claims_supported_by_retrieval() -> None:
    task = research_task()
    retrieved = Evidence(
        evidence_id=uuid4(),
        title="Liquidity guidance",
        publisher="Investor.gov",
        url="https://www.investor.gov/example",
        text="Shorter-term goals generally require attention to liquidity.",
        source="vector",
    )
    fabricated = Evidence(
        evidence_id=uuid4(),
        title="Fabricated source",
        publisher="Unknown",
        text="Unsupported content.",
        source="web",
    )
    draft_brief = ResearchBrief(
        findings=[
            CitedText(
                text="Preserve near-term liquidity.",
                evidence_ids=[retrieved.evidence_id],
            ),
            CitedText(text="Unsupported claim.", evidence_ids=[fabricated.evidence_id]),
        ],
        scenario_comparisons=[
            CitedText(
                text="This comparison has no retrieved evidence.",
                evidence_ids=[fabricated.evidence_id],
            )
        ],
        evidence=[retrieved, fabricated],
        limitations=["The evidence does not cover tax consequences."],
    )
    pipeline = StubRetrievalPipeline([retrieved])
    agent = create_analyst_agent(
        "test-model",
        retrieval_pipeline=cast(Any, pipeline),
    )
    state: dict[str, object] = {}
    context = callback_context(task, state)

    retrieve_before_agent = cast(Any, agent.before_agent_callback)
    assert retrieve_before_agent(callback_context=context) is None
    assert RETRIEVED_EVIDENCE_STATE_KEY in state
    state[ANALYST_DRAFT_STATE_KEY] = draft_brief.model_dump(mode="json")

    finalized_response = _ground_after_agent(context)

    result = ResearchBrief.model_validate_json(response_text(finalized_response))
    assert [finding.text for finding in result.findings] == ["Preserve near-term liquidity."]
    assert result.scenario_comparisons == []
    assert result.evidence == [retrieved]
    assert result.limitations == ["The evidence does not cover tax consequences."]
    assert pipeline.calls == [(task.question, task.retrieval_paths)]
