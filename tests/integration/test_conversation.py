import asyncio
import json
from collections.abc import AsyncGenerator, Sequence
from pathlib import Path
from typing import cast
from uuid import uuid4

from google.adk.models import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService
from google.genai import types
from pydantic import Field

from financial_advisor.agents.advisor.agent import (
    create_advisor_agent,
)
from financial_advisor.agents.analyst.agent import create_analyst_agent
from financial_advisor.agents.client import create_client_agent
from financial_advisor.config import (
    CONVERSATION_STATUS_STATE_KEY,
    FINAL_RECOMMENDATION_STATE_KEY,
    TRUSTED_RESEARCH_BRIEF_STATE_KEY,
)
from financial_advisor.contracts import (
    CitedText,
    ClientResult,
    Evidence,
    ResearchBrief,
)
from financial_advisor.retrieval.pipeline import RetrievalPipeline
from financial_advisor.runtime import APP_NAME, run_conversation


class ScriptedModel(BaseLlm):
    """Return deterministic responses while exercising the real ADK runner."""

    responses: list[types.Content]
    requests: list[LlmRequest] = Field(default_factory=list)

    async def generate_content_async(
        self,
        llm_request: LlmRequest,
        stream: bool = False,
    ) -> AsyncGenerator[LlmResponse, None]:
        del stream
        self.requests.append(llm_request)
        yield LlmResponse(content=self.responses.pop(0))


class FixedRetrievalPipeline:
    def __init__(self, evidence: Evidence) -> None:
        self.evidence = evidence

    def retrieve(
        self,
        query: str,
        paths: Sequence[str],
    ) -> list[Evidence]:
        assert query
        assert paths == ["local_hybrid"]
        return [self.evidence]


def model_text(text: str) -> types.Content:
    return types.Content(role="model", parts=[types.Part.from_text(text=text)])


def tool_call(name: str, args: dict[str, object]) -> types.Content:
    return types.Content(
        role="model",
        parts=[types.Part.from_function_call(name=name, args=args)],
    )


def test_complete_conversation_resolves_with_grounded_citations(tmp_path: Path) -> None:
    evidence = Evidence(
        evidence_id=uuid4(),
        title="Liquidity guidance",
        publisher="Investor.gov",
        url="https://www.investor.gov/example",
        text="Near-term goals require attention to liquidity.",
        source="vector",
    )
    cited_text = CitedText(
        text="Preserve liquidity for the near-term home goal.",
        evidence_ids=[evidence.evidence_id],
    )
    draft_brief = ResearchBrief(
        findings=[cited_text],
        evidence=[evidence],
    )
    recommendation_draft = {
        "summary": cited_text.model_dump(mode="json"),
        "options": {"Preserve liquidity": cited_text.model_dump(mode="json")},
        "assumptions": ["The five-year goal remains unchanged."],
        "risks": [cited_text.model_dump(mode="json")],
        "next_steps": ["Confirm the required down payment."],
    }

    advisor_model = ScriptedModel(
        model="scripted-advisor",
        responses=[
            tool_call("client_agent", {}),
            tool_call(
                "analyst_agent",
                {
                    "question": "How should the client balance liquidity and investing?",
                    "retrieval_paths": ["local_hybrid"],
                },
            ),
            tool_call(
                "finalize_recommendation",
                {"recommendation_draft_json": json.dumps(recommendation_draft)},
            ),
            tool_call("client_agent", {}),
        ],
    )
    client_model = ScriptedModel(
        model="scripted-client",
        responses=[
            model_text(
                ClientResult(
                    action="question",
                    message="How should I balance liquidity and investing?",
                ).model_dump_json()
            ),
            model_text(
                ClientResult(
                    action="accept",
                    message="This addresses my goal and explains the trade-offs.",
                ).model_dump_json()
            ),
        ],
    )
    analyst_model = ScriptedModel(
        model="scripted-analyst",
        responses=[model_text(draft_brief.model_dump_json())],
    )

    client_agent = create_client_agent(client_model)
    analyst_agent = create_analyst_agent(
        analyst_model,
        cast(RetrievalPipeline, FixedRetrievalPipeline(evidence)),
    )
    advisor_agent = create_advisor_agent(
        advisor_model,
        client_agent,
        analyst_agent,
    )
    runner = Runner(
        app_name=APP_NAME,
        agent=advisor_agent,
        session_service=DatabaseSessionService(
            db_url=f"sqlite+aiosqlite:///{tmp_path / 'sessions.db'}"
        ),
    )

    async def run() -> None:
        session_id = str(uuid4())
        await runner.session_service.create_session(
            app_name=APP_NAME,
            user_id="integration-user",
            session_id=session_id,
            state={
                CONVERSATION_STATUS_STATE_KEY: "active",
            },
        )
        await run_conversation(
            runner,
            "integration-user",
            session_id,
        )

        stored_session = await runner.session_service.get_session(
            app_name=APP_NAME,
            user_id="integration-user",
            session_id=session_id,
        )
        assert stored_session is not None
        assert stored_session.state[CONVERSATION_STATUS_STATE_KEY] == "resolved"
        assert stored_session.state[FINAL_RECOMMENDATION_STATE_KEY]["citations"][0][
            "evidence_id"
        ] == str(evidence.evidence_id)
        assert TRUSTED_RESEARCH_BRIEF_STATE_KEY in stored_session.state
        tool_calls = [
            call.name for event in stored_session.events for call in event.get_function_calls()
        ]
        function_responses = [
            response
            for event in stored_session.events
            for response in event.get_function_responses()
        ]
        client_results = [
            response for response in function_responses if response.name == "client_agent"
        ]
        assert tool_calls == [
            "client_agent",
            "analyst_agent",
            "finalize_recommendation",
            "client_agent",
        ]
        assert [response.name for response in function_responses] == tool_calls
        assert len(client_results) == 2
        assert client_results[0].response["action"] == "question"
        assert client_results[1].response["action"] == "accept"
        finalizer_response = next(
            response.response
            for response in function_responses
            if response.name == "finalize_recommendation"
        )
        assert finalizer_response == stored_session.state[FINAL_RECOMMENDATION_STATE_KEY]
        assert all(event.error_code is None for event in stored_session.events)
        assert not advisor_model.responses
        assert not client_model.responses
        assert not analyst_model.responses
        for request in advisor_model.requests:
            for tool in request.config.tools or []:
                assert "additional_properties" not in tool.model_dump_json(exclude_none=True)
        await runner.close()

    asyncio.run(run())
