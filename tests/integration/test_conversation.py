import asyncio
import json
from collections.abc import AsyncGenerator, Sequence
from decimal import Decimal
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
    CLIENT_RESULT_HISTORY_STATE_KEY,
    CONVERSATION_STATUS_STATE_KEY,
    FINAL_ADVISOR_RECOMMENDATION_STATE_KEY,
    create_advisor_agent,
)
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
    Finding,
    RecommendationOption,
    ResearchBrief,
    ResearchResult,
    RetrievalChannel,
    RetrievalPath,
    RetrievalResult,
)
from financial_advisor.retrieval.pipeline import RetrievalPipeline
from financial_advisor.runtime import (
    APP_NAME,
    create_conversation_session,
    run_conversation,
)


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
        paths: Sequence[RetrievalPath],
    ) -> RetrievalResult:
        assert query
        assert paths == [RetrievalPath.LOCAL_HYBRID]
        return RetrievalResult(evidence=[self.evidence])


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
        source=RetrievalChannel.VECTOR,
    )
    cited_text = CitedText(
        text="Preserve liquidity for the near-term home goal.",
        evidence_ids=[evidence.evidence_id],
    )
    draft_brief = ResearchBrief(
        findings=[Finding(text=cited_text.text, evidence_ids=cited_text.evidence_ids)],
        evidence=[evidence],
    )
    recommendation_draft = {
        "summary": cited_text.model_dump(mode="json"),
        "options": [
            RecommendationOption(
                title="Preserve liquidity",
                description=cited_text,
            ).model_dump(mode="json")
        ],
        "assumptions": ["The five-year goal remains unchanged."],
        "risks": [cited_text.model_dump(mode="json")],
        "next_steps": ["Confirm the required down payment."],
    }

    advisor_model = ScriptedModel(
        model="scripted-advisor",
        responses=[
            tool_call("client_agent", {"client_profile_json": "{}"}),
            tool_call(
                "analyst_agent",
                {
                    "question": "How should the client balance liquidity and investing?",
                    "client_profile_json": "{}",
                    "retrieval_paths": [RetrievalPath.LOCAL_HYBRID.value],
                },
            ),
            tool_call(
                "finalize_recommendation",
                {"draft_json": json.dumps(recommendation_draft)},
            ),
            tool_call("client_agent", {"client_profile_json": "{}"}),
            model_text("The Client accepted the recommendation."),
        ],
    )
    client_model = ScriptedModel(
        model="scripted-client",
        responses=[
            model_text(
                ClientResult(
                    action=ClientAction.QUESTION,
                    message="How should I balance liquidity and investing?",
                ).model_dump_json()
            ),
            model_text(
                ClientResult(
                    action=ClientAction.ACCEPT,
                    message="This addresses my goal and explains the trade-offs.",
                ).model_dump_json()
            ),
        ],
    )
    analyst_model = ScriptedModel(
        model="scripted-analyst",
        responses=[
            model_text(
                ResearchResult(success=True, brief=draft_brief).model_dump_json()
            )
        ],
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
    profile = ClientProfile(
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

    async def run() -> None:
        session_id = await create_conversation_session(
            runner,
            profile,
            "integration-user",
        )
        result = await run_conversation(
            runner,
            profile,
            "integration-user",
            session_id,
        )

        assert result.status is ConversationStatus.RESOLVED
        assert result.recommendation is not None
        assert result.recommendation.citations[0].evidence_id == evidence.evidence_id
        stored_session = await runner.session_service.get_session(
            app_name=APP_NAME,
            user_id="integration-user",
            session_id=session_id,
        )
        assert stored_session is not None
        assert stored_session.state[CONVERSATION_STATUS_STATE_KEY] == (
            ConversationStatus.RESOLVED
        )
        assert FINAL_ADVISOR_RECOMMENDATION_STATE_KEY in stored_session.state
        assert TRUSTED_ANALYST_BRIEF_STATE_KEY in stored_session.state
        assert len(stored_session.state[CLIENT_RESULT_HISTORY_STATE_KEY]) == 2
        assert all(event.error_code is None for event in stored_session.events)
        assert not advisor_model.responses
        assert not client_model.responses
        assert not analyst_model.responses
        for request in advisor_model.requests:
            for tool in request.config.tools or []:
                assert "additional_properties" not in tool.model_dump_json(
                    exclude_none=True
                )
        await runner.close()

    asyncio.run(run())
