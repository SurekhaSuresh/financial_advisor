import asyncio
import os
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock
from uuid import uuid4

import pytest
from google.adk.agents import RunConfig
from google.adk.events import Event, EventActions
from google.adk.models import Gemini
from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService
from google.adk.tools import AgentTool
from google.genai import types

from financial_advisor import runtime as runtime_module
from financial_advisor.config import (
    CONVERSATION_STATUS_STATE_KEY,
    ESCALATION_MESSAGE,
    FINAL_RECOMMENDATION_STATE_KEY,
    MAX_CONVERSATION_LLM_CALLS,
    MODEL_REQUEST_ATTEMPTS,
    MODEL_REQUEST_TIMEOUT_MILLISECONDS,
    ApplicationSettings,
)
from financial_advisor.contracts import (
    CitedText,
    EvidenceCitation,
    Recommendation,
)
from financial_advisor.runtime import (
    APP_NAME,
    create_runner,
    run_conversation,
)


def recommendation() -> Recommendation:
    evidence_id = uuid4()
    supported_text = CitedText(
        text="Preserve liquidity for the home purchase.",
        evidence_ids=[evidence_id],
    )
    return Recommendation(
        summary=supported_text,
        options={"Preserve liquidity": supported_text},
        assumptions=["The five-year goal remains unchanged."],
        risks=[supported_text],
        next_steps=["Confirm the required down payment."],
        citations=[
            EvidenceCitation(
                evidence_id=evidence_id,
                title="Liquidity guidance",
                publisher="Investor.gov",
                url="https://www.investor.gov/example",
            )
        ],
    )


class StubRunner:
    def __init__(
        self,
        session_service: DatabaseSessionService,
        *,
        result: Recommendation | None = None,
        error: Exception | None = None,
        delay_seconds: float = 0,
    ) -> None:
        self.session_service = session_service
        self.result = result
        self.error = error
        self.delay_seconds = delay_seconds
        self.run_config: RunConfig | None = None
        self.initial_status: object | None = None

    async def run_async(
        self,
        *,
        user_id: str,
        session_id: str,
        new_message: types.Content,
        run_config: RunConfig,
    ) -> AsyncIterator[Event]:
        del new_message
        self.run_config = run_config
        session = await self.session_service.get_session(
            app_name=APP_NAME,
            user_id=user_id,
            session_id=session_id,
        )
        assert session is not None
        self.initial_status = session.state.get(CONVERSATION_STATUS_STATE_KEY)
        if self.error is not None:
            event = Event(
                invocation_id="test-invocation",
                author="advisor_agent",
            )
            await self.session_service.append_event(
                session,
                event,
            )
            yield event
            raise self.error
        if self.delay_seconds:
            await asyncio.sleep(self.delay_seconds)

        state_delta = (
            {
                CONVERSATION_STATUS_STATE_KEY: "resolved",
                FINAL_RECOMMENDATION_STATE_KEY: self.result.model_dump(mode="json"),
            }
            if self.result is not None
            else {}
        )
        event = Event(
            invocation_id="test-invocation",
            author="advisor_agent",
            actions=EventActions(state_delta=state_delta),
        )
        await self.session_service.append_event(session, event)
        yield event


async def run_test_conversation(runner: StubRunner) -> str:
    session_id = str(uuid4())
    await runner.session_service.create_session(
        app_name=APP_NAME,
        user_id="test-user",
        session_id=session_id,
        state={
            CONVERSATION_STATUS_STATE_KEY: "active",
        },
    )
    await run_conversation(
        cast(Runner, runner),
        "test-user",
        session_id,
    )
    return session_id


def test_runner_assembles_configured_application(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    knowledge_database_path = tmp_path / "knowledge"
    knowledge_database_path.mkdir()
    retrieval_models = SimpleNamespace(
        encode=lambda _text: [],
        decode=lambda _token_ids: "",
        embed=lambda _texts: [],
        rerank=lambda _query, _documents: [],
    )
    local_retriever = object()
    web_retriever = object()
    retrieval_pipeline = object()
    runner = object()
    model_factory = Mock(return_value=retrieval_models)
    local_retriever_factory = Mock(return_value=local_retriever)
    web_retriever_factory = Mock(return_value=web_retriever)
    web_discovery = object()
    web_discovery_factory = Mock(return_value=web_discovery)
    pipeline_factory = Mock(return_value=retrieval_pipeline)
    runner_factory = Mock(return_value=runner)
    session_service = object()
    session_service_factory = Mock(return_value=session_service)

    monkeypatch.setattr(runtime_module, "LocalRetrievalModels", model_factory)
    monkeypatch.setattr(runtime_module, "LocalHybridRetriever", local_retriever_factory)
    monkeypatch.setattr(runtime_module, "WebCandidateRetriever", web_retriever_factory)
    monkeypatch.setattr(runtime_module, "WebSourceDiscovery", web_discovery_factory)
    monkeypatch.setattr(runtime_module, "RetrievalPipeline", pipeline_factory)
    monkeypatch.setattr(runtime_module, "DatabaseSessionService", session_service_factory)
    monkeypatch.setattr(runtime_module, "Runner", runner_factory)

    settings = ApplicationSettings(
        _env_file=None,
        google_api_key="google-key",
        exa_api_key="exa-key",
        brave_search_api_key="brave-key",
        advisor_analyst_model_name="advisor-model",
        client_model_name="client-model",
        session_database_path=Path("sessions.db"),
        knowledge_database_path=knowledge_database_path,
        model_cache_directory=tmp_path / "models",
    )

    result = create_runner(settings)

    assert result is runner
    assert os.environ["GOOGLE_API_KEY"] == "google-key"
    model_factory.assert_called_once_with(tmp_path / "models")
    local_retriever_factory.assert_called_once_with(
        knowledge_database_path,
        retrieval_models.embed,
    )
    web_discovery_factory.assert_called_once_with("exa-key", "brave-key")
    web_retriever_factory.assert_called_once_with(
        retrieval_models,
        web_discovery,
    )
    pipeline_factory.assert_called_once_with(
        local_retriever,
        retrieval_models.rerank,
        web_retriever,
    )
    session_service_factory.assert_called_once_with(
        db_url="sqlite+aiosqlite:///data/adk_sessions.db"
    )
    assert runner_factory.call_args.kwargs["app_name"] == APP_NAME
    assert runner_factory.call_args.kwargs["session_service"] is session_service


def test_runner_allows_unconfigured_retrieval_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retrieval_models = SimpleNamespace(embed=lambda _texts: [], rerank=lambda _query, _docs: [])
    pipeline_factory = Mock(return_value=object())
    local_retriever_factory = Mock(return_value=object())
    monkeypatch.setattr(runtime_module, "LocalRetrievalModels", Mock(return_value=retrieval_models))
    monkeypatch.setattr(runtime_module, "LocalHybridRetriever", local_retriever_factory)
    monkeypatch.setattr(runtime_module, "RetrievalPipeline", pipeline_factory)
    monkeypatch.setattr(runtime_module, "Runner", Mock(return_value=object()))

    create_runner(
        ApplicationSettings(
            _env_file=None,
            google_api_key="google-key",
            knowledge_database_path=tmp_path / "missing-knowledge",
        )
    )

    local_retriever_factory.assert_not_called()
    assert pipeline_factory.call_args.args[0] is None
    assert pipeline_factory.call_args.args[2] is None


def test_runner_persists_session_events_and_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_url = f"sqlite+aiosqlite:///{tmp_path / 'sessions.db'}"
    monkeypatch.setattr(runtime_module, "SESSION_DATABASE_URL", database_url)
    monkeypatch.setattr(
        runtime_module,
        "LocalRetrievalModels",
        Mock(
            return_value=SimpleNamespace(
                embed=lambda _texts: [],
                rerank=lambda _query, _docs: [],
            )
        ),
    )
    runner = create_runner(
        ApplicationSettings(
            _env_file=None,
            google_api_key="google-key",
            advisor_analyst_model_name="test-advisor-analyst-model",
            client_model_name="test-client-model",
            session_database_path=tmp_path / "sessions.db",
            knowledge_database_path=tmp_path / "missing-knowledge",
        )
    )
    advisor = runner.agent
    assert isinstance(advisor.model, Gemini)
    assert advisor.model.retry_options is not None
    assert advisor.model.retry_options.attempts == MODEL_REQUEST_ATTEMPTS
    assert advisor.generate_content_config is not None
    assert advisor.generate_content_config.http_options is not None
    assert (
        advisor.generate_content_config.http_options.timeout == MODEL_REQUEST_TIMEOUT_MILLISECONDS
    )

    client_tool, analyst_tool = advisor.tools[:2]
    assert isinstance(client_tool, AgentTool)
    assert isinstance(client_tool.agent.model, Gemini)
    assert client_tool.agent.model.model == "test-client-model"
    assert client_tool.agent.model is not advisor.model
    assert isinstance(analyst_tool, AgentTool)
    assert analyst_tool.agent.model is advisor.model

    for tool in (client_tool, analyst_tool):
        assert isinstance(tool, AgentTool)
        assert tool.agent.generate_content_config is not None
        assert tool.agent.generate_content_config.http_options is not None
        assert (
            tool.agent.generate_content_config.http_options.timeout
            == MODEL_REQUEST_TIMEOUT_MILLISECONDS
        )

    async def verify_persistence() -> None:
        session = await runner.session_service.create_session(
            app_name=APP_NAME,
            user_id="test-user",
            session_id="test-session",
        )
        await runner.session_service.append_event(
            session,
            Event(
                invocation_id="test-invocation",
                author="advisor_agent",
                actions=EventActions(state_delta={"status": "active"}),
            ),
        )
        await runner.session_service.close()

        recovered_service = DatabaseSessionService(db_url=database_url)
        recovered_session = await recovered_service.get_session(
            app_name=APP_NAME,
            user_id="test-user",
            session_id="test-session",
        )

        assert recovered_session is not None
        assert recovered_session.state == {"status": "active"}
        assert len(recovered_session.events) == 1
        assert recovered_session.events[0].invocation_id == "test-invocation"
        await recovered_service.close()

    asyncio.run(verify_persistence())


def test_conversation_keeps_the_persisted_recommendation(tmp_path: Path) -> None:
    database_url = f"sqlite+aiosqlite:///{tmp_path / 'conversation.db'}"
    session_service = DatabaseSessionService(db_url=database_url)
    expected_recommendation = recommendation()
    runner = StubRunner(session_service, result=expected_recommendation)

    async def run() -> None:
        session_id = await run_test_conversation(runner)
        stored_session = await session_service.get_session(
            app_name=APP_NAME,
            user_id="test-user",
            session_id=session_id,
        )

        assert stored_session is not None
        assert stored_session.state[CONVERSATION_STATUS_STATE_KEY] == "resolved"
        assert (
            Recommendation.model_validate(stored_session.state[FINAL_RECOMMENDATION_STATE_KEY])
            == expected_recommendation
        )
        assert runner.run_config is not None
        assert runner.run_config.max_llm_calls == MAX_CONVERSATION_LLM_CALLS
        await session_service.close()

    asyncio.run(run())


def test_conversation_without_a_recommendation_is_escalated(tmp_path: Path) -> None:
    database_url = f"sqlite+aiosqlite:///{tmp_path / 'no-recommendation.db'}"
    session_service = DatabaseSessionService(db_url=database_url)
    runner = StubRunner(session_service)

    async def run() -> None:
        session_id = await run_test_conversation(runner)
        stored_session = await session_service.get_session(
            app_name=APP_NAME,
            user_id="test-user",
            session_id=session_id,
        )

        assert stored_session is not None
        assert all(event.error_code is None for event in stored_session.events)
        assert stored_session.state[CONVERSATION_STATUS_STATE_KEY] == "escalated"
        await session_service.close()

    asyncio.run(run())


def test_new_conversation_starts_active(tmp_path: Path) -> None:
    database_url = f"sqlite+aiosqlite:///{tmp_path / 'active.db'}"
    session_service = DatabaseSessionService(db_url=database_url)
    runner = StubRunner(session_service, error=ConnectionError("stop after creation"))

    async def run() -> None:
        session_id = await run_test_conversation(runner)
        stored_session = await session_service.get_session(
            app_name=APP_NAME,
            user_id="test-user",
            session_id=session_id,
        )

        assert stored_session is not None
        assert runner.initial_status == "active"
        assert stored_session.state[CONVERSATION_STATUS_STATE_KEY] == "escalated"
        await session_service.close()

    asyncio.run(run())


def test_conversation_persists_unexpected_failures(tmp_path: Path) -> None:
    database_url = f"sqlite+aiosqlite:///{tmp_path / 'failure.db'}"
    session_service = DatabaseSessionService(db_url=database_url)
    runner = StubRunner(session_service, error=ConnectionError("model unavailable"))

    async def run() -> None:
        session_id = await run_test_conversation(runner)
        stored_session = await session_service.get_session(
            app_name=APP_NAME,
            user_id="test-user",
            session_id=session_id,
        )

        assert stored_session is not None
        assert stored_session.state[CONVERSATION_STATUS_STATE_KEY] == "escalated"
        assert stored_session.events[-1].error_code == "ConnectionError"
        assert stored_session.events[-1].error_message == "model unavailable"
        assert stored_session.events[-1].content.parts[0].text == ESCALATION_MESSAGE
        await session_service.close()

    asyncio.run(run())


def test_conversation_timeout_is_persisted_as_an_escalation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_url = f"sqlite+aiosqlite:///{tmp_path / 'timeout.db'}"
    session_service = DatabaseSessionService(db_url=database_url)
    runner = StubRunner(session_service, delay_seconds=0.1)
    monkeypatch.setattr(runtime_module, "CONVERSATION_TIMEOUT_SECONDS", 0.01)

    async def run() -> None:
        session_id = await run_test_conversation(runner)
        stored_session = await session_service.get_session(
            app_name=APP_NAME,
            user_id="test-user",
            session_id=session_id,
        )

        assert stored_session is not None
        assert stored_session.events[-1].error_code == "TimeoutError"
        await session_service.close()

    asyncio.run(run())
