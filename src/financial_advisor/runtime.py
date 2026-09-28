"""Build and run the financial-advisor application."""

import os
from asyncio import timeout

from google.adk.agents import RunConfig
from google.adk.events import Event, EventActions
from google.adk.models import Gemini
from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService
from google.genai import types

from financial_advisor.agents.advisor.agent import create_advisor_agent
from financial_advisor.agents.analyst.agent import create_analyst_agent
from financial_advisor.agents.client import create_client_agent
from financial_advisor.config import (
    CONVERSATION_STATUS_STATE_KEY,
    CONVERSATION_TIMEOUT_SECONDS,
    ESCALATION_MESSAGE,
    MAX_CONVERSATION_LLM_CALLS,
    MODEL_RETRY_OPTIONS,
    SESSION_DATABASE_URL,
    ApplicationSettings,
)
from financial_advisor.contracts import ConversationStatus
from financial_advisor.retrieval.hybrid_search import LocalHybridRetriever
from financial_advisor.retrieval.pipeline import RetrievalPipeline
from financial_advisor.retrieval.text_models import LocalRetrievalModels
from financial_advisor.retrieval.web.discovery import WebSourceDiscovery
from financial_advisor.retrieval.web.web_search import WebCandidateRetriever

APP_NAME = "financial_advisor"


def create_runner(settings: ApplicationSettings) -> Runner:
    """Assemble retrieval, agents, and durable ADK session storage."""

    os.environ["GOOGLE_API_KEY"] = settings.google_api_key.get_secret_value()

    retrieval_models = LocalRetrievalModels(settings.model_cache_directory)
    local_hybrid_retriever = (
        LocalHybridRetriever(
            settings.knowledge_database_path,
            retrieval_models.embed,
        )
        if settings.knowledge_database_path.exists()
        else None
    )

    exa_api_key = settings.exa_api_key.get_secret_value() if settings.exa_api_key else None
    brave_api_key = (
        settings.brave_search_api_key.get_secret_value() if settings.brave_search_api_key else None
    )

    web_candidate_retriever = (
        WebCandidateRetriever(
            retrieval_models,
            WebSourceDiscovery(exa_api_key, brave_api_key),
        )
        if exa_api_key or brave_api_key
        else None
    )
    retrieval_pipeline = RetrievalPipeline(
        local_hybrid_retriever,
        retrieval_models.rerank,
        web_candidate_retriever,
    )
    advisor_analyst_model = Gemini(
        model=settings.advisor_analyst_model_name,
        retry_options=MODEL_RETRY_OPTIONS,
    )
    client_model = Gemini(model=settings.client_model_name, retry_options=MODEL_RETRY_OPTIONS)

    client_agent = create_client_agent(client_model)
    analyst_agent = create_analyst_agent(advisor_analyst_model, retrieval_pipeline)
    advisor_agent = create_advisor_agent(
        advisor_analyst_model,
        client_agent,
        analyst_agent,
    )

    return Runner(
        app_name=APP_NAME,
        agent=advisor_agent,
        session_service=DatabaseSessionService(db_url=SESSION_DATABASE_URL),
    )


async def run_conversation(
    runner: Runner,
    user_id: str,
    session_id: str,
) -> None:
    """Run an active conversation to a resolved or escalated state."""

    runtime_error: Exception | None = None

    try:
        async with timeout(CONVERSATION_TIMEOUT_SECONDS):
            async for _event in runner.run_async(
                user_id=user_id,
                session_id=session_id,
                new_message=types.Content(
                    role="user",
                    parts=[types.Part.from_text(text="Start the financial-planning scenario.")],
                ),
                run_config=RunConfig(max_llm_calls=MAX_CONVERSATION_LLM_CALLS),
            ):
                pass

        completed_session = await runner.session_service.get_session(
            app_name=APP_NAME,
            user_id=user_id,
            session_id=session_id,
        )
        if completed_session is None:
            raise RuntimeError("The completed ADK session could not be loaded.")
    except Exception as error:
        runtime_error = error
        completed_session = await runner.session_service.get_session(
            app_name=APP_NAME,
            user_id=user_id,
            session_id=session_id,
        )
    # Escalate if the Advisor ends without resolving or escalating the conversation.
    if (
        completed_session is not None
        and completed_session.state.get(CONVERSATION_STATUS_STATE_KEY) == ConversationStatus.ACTIVE
    ):
        await runner.session_service.append_event(
            completed_session,
            Event(
                author="runtime",
                content=types.Content(
                    role="model",
                    parts=[types.Part.from_text(text=ESCALATION_MESSAGE)],
                ),
                error_code=type(runtime_error).__name__ if runtime_error else None,
                error_message=str(runtime_error) if runtime_error else None,
                actions=EventActions(
                    state_delta={CONVERSATION_STATUS_STATE_KEY: ConversationStatus.ESCALATED.value}
                ),
            ),
        )
