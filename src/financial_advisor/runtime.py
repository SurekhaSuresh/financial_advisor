"""Build and run the financial-advisor application."""

import os
from asyncio import timeout
from contextlib import aclosing
from uuid import uuid4

from google.adk.agents import RunConfig
from google.adk.events import Event, EventActions
from google.adk.models import Gemini
from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService, Session
from google.genai import types

from financial_advisor.agents.advisor.agent import (
    CONVERSATION_STATUS_STATE_KEY,
    FINAL_ADVISOR_RECOMMENDATION_STATE_KEY,
    create_advisor_agent,
)
from financial_advisor.agents.analyst.agent import create_analyst_agent
from financial_advisor.agents.client import create_client_agent
from financial_advisor.config import (
    CONVERSATION_TIMEOUT_SECONDS,
    MAX_ADVISOR_LLM_CALLS,
    MODEL_REQUEST_ATTEMPTS,
    MODEL_RETRY_DELAY_SECONDS,
    ApplicationSettings,
)
from financial_advisor.contracts import (
    ClientProfile,
    ConversationResult,
    ConversationStatus,
    Recommendation,
)
from financial_advisor.retrieval.hybrid_search import LocalHybridRetriever
from financial_advisor.retrieval.pipeline import RetrievalPipeline
from financial_advisor.retrieval.text_models import LocalRetrievalModels
from financial_advisor.retrieval.web.providers import brave_provider, exa_provider
from financial_advisor.retrieval.web.web_search import WebCandidateRetriever

APP_NAME = "financial_advisor"
CLIENT_PROFILE_STATE_KEY = "client_profile"


def create_application_runner(settings: ApplicationSettings) -> Runner:
    """Assemble retrieval, agents, and durable ADK session storage."""

    if settings.google_api_key is None:
        raise ValueError("GOOGLE_API_KEY is required.")
    os.environ["GOOGLE_API_KEY"] = settings.google_api_key.get_secret_value()

    retrieval_models = LocalRetrievalModels(settings.model_cache_directory)
    local_hybrid_retriever = (
        LocalHybridRetriever(
            settings.knowledge_database_path,
            retrieval_models.embed,
        )
        if settings.knowledge_database_path is not None
        and settings.knowledge_database_path.exists()
        else None
    )

    web_providers = []
    if settings.exa_api_key is not None:
        web_providers.append(exa_provider(settings.exa_api_key.get_secret_value()))
    if settings.brave_search_api_key is not None:
        web_providers.append(
            brave_provider(settings.brave_search_api_key.get_secret_value())
        )

    web_candidate_retriever = (
        WebCandidateRetriever(
            web_providers,
            retrieval_models.encode,
            retrieval_models.decode,
            retrieval_models.embed,
        )
        if web_providers
        else None
    )
    retrieval_pipeline = RetrievalPipeline(
        local_hybrid_retriever,
        retrieval_models.rerank,
        web_candidate_retriever,
    )
    return create_runner(
        settings.advisor_analyst_model_name,
        settings.client_model_name,
        retrieval_pipeline,
        settings.session_database_url,
    )


def create_runner(
    advisor_analyst_model_name: str,
    client_model_name: str,
    retrieval_pipeline: RetrievalPipeline,
    session_database_url: str,
) -> Runner:
    """Create one Runner using separate Client and Advisor/Analyst models."""

    retry_options = types.HttpRetryOptions(
        attempts=MODEL_REQUEST_ATTEMPTS,
        initial_delay=MODEL_RETRY_DELAY_SECONDS,
    )
    advisor_analyst_model = Gemini(
        model=advisor_analyst_model_name,
        retry_options=retry_options,
    )
    client_model = Gemini(model=client_model_name, retry_options=retry_options)

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
        session_service=DatabaseSessionService(db_url=session_database_url),
    )


async def create_conversation_session(
    runner: Runner,
    client_profile: ClientProfile,
    user_id: str,
) -> str:
    """Create a new active session and return its UI-visible identifier."""

    session_id = str(uuid4())
    await runner.session_service.create_session(
        app_name=APP_NAME,
        user_id=user_id,
        session_id=session_id,
        state={
            CLIENT_PROFILE_STATE_KEY: client_profile.model_dump(mode="json"),
            CONVERSATION_STATUS_STATE_KEY: ConversationStatus.ACTIVE.value,
        },
    )
    return session_id


async def run_conversation(
    runner: Runner,
    client_profile: ClientProfile,
    user_id: str,
    session_id: str,
) -> ConversationResult:
    """Run one previously created conversation to a terminal state."""

    invocation_id = str(uuid4())
    failure: Exception | None = None
    session_to_update: Session | None = None

    try:
        async with timeout(CONVERSATION_TIMEOUT_SECONDS):
            async with aclosing(
                runner.run_async(
                    user_id=user_id,
                    session_id=session_id,
                    invocation_id=invocation_id,
                    new_message=types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=client_profile.model_dump_json())],
                    ),
                    run_config=RunConfig(max_llm_calls=MAX_ADVISOR_LLM_CALLS),
                )
            ) as events:
                async for _event in events:
                    pass

        completed_session = await runner.session_service.get_session(
            app_name=APP_NAME,
            user_id=user_id,
            session_id=session_id,
        )
        if completed_session is None:
            raise RuntimeError("The completed ADK session could not be loaded.")

        if (
            completed_session.state.get(CONVERSATION_STATUS_STATE_KEY)
            == ConversationStatus.RESOLVED
        ):
            stored_recommendation = completed_session.state.get(
                FINAL_ADVISOR_RECOMMENDATION_STATE_KEY
            )
            if stored_recommendation is None:
                raise RuntimeError("A resolved conversation requires a recommendation.")
            return ConversationResult(
                session_id=session_id,
                status=ConversationStatus.RESOLVED,
                recommendation=Recommendation.model_validate(stored_recommendation),
            )
        session_to_update = completed_session
    except Exception as error:
        failure = error
        session_to_update = await runner.session_service.get_session(
            app_name=APP_NAME,
            user_id=user_id,
            session_id=session_id,
        )

    if session_to_update is not None:
        await runner.session_service.append_event(
            session_to_update,
            Event(
                invocation_id=invocation_id,
                author="runtime",
                error_code=type(failure).__name__ if failure else None,
                error_message=str(failure) if failure else None,
                actions=EventActions(
                    state_delta={
                        CONVERSATION_STATUS_STATE_KEY: ConversationStatus.ESCALATED.value
                    }
                ),
            ),
        )

    return ConversationResult(
        session_id=session_id,
        status=ConversationStatus.ESCALATED,
    )
