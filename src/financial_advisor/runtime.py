"""Build the financial-advisor agent runtime."""

from uuid import uuid4

from google.adk.agents import RunConfig
from google.adk.events import Event
from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService
from google.genai import types

from financial_advisor.agents.advisor.agent import (
    FINAL_ADVISOR_RECOMMENDATION_STATE_KEY,
    create_advisor_agent,
)
from financial_advisor.agents.analyst.agent import create_analyst_agent
from financial_advisor.agents.client import create_client_agent
from financial_advisor.config import MAX_INVOCATION_LLM_CALLS
from financial_advisor.contracts import ClientProfile, Recommendation
from financial_advisor.retrieval.pipeline import RetrievalPipeline

APP_NAME = "financial_advisor"


def create_runner(
    model: str,
    retrieval_pipeline: RetrievalPipeline,
    session_database_url: str,
) -> Runner:
    """Create one Runner for the Advisor and its Client and Analyst tools."""

    client_agent = create_client_agent(model)
    analyst_agent = create_analyst_agent(model, retrieval_pipeline)
    advisor_agent = create_advisor_agent(model, client_agent, analyst_agent)

    return Runner(
        app_name=APP_NAME,
        agent=advisor_agent,
        session_service=DatabaseSessionService(db_url=session_database_url),
    )


async def run_conversation(
    runner: Runner,
    client_profile: ClientProfile,
    user_id: str,
    session_id: str,
) -> Recommendation | None:
    """Run one conversation and return its trusted recommendation when available."""

    invocation_id = str(uuid4())
    try:
        existing_session = await runner.session_service.get_session(
            app_name=APP_NAME,
            user_id=user_id,
            session_id=session_id,
        )
        if existing_session is None:
            await runner.session_service.create_session(
                app_name=APP_NAME,
                user_id=user_id,
                session_id=session_id,
            )

        async for _event in runner.run_async(
            user_id=user_id,
            session_id=session_id,
            invocation_id=invocation_id,
            new_message=types.Content(
                role="user",
                parts=[types.Part.from_text(text=client_profile.model_dump_json())],
            ),
            run_config=RunConfig(max_llm_calls=MAX_INVOCATION_LLM_CALLS),
        ):
            pass

        completed_session = await runner.session_service.get_session(
            app_name=APP_NAME,
            user_id=user_id,
            session_id=session_id,
        )
        if completed_session is None:
            raise RuntimeError("The completed ADK session could not be loaded.")

        stored_recommendation = completed_session.state.get(
            FINAL_ADVISOR_RECOMMENDATION_STATE_KEY
        )
        if stored_recommendation is None:
            return None
        return Recommendation.model_validate(stored_recommendation)
    except Exception as error:
        failed_session = await runner.session_service.get_session(
            app_name=APP_NAME,
            user_id=user_id,
            session_id=session_id,
        )
        if failed_session is not None:
            await runner.session_service.append_event(
                failed_session,
                Event(
                    invocation_id=invocation_id,
                    author="runtime",
                    error_code=type(error).__name__,
                    error_message=str(error),
                ),
            )
        return None
