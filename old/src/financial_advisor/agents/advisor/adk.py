"""Google ADK adapters for Advisor planning and client-response synthesis."""

import asyncio
from collections.abc import Callable
from typing import Final, TypeVar, cast
from uuid import uuid4

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from financial_advisor.agents.advisor.contracts import (
    AdvisorPlanningRequest,
    AdvisorRecommendationDraft,
    AdvisorResearchPlanDraft,
    AdvisorResponseRequest,
)
from financial_advisor.agents.advisor.prompt import (
    ADVISOR_CITATION_REPAIR_INSTRUCTION,
    ADVISOR_PLANNING_INSTRUCTION,
    ADVISOR_RESPONSE_INSTRUCTION,
    build_advisor_knowledge_instruction,
    build_response_context,
)
from financial_advisor.config import GeminiSettings

_PLANNING_OUTPUT_KEY: Final = "advisor_research_plan_draft"
_RESPONSE_OUTPUT_KEY: Final = "advisor_recommendation_draft"
_APP_NAME: Final = "financial_advisor"
_USER_ID: Final = "workflow"


class AdvisorModelError(RuntimeError):
    """Raised when ADK cannot produce one validated Advisor structured result."""


class AdvisorOutputValidationError(AdvisorModelError):
    """Raised when a model response exists but violates an Advisor output contract."""


_AdvisorOutput = TypeVar("_AdvisorOutput")

def _generation_config(settings: GeminiSettings) -> types.GenerateContentConfig:
    """Create the same bounded generation settings for both Advisor agent roles."""

    return types.GenerateContentConfig(
        temperature=settings.temperature,
        max_output_tokens=settings.advisor_max_output_tokens,
        http_options=types.HttpOptions(
            retry_options=types.HttpRetryOptions(initial_delay=1, attempts=2)
        ),
    )


def create_advisor_planning_agent(
    settings: GeminiSettings, *, model: str | None = None
) -> LlmAgent:
    """Build the internal Advisor agent that decides the next workflow action."""

    return LlmAgent(
        name="advisor_planning_agent",
        description="Creates bounded research decisions for the financial-advice workflow.",
        model=model or settings.advisor_model,
        instruction=ADVISOR_PLANNING_INSTRUCTION,
        output_schema=AdvisorResearchPlanDraft,
        output_key=_PLANNING_OUTPUT_KEY,
        generate_content_config=_generation_config(settings),
        disallow_transfer_to_parent=True,
        disallow_transfer_to_peers=True,
    )


def create_advisor_response_agent(
    settings: GeminiSettings, *, model: str | None = None
) -> LlmAgent:
    """Build the Advisor agent that writes the only client-facing recommendation."""

    return LlmAgent(
        name="advisor_response_agent",
        description="Turns validated Analyst briefs into transparent client recommendations.",
        model=model or settings.advisor_model,
        instruction=ADVISOR_RESPONSE_INSTRUCTION,
        output_schema=AdvisorRecommendationDraft,
        output_key=_RESPONSE_OUTPUT_KEY,
        generate_content_config=_generation_config(settings),
        disallow_transfer_to_parent=True,
        disallow_transfer_to_peers=True,
    )


class AdkAdvisorService:
    """Run bounded structured Advisor planning and recommendation calls through ADK."""

    def __init__(self, settings: GeminiSettings) -> None:
        self._settings = settings
        self._planning_agent = create_advisor_planning_agent(settings)
        self._response_agent = create_advisor_response_agent(settings)
        self._fallback_planning_agent = create_advisor_planning_agent(
            settings, model=settings.fallback_model
        )
        self._fallback_response_agent = create_advisor_response_agent(
            settings, model=settings.fallback_model
        )
        # Google ADK currently exposes this constructor without type hints.
        self._session_service = InMemorySessionService()  # type: ignore[no-untyped-call]

    async def plan(self, request: AdvisorPlanningRequest) -> AdvisorResearchPlanDraft:
        """Ask the Advisor to choose the next validated research action."""

        try:
            return await asyncio.wait_for(
                self._run_with_policy(
                    primary_agent=self._planning_agent,
                    fallback_agent=self._fallback_planning_agent,
                    output_key=_PLANNING_OUTPUT_KEY,
                    context=build_advisor_knowledge_instruction(request),
                    repair_instruction=(
                        "Your prior planning output could not be accepted as the required "
                        "structured schema. Return one complete schema-valid plan."
                    ),
                    parser=AdvisorResearchPlanDraft.model_validate,
                ),
                timeout=self._settings.advisor_turn_timeout_seconds,
            )
        except TimeoutError as error:
            raise AdvisorModelError("Advisor planning deadline exceeded.") from error

    async def recommend(
        self, request: AdvisorResponseRequest, *, repair: bool = False
    ) -> AdvisorRecommendationDraft:
        """Ask the Advisor to create client wording from an evidence-backed brief."""

        try:
            context = build_response_context(request)
            if repair:
                context = f"{context}\n\n{ADVISOR_CITATION_REPAIR_INSTRUCTION}"
            return await asyncio.wait_for(
                self._run_with_policy(
                    primary_agent=self._response_agent,
                    fallback_agent=self._fallback_response_agent,
                    output_key=_RESPONSE_OUTPUT_KEY,
                    context=context,
                    repair_instruction=ADVISOR_CITATION_REPAIR_INSTRUCTION,
                    parser=AdvisorRecommendationDraft.model_validate,
                ),
                timeout=self._settings.advisor_turn_timeout_seconds,
            )
        except TimeoutError as error:
            raise AdvisorModelError("Advisor recommendation deadline exceeded.") from error

    async def _run_with_policy(
        self,
        *,
        primary_agent: LlmAgent,
        fallback_agent: LlmAgent,
        output_key: str,
        context: str,
        repair_instruction: str,
        parser: Callable[[dict[str, object]], _AdvisorOutput],
    ) -> _AdvisorOutput:
        """Retry one primary model, repair invalid output once, then try a fallback model."""

        last_error: AdvisorModelError | None = None
        repair_used = False
        for _ in range(self._settings.advisor_primary_attempts):
            try:
                return await self._attempt(
                    primary_agent, output_key, context, parser
                )
            except AdvisorOutputValidationError as error:
                last_error = error
                if repair_used:
                    continue
                repair_used = True
                try:
                    return await self._attempt(
                        primary_agent,
                        output_key,
                        f"{context}\n\n{repair_instruction}",
                        parser,
                    )
                except AdvisorModelError as repair_error:
                    last_error = repair_error
            except AdvisorModelError as error:
                last_error = error
        try:
            return await self._attempt(fallback_agent, output_key, context, parser)
        except AdvisorModelError as error:
            raise AdvisorModelError(
                "Advisor primary and fallback models were unavailable or invalid."
            ) from (last_error or error)

    async def _attempt(
        self,
        agent: LlmAgent,
        output_key: str,
        context: str,
        parser: Callable[[dict[str, object]], _AdvisorOutput],
    ) -> _AdvisorOutput:
        """Invoke one model and distinguish output validation from invocation failure."""

        output = await self._run(agent=agent, output_key=output_key, context=context)
        try:
            return parser(output)
        except Exception as error:
            raise AdvisorOutputValidationError(
                "Advisor model output did not satisfy the required structured schema."
            ) from error

    async def _run(self, *, agent: LlmAgent, output_key: str, context: str) -> dict[str, object]:
        """Run one ephemeral ADK session and return its structured state value."""

        session_id = str(uuid4())
        try:
            await self._session_service.create_session(
                app_name=_APP_NAME,
                user_id=_USER_ID,
                session_id=session_id,
            )
            runner = Runner(
                app_name=_APP_NAME,
                agent=agent,
                session_service=self._session_service,
            )
            message = types.Content(role="user", parts=[types.Part(text=context)])
            async for _ in runner.run_async(
                user_id=_USER_ID,
                session_id=session_id,
                new_message=message,
            ):
                pass
            session = await self._session_service.get_session(
                app_name=_APP_NAME,
                user_id=_USER_ID,
                session_id=session_id,
            )
            if session is None or not isinstance(session.state.get(output_key), dict):
                raise AdvisorModelError("Advisor model did not return structured output.")
            return cast(dict[str, object], session.state[output_key])
        except AdvisorModelError:
            raise
        except Exception as error:
            raise AdvisorModelError("Advisor model invocation failed.") from error
