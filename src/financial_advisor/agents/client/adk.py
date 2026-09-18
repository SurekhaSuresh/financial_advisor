"""Google ADK adapter for the structured simulated Client Agent."""

import asyncio
from collections.abc import Callable
from typing import Final, TypeVar, cast
from uuid import UUID, uuid4

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from financial_advisor.agents.client.contracts import (
    ClientOpeningDraft,
    ClientReviewDraft,
    ClientReviewRequest,
    create_client_review,
    create_opening_message,
)
from financial_advisor.agents.client.prompt import (
    CLIENT_OPENING_INSTRUCTION,
    CLIENT_REVIEW_INSTRUCTION,
    build_opening_context,
    build_review_context,
)
from financial_advisor.config import GeminiSettings
from financial_advisor.domain import ClientMessage, ClientProfile, ClientReview

_APP_NAME: Final = "financial_advisor"
_OPENING_OUTPUT_KEY: Final = "client_opening_draft"
_REVIEW_OUTPUT_KEY: Final = "client_review_draft"
_USER_ID: Final = "workflow"


class ClientModelError(RuntimeError):
    """Raised when ADK cannot produce one validated Client output."""


class ClientOutputValidationError(ClientModelError):
    """Raised when a model response exists but violates a Client output contract."""


_ClientOutput = TypeVar("_ClientOutput")

def _generation_config(settings: GeminiSettings) -> types.GenerateContentConfig:
    """Create bounded low-variance settings for simulated Client behavior."""

    return types.GenerateContentConfig(
        temperature=settings.client_temperature,
        max_output_tokens=settings.client_max_output_tokens,
        http_options=types.HttpOptions(
            retry_options=types.HttpRetryOptions(initial_delay=1, attempts=2)
        ),
    )


def create_client_opening_agent(settings: GeminiSettings, *, model: str | None = None) -> LlmAgent:
    """Build the Client Agent's profile-bound opening-question role."""

    return LlmAgent(
        name="client_opening_agent",
        description="Creates one profile-bound financial question for the simulation.",
        model=model or settings.client_model,
        instruction=CLIENT_OPENING_INSTRUCTION,
        output_schema=ClientOpeningDraft,
        output_key=_OPENING_OUTPUT_KEY,
        generate_content_config=_generation_config(settings),
        disallow_transfer_to_parent=True,
        disallow_transfer_to_peers=True,
    )


def create_client_review_agent(settings: GeminiSettings, *, model: str | None = None) -> LlmAgent:
    """Build the Client Agent's bounded recommendation-review role."""

    return LlmAgent(
        name="client_review_agent",
        description="Accepts a recommendation or asks one focused follow-up.",
        model=model or settings.client_model,
        instruction=CLIENT_REVIEW_INSTRUCTION,
        output_schema=ClientReviewDraft,
        output_key=_REVIEW_OUTPUT_KEY,
        generate_content_config=_generation_config(settings),
        disallow_transfer_to_parent=True,
        disallow_transfer_to_peers=True,
    )


class AdkClientService:
    """Run bounded structured Client opening and review calls through ADK."""

    def __init__(self, settings: GeminiSettings) -> None:
        self._settings = settings
        self._opening_agent = create_client_opening_agent(settings)
        self._review_agent = create_client_review_agent(settings)
        self._fallback_opening_agent = create_client_opening_agent(
            settings, model=settings.fallback_model
        )
        self._fallback_review_agent = create_client_review_agent(
            settings, model=settings.fallback_model
        )
        self._session_service = InMemorySessionService()  # type: ignore[no-untyped-call]

    async def open_conversation(self, profile: ClientProfile, session_id: UUID) -> ClientMessage:
        """Create the Client's opening message from its synthetic profile."""

        try:
            draft = await asyncio.wait_for(
                self._run_with_policy(
                    primary_agent=self._opening_agent,
                    fallback_agent=self._fallback_opening_agent,
                    output_key=_OPENING_OUTPUT_KEY,
                    context=build_opening_context(profile),
                    repair_instruction=(
                        "Your prior Client opening output could not be accepted as the "
                        "required structured schema. Return one complete schema-valid draft."
                    ),
                    parser=ClientOpeningDraft.model_validate,
                ),
                timeout=self._settings.client_turn_timeout_seconds,
            )
        except TimeoutError as error:
            raise ClientModelError("Client opening deadline exceeded.") from error
        return create_opening_message(session_id, draft)

    async def review_recommendation(self, request: ClientReviewRequest) -> ClientReview:
        """Accept a recommendation or create one bounded follow-up question."""

        try:
            draft = await asyncio.wait_for(
                self._run_with_policy(
                    primary_agent=self._review_agent,
                    fallback_agent=self._fallback_review_agent,
                    output_key=_REVIEW_OUTPUT_KEY,
                    context=build_review_context(request),
                    repair_instruction=(
                        "Your prior Client review output could not be accepted as the "
                        "required structured schema. Return one complete schema-valid draft."
                    ),
                    parser=ClientReviewDraft.model_validate,
                ),
                timeout=self._settings.client_turn_timeout_seconds,
            )
        except TimeoutError as error:
            raise ClientModelError("Client review deadline exceeded.") from error
        return create_client_review(request, draft)

    async def _run_with_policy(
        self,
        *,
        primary_agent: LlmAgent,
        fallback_agent: LlmAgent,
        output_key: str,
        context: str,
        repair_instruction: str,
        parser: Callable[[dict[str, object]], _ClientOutput],
    ) -> _ClientOutput:
        """Retry the primary model, repair invalid output once, then use a fallback model."""

        last_error: ClientModelError | None = None
        repair_used = False
        for _ in range(self._settings.client_primary_attempts):
            try:
                return await self._attempt(primary_agent, output_key, context, parser)
            except ClientOutputValidationError as error:
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
                except ClientModelError as repair_error:
                    last_error = repair_error
            except ClientModelError as error:
                last_error = error
        try:
            return await self._attempt(fallback_agent, output_key, context, parser)
        except ClientModelError as error:
            raise ClientModelError(
                "Client primary and fallback models were unavailable or invalid."
            ) from (last_error or error)

    async def _attempt(
        self,
        agent: LlmAgent,
        output_key: str,
        context: str,
        parser: Callable[[dict[str, object]], _ClientOutput],
    ) -> _ClientOutput:
        """Invoke one model and distinguish output validation from invocation failure."""

        output = await self._run(agent, output_key, context)
        try:
            return parser(output)
        except Exception as error:
            raise ClientOutputValidationError(
                "Client model output did not satisfy the required structured schema."
            ) from error

    async def _run(self, agent: LlmAgent, output_key: str, context: str) -> dict[str, object]:
        """Run one ephemeral ADK Client turn and return its structured output."""

        session_id = str(uuid4())
        try:
            await self._session_service.create_session(
                app_name=_APP_NAME, user_id=_USER_ID, session_id=session_id
            )
            runner = Runner(app_name=_APP_NAME, agent=agent, session_service=self._session_service)
            message = types.Content(role="user", parts=[types.Part(text=context)])
            async for _ in runner.run_async(
                user_id=_USER_ID, session_id=session_id, new_message=message
            ):
                pass
            session = await self._session_service.get_session(
                app_name=_APP_NAME, user_id=_USER_ID, session_id=session_id
            )
            if session is None or not isinstance(session.state.get(output_key), dict):
                raise ClientModelError("Client model did not return structured output.")
            return cast(dict[str, object], session.state[output_key])
        except ClientModelError:
            raise
        except Exception as error:
            raise ClientModelError("Client model invocation failed.") from error
