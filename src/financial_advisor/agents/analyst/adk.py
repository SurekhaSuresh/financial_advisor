"""Google ADK adapter for bounded, structured Analyst reasoning."""

import asyncio
from typing import Final, cast
from uuid import uuid4

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from financial_advisor.agents.analyst.contracts import (
    AnalystClaimIntegrityError,
    AnalystResearchDraft,
    AnalystResearchRequest,
    create_research_brief,
)
from financial_advisor.agents.analyst.prompt import (
    ANALYST_INSTRUCTION,
    build_analyst_knowledge_instruction,
)
from financial_advisor.config import GeminiSettings
from financial_advisor.domain import ResearchBrief

_APP_NAME: Final = "financial_advisor"
_OUTPUT_KEY: Final = "analyst_research_draft"
_USER_ID: Final = "workflow"
_REPAIR_INSTRUCTION: Final = (
    "Your previous output could not be accepted as the required structured schema. "
    "Return a complete schema-valid draft grounded only in the supplied evidence."
)


class AnalystModelError(RuntimeError):
    """Raised when ADK cannot produce a validated Analyst research draft."""


class AnalystOutputValidationError(AnalystModelError):
    """Raised when a model response exists but does not satisfy the output contract."""


def _generation_config(settings: GeminiSettings) -> types.GenerateContentConfig:
    """Create bounded, low-variance settings shared by primary and fallback calls."""

    return types.GenerateContentConfig(
        temperature=settings.temperature,
        max_output_tokens=settings.analyst_max_output_tokens,
        http_options=types.HttpOptions(
            retry_options=types.HttpRetryOptions(initial_delay=1, attempts=2)
        ),
    )


def create_analyst_agent(settings: GeminiSettings, *, model: str | None = None) -> LlmAgent:
    """Build an internal Analyst with typed, evidence-grounded output."""

    return LlmAgent(
        name="analyst_agent",
        description=(
            "Synthesizes server-selected financial research evidence into a "
            "structured internal brief for the Advisor Agent."
        ),
        model=model or settings.analyst_model,
        instruction=ANALYST_INSTRUCTION,
        output_schema=AnalystResearchDraft,
        output_key=_OUTPUT_KEY,
        generate_content_config=_generation_config(settings),
        disallow_transfer_to_parent=True,
        disallow_transfer_to_peers=True,
    )


class AdkAnalystService:
    """Run primary, repair, and fallback Analyst synthesis within one deadline."""

    def __init__(self, settings: GeminiSettings) -> None:
        self._settings = settings
        self._primary_agent = create_analyst_agent(settings)
        self._fallback_agent = create_analyst_agent(settings, model=settings.fallback_model)
        # Google ADK currently exposes this constructor without type hints.
        self._session_service = InMemorySessionService()  # type: ignore[no-untyped-call]

    async def research(self, request: AnalystResearchRequest) -> ResearchBrief:
        """Return a validated brief, bounded by a task-wide deadline."""

        try:
            return await asyncio.wait_for(
                self._research_with_policy(request),
                timeout=self._settings.analyst_task_timeout_seconds,
            )
        except TimeoutError as error:
            raise AnalystModelError("Analyst synthesis deadline exceeded.") from error

    async def _research_with_policy(self, request: AnalystResearchRequest) -> ResearchBrief:
        """Try the primary model, one repair, then the configured fallback model."""

        last_error: AnalystModelError | None = None
        repair_used = False
        for _ in range(self._settings.analyst_primary_attempts):
            try:
                return await self._invoke(self._primary_agent, request)
            except AnalystOutputValidationError as error:
                last_error = error
                if repair_used:
                    continue
                repair_used = True
                try:
                    return await self._invoke(self._primary_agent, request, repair=True)
                except AnalystModelError as repair_error:
                    last_error = repair_error
            except AnalystModelError as error:
                last_error = error

        try:
            return await self._invoke(self._fallback_agent, request)
        except AnalystModelError as error:
            raise AnalystModelError(
                "Analyst primary and fallback models were unavailable or invalid."
            ) from (last_error or error)

    async def _invoke(
        self,
        agent: LlmAgent,
        request: AnalystResearchRequest,
        *,
        repair: bool = False,
    ) -> ResearchBrief:
        """Run one ephemeral ADK invocation and validate its structured draft."""

        session_id = str(uuid4())
        context = build_analyst_knowledge_instruction(request)
        if repair:
            context = f"{context}\n\n{_REPAIR_INSTRUCTION}"
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
            if session is None:
                raise AnalystModelError("Analyst model session was not available after invocation.")
            output = session.state.get(_OUTPUT_KEY)
            if not isinstance(output, dict):
                raise AnalystOutputValidationError(
                    "Analyst model did not return a valid structured draft."
                )
            try:
                draft = AnalystResearchDraft.model_validate(cast(dict[str, object], output))
            except Exception as error:
                raise AnalystOutputValidationError(
                    "Analyst model output did not satisfy the research-draft contract."
                ) from error
            try:
                return create_research_brief(request, draft)
            except AnalystClaimIntegrityError as error:
                raise AnalystOutputValidationError(
                    "Analyst output contained no valid evidence-backed findings."
                ) from error
        except AnalystModelError:
            raise
        except Exception as error:
            raise AnalystModelError("Analyst model invocation failed.") from error
