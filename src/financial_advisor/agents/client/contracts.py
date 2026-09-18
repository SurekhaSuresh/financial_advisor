"""Typed Client Agent inputs and outputs for the simulated conversation."""

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from financial_advisor.domain import ClientMessage, ClientProfile, ClientReview, Recommendation


class ClientOpeningDraft(BaseModel):
    """Opening investment question created by the simulated Client."""

    model_config = ConfigDict(frozen=True)

    question: str = Field(min_length=1, max_length=2_000)


class ClientReviewDraft(BaseModel):
    """Client acceptance decision or one focused follow-up question."""

    model_config = ConfigDict(frozen=True)

    accepted: bool
    follow_up_question: str | None = Field(default=None, min_length=1, max_length=2_000)

    @model_validator(mode="after")
    def validate_review_shape(self) -> "ClientReviewDraft":
        """Require a question only when the Client has not accepted the response."""

        if self.accepted and self.follow_up_question is not None:
            raise ValueError("An accepted Client review cannot include a follow-up question.")
        if not self.accepted and self.follow_up_question is None:
            raise ValueError("A non-accepted Client review requires a follow-up question.")
        return self


class ClientConversationTurn(BaseModel):
    """A completed Client/Advisor exchange available to the Client review role."""

    model_config = ConfigDict(frozen=True)

    client_question: str = Field(min_length=1, max_length=2_000)
    advisor_recommendation: Recommendation


class ClientReviewRequest(BaseModel):
    """Profile-aware recommendation review input, with the remaining turn budget."""

    model_config = ConfigDict(frozen=True)

    profile: ClientProfile
    recommendation: Recommendation
    completed_follow_up_count: int = Field(ge=0)
    max_follow_ups: int = Field(ge=0)
    prior_chat_history: list[ClientConversationTurn] = Field(default_factory=list, max_length=2)


def create_opening_message(session_id: UUID, draft: ClientOpeningDraft) -> ClientMessage:
    """Bind a Client-generated opening question to the server-owned session."""

    return ClientMessage(session_id=session_id, text=draft.question, follow_up_number=0)


def create_client_review(request: ClientReviewRequest, draft: ClientReviewDraft) -> ClientReview:
    """Create a server-bound review and enforce the Client's bounded turn budget."""

    if request.completed_follow_up_count >= request.max_follow_ups:
        return ClientReview(
            session_id=request.recommendation.session_id,
            follow_up_number=request.completed_follow_up_count,
            accepted=True,
            message="The follow-up limit is complete; the simulated Client accepts the response.",
        )
    return ClientReview(
        session_id=request.recommendation.session_id,
        follow_up_number=(
            request.completed_follow_up_count
            if draft.accepted
            else request.completed_follow_up_count + 1
        ),
        accepted=draft.accepted,
        message=(
            "The recommendation addresses my simulated planning question."
            if draft.accepted
            else draft.follow_up_question or ""
        ),
    )
