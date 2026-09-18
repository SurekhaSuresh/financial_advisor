"""Tests for the bounded simulated Client Agent contracts."""

import pytest
from pydantic import ValidationError

from financial_advisor.agents.client.contracts import ClientReviewDraft


def test_client_review_requires_a_follow_up_when_not_accepted() -> None:
    """A dissatisfied Client must provide one focused question to the Advisor."""

    with pytest.raises(ValidationError, match="requires a follow-up question"):
        ClientReviewDraft(accepted=False)


def test_client_review_rejects_a_follow_up_when_accepted() -> None:
    """An accepted recommendation closes the current Client review turn."""

    with pytest.raises(ValidationError, match="cannot include a follow-up question"):
        ClientReviewDraft(accepted=True, follow_up_question="Can you clarify this?")
