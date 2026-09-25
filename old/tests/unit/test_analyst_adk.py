"""Tests for Analyst ADK construction without a live model call."""

from financial_advisor.agents.analyst.adk import create_analyst_agent
from financial_advisor.agents.analyst.contracts import AnalystResearchDraft
from financial_advisor.config import GeminiSettings


def test_create_analyst_agent_has_structured_internal_output() -> None:
    settings = GeminiSettings(
        analyst_model="test-analyst-model",
        temperature=0.2,
        analyst_max_output_tokens=1_024,
    )

    agent = create_analyst_agent(settings)

    assert agent.name == "analyst_agent"
    assert agent.model == "test-analyst-model"
    assert agent.output_schema is AnalystResearchDraft
    assert agent.output_key == "analyst_research_draft"
    assert agent.tools == []
    assert agent.disallow_transfer_to_parent is True
    assert agent.disallow_transfer_to_peers is True
