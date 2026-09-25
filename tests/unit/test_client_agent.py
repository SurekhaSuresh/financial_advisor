from google.adk.tools import AgentTool

from financial_advisor.agents.client import CLIENT_INSTRUCTION, create_client_agent
from financial_advisor.contracts import ClientResult, ClientTask


def test_client_agent_has_one_explicit_input_and_output_contract() -> None:
    agent = create_client_agent("test-model")

    assert agent.name == "client_agent"
    assert agent.model == "test-model"
    assert agent.input_schema is ClientTask
    assert agent.output_schema is ClientResult
    assert agent.tools == []
    assert agent.output_key is None
    assert agent.include_contents == "default"
    assert not agent.disallow_transfer_to_parent
    assert not agent.disallow_transfer_to_peers


def test_client_agent_can_be_called_directly_as_an_agent_tool() -> None:
    agent = create_client_agent("test-model")

    tool = AgentTool(agent)

    assert tool.name == "client_agent"
    assert tool.agent is agent


def test_client_prompt_enforces_the_deterministic_follow_up_limit() -> None:
    assert "follow_up_count is 2 or more, accept" in CLIENT_INSTRUCTION
