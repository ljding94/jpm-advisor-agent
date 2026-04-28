"""ClientAgent tests with FakeLLM."""
from __future__ import annotations

from src.agents.client import ClientAgent
from src.graph.state import ConversationStatus, initial_state
from src.schemas import AgentMessage, AgentRole, MessageType


def test_client_loads_from_persona_file(fake_llm):
    agent = ClientAgent.from_persona_file("data/personas/david_moderate.json", llm=fake_llm)
    assert agent.profile.name == "David Patel"
    assert agent.profile.risk_tolerance == "moderate"
    assert agent.role is AgentRole.CLIENT


def test_client_answers_advisor_question(david_profile, fake_llm):
    fake_llm.script.append("I'm hoping to retire at 60 and put both kids through college.")
    agent = ClientAgent(profile=david_profile, llm=fake_llm)
    state = initial_state(david_profile)
    state["conversation_history"].append(
        AgentMessage(
            sender=AgentRole.ADVISOR,
            recipient=AgentRole.CLIENT,
            content="What are your top financial goals?",
            message_type=MessageType.QUESTION,
        )
    )
    new_state = agent.process(state)
    last = new_state["conversation_history"][-1]
    assert last.sender is AgentRole.CLIENT
    assert last.recipient is AgentRole.ADVISOR
    assert last.message_type is MessageType.ANSWER
    assert "retire at 60" in last.content


def test_client_confirms_advice_resolves_state(david_profile, fake_llm):
    fake_llm.script.append("[CONFIRM] looks great, that matches my goals.")
    agent = ClientAgent(profile=david_profile, llm=fake_llm)
    state = initial_state(david_profile)
    state["conversation_history"].append(
        AgentMessage(
            sender=AgentRole.ADVISOR,
            recipient=AgentRole.CLIENT,
            content="Recommendation: 60/30/10 stocks/bonds/cash.",
            message_type=MessageType.ADVICE,
        )
    )
    new_state = agent.process(state)
    assert new_state["status"] is ConversationStatus.RESOLVED
    assert new_state["conversation_history"][-1].metadata["confirmed"] is True


def test_client_rejects_advice_loops_back(david_profile, fake_llm):
    fake_llm.script.append("[REJECT] too aggressive given my horizon.")
    agent = ClientAgent(profile=david_profile, llm=fake_llm)
    state = initial_state(david_profile)
    state["conversation_history"].append(
        AgentMessage(
            sender=AgentRole.ADVISOR,
            recipient=AgentRole.CLIENT,
            content="Recommendation: 95% equities.",
            message_type=MessageType.ADVICE,
        )
    )
    new_state = agent.process(state)
    assert new_state["status"] is ConversationStatus.ANALYZE
    assert new_state["conversation_history"][-1].metadata["confirmed"] is False


def test_client_no_op_when_last_message_not_for_client(david_profile, fake_llm):
    agent = ClientAgent(profile=david_profile, llm=fake_llm)
    state = initial_state(david_profile)
    # advisor → analyst (not for the client)
    state["conversation_history"].append(
        AgentMessage(
            sender=AgentRole.ADVISOR,
            recipient=AgentRole.ANALYST,
            content="please research...",
            message_type=MessageType.TASK,
        )
    )
    new_state = agent.process(state)
    assert new_state == state  # unchanged


def test_client_open_conversation_seeds_history(david_profile, fake_llm):
    agent = ClientAgent(profile=david_profile, llm=fake_llm)
    state = agent.open_conversation(initial_state(david_profile))
    assert len(state["conversation_history"]) == 1
    msg = state["conversation_history"][0]
    assert msg.sender is AgentRole.CLIENT
    assert msg.recipient is AgentRole.ADVISOR
    assert "David Patel" in msg.content
