"""Tests for src/ui/human_client.py — HumanClientAgent."""
from __future__ import annotations

import queue
import threading

from src.graph.state import ConversationStatus, initial_state
from src.schemas import (
    AgentMessage,
    AgentRole,
    MessageType,
)
from src.ui.human_client import HumanClientAgent


def _seed_with_question(state, question="What's your goal?"):
    msg = AgentMessage(
        sender=AgentRole.ADVISOR,
        recipient=AgentRole.CLIENT,
        content=question,
        message_type=MessageType.QUESTION,
    )
    history = list(state.get("conversation_history", []))
    history.append(msg)
    return {**state, "conversation_history": history}


def _seed_with_advice(state, advice="Buy diversified ETFs."):
    msg = AgentMessage(
        sender=AgentRole.ADVISOR,
        recipient=AgentRole.CLIENT,
        content=advice,
        message_type=MessageType.ADVICE,
    )
    history = list(state.get("conversation_history", []))
    history.append(msg)
    return {**state, "conversation_history": history}


def _agent_with_response(profile, response: str) -> HumanClientAgent:
    from tests.conftest import FakeLLM

    prompts: queue.Queue = queue.Queue()
    responses: queue.Queue = queue.Queue()
    agent = HumanClientAgent(
        profile=profile, llm=FakeLLM(),
        prompts_out=prompts, responses_in=responses,
    )
    responses.put(response)
    return agent


def test_human_client_question_uses_typed_response(david_profile):
    agent = _agent_with_response(david_profile, "I want to retire at 60 comfortably.")
    state = _seed_with_question(initial_state(david_profile))
    new_state = agent.process(state)
    history = new_state["conversation_history"]
    assert history[-1].sender is AgentRole.CLIENT
    assert history[-1].content == "I want to retire at 60 comfortably."
    assert history[-1].message_type is MessageType.ANSWER


def test_human_client_pushes_prompt_for_ui(david_profile):
    from tests.conftest import FakeLLM

    prompts: queue.Queue = queue.Queue()
    responses: queue.Queue = queue.Queue()
    agent = HumanClientAgent(
        profile=david_profile, llm=FakeLLM(),
        prompts_out=prompts, responses_in=responses,
    )
    state = _seed_with_question(initial_state(david_profile), question="age?")

    # Run the agent in a thread; it will block on responses.get().
    holder: dict = {}
    t = threading.Thread(target=lambda: holder.setdefault("state", agent.process(state)))
    t.start()

    pushed = prompts.get(timeout=2.0)
    assert pushed["kind"] == "question"
    assert pushed["content"] == "age?"

    responses.put("42")
    t.join(timeout=2.0)
    assert holder["state"]["conversation_history"][-1].content == "42"


def test_human_client_confirm_resolves(david_profile):
    agent = _agent_with_response(david_profile, "[CONFIRM] looks good")
    state = _seed_with_advice(initial_state(david_profile))
    new_state = agent.process(state)
    last = new_state["conversation_history"][-1]
    assert last.message_type is MessageType.CONFIRMATION
    assert last.metadata["confirmed"] is True
    assert new_state["status"] is ConversationStatus.RESOLVED


def test_human_client_reject_loops_back(david_profile):
    agent = _agent_with_response(david_profile, "[REJECT] not aggressive enough")
    state = _seed_with_advice(initial_state(david_profile))
    new_state = agent.process(state)
    assert new_state["status"] is ConversationStatus.ANALYZE
    assert new_state["conversation_history"][-1].metadata["confirmed"] is False


def test_human_client_normalizes_plain_yes(david_profile):
    agent = _agent_with_response(david_profile, "yes that's great")
    state = _seed_with_advice(initial_state(david_profile))
    new_state = agent.process(state)
    last = new_state["conversation_history"][-1]
    assert last.content.startswith("[CONFIRM]")
    assert new_state["status"] is ConversationStatus.RESOLVED


def test_human_client_normalizes_plain_no(david_profile):
    agent = _agent_with_response(david_profile, "no thanks")
    state = _seed_with_advice(initial_state(david_profile))
    new_state = agent.process(state)
    last = new_state["conversation_history"][-1]
    assert last.content.startswith("[REJECT]")
    assert new_state["status"] is ConversationStatus.ANALYZE
