"""Routing function tests."""
from __future__ import annotations

from src.graph.routing import route_next
from src.graph.state import ConversationStatus, initial_state
from src.guardrails.limits import MAX_TURNS
from src.schemas import AgentMessage, AgentRole, MessageType


def _state_with_last(role_pair, profile):
    state = initial_state(profile)
    state["conversation_history"].append(
        AgentMessage(
            sender=role_pair[0],
            recipient=role_pair[1],
            content="...",
            message_type=MessageType.QUESTION,
        )
    )
    return state


def test_empty_history_routes_to_advisor(david_profile):
    state = initial_state(david_profile)
    assert route_next(state) == "advisor"


def test_resolved_routes_to_end(david_profile):
    state = initial_state(david_profile)
    state["status"] = ConversationStatus.RESOLVED
    assert route_next(state) == "__end__"


def test_terminated_routes_to_end(david_profile):
    state = initial_state(david_profile)
    state["status"] = ConversationStatus.TERMINATED
    assert route_next(state) == "__end__"


def test_max_turns_alone_does_not_force_end_in_routing(david_profile):
    """Routing only ends on RESOLVED/TERMINATED. Hard-limit enforcement happens in
    the node wrapper which sets status=TERMINATED before routing runs again."""
    state = initial_state(david_profile)
    state["turn_count"] = MAX_TURNS
    # Status is still GATHER_PROFILE → routing falls through to last-message dispatch
    # (or 'advisor' on empty history). The wrapper will trip on the next node call.
    assert route_next(state) == "advisor"


def test_routes_to_client(david_profile):
    state = _state_with_last((AgentRole.ADVISOR, AgentRole.CLIENT), david_profile)
    assert route_next(state) == "client"


def test_routes_to_advisor(david_profile):
    state = _state_with_last((AgentRole.CLIENT, AgentRole.ADVISOR), david_profile)
    assert route_next(state) == "advisor"


def test_routes_to_analyst(david_profile):
    state = _state_with_last((AgentRole.ADVISOR, AgentRole.ANALYST), david_profile)
    assert route_next(state) == "analyst"
