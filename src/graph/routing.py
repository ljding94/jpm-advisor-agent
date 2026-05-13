"""Routing function for the LangGraph state machine."""
from __future__ import annotations

from typing import Literal

from src.graph.state import AdvisorState, ConversationStatus
from src.schemas import AgentRole

NextNode = Literal["client", "advisor", "analyst", "reviewer", "__end__"]


def route_next(state: AdvisorState) -> NextNode:
    """Decide which agent node should run next based on state.

    Routing rules:
    - If status is RESOLVED or TERMINATED → END.
    - Otherwise dispatch to whichever agent the last message is *for*.

    Hard-limit termination is enforced by the node-wrapper in `builder.py`,
    which marks status=TERMINATED — routing then sees it and exits.
    """
    status = state.get("status", ConversationStatus.GATHER_PROFILE)
    if status in (ConversationStatus.RESOLVED, ConversationStatus.TERMINATED):
        return "__end__"

    history = state.get("conversation_history", [])
    if not history:
        return "advisor"  # advisor opens if no history yet

    last = history[-1]
    if last.recipient is AgentRole.CLIENT:
        return "client"
    if last.recipient is AgentRole.ANALYST:
        return "analyst"
    if last.recipient is AgentRole.ADVISOR:
        return "advisor"
    if last.recipient is AgentRole.REVIEWER:
        return "reviewer"
    return "__end__"


__all__ = ["NextNode", "route_next"]
