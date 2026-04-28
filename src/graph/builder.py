"""LangGraph StateGraph builder."""
from __future__ import annotations

from typing import Any, Callable

from langgraph.graph import END, StateGraph

from src.agents.advisor import AdvisorAgent
from src.agents.analyst import AnalystAgent
from src.agents.client import ClientAgent
from src.graph.routing import route_next
from src.graph.state import AdvisorState, ConversationStatus
from src.guardrails.limits import LimitState, check_limits


def _wrap_with_limit_check(node_fn: Callable[[AdvisorState], AdvisorState]) -> Callable[[AdvisorState], AdvisorState]:
    """Marks state TERMINATED if limits are breached before running the node."""

    def wrapper(state: AdvisorState) -> AdvisorState:
        breach = check_limits(LimitState(turn_count=state.get("turn_count", 0)))
        if breach is not None:
            return {
                **state,
                "status": ConversationStatus.TERMINATED,
                "termination_reason": breach.detail,
            }
        return node_fn(state)

    return wrapper


def build_graph(
    client: ClientAgent,
    advisor: AdvisorAgent,
    analyst: AnalystAgent,
) -> Any:
    """Build the StateGraph and compile it. Returns a runnable graph."""
    graph: StateGraph = StateGraph(AdvisorState)
    graph.add_node("client", _wrap_with_limit_check(client.process))
    graph.add_node("advisor", _wrap_with_limit_check(advisor.process))
    graph.add_node("analyst", _wrap_with_limit_check(analyst.process))

    graph.set_entry_point("advisor")

    # Conditional edges from each node use the same routing function.
    cond_map: dict[Any, str] = {
        "client": "client",
        "advisor": "advisor",
        "analyst": "analyst",
        "__end__": END,
    }
    for node in ("client", "advisor", "analyst"):
        graph.add_conditional_edges(node, route_next, cond_map)

    return graph.compile()


__all__ = ["build_graph"]
