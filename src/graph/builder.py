"""LangGraph StateGraph builder."""
from __future__ import annotations

import sys
import time
from typing import Any, Callable

from langgraph.graph import END, StateGraph

from src.agents.advisor import AdvisorAgent
from src.agents.analyst import AnalystAgent
from src.agents.base import BaseAgent
from src.agents.client import ClientAgent
from src.agents.reviewer import ReviewerAgent
from src.graph.routing import route_next
from src.graph.state import AdvisorState, ConversationStatus, append_error
from src.guardrails.limits import LimitState, check_limits
from src.observability.logger import TurnLogger


def _wrap_node(
    agent: BaseAgent,
    *,
    label: str,
    verbose: bool = False,
    turn_logger: TurnLogger | None = None,
) -> Callable[[AdvisorState], AdvisorState]:
    """Wrap an agent.process so the runtime enforces limits and records turns.

    - Trips MAX_TURNS / MAX_TOTAL_COST_USD before invoking the node.
    - Reads `agent.llm.last_usage` after the node runs to attribute tokens/cost.
    - Records a TurnRecord per node call (Observer pattern).
    - Optionally prints stderr breadcrumbs (verbose=True).
    """

    def wrapper(state: AdvisorState) -> AdvisorState:
        running_cost = turn_logger.total_cost_usd if turn_logger is not None else 0.0
        breach = check_limits(
            LimitState(
                turn_count=state.get("turn_count", 0),
                total_cost_usd=running_cost,
            )
        )
        if breach is not None:
            terminated: AdvisorState = {
                **state,
                "status": ConversationStatus.TERMINATED,
                "termination_reason": breach.detail,
            }
            return append_error(terminated, source="limits", detail=f"{breach.name}: {breach.detail}")

        if verbose:
            turn = state.get("turn_count", 0) + 1
            print(f"  [{turn:02d}] {label} thinking...", file=sys.stderr, flush=True)

        # Snapshot CUMULATIVE usage so multi-call nodes attribute correctly.
        before_in = int(getattr(agent.llm, "cumulative_prompt_tokens", 0))
        before_out = int(getattr(agent.llm, "cumulative_completion_tokens", 0))
        before_cost = float(getattr(agent.llm, "cumulative_cost_usd", 0.0))
        started_at = time.monotonic()

        new_state = agent.process(state)

        if turn_logger is not None:
            after_in = int(getattr(agent.llm, "cumulative_prompt_tokens", 0))
            after_out = int(getattr(agent.llm, "cumulative_completion_tokens", 0))
            after_cost = float(getattr(agent.llm, "cumulative_cost_usd", 0.0))
            turn_logger.record_turn(
                agent=label,
                action=_infer_action(state, new_state),
                started_at=started_at,
                input_tokens=max(0, after_in - before_in),
                output_tokens=max(0, after_out - before_out),
                cost_usd=max(0.0, after_cost - before_cost),
            )

        if verbose:
            history = new_state.get("conversation_history", [])
            if history:
                last = history[-1]
                preview = last.content.replace("\n", " ")[:120]
                suffix = "..." if len(last.content) > 120 else ""
                print(
                    f"       {last.sender.value} -> {last.recipient.value} "
                    f"({last.message_type.value}): {preview}{suffix}",
                    file=sys.stderr,
                    flush=True,
                )
        return new_state

    return wrapper


def _infer_action(prev: AdvisorState, new: AdvisorState) -> str:
    prev_n = len(prev.get("conversation_history", []))
    new_n = len(new.get("conversation_history", []))
    if new_n > prev_n:
        last = new["conversation_history"][-1]
        return f"{last.message_type.value}->{last.recipient.value}"
    return "no_op"


def build_graph(
    client: ClientAgent,
    advisor: AdvisorAgent,
    analyst: AnalystAgent,
    reviewer: ReviewerAgent,
    *,
    verbose: bool = False,
    turn_logger: TurnLogger | None = None,
) -> Any:
    """Build the StateGraph and compile it. Returns a runnable graph."""
    graph: StateGraph = StateGraph(AdvisorState)
    graph.add_node(
        "client",
        _wrap_node(client, label="client", verbose=verbose, turn_logger=turn_logger),
    )
    graph.add_node(
        "advisor",
        _wrap_node(advisor, label="advisor", verbose=verbose, turn_logger=turn_logger),
    )
    graph.add_node(
        "analyst",
        _wrap_node(analyst, label="analyst", verbose=verbose, turn_logger=turn_logger),
    )
    graph.add_node(
        "reviewer",
        _wrap_node(reviewer, label="reviewer", verbose=verbose, turn_logger=turn_logger),
    )

    graph.set_entry_point("advisor")

    cond_map: dict[Any, str] = {
        "client": "client",
        "advisor": "advisor",
        "analyst": "analyst",
        "reviewer": "reviewer",
        "__end__": END,
    }
    for node in ("client", "advisor", "analyst", "reviewer"):
        graph.add_conditional_edges(node, route_next, cond_map)

    return graph.compile()


__all__ = ["build_graph"]
