"""Hard runtime limits."""
from __future__ import annotations

from dataclasses import dataclass

MAX_TURNS = 20
MAX_TOKENS_PER_CALL = 4000
MAX_TOTAL_COST_USD = 2.00
PER_CALL_TIMEOUT_SECONDS = 60.0


@dataclass
class LimitState:
    turn_count: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0


@dataclass
class LimitBreach:
    name: str
    detail: str


def check_limits(state: LimitState) -> LimitBreach | None:
    if state.turn_count >= MAX_TURNS:
        return LimitBreach(
            name="max_turns",
            detail=f"turn_count={state.turn_count} reached MAX_TURNS={MAX_TURNS}",
        )
    if state.total_cost_usd >= MAX_TOTAL_COST_USD:
        return LimitBreach(
            name="max_cost",
            detail=f"total_cost_usd={state.total_cost_usd:.2f} reached MAX_TOTAL_COST_USD={MAX_TOTAL_COST_USD:.2f}",
        )
    return None


__all__ = [
    "LimitBreach",
    "LimitState",
    "MAX_TOKENS_PER_CALL",
    "MAX_TOTAL_COST_USD",
    "MAX_TURNS",
    "PER_CALL_TIMEOUT_SECONDS",
    "check_limits",
]
