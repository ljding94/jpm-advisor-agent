"""Limit checks."""
from __future__ import annotations

from src.guardrails.limits import (
    MAX_TOTAL_COST_USD,
    MAX_TURNS,
    LimitState,
    check_limits,
)


def test_no_breach_at_zero():
    assert check_limits(LimitState()) is None


def test_max_turns_breach():
    breach = check_limits(LimitState(turn_count=MAX_TURNS))
    assert breach is not None
    assert breach.name == "max_turns"


def test_max_cost_breach():
    breach = check_limits(LimitState(total_cost_usd=MAX_TOTAL_COST_USD + 0.01))
    assert breach is not None
    assert breach.name == "max_cost"


def test_below_limits_no_breach():
    assert check_limits(LimitState(turn_count=MAX_TURNS - 1, total_cost_usd=MAX_TOTAL_COST_USD - 0.5)) is None
