"""Risk strategy tests."""
from __future__ import annotations

import pytest

from src.schemas import ClientProfile
from src.strategies.risk_profile import (
    AggressiveStrategy,
    ConservativeStrategy,
    ModerateStrategy,
    RiskStrategy,
    get_strategy,
)


@pytest.fixture
def stub_profile() -> ClientProfile:
    return ClientProfile(
        name="Stub", age=40, risk_tolerance="moderate",
        time_horizon_years=20, annual_income=100_000,
    )


@pytest.mark.parametrize(
    "risk,strategy_cls",
    [
        ("conservative", ConservativeStrategy),
        ("moderate", ModerateStrategy),
        ("aggressive", AggressiveStrategy),
    ],
)
def test_get_strategy_dispatches_correctly(risk, strategy_cls):
    strat = get_strategy(risk)
    assert isinstance(strat, strategy_cls)


def test_get_strategy_unknown_raises():
    with pytest.raises(ValueError):
        get_strategy("balanced")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "strategy",
    [ConservativeStrategy(), ModerateStrategy(), AggressiveStrategy()],
)
def test_allocations_sum_to_100(strategy: RiskStrategy, stub_profile: ClientProfile):
    alloc = strategy.recommended_allocation(stub_profile)
    assert sum(alloc.values()) == pytest.approx(100.0)
    for v in alloc.values():
        assert 0 <= v <= 100


def test_conservative_is_more_bond_heavy_than_aggressive(stub_profile: ClientProfile):
    cons = ConservativeStrategy().recommended_allocation(stub_profile)
    aggr = AggressiveStrategy().recommended_allocation(stub_profile)
    assert cons["bonds"] > aggr["bonds"]
    assert aggr["equities"] > cons["equities"]


@pytest.mark.parametrize(
    "strategy",
    [ConservativeStrategy(), ModerateStrategy(), AggressiveStrategy()],
)
def test_headline_advice_non_empty(strategy: RiskStrategy, stub_profile: ClientProfile):
    bullets = strategy.headline_advice(stub_profile)
    assert len(bullets) >= 3
    assert all(isinstance(b, str) and b.strip() for b in bullets)
