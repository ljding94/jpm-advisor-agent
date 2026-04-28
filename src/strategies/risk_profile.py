"""Risk-profile strategies (Strategy pattern)."""
from __future__ import annotations

from abc import ABC, abstractmethod

from src.schemas import ClientProfile, RiskTolerance


class RiskStrategy(ABC):
    name: str

    @abstractmethod
    def recommended_allocation(self, profile: ClientProfile) -> dict[str, float]:
        """Return a target allocation as percentages summing to ~100."""

    @abstractmethod
    def headline_advice(self, profile: ClientProfile) -> list[str]:
        """Return a few qualitative bullets specific to this risk profile."""


class ConservativeStrategy(RiskStrategy):
    name = "conservative"

    def recommended_allocation(self, profile: ClientProfile) -> dict[str, float]:
        # Heavier in bonds/cash; small equity sleeve for inflation hedge.
        return {"equities": 35.0, "bonds": 55.0, "cash": 10.0}

    def headline_advice(self, profile: ClientProfile) -> list[str]:
        return [
            "Prioritize capital preservation over growth.",
            "Hold 1–2 years of essential spending in cash equivalents to avoid sequence-of-returns risk.",
            "Use a Treasury bond ladder or short-duration aggregate bond fund for stable income.",
            "Keep an inflation hedge via a small equity sleeve and TIPS exposure.",
        ]


class ModerateStrategy(RiskStrategy):
    name = "moderate"

    def recommended_allocation(self, profile: ClientProfile) -> dict[str, float]:
        return {"equities": 65.0, "bonds": 30.0, "cash": 5.0}

    def headline_advice(self, profile: ClientProfile) -> list[str]:
        return [
            "Balance growth and stability for a multi-decade horizon.",
            "Diversify equities globally: ~65% domestic, ~30% international developed, ~5% emerging.",
            "Use tax-advantaged accounts first (401(k) match → HSA → IRA → remaining 401(k)).",
            "Rebalance annually or when allocations drift more than 5 percentage points.",
        ]


class AggressiveStrategy(RiskStrategy):
    name = "aggressive"

    def recommended_allocation(self, profile: ClientProfile) -> dict[str, float]:
        return {"equities": 90.0, "bonds": 5.0, "cash": 5.0}

    def headline_advice(self, profile: ClientProfile) -> list[str]:
        return [
            "Maximize long-term growth with broad equity exposure.",
            "Cap any speculative sleeve (single names, crypto) at 10% of investable assets.",
            "Maintain a fully funded emergency reserve so you never have to sell equities in a drawdown.",
            "Stress-test emotionally: a 90/10 portfolio can fall 40%+ in a bear market.",
        ]


_STRATEGIES: dict[str, type[RiskStrategy]] = {
    "conservative": ConservativeStrategy,
    "moderate": ModerateStrategy,
    "aggressive": AggressiveStrategy,
}


def get_strategy(risk_tolerance: RiskTolerance) -> RiskStrategy:
    cls = _STRATEGIES.get(risk_tolerance)
    if cls is None:
        raise ValueError(f"unknown risk_tolerance: {risk_tolerance}")
    return cls()


__all__ = [
    "AggressiveStrategy",
    "ConservativeStrategy",
    "ModerateStrategy",
    "RiskStrategy",
    "get_strategy",
]
