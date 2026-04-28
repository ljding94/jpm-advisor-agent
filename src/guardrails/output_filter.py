"""Output filter — blocks banned phrases, named tickers, enforces disclaimer."""
from __future__ import annotations

import re
from dataclasses import dataclass

from src.schemas import STANDARD_DISCLAIMER, AdviceOutput

_BANNED_PHRASES = (
    "guaranteed return",
    "guaranteed returns",
    "risk-free",
    "risk free",
    "can't lose",
    "cannot lose",
    "no risk",
)

# A 1–5 char all-caps run that looks like a ticker. We allow common false-positives.
_TICKER_RE = re.compile(r"\b[A-Z]{2,5}\b")
_TICKER_WHITELIST = {
    "USA", "US", "UK", "EU", "ETF", "ETFS", "IRA", "HSA",
    "CD", "CDS", "TIPS", "AGI", "AUM", "APR", "APY",
    "GDP", "CPI", "FED", "FOMC", "SP", "NYSE", "NASDAQ", "OTC",
    "ESG", "REIT", "REITS", "FIRE", "FY", "Q1", "Q2", "Q3", "Q4",
    "AI", "ML", "AM", "PM", "USD", "EUR", "JPY", "GBP",
    "PMI", "RSU", "RSUS", "ISO", "ISOS", "NSO", "NSOS",
    "CONFIRM", "REJECT",
}


@dataclass
class FilterResult:
    blocked: bool
    reasons: list[str]
    cleaned_text: str | None = None


def find_banned_phrases(text: str) -> list[str]:
    lowered = text.lower()
    return [p for p in _BANNED_PHRASES if p in lowered]


def find_named_tickers(text: str) -> list[str]:
    candidates = _TICKER_RE.findall(text)
    return [c for c in candidates if c not in _TICKER_WHITELIST]


def filter_text(text: str, *, allow_disclaimer_missing: bool = False) -> FilterResult:
    """Inspect `text`. Returns `blocked=True` and reasons if it violates rules."""
    reasons: list[str] = []
    banned = find_banned_phrases(text)
    if banned:
        reasons.append(f"banned phrases: {', '.join(sorted(set(banned)))}")
    tickers = find_named_tickers(text)
    if tickers:
        reasons.append(f"named tickers: {', '.join(sorted(set(tickers)))}")
    if not allow_disclaimer_missing and "not financial advice" not in text.lower():
        reasons.append("missing standard disclaimer")
    return FilterResult(blocked=bool(reasons), reasons=reasons, cleaned_text=text)


def enforce_advice_disclaimer(advice: AdviceOutput) -> AdviceOutput:
    """Force-append the standard disclaimer if missing."""
    if not any("not financial advice" in d.lower() for d in advice.disclaimers):
        advice = advice.model_copy(update={
            "disclaimers": [*advice.disclaimers, STANDARD_DISCLAIMER]
        })
    return advice


__all__ = [
    "FilterResult",
    "enforce_advice_disclaimer",
    "filter_text",
    "find_banned_phrases",
    "find_named_tickers",
]
