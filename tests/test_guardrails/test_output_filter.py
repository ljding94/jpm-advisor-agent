"""Output filter tests."""
from __future__ import annotations

import pytest

from src.guardrails.output_filter import (
    enforce_advice_disclaimer,
    filter_text,
    find_banned_phrases,
    find_named_tickers,
)
from src.schemas import AdviceOutput


@pytest.mark.parametrize(
    "text,phrase",
    [
        ("This product offers a guaranteed return.", "guaranteed return"),
        ("100% risk-free investment.", "risk-free"),
        ("You can't lose money.", "can't lose"),
    ],
)
def test_banned_phrases_detected(text, phrase):
    assert phrase in find_banned_phrases(text)


def test_named_ticker_detected():
    assert "AAPL" in find_named_tickers("Buy AAPL now.")


def test_named_ticker_whitelist():
    assert find_named_tickers("Use an IRA or HSA for tax advantages.") == []


def test_filter_text_blocks_banned_phrase():
    res = filter_text("This is a guaranteed return investment. Not financial advice.")
    assert res.blocked is True
    assert any("banned phrase" in r for r in res.reasons)


def test_filter_text_blocks_named_ticker():
    res = filter_text("I recommend AAPL. Not financial advice.")
    assert res.blocked is True
    assert any("named ticker" in r for r in res.reasons)


def test_filter_text_requires_disclaimer():
    res = filter_text("60/30/10 stocks/bonds/cash.")
    assert res.blocked is True
    assert any("disclaimer" in r for r in res.reasons)


def test_filter_text_clean_passes():
    res = filter_text(
        "Diversify across global equities and bonds. This is not financial advice."
    )
    assert res.blocked is False
    assert res.reasons == []


def test_enforce_advice_disclaimer_appends_when_missing():
    advice = AdviceOutput(
        recommendations=["Diversify."],
        rationale="Risk reduction.",
        disclaimers=["Custom note."],  # no standard disclaimer
    )
    # AdviceOutput already auto-appends, so test the explicit enforcement is idempotent.
    out = enforce_advice_disclaimer(advice)
    assert any("not financial advice" in d.lower() for d in out.disclaimers)


def test_enforce_advice_disclaimer_idempotent():
    advice = AdviceOutput(
        recommendations=["Diversify."], rationale="r",
        disclaimers=["This is not financial advice — see a pro."],
    )
    before = len(advice.disclaimers)
    out = enforce_advice_disclaimer(advice)
    assert len(out.disclaimers) == before
