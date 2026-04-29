"""Tests for src/eval/deterministic.py."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.eval.deterministic import (
    DEFAULT_CHECKS,
    all_three_agents_spoke,
    analyst_cited_sources,
    disclaimer_present,
    no_banned_phrases,
    no_named_tickers,
    no_pii_leaked,
    run_all_checks,
    state_errors_clean,
    status_resolved,
    under_token_budget,
)
from src.graph.state import ConversationStatus, initial_state
from src.observability.logger import TurnLogger
from src.schemas import (
    AdviceOutput,
    AgentMessage,
    AgentRole,
    AnalystReport,
    MessageType,
    Source,
)


def _profile(all_personas):
    return all_personas["david"]


def _msg(sender: AgentRole, recipient: AgentRole, content: str) -> AgentMessage:
    return AgentMessage(
        sender=sender,
        recipient=recipient,
        content=content,
        message_type=MessageType.QUESTION,
        timestamp=datetime.now(timezone.utc),
    )


def _good_state(all_personas):
    state = initial_state(_profile(all_personas))
    state["status"] = ConversationStatus.RESOLVED
    state["conversation_history"] = [
        _msg(AgentRole.CLIENT, AgentRole.ADVISOR, "I want to plan for retirement."),
        _msg(AgentRole.ADVISOR, AgentRole.ANALYST, "Research balanced allocations."),
        _msg(AgentRole.ANALYST, AgentRole.ADVISOR, "Diversified mix is appropriate."),
        _msg(AgentRole.ADVISOR, AgentRole.CLIENT, "Here is my recommendation."),
    ]
    state["analyst_findings"] = AnalystReport(
        query="balanced allocation",
        findings="A diversified portfolio across stocks, bonds, and cash is appropriate.",
        sources=[Source(title="Asset allocation primer", url="https://kb/alloc")],
        confidence=0.8,
    )
    state["draft_advice"] = AdviceOutput(
        recommendations=[
            "Maintain a diversified mix across equities and fixed income.",
            "Increase your tax-advantaged contributions.",
        ],
        rationale="Diversification matches your moderate risk tolerance and long horizon.",
        sources=[Source(title="kb")],
    )
    return state


def test_status_resolved_pass(all_personas):
    s = _good_state(all_personas)
    assert status_resolved(s).passed


def test_status_resolved_fail(all_personas):
    s = _good_state(all_personas)
    s["status"] = ConversationStatus.TERMINATED
    r = status_resolved(s)
    assert not r.passed
    assert "terminated" in r.detail


def test_all_three_agents_spoke(all_personas):
    s = _good_state(all_personas)
    assert all_three_agents_spoke(s).passed


def test_all_three_agents_missing(all_personas):
    s = _good_state(all_personas)
    # Remove the analyst message
    s["conversation_history"] = [m for m in s["conversation_history"]
                                 if m.sender is not AgentRole.ANALYST]
    r = all_three_agents_spoke(s)
    assert not r.passed
    assert "analyst" in r.detail


def test_disclaimer_present(all_personas):
    s = _good_state(all_personas)
    assert disclaimer_present(s).passed  # AdviceOutput auto-appends it


def test_disclaimer_missing(all_personas):
    s = _good_state(all_personas)
    s["draft_advice"] = None
    assert not disclaimer_present(s).passed


def test_no_pii_leaked(all_personas):
    s = _good_state(all_personas)
    assert no_pii_leaked(s).passed


def test_pii_leak_detected(all_personas):
    s = _good_state(all_personas)
    s["conversation_history"].append(
        _msg(AgentRole.CLIENT, AgentRole.ADVISOR, "My ssn is 123-45-6789.")
    )
    r = no_pii_leaked(s)
    assert not r.passed
    assert "ssn" in r.detail


def test_no_banned_phrases(all_personas):
    s = _good_state(all_personas)
    assert no_banned_phrases(s).passed


def test_banned_phrase_detected(all_personas):
    s = _good_state(all_personas)
    s["draft_advice"] = s["draft_advice"].model_copy(update={
        "rationale": "This portfolio offers guaranteed returns over the long term.",
    })
    r = no_banned_phrases(s)
    assert not r.passed
    assert "guaranteed returns" in r.detail


def test_no_named_tickers(all_personas):
    s = _good_state(all_personas)
    assert no_named_tickers(s).passed


def test_named_ticker_detected(all_personas):
    s = _good_state(all_personas)
    s["draft_advice"] = s["draft_advice"].model_copy(update={
        "rationale": "Buy MSFT and AAPL for tech exposure.",
    })
    r = no_named_tickers(s)
    assert not r.passed


def test_analyst_cited_sources(all_personas):
    s = _good_state(all_personas)
    assert analyst_cited_sources(s).passed


def test_analyst_no_findings(all_personas):
    s = _good_state(all_personas)
    s["analyst_findings"] = None
    assert not analyst_cited_sources(s).passed


def test_under_token_budget_no_logger(all_personas):
    s = _good_state(all_personas)
    assert under_token_budget(s, turn_logger=None).passed


def test_under_token_budget_pass(all_personas):
    s = _good_state(all_personas)
    tl = TurnLogger()
    tl.total_cost_usd = 0.05
    assert under_token_budget(s, turn_logger=tl, budget_usd=2.0).passed


def test_under_token_budget_exceeded(all_personas):
    s = _good_state(all_personas)
    tl = TurnLogger()
    tl.total_cost_usd = 5.0
    r = under_token_budget(s, turn_logger=tl, budget_usd=2.0)
    assert not r.passed
    assert "5.00" in r.detail


def test_state_errors_clean(all_personas):
    s = _good_state(all_personas)
    assert state_errors_clean(s).passed


def test_state_errors_dirty(all_personas):
    s = _good_state(all_personas)
    s["errors"] = ["2026-04-29T00:00:00 [analyst.kb] timed out"]
    r = state_errors_clean(s)
    assert not r.passed
    assert "1 errors" in r.detail


def test_run_all_checks(all_personas):
    s = _good_state(all_personas)
    results = run_all_checks(s, turn_logger=None)
    assert len(results) == len(DEFAULT_CHECKS)
    assert all(c.passed for c in results)
