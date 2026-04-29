"""Tests for src/eval/judge.py."""
from __future__ import annotations

import json

from src.eval.judge import (
    JUDGE_SYSTEM_PROMPT,
    LLMJudge,
    JudgeScore,
    aggregate,
    _build_user_prompt,
    _parse_score,
)
from src.graph.state import initial_state
from src.schemas import AdviceOutput, AnalystReport, Source


def _state_with_advice(profile):
    state = initial_state(profile)
    state["analyst_findings"] = AnalystReport(
        query="allocation",
        findings="Diversified equities/bonds.",
        sources=[Source(title="kb")],
        confidence=0.7,
    )
    state["draft_advice"] = AdviceOutput(
        recommendations=["Maintain diversified portfolio."],
        rationale="Matches your risk tolerance.",
    )
    return state


def test_parse_score_clean_json():
    raw = json.dumps({
        "risk_alignment": 5, "goal_alignment": 4, "specificity": 3,
        "coherence": 4, "safety": 5, "notes": "ok",
    })
    score, err = _parse_score(raw)
    assert err is None
    assert score.risk_alignment == 5
    assert score.notes == "ok"
    assert score.mean == (5 + 4 + 3 + 4 + 5) / 5


def test_parse_score_with_fence():
    raw = "Sure!\n```json\n{\"risk_alignment\": 4, \"goal_alignment\": 3, \"specificity\": 3, \"coherence\": 4, \"safety\": 5, \"notes\": \"x\"}\n```"
    score, err = _parse_score(raw)
    assert err is None
    assert score.risk_alignment == 4


def test_parse_score_clamps_out_of_range():
    raw = json.dumps({
        "risk_alignment": 99, "goal_alignment": 0, "specificity": -1,
        "coherence": 3, "safety": 7, "notes": "",
    })
    score, err = _parse_score(raw)
    assert err is None
    assert score.risk_alignment == 5
    assert score.goal_alignment == 1  # 0 clamped up to 1
    assert score.specificity == 1
    assert score.safety == 5


def test_parse_score_garbage():
    score, err = _parse_score("not json at all")
    assert err is not None
    assert score.risk_alignment == 0


def test_judge_uses_llm(david_profile):
    from tests.conftest import FakeLLM

    canned = json.dumps({
        "risk_alignment": 4, "goal_alignment": 4, "specificity": 3,
        "coherence": 4, "safety": 5, "notes": "decent",
    })
    fake = FakeLLM(default=canned)
    judge = LLMJudge(fake)
    state = _state_with_advice(david_profile)
    result = judge.score(state)
    assert result.error is None
    assert result.score.risk_alignment == 4
    assert result.score.notes == "decent"
    # Confirm the system prompt was passed.
    assert any("strict but fair" in m["content"]
               for m in fake.calls[0]["messages"])


def test_judge_no_advice(david_profile):
    from tests.conftest import FakeLLM

    judge = LLMJudge(FakeLLM())
    state = initial_state(david_profile)  # no advice
    result = judge.score(state)
    assert result.error == "no advice to judge"


def test_judge_handles_llm_exception(david_profile):
    class _BoomLLM:
        model = "boom"
        last_usage = None
        cumulative_prompt_tokens = 0
        cumulative_completion_tokens = 0
        cumulative_cost_usd = 0.0

        def complete(self, *_a, **_kw):
            raise RuntimeError("network")

    judge = LLMJudge(_BoomLLM())  # type: ignore[arg-type]
    state = _state_with_advice(david_profile)
    result = judge.score(state)
    assert result.error is not None
    assert "network" in result.error


def test_aggregate():
    scores = [
        JudgeScore(5, 4, 3, 4, 5),
        JudgeScore(3, 4, 5, 4, 5),
    ]
    out = aggregate(scores)
    assert out["risk_alignment"] == 4.0
    assert out["mean"] == round((4.0 + 4.0 + 4.0 + 4.0 + 5.0) / 5, 3)


def test_aggregate_empty():
    out = aggregate([])
    assert out["mean"] == 0.0


def test_user_prompt_includes_profile(david_profile):
    state = _state_with_advice(david_profile)
    text = _build_user_prompt(state)
    assert david_profile.name in text
    assert "moderate" in text
    assert "Maintain diversified portfolio" in text
