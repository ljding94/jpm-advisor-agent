"""End-to-end eval smoke: run one persona via FakeLLM, write a report."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.judge import LLMJudge
from src.eval.report import write_report
from src.eval.runner import run_one
from src.tools.web_search import FakeWebSearchProvider, WebResult


def _scripted_main_llm():
    from tests.conftest import FakeLLM

    return FakeLLM(
        marker_responses={
            "decide the advisor's next action": json.dumps({
                "next_action": "dispatch_analyst",
                "target": "analyst",
                "message": "Recommend a balanced portfolio.",
            }),
            "research task from the advisor": (
                "A diversified mix of equities and bonds is appropriate."
            ),
            "reply on a single line starting with [confirm] or [reject]": (
                "[CONFIRM] this matches my goals."
            ),
            "answer in 1–4 sentences in first person": (
                "That sounds right — please proceed."
            ),
        },
        default=json.dumps({"next_action": "draft_advice", "target": "client", "message": ""}),
    )


def _judge_llm():
    from tests.conftest import FakeLLM

    canned = json.dumps({
        "risk_alignment": 4, "goal_alignment": 4, "specificity": 3,
        "coherence": 4, "safety": 5,
        "notes": "Reasonable balanced advice.",
    })
    return FakeLLM(default=canned)


@pytest.fixture
def fake_web():
    return FakeWebSearchProvider(canned=[
        WebResult(title="Diversification 101", url="https://kb/div",
                  snippet="Spread risk across asset classes."),
    ])


def test_run_one_with_judge_smoke(tmp_path, monkeypatch, fake_embedder, fake_web):
    """A single offline run produces a populated RunResult."""
    monkeypatch.chdir(tmp_path)
    real_root = Path(__file__).resolve().parents[2]
    (tmp_path / "data" / "personas").mkdir(parents=True)
    (tmp_path / "data" / "knowledge_base").mkdir(parents=True)
    for f in (real_root / "data" / "personas").iterdir():
        (tmp_path / "data" / "personas" / f.name).write_text(f.read_text())
    for f in (real_root / "data" / "knowledge_base").iterdir():
        (tmp_path / "data" / "knowledge_base" / f.name).write_text(f.read_text())

    main_llm = _scripted_main_llm()
    judge = LLMJudge(_judge_llm())

    # Skip the chroma KB to avoid network/embeddings — the analyst can fall back to web/principles.
    result = run_one(
        "david",
        run_index=0,
        llm=main_llm,
        judge=judge,
        web=fake_web,
        kb=None,
    )

    assert result.persona == "david"
    assert result.final_status == "resolved"
    assert result.deterministic_total > 0
    # Most checks should pass; tolerate the occasional miss (e.g., analyst_cited_sources).
    assert result.deterministic_passed >= result.deterministic_total - 2
    assert result.judge is not None
    assert result.judge["score"]["risk_alignment"] == 4
    assert result.usage["turns"] > 0
    assert len(result.transcript) > 0


def test_write_report_creates_files(tmp_path):
    from src.eval.runner import RunResult

    results = [
        RunResult(
            persona="david",
            run_index=0,
            final_status="resolved",
            termination_reason=None,
            turn_count=8,
            checks=[
                {"name": "status_resolved", "passed": True, "detail": "ok"},
                {"name": "no_pii_leaked", "passed": True, "detail": "clean"},
            ],
            judge={
                "score": {
                    "risk_alignment": 4, "goal_alignment": 4, "specificity": 3,
                    "coherence": 4, "safety": 5, "notes": "balanced", "mean": 4.0,
                },
                "raw": "{}", "error": None,
            },
            usage={"turns": 8, "total_input_tokens": 100, "total_output_tokens": 50,
                   "total_cost_usd": 0.01},
            duration_s=1.2,
        ),
        RunResult(
            persona="priya",
            run_index=0,
            final_status="terminated",
            termination_reason="exceeded MAX_TURNS",
            turn_count=20,
            checks=[
                {"name": "status_resolved", "passed": False, "detail": "terminated"},
            ],
            usage={"turns": 20, "total_input_tokens": 1000, "total_output_tokens": 200,
                   "total_cost_usd": 0.05},
            duration_s=2.5,
        ),
    ]
    out_dir = write_report(results, out_root=tmp_path / "reports", label="test-run")
    assert (out_dir / "results.json").exists()
    assert (out_dir / "report.md").exists()
    md = (out_dir / "report.md").read_text()
    assert "david" in md
    assert "priya" in md
    assert "## Aggregates" in md
    assert "## Deterministic check failures" in md  # priya failed
