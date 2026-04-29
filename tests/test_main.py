"""Smoke test for `python -m src.main`. Patches LLM/web/embedder so no key needed."""
from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from src.graph.state import ConversationStatus


def _scripted_llm(persona_key: str):
    from tests.conftest import FakeLLM

    return FakeLLM(
        marker_responses={
            "decide the advisor's next action": json.dumps({
                "next_action": "dispatch_analyst",
                "target": "analyst",
                "message": f"Research a sensible portfolio for {persona_key}.",
            }),
            "research task from the advisor": (
                "A balanced allocation appropriate to the client's risk tolerance "
                "is well supported by standard planning principles."
            ),
            "reply on a single line starting with [confirm] or [reject]": (
                "[CONFIRM] this matches my goals."
            ),
            "answer in 1–4 sentences in first person": (
                "That sounds right — please go ahead."
            ),
        },
        default=json.dumps({"next_action": "draft_advice", "target": "client", "message": ""}),
    )


@pytest.mark.parametrize("persona_key", ["margaret", "david", "priya"])
def test_main_run_writes_transcript(persona_key, tmp_path, monkeypatch, fake_embedder):
    """Acceptance #3: each persona produces examples/sample_conversation_<persona>.md."""
    monkeypatch.chdir(tmp_path)

    # Stage the project files (data/personas, data/knowledge_base) under tmp_path.
    project_root = Path(os.environ.get("PYTEST_PROJECT_ROOT", os.getcwd()))
    # We need to reach back to the actual project. Use the directory of this file.
    real_root = Path(__file__).resolve().parents[1]
    (tmp_path / "data").mkdir(exist_ok=True)
    (tmp_path / "data" / "personas").mkdir(exist_ok=True)
    (tmp_path / "data" / "knowledge_base").mkdir(exist_ok=True)
    for f in (real_root / "data" / "personas").iterdir():
        (tmp_path / "data" / "personas" / f.name).write_text(f.read_text())
    for f in (real_root / "data" / "knowledge_base").iterdir():
        (tmp_path / "data" / "knowledge_base" / f.name).write_text(f.read_text())

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    from src import main as main_mod

    fake_llm = _scripted_llm(persona_key)

    class _NoOpWeb:
        def search(self, query, max_results=5):
            return []

    monkeypatch.setattr(main_mod, "get_llm_provider", lambda: fake_llm)
    monkeypatch.setattr(main_mod, "DDGSearchProvider", lambda: _NoOpWeb())
    monkeypatch.setattr(main_mod, "get_embedding_provider", lambda: fake_embedder)

    final = main_mod.run(persona_key)
    assert final["status"] is ConversationStatus.RESOLVED

    out = tmp_path / "examples" / f"sample_conversation_{persona_key}.md"
    assert out.exists()
    text = out.read_text()
    assert "Final Advice" in text
    assert "not financial advice" in text.lower()


def test_main_exits_without_api_key(monkeypatch, capsys):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    from src import main as main_mod

    with pytest.raises(SystemExit) as exc_info:
        main_mod.run("david")
    assert exc_info.value.code == 2
    err = capsys.readouterr().err
    assert "OPENROUTER_API_KEY" in err
