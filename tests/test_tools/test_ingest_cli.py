"""Smoke test for the ingest CLI."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from src.tools import ingest as ingest_mod


def test_ingest_cli_runs(tmp_path: Path, fake_embedder, monkeypatch, capsys):
    docs = tmp_path / "kb"
    docs.mkdir()
    (docs / "x.md").write_text("Diversification reduces unsystematic risk.")

    # Patch get_embedding_provider so we don't download a model.
    with patch("src.tools.knowledge_store.get_embedding_provider", return_value=fake_embedder):
        monkeypatch.setattr(
            "sys.argv",
            [
                "src.tools.ingest",
                "--source", str(docs),
                "--persist", str(tmp_path / "chroma"),
                "--reset",
            ],
        )
        rc = ingest_mod.main()
    assert rc == 0
    out = capsys.readouterr().out
    assert "Ingested" in out
    assert "Collection size" in out
