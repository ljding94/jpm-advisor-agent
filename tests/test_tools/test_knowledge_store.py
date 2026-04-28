"""Knowledge store ingestion + retrieval (with fake embedder)."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.tools.knowledge_store import KnowledgeStore, chunk_tokens_for_test


def test_chunker_handles_short_text():
    chunks = chunk_tokens_for_test("just a few words", chunk_size=500, overlap=50)
    assert chunks == ["just a few words"]


def test_chunker_overlaps():
    text = " ".join(f"w{i}" for i in range(1200))
    chunks = chunk_tokens_for_test(text, chunk_size=500, overlap=50)
    # 1200 words with 500/450 step → at least 3 chunks.
    assert len(chunks) >= 3
    # Each chunk should be word-aligned.
    for c in chunks:
        assert all(piece.startswith("w") for piece in c.split())


def test_chunker_empty():
    assert chunk_tokens_for_test("", chunk_size=500) == []


def test_ingest_then_search(tmp_path: Path, fake_embedder):
    """Acceptance: similarity_search returns chunks with non-empty source metadata."""
    docs = tmp_path / "kb"
    docs.mkdir()
    (docs / "alpha.md").write_text(
        "Asset allocation divides the portfolio across stocks bonds and cash. "
        "It is the most important investment decision an investor will ever make."
    )
    (docs / "beta.md").write_text(
        "An emergency fund holds three to six months of essential expenses in a "
        "high yield savings account so that forced selling never happens."
    )

    store = KnowledgeStore(persist_path=tmp_path / "chroma", embedder=fake_embedder)
    store.reset()
    n = store.ingest_directory(docs)
    assert n >= 2
    assert store.count() >= 2

    results = store.similarity_search("how should I allocate my portfolio", k=2)
    assert len(results) <= 2
    assert all(r.source for r in results)
    assert all(0.0 <= r.score <= 1.0 for r in results)


def test_search_on_empty_collection(tmp_path: Path, fake_embedder):
    store = KnowledgeStore(persist_path=tmp_path / "chroma", embedder=fake_embedder)
    store.reset()
    assert store.similarity_search("anything", k=4) == []


def test_search_blank_query(tmp_path: Path, fake_embedder):
    store = KnowledgeStore(persist_path=tmp_path / "chroma", embedder=fake_embedder)
    store.reset()
    assert store.similarity_search("   ", k=4) == []


def test_ingest_missing_directory(tmp_path: Path, fake_embedder):
    store = KnowledgeStore(persist_path=tmp_path / "chroma", embedder=fake_embedder)
    with pytest.raises(FileNotFoundError):
        store.ingest_directory(tmp_path / "does_not_exist")
