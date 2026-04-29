"""ChromaDB-backed knowledge store for finance markdown docs."""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from src.providers.embeddings import EmbeddingProvider, get_embedding_provider

# Silence chromadb's posthog telemetry warnings before any chromadb import.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("CHROMA_TELEMETRY_DISABLED", "true")
os.environ.setdefault("POSTHOG_DISABLED", "true")
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)
logging.getLogger("posthog").setLevel(logging.CRITICAL)


@dataclass
class RetrievedChunk:
    text: str
    source: str
    score: float


def _chunk_tokens(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Word-level chunking — token-approximate, deterministic, dependency-free."""
    words = re.findall(r"\S+|\n", text)
    if not words:
        return []
    chunks: list[str] = []
    start = 0
    step = max(1, chunk_size - overlap)
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(w for w in words[start:end] if w != "\n")
        chunk = re.sub(r"\s+", " ", chunk).strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(words):
            break
        start += step
    return chunks


class KnowledgeStore:
    """Chroma-backed similarity store with pluggable embeddings."""

    DEFAULT_COLLECTION = "knowledge_base"

    def __init__(
        self,
        persist_path: str | os.PathLike[str] | None = None,
        embedder: EmbeddingProvider | None = None,
        collection_name: str | None = None,
    ) -> None:
        import chromadb
        from chromadb.config import Settings

        self.persist_path = str(persist_path or os.getenv("CHROMA_PATH", "data/chroma"))
        Path(self.persist_path).mkdir(parents=True, exist_ok=True)
        self.embedder = embedder or get_embedding_provider()
        self.collection_name = collection_name or self.DEFAULT_COLLECTION
        self._client = chromadb.PersistentClient(
            path=self.persist_path,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(self.collection_name)

    # ----------- ingestion -----------

    def ingest_directory(
        self,
        directory: str | os.PathLike[str],
        glob: str = "*.md",
        chunk_size: int = 500,
        overlap: int = 50,
    ) -> int:
        """Ingest every file matching `glob` in `directory`. Returns chunks added."""
        path = Path(directory)
        if not path.exists():
            raise FileNotFoundError(f"knowledge directory not found: {path}")
        files = sorted(path.glob(glob))
        added = 0
        for fp in files:
            text = fp.read_text(encoding="utf-8")
            added += self._ingest_text(
                text=text, source=fp.name, chunk_size=chunk_size, overlap=overlap
            )
        return added

    def _ingest_text(
        self, text: str, source: str, chunk_size: int = 500, overlap: int = 50
    ) -> int:
        chunks = _chunk_tokens(text, chunk_size=chunk_size, overlap=overlap)
        if not chunks:
            return 0
        embeddings = self.embedder.embed(chunks)
        ids = [f"{source}:{i}" for i in range(len(chunks))]
        metadatas = [{"source": source, "chunk_index": i} for i in range(len(chunks))]
        self._collection.upsert(
            ids=ids,
            documents=chunks,
            embeddings=cast(Any, embeddings),
            metadatas=cast(Any, metadatas),
        )
        return len(chunks)

    # ----------- retrieval -----------

    def similarity_search(self, query: str, k: int = 4) -> list[RetrievedChunk]:
        if not query.strip():
            return []
        if self._collection.count() == 0:
            return []
        embedding = self.embedder.embed([query])[0]
        result = self._collection.query(
            query_embeddings=cast(Any, [embedding]),
            n_results=k,
            include=cast(Any, ["documents", "metadatas", "distances"]),
        )
        out: list[RetrievedChunk] = []
        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        dists = (result.get("distances") or [[]])[0]
        for doc, meta, dist in zip(docs, metas, dists):
            score = 1.0 / (1.0 + float(dist))
            out.append(
                RetrievedChunk(
                    text=doc,
                    source=str((meta or {}).get("source", "unknown")),
                    score=score,
                )
            )
        return out

    def count(self) -> int:
        return self._collection.count()

    def reset(self) -> None:
        try:
            self._client.delete_collection(self.collection_name)
        except Exception:
            pass
        self._collection = self._client.get_or_create_collection(self.collection_name)


def chunk_tokens_for_test(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Public helper purely for unit-testing the chunker."""
    return _chunk_tokens(text, chunk_size=chunk_size, overlap=overlap)
