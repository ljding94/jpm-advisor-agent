"""Embedding provider abstraction with OpenRouter and local fallback."""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Sequence


class EmbeddingProvider(ABC):
    """Abstract embedding provider."""

    dimension: int

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        ...


class OpenRouterEmbeddings(EmbeddingProvider):
    """Embeddings via OpenRouter (OpenAI-compatible)."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ) -> None:
        from openai import OpenAI

        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY not set")
        self.base_url = base_url or os.getenv(
            "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
        )
        self.model = model or os.getenv("EMBEDDING_MODEL", "openai/text-embedding-3-small")
        self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        # text-embedding-3-small returns 1536-dim vectors.
        self.dimension = 1536

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        resp = self._client.embeddings.create(model=self.model, input=list(texts))
        return [item.embedding for item in resp.data]


class LocalEmbeddings(EmbeddingProvider):
    """Sentence-transformers fallback. No network/API key required."""

    def __init__(self, model_name: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name or os.getenv(
            "LOCAL_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
        self._model = SentenceTransformer(self.model_name)
        dim = self._model.get_sentence_embedding_dimension()
        self.dimension = int(dim) if dim is not None else 384

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(list(texts), show_progress_bar=False)
        return [v.tolist() for v in vectors]


def get_embedding_provider() -> EmbeddingProvider:
    """Return OpenRouter embeddings if API key is set, else local."""
    if os.getenv("OPENROUTER_API_KEY"):
        try:
            return OpenRouterEmbeddings()
        except Exception:  # pragma: no cover - defensive fallback
            pass
    return LocalEmbeddings()
