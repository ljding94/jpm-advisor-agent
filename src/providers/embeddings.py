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


class OpenAIEmbeddings(EmbeddingProvider):
    """Native OpenAI embeddings (api.openai.com)."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
    ) -> None:
        from openai import OpenAI

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        self.model = model or os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        if self.base_url:
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        else:
            self._client = OpenAI(api_key=self.api_key)
        # text-embedding-3-small = 1536 dims; -large = 3072. Adjust if needed.
        self.dimension = 3072 if "large" in self.model else 1536

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
    """Dispatch on `EMBEDDING_PROVIDER` env var.

    Values: "auto" (default — OpenRouter if key present, else local),
    "openrouter", "openai", "local".
    """
    name = (os.getenv("EMBEDDING_PROVIDER") or "auto").lower().strip()
    if name == "openrouter":
        return OpenRouterEmbeddings()
    if name == "openai":
        return OpenAIEmbeddings()
    if name == "local":
        return LocalEmbeddings()
    if name == "auto":
        if os.getenv("OPENROUTER_API_KEY"):
            try:
                return OpenRouterEmbeddings()
            except Exception:  # pragma: no cover - defensive fallback
                pass
        return LocalEmbeddings()
    raise ValueError(
        f"Unknown EMBEDDING_PROVIDER={name!r}. "
        "Expected one of: auto, openrouter, openai, local."
    )
