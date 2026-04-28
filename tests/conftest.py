"""Shared pytest fixtures — FakeLLM, FakeEmbedder, persona loader."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

import pytest

from src.providers.embeddings import EmbeddingProvider
from src.schemas import ClientProfile


# --------------------- Fake embedder ---------------------

class FakeEmbedder(EmbeddingProvider):
    """Deterministic hashing embedder. No network, no model download."""

    def __init__(self, dim: int = 64) -> None:
        self.dimension = dim

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for t in texts:
            digest = hashlib.sha256(t.encode("utf-8")).digest()
            words = [t.lower()]
            for w in t.lower().split():
                words.append(w)
            vec = [0.0] * self.dimension
            for w in words:
                h = hashlib.sha256(w.encode("utf-8")).digest()
                for i in range(self.dimension):
                    vec[i] += (h[i % len(h)] - 128) / 128.0
            for i in range(self.dimension):
                vec[i] += (digest[i % len(digest)] - 128) / 256.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


# --------------------- Fake LLM ---------------------

class FakeLLM:
    """Scripted LLM that returns canned responses keyed by call index or marker.

    Use `script` for ordered responses, `marker_responses` for prompt-keyword
    matching, and `default` as a final fallback.
    """

    def __init__(
        self,
        script: list[str] | None = None,
        marker_responses: dict[str, str] | None = None,
        default: str = "{}",
    ) -> None:
        self.script = list(script or [])
        self.marker_responses = dict(marker_responses or {})
        self.default = default
        self.calls: list[dict[str, Any]] = []

    def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        self.calls.append({"messages": messages, "kwargs": kwargs})
        joined = "\n".join(m.get("content", "") for m in messages).lower()
        for marker, response in self.marker_responses.items():
            if marker.lower() in joined:
                return response
        if self.script:
            return self.script.pop(0)
        return self.default


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


# --------------------- Personas ---------------------

PERSONA_DIR = Path("data/personas")
PERSONA_FILES = {
    "margaret": "margaret_conservative.json",
    "david": "david_moderate.json",
    "priya": "priya_aggressive.json",
}


@pytest.fixture
def all_personas() -> dict[str, ClientProfile]:
    out: dict[str, ClientProfile] = {}
    for key, fname in PERSONA_FILES.items():
        data = json.loads((PERSONA_DIR / fname).read_text())
        out[key] = ClientProfile(**data)
    return out


@pytest.fixture
def david_profile(all_personas: dict[str, ClientProfile]) -> ClientProfile:
    return all_personas["david"]
