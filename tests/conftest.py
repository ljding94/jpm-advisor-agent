"""Shared pytest fixtures — FakeLLM, FakeEmbedder, persona loader."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

import pytest

from src.providers.embeddings import EmbeddingProvider
from src.providers.llm import Usage
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

    model = "fake/llm"

    def __init__(
        self,
        script: list[str] | None = None,
        marker_responses: dict[str, str] | None = None,
        default: str = "{}",
        cost_per_call_usd: float = 0.001,
    ) -> None:
        self.script = list(script or [])
        self.marker_responses = dict(marker_responses or {})
        self.default = default
        self.calls: list[dict[str, Any]] = []
        self.cost_per_call_usd = cost_per_call_usd
        self.last_usage = Usage(model=self.model)
        self.cumulative_prompt_tokens = 0
        self.cumulative_completion_tokens = 0
        self.cumulative_cost_usd = 0.0

    def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        self.calls.append({"messages": messages, "kwargs": kwargs})
        joined = "\n".join(m.get("content", "") for m in messages).lower()
        response = self.default
        for marker, resp in self.marker_responses.items():
            if marker.lower() in joined:
                response = resp
                break
        else:
            if self.script:
                response = self.script.pop(0)
        prompt_tokens = max(1, sum(len(m.get("content", "")) for m in messages) // 4)
        completion_tokens = max(1, len(response) // 4)
        self.last_usage = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=self.cost_per_call_usd,
            model=self.model,
        )
        self.cumulative_prompt_tokens += prompt_tokens
        self.cumulative_completion_tokens += completion_tokens
        self.cumulative_cost_usd += self.cost_per_call_usd
        return response


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
