"""LLM provider abstraction. OpenRouter (OpenAI SDK) by default."""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    model: str = ""

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


# OpenRouter price table (USD per 1M tokens). Approximate, updated 2025-Q1.
# Used as a heuristic when the API doesn't return an explicit cost field.
# Format: {model_id: (input_price_per_1m, output_price_per_1m)}
MODEL_PRICES: dict[str, tuple[float, float]] = {
    "anthropic/claude-sonnet-4": (3.00, 15.00),
    "anthropic/claude-opus-4": (15.00, 75.00),
    "anthropic/claude-haiku-4": (0.80, 4.00),
    "openai/gpt-4o": (2.50, 10.00),
    "openai/gpt-4o-mini": (0.15, 0.60),
    "openai/o1-mini": (1.10, 4.40),
    "google/gemini-2.0-flash-001": (0.10, 0.40),
    "meta-llama/llama-3.1-70b-instruct": (0.40, 0.40),
}
DEFAULT_PRICE = (1.00, 3.00)  # cautious fallback for unknown models


def estimate_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    in_price, out_price = MODEL_PRICES.get(model, DEFAULT_PRICE)
    return (prompt_tokens / 1_000_000.0) * in_price + (completion_tokens / 1_000_000.0) * out_price


class LLMProvider(ABC):
    """Abstract LLM provider — chat-completion style.

    Implementations MUST update `last_usage` and call `_accumulate(usage)` after
    every successful call so the runtime can attribute tokens/cost to nodes.
    """

    model: str
    last_usage: Usage = field(default_factory=Usage)
    cumulative_prompt_tokens: int = 0
    cumulative_completion_tokens: int = 0
    cumulative_cost_usd: float = 0.0

    @abstractmethod
    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.2,
        response_format: dict[str, Any] | None = None,
        timeout: float = 60.0,
    ) -> str:
        ...

    def _accumulate(self, usage: Usage) -> None:
        self.cumulative_prompt_tokens += usage.prompt_tokens
        self.cumulative_completion_tokens += usage.completion_tokens
        self.cumulative_cost_usd += usage.cost_usd


class OpenRouterLLM(LLMProvider):
    """OpenAI-compatible client pointed at OpenRouter."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        default_max_tokens: int = 1024,
    ) -> None:
        from openai import OpenAI

        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY not set. Either set it in .env or use a FakeLLM."
            )
        self.base_url = base_url or os.getenv(
            "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
        )
        self.model = model or os.getenv("LLM_MODEL", "anthropic/claude-sonnet-4")
        self.default_max_tokens = default_max_tokens
        self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.last_usage = Usage(model=self.model)
        self.cumulative_prompt_tokens = 0
        self.cumulative_completion_tokens = 0
        self.cumulative_cost_usd = 0.0

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=4.0),
        retry=retry_if_exception_type(Exception),
    )
    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.2,
        response_format: dict[str, Any] | None = None,
        timeout: float = 60.0,
    ) -> str:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "timeout": timeout,
        }
        if response_format is not None:
            kwargs["response_format"] = response_format
        resp = self._client.chat.completions.create(**kwargs)
        usage = getattr(resp, "usage", None)
        if usage is not None:
            prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
            completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
            self.last_usage = Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cost_usd=estimate_cost_usd(self.model, prompt_tokens, completion_tokens),
                model=self.model,
            )
            self._accumulate(self.last_usage)
        choice = resp.choices[0]
        return choice.message.content or ""


def get_llm_provider() -> LLMProvider:
    return OpenRouterLLM()
