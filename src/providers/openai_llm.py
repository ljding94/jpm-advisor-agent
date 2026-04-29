"""Native OpenAI LLM provider (separate from OpenRouter)."""
from __future__ import annotations

import os
from typing import Any

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.providers.llm import LLMProvider, Usage, estimate_cost_usd


class OpenAILLM(LLMProvider):
    """OpenAI Chat Completions client (api.openai.com)."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
    ) -> None:
        from openai import OpenAI

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "OPENAI_API_KEY not set. Either set it in .env or pick a different LLM_PROVIDER."
            )
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        if self.base_url:
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        else:
            self._client = OpenAI(api_key=self.api_key)
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
