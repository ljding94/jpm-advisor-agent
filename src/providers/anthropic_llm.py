"""Native Anthropic LLM provider."""
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


def _split_system_and_messages(
    messages: list[dict[str, str]],
) -> tuple[str | None, list[dict[str, str]]]:
    """Anthropic takes `system` as a top-level arg, not a message role.

    Concatenates any system messages into a single system string and returns
    the remaining messages with their roles preserved.
    """
    system_parts: list[str] = []
    rest: list[dict[str, str]] = []
    for m in messages:
        role = m.get("role", "user")
        if role == "system":
            content = m.get("content", "")
            if content:
                system_parts.append(content)
        else:
            rest.append({"role": role, "content": m.get("content", "")})
    system = "\n\n".join(system_parts) if system_parts else None
    return system, rest


class AnthropicLLM(LLMProvider):
    """Native Anthropic Messages API client."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        try:
            import anthropic  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "The `anthropic` package is required for AnthropicLLM. "
                "Install with `pip install anthropic`."
            ) from exc
        from anthropic import Anthropic

        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set. Either set it in .env or pick a different LLM_PROVIDER."
            )
        self.model = model or os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
        self._client = Anthropic(api_key=self.api_key)
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
        system, msgs = _split_system_and_messages(messages)
        # Anthropic doesn't have an explicit `response_format=json_object` flag;
        # if the caller wants JSON, append a system-level nudge.
        if response_format and response_format.get("type") == "json_object":
            json_nudge = "Respond with a single valid JSON object and nothing else."
            system = f"{system}\n\n{json_nudge}" if system else json_nudge

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": msgs,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "timeout": timeout,
        }
        if system is not None:
            kwargs["system"] = system

        resp = self._client.messages.create(**kwargs)
        usage = getattr(resp, "usage", None)
        if usage is not None:
            prompt_tokens = int(getattr(usage, "input_tokens", 0) or 0)
            completion_tokens = int(getattr(usage, "output_tokens", 0) or 0)
            self.last_usage = Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cost_usd=estimate_cost_usd(self.model, prompt_tokens, completion_tokens),
                model=self.model,
            )
            self._accumulate(self.last_usage)

        # Anthropic returns content as a list of blocks; concatenate text blocks.
        parts: list[str] = []
        for block in getattr(resp, "content", []) or []:
            text = getattr(block, "text", None)
            if text:
                parts.append(text)
        return "".join(parts)
