"""Local Ollama LLM provider — http://localhost:11434 by default."""
from __future__ import annotations

import json
import os
from typing import Any

import urllib.error
import urllib.request

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.providers.llm import LLMProvider, Usage


class OllamaLLM(LLMProvider):
    """Talks to a local Ollama server. No API key required, no cost."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
    ) -> None:
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
        self.model = model or os.getenv("OLLAMA_MODEL", "llama3.1:8b")
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
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        }
        # Ollama supports `format: "json"` for structured output.
        if response_format and response_format.get("type") == "json_object":
            body["format"] = "json"

        url = f"{self.base_url}/api/chat"
        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Could not reach Ollama at {self.base_url}. Is the server running? ({exc})"
            ) from exc

        prompt_tokens = int(payload.get("prompt_eval_count") or 0)
        completion_tokens = int(payload.get("eval_count") or 0)
        self.last_usage = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=0.0,  # local inference is free
            model=self.model,
        )
        self._accumulate(self.last_usage)

        message = payload.get("message") or {}
        return message.get("content", "") or ""
