"""Abstract base agent."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from src.providers.llm import LLMProvider
from src.schemas import AgentRole

if TYPE_CHECKING:
    from src.graph.state import AdvisorState


class BaseAgent(ABC):
    """All agents implement `process(state) -> state`."""

    role: AgentRole

    def __init__(
        self,
        name: str,
        system_prompt: str,
        llm: LLMProvider,
        tools: dict[str, Any] | None = None,
    ) -> None:
        self.name = name
        self.system_prompt = system_prompt
        self.llm = llm
        self.tools = tools or {}

    @abstractmethod
    def process(self, state: "AdvisorState") -> "AdvisorState":
        """Run one turn for this agent and return the updated state dict."""

    def _call_llm(
        self,
        user_prompt: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.2,
        response_format: dict[str, Any] | None = None,
        extra_messages: list[dict[str, str]] | None = None,
    ) -> str:
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self.system_prompt}
        ]
        if extra_messages:
            messages.extend(extra_messages)
        messages.append({"role": "user", "content": user_prompt})
        return self.llm.complete(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            response_format=response_format,
        )
