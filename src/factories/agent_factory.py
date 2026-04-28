"""Factory for constructing agents (Mediator/Strategy/Factory/Observer/State)."""
from __future__ import annotations

from typing import Any

from src.agents.base import BaseAgent
from src.providers.llm import LLMProvider, get_llm_provider
from src.schemas import AgentRole


class AgentFactory:
    """Builds agents from a role + config dict.

    Lazy-imports concrete agent classes so the factory module stays import-safe
    even before all agent files exist.
    """

    @staticmethod
    def create(
        role: AgentRole, config: dict[str, Any] | None = None, *, llm: LLMProvider | None = None
    ) -> BaseAgent:
        config = dict(config or {})
        llm = llm or config.pop("llm", None) or get_llm_provider()

        if role is AgentRole.CLIENT:
            from src.agents.client import ClientAgent

            return ClientAgent(llm=llm, **config)

        if role is AgentRole.ADVISOR:
            from src.agents.advisor import AdvisorAgent

            return AdvisorAgent(llm=llm, **config)

        if role is AgentRole.ANALYST:
            from src.agents.analyst import AnalystAgent

            return AnalystAgent(llm=llm, **config)

        raise ValueError(f"unsupported agent role: {role}")
