"""ClientAgent — answers questions consistently with its persona and confirms advice."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.agents.base import BaseAgent
from src.graph.state import AdvisorState, ConversationStatus
from src.providers.llm import LLMProvider
from src.schemas import (
    STANDARD_DISCLAIMER,
    AgentMessage,
    AgentRole,
    ClientProfile,
    MessageType,
)

CLIENT_SYSTEM_PROMPT = """You are simulating a real human financial-advisory client.
You will receive your persona below. Stay in character at all times.

Behavior rules:
- Answer the advisor's questions truthfully according to your persona.
- Be conversational, not list-like. 1–4 sentences.
- If asked something not in your persona, give a plausible answer consistent with it.
- If the advisor presents final recommendations, decide whether to CONFIRM or REJECT
  by emitting a single line that begins with `[CONFIRM]` or `[REJECT]` followed by a
  short reason. Confirm if the advice is consistent with your stated risk tolerance,
  goals, and time horizon. Only reject for substantive reasons, not nitpicks.
"""


class ClientAgent(BaseAgent):
    role = AgentRole.CLIENT

    def __init__(
        self,
        profile: ClientProfile,
        llm: LLMProvider,
        name: str | None = None,
        system_prompt: str | None = None,
    ) -> None:
        super().__init__(
            name=name or profile.name,
            system_prompt=system_prompt or CLIENT_SYSTEM_PROMPT,
            llm=llm,
        )
        self.profile = profile

    # -------- factory helpers --------

    @classmethod
    def from_persona_file(
        cls, path: str | Path, llm: LLMProvider, **kwargs: Any
    ) -> "ClientAgent":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        profile = ClientProfile(**data)
        return cls(profile=profile, llm=llm, **kwargs)

    # -------- main loop entry --------

    def process(self, state: AdvisorState) -> AdvisorState:
        history: list[AgentMessage] = list(state.get("conversation_history", []))
        last = history[-1] if history else None

        if last is None or last.recipient != AgentRole.CLIENT:
            return state

        if last.message_type == MessageType.ADVICE:
            return self._respond_to_advice(state, last)

        return self._respond_to_question(state, last)

    # -------- responders --------

    def _respond_to_question(self, state: AdvisorState, last: AgentMessage) -> AdvisorState:
        prompt = (
            f"Persona:\n{self.profile.model_dump_json(indent=2)}\n\n"
            f"Advisor's question:\n{last.content}\n\n"
            "Answer in 1–4 sentences in first person."
        )
        response = self._call_llm(prompt, max_tokens=400)
        msg = AgentMessage(
            sender=AgentRole.CLIENT,
            recipient=AgentRole.ADVISOR,
            content=response.strip() or "(no response)",
            message_type=MessageType.ANSWER,
        )
        history = list(state.get("conversation_history", []))
        history.append(msg)
        new_state: AdvisorState = {**state, "conversation_history": history}
        new_state["turn_count"] = state.get("turn_count", 0) + 1
        return new_state

    def _respond_to_advice(self, state: AdvisorState, last: AgentMessage) -> AdvisorState:
        prompt = (
            f"Persona:\n{self.profile.model_dump_json(indent=2)}\n\n"
            f"Advisor's recommendations:\n{last.content}\n\n"
            "Reply on a single line starting with [CONFIRM] or [REJECT] followed by a brief reason."
        )
        response = self._call_llm(prompt, max_tokens=200).strip()
        if not response:
            response = "[CONFIRM] sounds reasonable."
        confirmed = response.upper().startswith("[CONFIRM]")
        msg = AgentMessage(
            sender=AgentRole.CLIENT,
            recipient=AgentRole.ADVISOR,
            content=response,
            message_type=MessageType.CONFIRMATION,
            metadata={"confirmed": confirmed},
        )
        history = list(state.get("conversation_history", []))
        history.append(msg)
        new_state: AdvisorState = {**state, "conversation_history": history}
        new_state["turn_count"] = state.get("turn_count", 0) + 1
        if confirmed:
            new_state["status"] = ConversationStatus.RESOLVED
        else:
            # Loop back into ANALYZE so the advisor revises.
            new_state["status"] = ConversationStatus.ANALYZE
        return new_state

    # Used by the runner to seed the conversation.
    def open_conversation(self, state: AdvisorState) -> AdvisorState:
        opener = (
            f"Hi, I'm {self.profile.name}. I'm {self.profile.age} and I'd like help "
            f"with my finances. {self.profile.notes or ''}".strip()
        )
        msg = AgentMessage(
            sender=AgentRole.CLIENT,
            recipient=AgentRole.ADVISOR,
            content=opener,
            message_type=MessageType.QUESTION,
        )
        history = list(state.get("conversation_history", []))
        history.append(msg)
        return {**state, "conversation_history": history}


# expose disclaimer constant for downstream prompts
__all__ = ["ClientAgent", "STANDARD_DISCLAIMER"]
