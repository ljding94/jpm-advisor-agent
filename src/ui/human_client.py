"""ClientAgent that reads responses from a thread-safe queue (the UI input box).

Lets a human drive the Client side of the conversation in the Streamlit app.
The graph thread calls `process(state)` as usual; when a response is needed
the agent blocks on `Queue.get()` until the UI puts a string on it.
"""
from __future__ import annotations

import queue
from typing import Any

from src.agents.client import ClientAgent
from src.graph.state import AdvisorState, ConversationStatus
from src.providers.llm import LLMProvider
from src.schemas import (
    AgentMessage,
    AgentRole,
    ClientProfile,
    MessageType,
)


class HumanClientAgent(ClientAgent):
    """Same routing/state behavior as ClientAgent, but text comes from the UI.

    Two queues:
    - `prompts_out` — the agent puts the advisor's question on this queue so the UI
      knows what to prompt the human with.
    - `responses_in` — the UI puts the human's typed response on this queue.

    Both have arbitrary capacity; the UI is responsible for draining `prompts_out`.
    """

    def __init__(
        self,
        profile: ClientProfile,
        llm: LLMProvider,
        *,
        prompts_out: queue.Queue[dict[str, Any]] | None = None,
        responses_in: queue.Queue[str] | None = None,
        timeout_s: float | None = None,
    ) -> None:
        # We still pass an LLM so BaseAgent's plumbing is happy, but never call it.
        super().__init__(profile=profile, llm=llm)
        self.prompts_out: queue.Queue[dict[str, Any]] = prompts_out or queue.Queue()
        self.responses_in: queue.Queue[str] = responses_in or queue.Queue()
        self.timeout_s = timeout_s

    # ---- override the two responder paths to read from the queue ----

    def _respond_to_question(self, state: AdvisorState, last: AgentMessage) -> AdvisorState:
        self.prompts_out.put({"kind": "question", "content": last.content})
        text = self._await_response()
        msg = AgentMessage(
            sender=AgentRole.CLIENT,
            recipient=AgentRole.ADVISOR,
            content=text or "(no response)",
            message_type=MessageType.ANSWER,
        )
        history = list(state.get("conversation_history", []))
        history.append(msg)
        new_state: AdvisorState = {**state, "conversation_history": history}
        new_state["turn_count"] = state.get("turn_count", 0) + 1
        return new_state

    def _respond_to_advice(self, state: AdvisorState, last: AgentMessage) -> AdvisorState:
        self.prompts_out.put({"kind": "advice", "content": last.content})
        text = (self._await_response() or "[CONFIRM] sounds reasonable.").strip()
        # Tolerate the human typing 'confirm' / 'yes' / 'reject' / 'no' instead of bracket form.
        upper = text.upper()
        if not (upper.startswith("[CONFIRM]") or upper.startswith("[REJECT]")):
            if upper.startswith(("YES", "CONFIRM", "OK")):
                text = f"[CONFIRM] {text}"
            elif upper.startswith(("NO", "REJECT")):
                text = f"[REJECT] {text}"
            else:
                text = f"[CONFIRM] {text}"
        confirmed = text.upper().startswith("[CONFIRM]")
        msg = AgentMessage(
            sender=AgentRole.CLIENT,
            recipient=AgentRole.ADVISOR,
            content=text,
            message_type=MessageType.CONFIRMATION,
            metadata={"confirmed": confirmed},
        )
        history = list(state.get("conversation_history", []))
        history.append(msg)
        new_state: AdvisorState = {**state, "conversation_history": history}
        new_state["turn_count"] = state.get("turn_count", 0) + 1
        new_state["status"] = (
            ConversationStatus.RESOLVED if confirmed else ConversationStatus.ANALYZE
        )
        return new_state

    # ---- helpers ----

    def _await_response(self) -> str:
        return self.responses_in.get(timeout=self.timeout_s)

    def open_conversation(self, state: AdvisorState) -> AdvisorState:
        """For the UI we still seed an opener so the graph has something to react to.

        The user can then steer the conversation via subsequent question turns.
        """
        return super().open_conversation(state)


__all__ = ["HumanClientAgent"]
