"""Agent messaging schemas with hard routing constraint."""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator


class AgentRole(str, Enum):
    CLIENT = "client"
    ADVISOR = "advisor"
    ANALYST = "analyst"
    SYSTEM = "system"


class MessageType(str, Enum):
    QUESTION = "question"
    ANSWER = "answer"
    TASK = "task"
    REPORT = "report"
    ADVICE = "advice"
    CONFIRMATION = "confirmation"
    SYSTEM = "system"


# Allowed (sender, recipient) pairs. Analyst MUST NOT talk to Client directly.
ALLOWED_ROUTES: frozenset[tuple[AgentRole, AgentRole]] = frozenset({
    (AgentRole.CLIENT, AgentRole.ADVISOR),
    (AgentRole.ADVISOR, AgentRole.CLIENT),
    (AgentRole.ADVISOR, AgentRole.ANALYST),
    (AgentRole.ANALYST, AgentRole.ADVISOR),
    (AgentRole.SYSTEM, AgentRole.CLIENT),
    (AgentRole.SYSTEM, AgentRole.ADVISOR),
    (AgentRole.SYSTEM, AgentRole.ANALYST),
})


class AgentMessage(BaseModel):
    sender: AgentRole
    recipient: AgentRole
    content: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    message_type: MessageType = MessageType.SYSTEM
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_route(self) -> "AgentMessage":
        if (self.sender, self.recipient) not in ALLOWED_ROUTES:
            raise ValueError(
                f"Illegal route {self.sender.value} -> {self.recipient.value}. "
                "Analyst must not communicate with Client directly."
            )
        return self
