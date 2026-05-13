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
    REVIEWER = "reviewer"
    SYSTEM = "system"


class MessageType(str, Enum):
    QUESTION = "question"
    ANSWER = "answer"
    TASK = "task"
    REPORT = "report"
    ADVICE = "advice"
    CONFIRMATION = "confirmation"
    REVIEW = "review"
    SYSTEM = "system"


# Allowed (sender, recipient) pairs.
# Hard constraints enforced at the schema layer:
#   - Analyst MUST NOT talk to Client directly (Advisor mediates).
#   - Advisor MUST NOT talk to Client directly (Reviewer mediates outbound traffic).
#   - All Advisor→Client traffic flows: ADVISOR → REVIEWER → CLIENT.
#   - Reviewer can bounce content back to Advisor with feedback.
ALLOWED_ROUTES: frozenset[tuple[AgentRole, AgentRole]] = frozenset({
    (AgentRole.CLIENT, AgentRole.ADVISOR),
    (AgentRole.ADVISOR, AgentRole.ANALYST),
    (AgentRole.ANALYST, AgentRole.ADVISOR),
    (AgentRole.ADVISOR, AgentRole.REVIEWER),
    (AgentRole.REVIEWER, AgentRole.ADVISOR),
    (AgentRole.REVIEWER, AgentRole.CLIENT),
    (AgentRole.SYSTEM, AgentRole.CLIENT),
    (AgentRole.SYSTEM, AgentRole.ADVISOR),
    (AgentRole.SYSTEM, AgentRole.ANALYST),
    (AgentRole.SYSTEM, AgentRole.REVIEWER),
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
                "Advisor and Analyst must not communicate with Client directly; "
                "all outbound traffic to Client flows through the Reviewer."
            )
        return self
