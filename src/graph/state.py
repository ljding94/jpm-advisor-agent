"""Graph state definitions."""
from __future__ import annotations

from enum import Enum
from typing import Optional, TypedDict

from src.schemas import AdviceOutput, AgentMessage, AnalystReport, ClientProfile


class ConversationStatus(str, Enum):
    GATHER_PROFILE = "gather_profile"
    ANALYZE = "analyze"
    ADVISE = "advise"
    CONFIRM = "confirm"
    RESOLVED = "resolved"
    TERMINATED = "terminated"


class AdvisorState(TypedDict, total=False):
    client_profile: ClientProfile
    conversation_history: list[AgentMessage]
    current_advisor_query: Optional[str]
    analyst_findings: Optional[AnalystReport]
    draft_advice: Optional[AdviceOutput]
    status: ConversationStatus
    turn_count: int
    errors: list[str]
    termination_reason: Optional[str]


def initial_state(profile: ClientProfile) -> AdvisorState:
    return AdvisorState(
        client_profile=profile,
        conversation_history=[],
        current_advisor_query=None,
        analyst_findings=None,
        draft_advice=None,
        status=ConversationStatus.GATHER_PROFILE,
        turn_count=0,
        errors=[],
        termination_reason=None,
    )
