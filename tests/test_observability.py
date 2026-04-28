"""Observability/transcript tests."""
from __future__ import annotations

import json
import time
from pathlib import Path

from src.graph.state import ConversationStatus, initial_state
from src.observability.logger import TurnLogger, export_transcript, render_transcript
from src.schemas import (
    AdviceOutput,
    AgentMessage,
    AgentRole,
    MessageType,
    Source,
)


def test_turn_logger_records_in_order():
    log = TurnLogger()
    t0 = time.monotonic()
    log.record_turn(agent="advisor", action="ask_client", started_at=t0)
    log.record_turn(agent="client", action="answer", started_at=t0)
    assert [r.turn for r in log.records] == [1, 2]
    assert log.records[0].agent == "advisor"
    jsonl = log.to_jsonl()
    parsed = [json.loads(line) for line in jsonl.splitlines()]
    assert parsed[0]["action"] == "ask_client"


def test_render_transcript_includes_messages_and_advice(david_profile):
    state = initial_state(david_profile)
    state["status"] = ConversationStatus.RESOLVED
    state["conversation_history"].extend(
        [
            AgentMessage(
                sender=AgentRole.CLIENT,
                recipient=AgentRole.ADVISOR,
                content="Hi, I want help.",
                message_type=MessageType.QUESTION,
            ),
            AgentMessage(
                sender=AgentRole.ADVISOR,
                recipient=AgentRole.ANALYST,
                content="Research moderate allocations.",
                message_type=MessageType.TASK,
            ),
            AgentMessage(
                sender=AgentRole.ANALYST,
                recipient=AgentRole.ADVISOR,
                content="Findings…",
                message_type=MessageType.REPORT,
            ),
        ]
    )
    state["draft_advice"] = AdviceOutput(
        recommendations=["Diversify globally."],
        rationale="Risk reduction.",
        sources=[Source(title="kb::01_asset_allocation.md")],
    )
    md = render_transcript(state, persona_key="david")
    assert "David Patel" in md
    assert "moderate" in md
    assert "Final Advice" in md
    assert "Diversify globally" in md
    assert "not financial advice" in md.lower()


def test_export_transcript_writes_file(tmp_path: Path, david_profile):
    state = initial_state(david_profile)
    state["status"] = ConversationStatus.RESOLVED
    state["conversation_history"].append(
        AgentMessage(
            sender=AgentRole.CLIENT,
            recipient=AgentRole.ADVISOR,
            content="Hi.",
            message_type=MessageType.QUESTION,
        )
    )
    out = export_transcript(state, persona_key="david", out_dir=tmp_path)
    assert out.exists()
    assert out.name == "sample_conversation_david.md"
    assert "David Patel" in out.read_text()
