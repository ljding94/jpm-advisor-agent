"""Structured turn logger + markdown transcript exporter (Observer pattern)."""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from src.graph.state import AdvisorState
from src.schemas import AgentMessage

logger = logging.getLogger("jpm_advisor")


@dataclass
class TurnRecord:
    turn: int
    agent: str
    action: str
    duration_ms: float
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    extra: dict = field(default_factory=dict)


class TurnLogger:
    """Observer: records each agent turn as a structured JSON line.

    Tracks cumulative tokens and cost so the runtime can enforce the
    `MAX_TOTAL_COST_USD` limit.
    """

    def __init__(self) -> None:
        self.records: list[TurnRecord] = []
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.total_cost_usd: float = 0.0

    def record_turn(
        self,
        *,
        agent: str,
        action: str,
        started_at: float,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost_usd: float = 0.0,
        **extra,
    ) -> TurnRecord:
        rec = TurnRecord(
            turn=len(self.records) + 1,
            agent=agent,
            action=action,
            duration_ms=(time.monotonic() - started_at) * 1000.0,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            extra=extra,
        )
        self.records.append(rec)
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost_usd += cost_usd
        logger.info("turn", extra={"record": asdict(rec)})
        return rec

    def to_jsonl(self) -> str:
        return "\n".join(json.dumps(asdict(r), default=str) for r in self.records)

    def write_jsonl(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(self.to_jsonl() + ("\n" if self.records else ""), encoding="utf-8")
        return out

    def summary(self) -> dict:
        return {
            "turns": len(self.records),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
        }


def render_transcript(state: AdvisorState, *, persona_key: str) -> str:
    """Render a markdown transcript for `examples/sample_conversation_<persona>.md`."""
    profile = state["client_profile"]
    history: list[AgentMessage] = state.get("conversation_history", [])
    advice = state.get("draft_advice")
    status = state.get("status")
    termination = state.get("termination_reason")

    lines = [
        f"# Sample Conversation — {profile.name}",
        "",
        f"- **Persona**: `{persona_key}` ({profile.risk_tolerance})",
        f"- **Age**: {profile.age}",
        f"- **Time horizon**: {profile.time_horizon_years} years",
        f"- **Annual income**: ${profile.annual_income:,.0f}",
        f"- **Total assets**: ${profile.total_assets:,.0f}",
        f"- **Final status**: `{status.value if status else 'unknown'}`",
    ]
    if termination:
        lines.append(f"- **Termination reason**: {termination}")
    lines.append("")
    lines.append("## Transcript")
    lines.append("")

    for i, msg in enumerate(history, start=1):
        header = (
            f"### Turn {i} — {msg.sender.value} → {msg.recipient.value} "
            f"({msg.message_type.value})"
        )
        lines.append(header)
        lines.append("")
        lines.append(msg.content.strip())
        lines.append("")

    if advice is not None:
        lines.append("## Final Advice")
        lines.append("")
        lines.append("**Recommendations:**")
        for r in advice.recommendations:
            lines.append(f"- {r}")
        lines.append("")
        lines.append(f"**Rationale:** {advice.rationale}")
        lines.append("")
        if advice.sources:
            lines.append("**Sources:**")
            for s in advice.sources:
                src_line = f"- {s.title}"
                if s.url:
                    src_line += f" ({s.url})"
                lines.append(src_line)
            lines.append("")
        lines.append("**Disclaimers:**")
        for d in advice.disclaimers:
            lines.append(f"- {d}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def export_transcript(state: AdvisorState, *, persona_key: str, out_dir: str | Path = "examples") -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"sample_conversation_{persona_key}.md"
    out_path.write_text(render_transcript(state, persona_key=persona_key), encoding="utf-8")
    return out_path


__all__ = [
    "TurnLogger",
    "TurnRecord",
    "export_transcript",
    "render_transcript",
]
