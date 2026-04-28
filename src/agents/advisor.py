"""AdvisorAgent — Mediator. Decides next action JSON and synthesizes advice."""
from __future__ import annotations

import json
from typing import Any, Literal

from src.agents.base import BaseAgent
from src.graph.state import AdvisorState, ConversationStatus
from src.guardrails.output_filter import enforce_advice_disclaimer, filter_text
from src.guardrails.pii import redact_text
from src.providers.llm import LLMProvider
from src.schemas import (
    AdviceOutput,
    AgentMessage,
    AgentRole,
    AnalystReport,
    ClientProfile,
    MessageType,
)
from src.strategies.risk_profile import RiskStrategy, get_strategy

ADVISOR_SYSTEM_PROMPT = """You are a financial advisor. You mediate between a Client and an Analyst.

Hard rules:
- The Analyst NEVER speaks to the Client directly; you are the only path between them.
- Decide one of four next actions per turn: ask_client, dispatch_analyst, draft_advice, finalize.
- Output JSON ONLY when asked for a decision: {"next_action": ..., "target": "client|analyst|none", "message": "..."}.
- When drafting advice for the client: focus on principles, not individual tickers.
- Always include the standard disclaimer that this is not financial advice.
"""

NextAction = Literal["ask_client", "dispatch_analyst", "draft_advice", "finalize"]


class AdvisorAgent(BaseAgent):
    role = AgentRole.ADVISOR

    def __init__(
        self,
        llm: LLMProvider,
        name: str = "Advisor",
        system_prompt: str | None = None,
    ) -> None:
        super().__init__(
            name=name,
            system_prompt=system_prompt or ADVISOR_SYSTEM_PROMPT,
            llm=llm,
        )

    # -------- main loop entry --------

    def process(self, state: AdvisorState) -> AdvisorState:
        history: list[AgentMessage] = list(state.get("conversation_history", []))
        last = history[-1] if history else None
        if last is None or last.recipient is not AgentRole.ADVISOR:
            return state

        if last.message_type is MessageType.REPORT:
            return self._after_analyst_report(state, last)

        # Otherwise (client question or rejection), decide next action.
        decision = self._decide(state)
        return self._apply_decision(state, decision)

    # -------- decision step --------

    def _decide(self, state: AdvisorState) -> dict[str, Any]:
        """Ask the LLM what to do next based on the state."""
        profile = state["client_profile"]
        history_text = self._format_history(state)
        prompt = (
            "Decide the advisor's next action.\n\n"
            f"Client profile:\n{profile.model_dump_json(indent=2)}\n\n"
            f"Conversation so far:\n{history_text}\n\n"
            "Choose ONE of:\n"
            "  - ask_client: ask one focused follow-up question.\n"
            "  - dispatch_analyst: send a research task to the analyst.\n"
            "  - draft_advice: synthesize recommendations directly.\n"
            "  - finalize: present already-drafted advice for confirmation.\n\n"
            "Reply ONLY with JSON of shape "
            '{"next_action": "...", "target": "client|analyst|none", "message": "..."}.'
        )
        raw = self._call_llm(
            redact_text(prompt),
            max_tokens=500,
            response_format={"type": "json_object"},
        )
        return self._parse_decision_json(raw, state=state)

    @staticmethod
    def _parse_decision_json(raw: str, state: AdvisorState) -> dict[str, Any]:
        raw = raw.strip()
        try:
            decoded = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            decoded = {}
        action = decoded.get("next_action")
        if action not in {"ask_client", "dispatch_analyst", "draft_advice", "finalize"}:
            # Default fallback: progress through the state machine.
            current = state.get("status", ConversationStatus.GATHER_PROFILE)
            if current == ConversationStatus.GATHER_PROFILE:
                action = "ask_client"
            elif current == ConversationStatus.ANALYZE:
                action = "dispatch_analyst"
            else:
                action = "draft_advice"
        return {
            "next_action": action,
            "target": decoded.get("target", "client" if action == "ask_client" else "analyst"),
            "message": decoded.get("message", ""),
        }

    def _apply_decision(
        self, state: AdvisorState, decision: dict[str, Any]
    ) -> AdvisorState:
        action = decision["next_action"]
        history = list(state.get("conversation_history", []))
        new_state: AdvisorState = {**state}

        if action == "ask_client":
            content = decision["message"] or "Could you tell me more about your goals?"
            history.append(
                AgentMessage(
                    sender=AgentRole.ADVISOR,
                    recipient=AgentRole.CLIENT,
                    content=content,
                    message_type=MessageType.QUESTION,
                )
            )
            new_state["status"] = ConversationStatus.GATHER_PROFILE

        elif action == "dispatch_analyst":
            query = decision["message"] or self._default_research_task(state)
            history.append(
                AgentMessage(
                    sender=AgentRole.ADVISOR,
                    recipient=AgentRole.ANALYST,
                    content=query,
                    message_type=MessageType.TASK,
                )
            )
            new_state["current_advisor_query"] = query
            new_state["status"] = ConversationStatus.ANALYZE

        elif action == "draft_advice":
            advice = self.synthesize_advice(state)
            new_state["draft_advice"] = advice
            history.append(
                AgentMessage(
                    sender=AgentRole.ADVISOR,
                    recipient=AgentRole.CLIENT,
                    content=self._render_advice(advice),
                    message_type=MessageType.ADVICE,
                )
            )
            new_state["status"] = ConversationStatus.CONFIRM

        elif action == "finalize":
            advice = state.get("draft_advice") or self.synthesize_advice(state)
            new_state["draft_advice"] = advice
            history.append(
                AgentMessage(
                    sender=AgentRole.ADVISOR,
                    recipient=AgentRole.CLIENT,
                    content=self._render_advice(advice),
                    message_type=MessageType.ADVICE,
                )
            )
            new_state["status"] = ConversationStatus.CONFIRM

        new_state["conversation_history"] = history
        new_state["turn_count"] = state.get("turn_count", 0) + 1
        return new_state

    # -------- after analyst report → draft advice next turn --------

    def _after_analyst_report(self, state: AdvisorState, last: AgentMessage) -> AdvisorState:
        """When the Advisor receives a REPORT, transition to ADVISE."""
        new_state: AdvisorState = {**state}
        new_state["status"] = ConversationStatus.ADVISE
        # We don't add a new message here; the next turn will draft advice.
        return new_state

    # -------- advice synthesis --------

    def synthesize_advice(self, state: AdvisorState) -> AdviceOutput:
        profile: ClientProfile = state["client_profile"]
        report: AnalystReport | None = state.get("analyst_findings")
        strategy = get_strategy(profile.risk_tolerance)
        recommendations = self._build_recommendations(profile, strategy, report)
        rationale = self._build_rationale(profile, strategy, report)
        sources = list(report.sources) if report else []
        advice = AdviceOutput(
            recommendations=recommendations,
            rationale=rationale,
            sources=sources,
            disclaimers=[],  # auto-appended by the schema
        )
        # Defense-in-depth: enforce again at the boundary.
        return enforce_advice_disclaimer(advice)

    def _build_recommendations(
        self,
        profile: ClientProfile,
        strategy: RiskStrategy,
        report: AnalystReport | None,
    ) -> list[str]:
        alloc = strategy.recommended_allocation(profile)
        allocation_line = (
            f"Target allocation: {alloc['equities']:.0f}% equities / "
            f"{alloc['bonds']:.0f}% bonds / {alloc['cash']:.0f}% cash."
        )
        bullets = [allocation_line, *strategy.headline_advice(profile)]
        if report:
            bullets.append(
                f"Research note (confidence {report.confidence:.2f}): {report.findings}"
            )

        # Strip anything the output filter would reject (named tickers, banned phrases).
        cleaned: list[str] = []
        for b in bullets:
            res = filter_text(b, allow_disclaimer_missing=True)
            cleaned.append(b if not res.blocked else self._scrub_bullet(b))
        return cleaned

    @staticmethod
    def _scrub_bullet(text: str) -> str:
        # Last-ditch scrub of forbidden tokens.
        from src.guardrails.output_filter import (
            find_banned_phrases,
            find_named_tickers,
        )

        for tkr in find_named_tickers(text):
            text = text.replace(tkr, "[TICKER]")
        for ph in find_banned_phrases(text):
            text = text.replace(ph, "[BANNED]")
        return text

    def _build_rationale(
        self,
        profile: ClientProfile,
        strategy: RiskStrategy,
        report: AnalystReport | None,
    ) -> str:
        parts = [
            f"{profile.name} is {profile.age} with a {strategy.name} risk tolerance "
            f"and a {profile.time_horizon_years}-year horizon. Goals: "
            f"{'; '.join(profile.goals) or 'not stated'}.",
            "The recommended allocation reflects standard rules of thumb for this "
            f"risk profile, balancing the stated goals against drawdown tolerance.",
        ]
        if report:
            parts.append(f"Analyst input was used to inform the recommendation: {report.findings}")
        return " ".join(parts)

    # -------- helpers --------

    @staticmethod
    def _format_history(state: AdvisorState) -> str:
        lines = []
        for m in state.get("conversation_history", []):
            lines.append(f"[{m.sender.value} → {m.recipient.value}] ({m.message_type.value}) {m.content}")
        return "\n".join(lines) or "(empty)"

    @staticmethod
    def _default_research_task(state: AdvisorState) -> str:
        profile: ClientProfile = state["client_profile"]
        return (
            f"Research the appropriate asset allocation for a {profile.age}-year-old "
            f"with {profile.risk_tolerance} risk tolerance, "
            f"a {profile.time_horizon_years}-year time horizon, and goals: "
            f"{'; '.join(profile.goals) or 'general planning'}."
        )

    @staticmethod
    def _render_advice(advice: AdviceOutput) -> str:
        recs = "\n".join(f"- {r}" for r in advice.recommendations)
        sources = "\n".join(
            f"- {s.title}" + (f" ({s.url})" if s.url else "") for s in advice.sources
        ) or "- (no external sources cited)"
        disclaimers = "\n".join(f"- {d}" for d in advice.disclaimers)
        return (
            "Recommendations:\n"
            f"{recs}\n\n"
            f"Rationale:\n{advice.rationale}\n\n"
            f"Sources:\n{sources}\n\n"
            f"Disclaimers:\n{disclaimers}"
        )

    # -------- runner helper --------

    def open_response(self, state: AdvisorState) -> AdvisorState:
        """Used by the runner once the client has spoken first."""
        decision = self._decide(state)
        return self._apply_decision(state, decision)


__all__ = ["AdvisorAgent", "ADVISOR_SYSTEM_PROMPT"]
