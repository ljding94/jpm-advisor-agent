"""ReviewerAgent — policy gate between Advisor and Client.

All Advisor-authored messages destined for the Client are intercepted here. The
reviewer applies a policy file (data/reviewer/policy.yaml) plus deterministic
guardrails, then emits one of three verdicts:

  - pass:   forwards the message verbatim to the Client (sender becomes REVIEWER).
  - revise: reviewer rewrites the content to fix minor violations, then forwards.
  - block:  bounces a REVIEW message back to the Advisor with reasons. After
            MAX_REVIEWER_RETRIES bounces, falls back to a safe canned message so
            the conversation never deadlocks.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

from src.agents.base import BaseAgent
from src.graph.state import AdvisorState, ConversationStatus, append_error
from src.guardrails.output_filter import (
    filter_text,
    find_banned_phrases,
    find_named_tickers,
)
from src.providers.llm import LLMProvider
from src.schemas import (
    STANDARD_DISCLAIMER,
    AgentMessage,
    AgentRole,
    MessageType,
)

MAX_REVIEWER_RETRIES = 1
DEFAULT_POLICY_PATH = Path("data/reviewer/policy.yaml")

Verdict = Literal["pass", "revise", "block"]

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", flags=re.DOTALL)
_JSON_OBJECT_RE = re.compile(r"\{.*\}", flags=re.DOTALL)


def _extract_json_object(text: str) -> str | None:
    fence = _JSON_FENCE_RE.search(text)
    if fence:
        return fence.group(1)
    obj = _JSON_OBJECT_RE.search(text)
    return obj.group(0) if obj else None


@dataclass(frozen=True)
class PolicyRule:
    id: str
    severity: Literal["block", "revise"]
    applies_to: tuple[str, ...]
    description: str


@dataclass
class ReviewVerdict:
    verdict: Verdict
    reasons: list[str]
    revised_content: str | None
    matched_rule_ids: list[str]

    def to_metadata(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "reasons": list(self.reasons),
            "matched_rule_ids": list(self.matched_rule_ids),
        }


REVIEWER_SYSTEM_PROMPT = """You are a compliance/quality reviewer for a financial-advisor agent.

You receive a candidate message authored by the advisor that is destined for the client.
You also receive a policy of rules. Decide whether the message complies.

Reply with JSON ONLY of shape:
  {"verdict": "pass" | "revise" | "block",
   "reasons": ["short reason", ...],
   "matched_rule_ids": ["rule.id", ...],
   "revised_content": "<rewritten message if verdict=revise, else empty string>"}

Verdict semantics:
- pass: no violations. Leave revised_content empty.
- revise: minor / fixable violations (missing disclaimer, named ticker, scope creep,
  hype tone). Provide a corrected revised_content that preserves the advisor's intent.
- block: serious / structural violations (guaranteed returns, unsuitable
  recommendation, soliciting sensitive PII). The advisor must redraft.
"""


class ReviewerAgent(BaseAgent):
    role = AgentRole.REVIEWER

    def __init__(
        self,
        llm: LLMProvider,
        name: str = "Reviewer",
        system_prompt: str | None = None,
        policy_path: Path | str | None = None,
        max_retries: int = MAX_REVIEWER_RETRIES,
    ) -> None:
        super().__init__(
            name=name,
            system_prompt=system_prompt or REVIEWER_SYSTEM_PROMPT,
            llm=llm,
        )
        self.policy_path = Path(policy_path) if policy_path else DEFAULT_POLICY_PATH
        self.max_retries = max_retries
        self.rules: tuple[PolicyRule, ...] = self._load_policy(self.policy_path)

    # -------- policy loading --------

    @staticmethod
    def _load_policy(path: Path) -> tuple[PolicyRule, ...]:
        if not path.exists():
            return tuple()
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        rules_raw = raw.get("rules", []) or []
        rules: list[PolicyRule] = []
        for r in rules_raw:
            rules.append(
                PolicyRule(
                    id=str(r["id"]),
                    severity=r.get("severity", "revise"),
                    applies_to=tuple(r.get("applies_to", [])),
                    description=str(r.get("description", "")).strip(),
                )
            )
        return tuple(rules)

    # -------- main loop entry --------

    def process(self, state: AdvisorState) -> AdvisorState:
        history: list[AgentMessage] = list(state.get("conversation_history", []))
        last = history[-1] if history else None
        if (
            last is None
            or last.recipient is not AgentRole.REVIEWER
            or last.sender is not AgentRole.ADVISOR
        ):
            return state

        verdict = self.review(last, state=state)
        return self._apply_verdict(state, original=last, verdict=verdict)

    # -------- review pipeline --------

    def review(self, message: AgentMessage, state: AdvisorState | None = None) -> ReviewVerdict:
        """Inspect an advisor-authored message and return a verdict."""
        msg_type = message.message_type.value
        applicable = [r for r in self.rules if not r.applies_to or msg_type in r.applies_to]

        # 1) Deterministic guardrails — fast, no LLM.
        det_reasons: list[str] = []
        det_matched: list[str] = []
        det_revised: str | None = None

        filter_result = filter_text(
            message.content,
            allow_disclaimer_missing=(message.message_type is not MessageType.ADVICE),
        )
        if filter_result.blocked:
            det_reasons.extend(filter_result.reasons)
            # Map filter reasons onto our rule IDs.
            if any("banned phrases" in r for r in filter_result.reasons):
                det_matched.append("claims.no_guarantees")
            if any("named tickers" in r for r in filter_result.reasons):
                det_matched.append("claims.no_specific_tickers")
            if any("missing standard disclaimer" in r for r in filter_result.reasons):
                det_matched.append("disclosure.standard_disclaimer")
            det_revised = self._deterministic_revise(message.content)

        # If deterministic checks alone flagged a `block`-severity rule, short-circuit.
        det_block = self._has_block_match(det_matched, applicable)

        # 2) LLM judgement against the full applicable policy.
        llm_verdict = self._llm_review(message, applicable, state=state)

        # 3) Combine. Block dominates revise dominates pass.
        combined_verdict: Verdict
        if det_block or llm_verdict.verdict == "block":
            combined_verdict = "block"
        elif det_reasons or llm_verdict.verdict == "revise":
            combined_verdict = "revise"
        else:
            combined_verdict = "pass"

        reasons = [*det_reasons, *llm_verdict.reasons]
        matched = list(dict.fromkeys([*det_matched, *llm_verdict.matched_rule_ids]))
        revised = llm_verdict.revised_content or det_revised
        if combined_verdict == "revise" and not revised:
            revised = self._deterministic_revise(message.content)
        return ReviewVerdict(
            verdict=combined_verdict,
            reasons=reasons,
            revised_content=revised,
            matched_rule_ids=matched,
        )

    @staticmethod
    def _has_block_match(matched_ids: list[str], rules: list[PolicyRule]) -> bool:
        idx = {r.id: r for r in rules}
        return any(idx.get(rid) and idx[rid].severity == "block" for rid in matched_ids)

    def _llm_review(
        self,
        message: AgentMessage,
        rules: list[PolicyRule],
        state: AdvisorState | None = None,
    ) -> ReviewVerdict:
        if not rules:
            return ReviewVerdict("pass", [], None, [])
        policy_text = "\n".join(
            f"- [{r.id}] (severity={r.severity}) {r.description}" for r in rules
        )
        prompt = (
            f"Message type: {message.message_type.value}\n"
            f"Advisor message to client:\n---\n{message.content}\n---\n\n"
            f"Applicable policy rules:\n{policy_text}\n\n"
            'Reply with JSON: {"verdict": "...", "reasons": [...], '
            '"matched_rule_ids": [...], "revised_content": "..."}'
        )
        try:
            raw = self._call_llm(
                prompt,
                max_tokens=600,
                response_format={"type": "json_object"},
            )
        except Exception as exc:
            if state is not None:
                append_error(state, source="reviewer.llm", detail=f"{type(exc).__name__}: {exc}")
            return ReviewVerdict("pass", [f"reviewer LLM error: {exc}"], None, [])

        return self._parse_verdict_json(raw, state=state)

    @staticmethod
    def _parse_verdict_json(raw: str, state: AdvisorState | None = None) -> ReviewVerdict:
        raw = (raw or "").strip()
        decoded: dict[str, Any] = {}
        try:
            if raw:
                decoded = json.loads(raw)
        except json.JSONDecodeError:
            extracted = _extract_json_object(raw)
            if extracted is not None:
                try:
                    decoded = json.loads(extracted)
                except json.JSONDecodeError as exc:
                    if state is not None:
                        append_error(
                            state, source="reviewer",
                            detail=f"verdict JSON malformed: {exc.msg}",
                        )
        verdict = decoded.get("verdict")
        if verdict not in ("pass", "revise", "block"):
            verdict = "pass"  # fail open — never deadlock on malformed reviewer output
            if state is not None and decoded:
                append_error(
                    state, source="reviewer",
                    detail=f"verdict missing/invalid; defaulting to pass: {raw[:120]!r}",
                )
        reasons = decoded.get("reasons") or []
        if not isinstance(reasons, list):
            reasons = [str(reasons)]
        matched = decoded.get("matched_rule_ids") or []
        if not isinstance(matched, list):
            matched = [str(matched)]
        revised = decoded.get("revised_content") or None
        if revised is not None and not isinstance(revised, str):
            revised = str(revised)
        if isinstance(revised, str) and not revised.strip():
            revised = None
        return ReviewVerdict(
            verdict=verdict,
            reasons=[str(r) for r in reasons],
            revised_content=revised,
            matched_rule_ids=[str(m) for m in matched],
        )

    @staticmethod
    def _deterministic_revise(text: str) -> str:
        """Apply mechanical fixes (strip tickers, banned phrases, append disclaimer)."""
        for tkr in find_named_tickers(text):
            text = text.replace(tkr, "[asset class]")
        for phrase in find_banned_phrases(text):
            text = re.sub(re.escape(phrase), "historical tendency", text, flags=re.IGNORECASE)
        if "not financial advice" not in text.lower():
            text = text.rstrip() + f"\n\nDisclaimer: {STANDARD_DISCLAIMER}"
        return text

    # -------- verdict application --------

    def _apply_verdict(
        self,
        state: AdvisorState,
        original: AgentMessage,
        verdict: ReviewVerdict,
    ) -> AdvisorState:
        history = list(state.get("conversation_history", []))
        new_state: AdvisorState = {**state}
        retries = int(state.get("reviewer_retries", 0))
        review_meta = {**verdict.to_metadata(), "original_message_type": original.message_type.value}

        if verdict.verdict == "pass":
            history.append(
                AgentMessage(
                    sender=AgentRole.REVIEWER,
                    recipient=AgentRole.CLIENT,
                    content=original.content,
                    message_type=original.message_type,
                    metadata={**original.metadata, "review": review_meta},
                )
            )
            new_state["reviewer_retries"] = 0

        elif verdict.verdict == "revise":
            content = verdict.revised_content or self._deterministic_revise(original.content)
            history.append(
                AgentMessage(
                    sender=AgentRole.REVIEWER,
                    recipient=AgentRole.CLIENT,
                    content=content,
                    message_type=original.message_type,
                    metadata={**original.metadata, "review": review_meta},
                )
            )
            new_state["reviewer_retries"] = 0

        else:  # block
            if retries < self.max_retries:
                history.append(
                    AgentMessage(
                        sender=AgentRole.REVIEWER,
                        recipient=AgentRole.ADVISOR,
                        content=self._render_block_feedback(verdict),
                        message_type=MessageType.REVIEW,
                        metadata={**review_meta, "blocked_original": original.content},
                    )
                )
                new_state["reviewer_retries"] = retries + 1
            else:
                # Retries exhausted — fall back to a safe, sanitized version so the
                # conversation can complete.
                fallback = (
                    verdict.revised_content
                    or self._deterministic_revise(original.content)
                )
                history.append(
                    AgentMessage(
                        sender=AgentRole.REVIEWER,
                        recipient=AgentRole.CLIENT,
                        content=fallback,
                        message_type=original.message_type,
                        metadata={
                            **original.metadata,
                            "review": {**review_meta, "fallback_after_retries": retries},
                        },
                    )
                )
                new_state["reviewer_retries"] = 0
                append_error(
                    new_state, source="reviewer",
                    detail=f"block retries exhausted; emitted sanitized fallback (rules={verdict.matched_rule_ids})",
                )

        new_state["conversation_history"] = history
        new_state["turn_count"] = state.get("turn_count", 0) + 1
        new_state["last_review"] = review_meta

        # Status bookkeeping: if we just forwarded ADVICE, move conversation to CONFIRM.
        if (
            verdict.verdict in ("pass", "revise")
            or new_state["reviewer_retries"] == 0  # fallback path
        ) and original.message_type is MessageType.ADVICE and history[-1].recipient is AgentRole.CLIENT:
            new_state["status"] = ConversationStatus.CONFIRM
        return new_state

    @staticmethod
    def _render_block_feedback(verdict: ReviewVerdict) -> str:
        reason_lines = "\n".join(f"- {r}" for r in verdict.reasons) or "- (no specific reason given)"
        rules_line = (
            f"Rules: {', '.join(verdict.matched_rule_ids)}"
            if verdict.matched_rule_ids
            else "Rules: (unmatched)"
        )
        return (
            "Compliance review BLOCKED this draft. Please redraft.\n\n"
            f"{rules_line}\n"
            "Reasons:\n"
            f"{reason_lines}"
        )


__all__ = [
    "MAX_REVIEWER_RETRIES",
    "PolicyRule",
    "REVIEWER_SYSTEM_PROMPT",
    "ReviewVerdict",
    "ReviewerAgent",
]
