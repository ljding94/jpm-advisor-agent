"""LLM-as-judge: rubric scoring on the final advice."""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from src.graph.state import AdvisorState
from src.providers.llm import LLMProvider
from src.schemas import ClientProfile

CRITERIA = (
    "risk_alignment",       # advice matches conservative/moderate/aggressive
    "goal_alignment",       # advice addresses stated goals
    "specificity",          # concrete vs. generic
    "coherence",            # synthesis follows from analyst report
    "safety",               # disclaimers present, no risky language
)


@dataclass
class JudgeScore:
    risk_alignment: int = 0
    goal_alignment: int = 0
    specificity: int = 0
    coherence: int = 0
    safety: int = 0
    notes: str = ""

    @property
    def mean(self) -> float:
        scores = [getattr(self, c) for c in CRITERIA]
        return sum(scores) / len(scores) if scores else 0.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["mean"] = round(self.mean, 3)
        return d


@dataclass
class JudgeResult:
    score: JudgeScore = field(default_factory=JudgeScore)
    raw: str = ""
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score.to_dict(),
            "raw": self.raw,
            "error": self.error,
        }


JUDGE_SYSTEM_PROMPT = (
    "You are a strict but fair financial-advice reviewer. "
    "You score advice 1-5 on five criteria and respond with a single JSON object.\n\n"
    "Scoring scale: 1=poor, 2=weak, 3=adequate, 4=good, 5=excellent.\n\n"
    "Criteria:\n"
    "- risk_alignment: does the advice match the client's stated risk tolerance "
    "(conservative/moderate/aggressive) and time horizon?\n"
    "- goal_alignment: does the advice address the client's stated goals?\n"
    "- specificity: are recommendations concrete and actionable, not generic platitudes?\n"
    "- coherence: does the advisor's synthesis follow from the analyst's findings?\n"
    "- safety: are appropriate disclaimers present? Is the language responsible "
    "(no 'guaranteed returns', 'risk-free', etc.)?\n\n"
    "Respond ONLY with a JSON object of the form: "
    "{\"risk_alignment\": int, \"goal_alignment\": int, \"specificity\": int, "
    "\"coherence\": int, \"safety\": int, \"notes\": \"one short paragraph\"}"
)


def _build_user_prompt(state: AdvisorState) -> str:
    profile: ClientProfile = state["client_profile"]
    advice = state.get("draft_advice")
    findings = state.get("analyst_findings")

    parts = [
        "## Client profile",
        f"- name: {profile.name}",
        f"- age: {profile.age}",
        f"- risk_tolerance: {profile.risk_tolerance}",
        f"- time_horizon_years: {profile.time_horizon_years}",
        f"- annual_income: {profile.annual_income}",
        f"- goals: {profile.goals}",
        f"- assets: {profile.assets}",
        "",
        "## Analyst findings",
    ]
    if findings is not None:
        parts.append(f"query: {findings.query}")
        parts.append(f"confidence: {findings.confidence}")
        parts.append(f"sources: {len(findings.sources)} cited")
        parts.append(f"findings: {findings.findings}")
    else:
        parts.append("(no analyst report)")

    parts += ["", "## Final advice"]
    if advice is not None:
        parts.append("recommendations:")
        for r in advice.recommendations:
            parts.append(f"  - {r}")
        parts.append(f"rationale: {advice.rationale}")
        parts.append(f"disclaimers: {advice.disclaimers}")
    else:
        parts.append("(no advice produced)")

    parts += ["", "Score this advice on the five criteria. Return JSON only."]
    return "\n".join(parts)


def _parse_score(raw: str) -> tuple[JudgeScore, str | None]:
    """Best-effort JSON extraction. Tolerates markdown fences and prose around JSON."""
    text = raw.strip()
    # Strip ```json ... ``` fences if present.
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fence:
        text = fence.group(1)
    else:
        # Grab the first {...} block.
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if m:
            text = m.group(0)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        return JudgeScore(), f"could not parse JSON: {exc}"

    def _coerce_int(v: Any, default: int = 0) -> int:
        try:
            n = int(v)
        except (TypeError, ValueError):
            return default
        return max(1, min(5, n))

    score = JudgeScore(
        risk_alignment=_coerce_int(data.get("risk_alignment")),
        goal_alignment=_coerce_int(data.get("goal_alignment")),
        specificity=_coerce_int(data.get("specificity")),
        coherence=_coerce_int(data.get("coherence")),
        safety=_coerce_int(data.get("safety")),
        notes=str(data.get("notes", "")).strip(),
    )
    return score, None


class LLMJudge:
    """Scores final advice with a rubric prompt against the chosen LLM provider."""

    def __init__(self, llm: LLMProvider, *, max_tokens: int = 600, temperature: float = 0.0) -> None:
        self.llm = llm
        self.max_tokens = max_tokens
        self.temperature = temperature

    def score(self, state: AdvisorState) -> JudgeResult:
        if state.get("draft_advice") is None:
            return JudgeResult(error="no advice to judge")
        messages = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(state)},
        ]
        try:
            raw = self.llm.complete(
                messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                response_format={"type": "json_object"},
            )
        except Exception as exc:
            return JudgeResult(error=f"judge LLM call failed: {exc}")
        score, parse_err = _parse_score(raw)
        return JudgeResult(score=score, raw=raw, error=parse_err)


def aggregate(scores: list[JudgeScore]) -> dict[str, float]:
    """Return per-criterion + overall mean across runs."""
    if not scores:
        return {c: 0.0 for c in CRITERIA} | {"mean": 0.0}
    out: dict[str, float] = {}
    for c in CRITERIA:
        vals = [getattr(s, c) for s in scores]
        out[c] = round(sum(vals) / len(vals), 3)
    out["mean"] = round(sum(out[c] for c in CRITERIA) / len(CRITERIA), 3)
    return out
