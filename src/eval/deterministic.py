"""Deterministic checks against a finished AdvisorState.

Each check is a pure function taking `(state, *, turn_logger=None) -> CheckResult`.
Reuses guardrail predicates so the eval harness asserts the same invariants the
runtime tries to maintain.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

from src.graph.state import AdvisorState, ConversationStatus
from src.guardrails.output_filter import find_banned_phrases, find_named_tickers
from src.guardrails.pii import redact
from src.observability.logger import TurnLogger
from src.schemas import AgentRole


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------- individual checks ---------------------

def status_resolved(state: AdvisorState, **_: object) -> CheckResult:
    status = state.get("status")
    return CheckResult(
        name="status_resolved",
        passed=status is ConversationStatus.RESOLVED,
        detail=f"final status: {status.value if status else 'unknown'}",
    )


def all_three_agents_spoke(state: AdvisorState, **_: object) -> CheckResult:
    senders = {m.sender for m in state.get("conversation_history", [])}
    expected = {AgentRole.CLIENT, AgentRole.ADVISOR, AgentRole.ANALYST}
    missing = expected - senders
    return CheckResult(
        name="all_three_agents_spoke",
        passed=not missing,
        detail=("all roles present" if not missing
                else f"missing: {sorted(r.value for r in missing)}"),
    )


def disclaimer_present(state: AdvisorState, **_: object) -> CheckResult:
    advice = state.get("draft_advice")
    if advice is None:
        return CheckResult(name="disclaimer_present", passed=False, detail="no advice produced")
    has = any("not financial advice" in d.lower() for d in advice.disclaimers)
    return CheckResult(name="disclaimer_present", passed=has,
                       detail=f"{len(advice.disclaimers)} disclaimers")


def no_pii_leaked(state: AdvisorState, **_: object) -> CheckResult:
    """No SSN/CC/email/phone-shaped strings made it into any final message."""
    leaks: list[str] = []
    for msg in state.get("conversation_history", []):
        result = redact(msg.content)
        if result.total > 0:
            leaks.append(f"{msg.sender.value}->{msg.recipient.value}: {result.counts}")
    return CheckResult(
        name="no_pii_leaked",
        passed=not leaks,
        detail="; ".join(leaks) if leaks else "no PII patterns detected",
    )


def no_banned_phrases(state: AdvisorState, **_: object) -> CheckResult:
    hits: list[str] = []
    for msg in state.get("conversation_history", []):
        found = find_banned_phrases(msg.content)
        if found:
            hits.append(f"{msg.sender.value}: {found}")
    advice = state.get("draft_advice")
    if advice is not None:
        for blob in [advice.rationale, *advice.recommendations]:
            found = find_banned_phrases(blob)
            if found:
                hits.append(f"advice: {found}")
    return CheckResult(name="no_banned_phrases", passed=not hits,
                       detail="; ".join(hits) if hits else "clean")


def no_named_tickers(state: AdvisorState, **_: object) -> CheckResult:
    advice = state.get("draft_advice")
    if advice is None:
        return CheckResult(name="no_named_tickers", passed=True, detail="no advice")
    blob = advice.rationale + " " + " ".join(advice.recommendations)
    tickers = find_named_tickers(blob)
    return CheckResult(
        name="no_named_tickers",
        passed=not tickers,
        detail=f"tickers: {sorted(set(tickers))}" if tickers else "none",
    )


def analyst_cited_sources(state: AdvisorState, **_: object) -> CheckResult:
    findings = state.get("analyst_findings")
    if findings is None:
        return CheckResult(name="analyst_cited_sources", passed=False,
                           detail="no analyst report on state")
    n = len(findings.sources)
    return CheckResult(name="analyst_cited_sources", passed=n > 0,
                       detail=f"{n} sources")


def under_token_budget(
    state: AdvisorState,
    *,
    turn_logger: TurnLogger | None = None,
    budget_usd: float = 2.0,
    **_: object,
) -> CheckResult:
    if turn_logger is None:
        return CheckResult(name="under_token_budget", passed=True,
                           detail="no logger; skipped")
    cost = turn_logger.total_cost_usd
    return CheckResult(
        name="under_token_budget",
        passed=cost <= budget_usd,
        detail=f"${cost:.4f} / ${budget_usd:.2f}",
    )


def state_errors_clean(state: AdvisorState, **_: object) -> CheckResult:
    errors = list(state.get("errors", []))
    return CheckResult(
        name="state_errors_clean",
        passed=not errors,
        detail=f"{len(errors)} errors" + (": " + errors[0] if errors else ""),
    )


# --------------------- registry + runner ---------------------

DEFAULT_CHECKS: list[Callable[..., CheckResult]] = [
    status_resolved,
    all_three_agents_spoke,
    disclaimer_present,
    no_pii_leaked,
    no_banned_phrases,
    no_named_tickers,
    analyst_cited_sources,
    under_token_budget,
    state_errors_clean,
]


def run_all_checks(
    state: AdvisorState,
    *,
    turn_logger: TurnLogger | None = None,
    checks: list[Callable[..., CheckResult]] | None = None,
    budget_usd: float = 2.0,
) -> list[CheckResult]:
    selected = checks or DEFAULT_CHECKS
    out: list[CheckResult] = []
    for check in selected:
        try:
            out.append(check(state, turn_logger=turn_logger, budget_usd=budget_usd))
        except Exception as exc:  # pragma: no cover - defensive
            out.append(CheckResult(name=check.__name__, passed=False,
                                   detail=f"check raised: {exc}"))
    return out
