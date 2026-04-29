"""Eval report writer — JSON + markdown."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from src.eval.judge import CRITERIA, JudgeScore, aggregate
from src.eval.runner import RunResult


def _now_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def write_report(
    results: list[RunResult],
    *,
    out_root: str | Path = "evals/reports",
    label: str | None = None,
) -> Path:
    """Write `results.json` + `report.md` under a timestamped directory."""
    slug = label or _now_slug()
    out_dir = Path(out_root) / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "results.json").write_text(
        json.dumps([r.to_dict() for r in results], indent=2, default=str),
        encoding="utf-8",
    )
    (out_dir / "report.md").write_text(_render_markdown(results), encoding="utf-8")
    return out_dir


def _render_markdown(results: list[RunResult]) -> str:
    lines: list[str] = []
    lines.append(f"# Eval report — {_now_slug()}")
    lines.append("")
    lines.append(f"Total runs: **{len(results)}**.")
    lines.append("")

    # ----- summary table -----
    lines.append("## Summary")
    lines.append("")
    lines.append("| Persona | Run | Status | Turns | Det. checks | Judge mean | Cost (USD) |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in results:
        det = f"{r.deterministic_passed}/{r.deterministic_total}"
        if r.judge and r.judge.get("score"):
            jmean = r.judge["score"].get("mean")
            jmean_str = f"{jmean:.2f}" if isinstance(jmean, (int, float)) else "—"
        else:
            jmean_str = "—"
        cost = r.usage.get("total_cost_usd", 0.0)
        lines.append(
            f"| {r.persona} | {r.run_index} | `{r.final_status}` | {r.turn_count} "
            f"| {det} | {jmean_str} | ${cost:.4f} |"
        )
    lines.append("")

    # ----- aggregates -----
    resolved = [r for r in results if r.final_status == "resolved"]
    lines.append("## Aggregates")
    lines.append("")
    lines.append(f"- Resolved: **{len(resolved)}/{len(results)}**.")
    if results:
        det_pass = sum(r.deterministic_passed for r in results)
        det_total = sum(r.deterministic_total for r in results)
        lines.append(f"- Deterministic checks: **{det_pass}/{det_total}** passed across all runs.")
        avg_cost = mean(r.usage.get("total_cost_usd", 0.0) for r in results)
        avg_turns = mean(r.turn_count for r in results)
        lines.append(f"- Average cost per run: **${avg_cost:.4f}**.")
        lines.append(f"- Average turns per run: **{avg_turns:.1f}**.")
    judge_scores: list[JudgeScore] = []
    for r in results:
        if r.judge and r.judge.get("score") and not r.judge.get("error"):
            s = r.judge["score"]
            judge_scores.append(JudgeScore(
                risk_alignment=s.get("risk_alignment", 0),
                goal_alignment=s.get("goal_alignment", 0),
                specificity=s.get("specificity", 0),
                coherence=s.get("coherence", 0),
                safety=s.get("safety", 0),
            ))
    if judge_scores:
        agg = aggregate(judge_scores)
        lines.append("")
        lines.append("### Judge scores (mean across runs, 1–5)")
        lines.append("")
        for c in CRITERIA:
            lines.append(f"- {c}: **{agg[c]:.2f}**")
        lines.append(f"- **overall: {agg['mean']:.2f}**")
    lines.append("")

    # ----- failures -----
    failures = []
    for r in results:
        for c in r.checks:
            if not c["passed"]:
                failures.append((r.persona, r.run_index, c))
    if failures:
        lines.append("## Deterministic check failures")
        lines.append("")
        for persona, idx, c in failures:
            lines.append(f"- `{persona}` run {idx}: **{c['name']}** — {c['detail']}")
        lines.append("")

    # ----- judge notes -----
    judge_notes = [(r, r.judge["score"].get("notes", "")) for r in results
                   if r.judge and r.judge.get("score") and r.judge["score"].get("notes")]
    if judge_notes:
        lines.append("## Judge notes")
        lines.append("")
        for r, note in judge_notes:
            lines.append(f"- `{r.persona}` run {r.run_index}: {note}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"
