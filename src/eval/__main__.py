"""CLI: `python -m src.eval` — run the eval suite, write a report."""
from __future__ import annotations

import argparse
import os
import sys

from src.eval.judge import LLMJudge
from src.eval.report import write_report
from src.eval.runner import PERSONA_FILES, run_suite
from src.providers.llm import get_llm_provider


def _build_judge(spec: str | None):
    """spec is one of: None (no judge), 'fake', 'same' (reuse main LLM), or 'provider/model'."""
    if not spec or spec.lower() == "none":
        return None, None
    if spec.lower() == "fake":
        # Offline judge that returns canned middling scores. Useful in CI.
        from tests.conftest import FakeLLM  # type: ignore[import-not-found]
        import json as _json

        canned = _json.dumps({
            "risk_alignment": 4, "goal_alignment": 4, "specificity": 3,
            "coherence": 4, "safety": 5,
            "notes": "Offline FakeLLM judge — canned scores.",
        })
        fake = FakeLLM(default=canned)
        return LLMJudge(fake), fake
    if spec.lower() == "same":
        llm = get_llm_provider()
        return LLMJudge(llm), llm
    # spec is a model id; build a provider from env, then override its model attribute.
    llm = get_llm_provider()
    llm.model = spec
    return LLMJudge(llm), llm


def _parse_personas(raw: str) -> list[str]:
    if raw == "all":
        return list(PERSONA_FILES.keys())
    return [p.strip() for p in raw.split(",") if p.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.eval")
    parser.add_argument("--personas", default="all",
                        help="comma list, e.g. david,priya — or 'all' (default).")
    parser.add_argument("--n", type=int, default=1,
                        help="runs per persona (default 1).")
    parser.add_argument("--judge", default="none",
                        help="'none' (default), 'fake' (offline canned), 'same' (reuse main LLM), or a model id.")
    parser.add_argument("--out", default="evals/reports",
                        help="output root directory (default: evals/reports).")
    parser.add_argument("--budget-usd", type=float, default=2.0,
                        help="per-run cost budget for the under_token_budget check.")
    parser.add_argument("--label", default=None,
                        help="optional label for the report directory; defaults to a UTC timestamp.")
    args = parser.parse_args(argv)

    personas = _parse_personas(args.personas)
    invalid = [p for p in personas if p not in PERSONA_FILES]
    if invalid:
        print(f"unknown personas: {invalid}. valid: {list(PERSONA_FILES)}", file=sys.stderr)
        return 2

    judge, _ = _build_judge(args.judge)

    # Single LLM instance reused across runs.
    main_llm = get_llm_provider()

    print(
        f"running suite: personas={personas} runs={args.n} judge={args.judge} "
        f"provider={os.getenv('LLM_PROVIDER', 'openrouter')} model={main_llm.model}",
        file=sys.stderr,
    )

    results = run_suite(
        personas,
        runs_per_persona=args.n,
        llm=main_llm,
        judge=judge,
        budget_usd=args.budget_usd,
    )
    out_dir = write_report(results, out_root=args.out, label=args.label)
    print(f"\nwrote {out_dir / 'report.md'}", file=sys.stderr)
    print(f"wrote {out_dir / 'results.json'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
