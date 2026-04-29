"""Eval runner: build runtime, invoke graph per (persona, seed), capture results."""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.agents.advisor import AdvisorAgent
from src.agents.analyst import AnalystAgent
from src.agents.client import ClientAgent
from src.eval.deterministic import run_all_checks
from src.eval.judge import JudgeResult, LLMJudge
from src.graph.builder import build_graph
from src.graph.state import AdvisorState, initial_state
from src.observability.logger import TurnLogger
from src.providers.embeddings import get_embedding_provider
from src.providers.llm import LLMProvider, get_llm_provider
from src.schemas import ClientProfile
from src.tools.knowledge_store import KnowledgeStore
from src.tools.web_search import DDGSearchProvider, WebSearchProvider

PERSONA_FILES = {
    "margaret": "data/personas/margaret_conservative.json",
    "david": "data/personas/david_moderate.json",
    "priya": "data/personas/priya_aggressive.json",
}


@dataclass
class RunResult:
    persona: str
    run_index: int
    final_status: str
    termination_reason: str | None
    turn_count: int
    checks: list[dict] = field(default_factory=list)
    judge: dict | None = None
    usage: dict = field(default_factory=dict)
    duration_s: float = 0.0
    transcript: list[dict] = field(default_factory=list)

    @property
    def deterministic_passed(self) -> int:
        return sum(1 for c in self.checks if c["passed"])

    @property
    def deterministic_total(self) -> int:
        return len(self.checks)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_persona(key: str) -> ClientProfile:
    path = Path(PERSONA_FILES[key])
    return ClientProfile(**json.loads(path.read_text(encoding="utf-8")))


def _build_kb_or_none() -> KnowledgeStore | None:
    """Best-effort knowledge store. Returns None if construction fails (no model available, etc.)."""
    try:
        embedder = get_embedding_provider()
        persist = os.getenv("CHROMA_PATH", "data/chroma")
        store = KnowledgeStore(persist_path=persist, embedder=embedder)
        if store.count() == 0:
            store.ingest_directory(Path("data/knowledge_base"))
        return store
    except Exception:
        return None


def _build_runtime(
    profile: ClientProfile,
    *,
    llm: LLMProvider,
    web: WebSearchProvider,
    kb: KnowledgeStore | None,
):
    client = ClientAgent(profile=profile, llm=llm)
    advisor = AdvisorAgent(llm=llm)
    analyst = AnalystAgent(llm=llm, knowledge_store=kb, web_search=web)
    return client, advisor, analyst


def _serialize_history(state: AdvisorState) -> list[dict]:
    return [
        {
            "sender": m.sender.value,
            "recipient": m.recipient.value,
            "type": m.message_type.value,
            "content": m.content,
        }
        for m in state.get("conversation_history", [])
    ]


def run_one(
    persona_key: str,
    *,
    run_index: int = 0,
    llm: LLMProvider | None = None,
    judge: LLMJudge | None = None,
    web: WebSearchProvider | None = None,
    kb: KnowledgeStore | None = None,
    budget_usd: float = 2.0,
    recursion_limit: int = 80,
    capture_transcript: bool = True,
) -> RunResult:
    started = time.monotonic()
    profile = _load_persona(persona_key)
    llm = llm or get_llm_provider()
    web = web or DDGSearchProvider()
    if kb is None:
        kb = _build_kb_or_none()
    client, advisor, analyst = _build_runtime(profile, llm=llm, web=web, kb=kb)

    state = initial_state(profile)
    state = client.open_conversation(state)
    turn_logger = TurnLogger()
    graph = build_graph(client=client, advisor=advisor, analyst=analyst,
                        verbose=False, turn_logger=turn_logger)
    final = graph.invoke(state, config={"recursion_limit": recursion_limit})

    deterministic = run_all_checks(final, turn_logger=turn_logger, budget_usd=budget_usd)
    judge_dict: dict | None = None
    if judge is not None:
        judge_result: JudgeResult = judge.score(final)
        judge_dict = judge_result.to_dict()

    status = final.get("status")
    return RunResult(
        persona=persona_key,
        run_index=run_index,
        final_status=status.value if status else "unknown",
        termination_reason=final.get("termination_reason"),
        turn_count=int(final.get("turn_count", 0)),
        checks=[c.to_dict() for c in deterministic],
        judge=judge_dict,
        usage=turn_logger.summary(),
        duration_s=round(time.monotonic() - started, 3),
        transcript=_serialize_history(final) if capture_transcript else [],
    )


def run_suite(
    personas: list[str],
    *,
    runs_per_persona: int = 1,
    llm: LLMProvider | None = None,
    judge: LLMJudge | None = None,
    web: WebSearchProvider | None = None,
    kb_factory=None,
    budget_usd: float = 2.0,
) -> list[RunResult]:
    """Run the full suite. `kb_factory` is a callable returning a fresh KB per call (or None).

    The same `llm` instance is reused across runs so cumulative usage across the
    whole suite is visible in the judge LLM's counters too if desired.
    """
    results: list[RunResult] = []
    for persona in personas:
        for idx in range(runs_per_persona):
            kb = kb_factory() if kb_factory else None
            try:
                res = run_one(
                    persona,
                    run_index=idx,
                    llm=llm,
                    judge=judge,
                    web=web,
                    kb=kb,
                    budget_usd=budget_usd,
                )
            except Exception as exc:  # pragma: no cover - defensive
                res = RunResult(
                    persona=persona,
                    run_index=idx,
                    final_status="error",
                    termination_reason=f"runner exception: {exc}",
                    turn_count=0,
                )
            results.append(res)
    return results


__all__ = [
    "PERSONA_FILES",
    "RunResult",
    "run_one",
    "run_suite",
]
