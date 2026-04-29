"""CLI runner: build the graph, run a persona to RESOLVED, write a transcript."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from src.agents.advisor import AdvisorAgent
from src.agents.analyst import AnalystAgent
from src.agents.client import ClientAgent
from src.graph.builder import build_graph
from src.graph.state import ConversationStatus, initial_state
from src.observability.logger import export_transcript
from src.providers.embeddings import get_embedding_provider
from src.providers.llm import LLMProvider, OpenRouterLLM
from src.schemas import ClientProfile
from src.tools.knowledge_store import KnowledgeStore
from src.tools.web_search import DDGSearchProvider

PERSONA_FILES = {
    "margaret": "data/personas/margaret_conservative.json",
    "david": "data/personas/david_moderate.json",
    "priya": "data/personas/priya_aggressive.json",
}


def _load_persona(key: str) -> ClientProfile:
    path = Path(PERSONA_FILES[key])
    if not path.exists():
        raise FileNotFoundError(f"persona file not found: {path}")
    return ClientProfile(**json.loads(path.read_text(encoding="utf-8")))


def _build_llm() -> LLMProvider:
    """Build the OpenRouter LLM. Exits with a clear setup message if key missing."""
    if not os.getenv("OPENROUTER_API_KEY"):
        print(
            "ERROR: OPENROUTER_API_KEY is not set.\n\n"
            "Set up:\n"
            "  1. Copy .env.example to .env\n"
            "  2. Get a key at https://openrouter.ai\n"
            "  3. Put it in .env as OPENROUTER_API_KEY=...\n"
            "  4. `source .env` (or use python-dotenv) and re-run.\n\n"
            "If you only want to run the test suite, you don't need an API key.",
            file=sys.stderr,
        )
        sys.exit(2)
    return OpenRouterLLM()


def _build_kb() -> KnowledgeStore:
    persist = os.getenv("CHROMA_PATH", "data/chroma")
    embedder = get_embedding_provider()
    store = KnowledgeStore(persist_path=persist, embedder=embedder)
    if store.count() == 0:
        # First run — ingest the bundled knowledge base.
        store.ingest_directory(Path("data/knowledge_base"))
    return store


def run(persona_key: str, *, max_turns_hint: int = 80, verbose: bool = True) -> dict[str, Any]:
    profile = _load_persona(persona_key)
    print(f"[1/3] Persona: {profile.name} ({persona_key}, {profile.risk_tolerance}, age {profile.age})", file=sys.stderr)
    print("[2/3] Initializing LLM, knowledge store, and web search...", file=sys.stderr, flush=True)
    llm = _build_llm()
    try:
        kb = _build_kb()
    except Exception as exc:
        print(f"WARN: knowledge store unavailable ({exc}); analyst will rely on web search.", file=sys.stderr)
        kb = None  # type: ignore[assignment]
    web = DDGSearchProvider()

    client = ClientAgent(profile=profile, llm=llm)
    advisor = AdvisorAgent(llm=llm)
    analyst = AnalystAgent(llm=llm, knowledge_store=kb, web_search=web)

    state = initial_state(profile)
    state = client.open_conversation(state)
    print("[3/3] Running conversation...", file=sys.stderr, flush=True)
    graph = build_graph(client=client, advisor=advisor, analyst=analyst, verbose=verbose)
    final = graph.invoke(state, config={"recursion_limit": max_turns_hint})

    out = export_transcript(final, persona_key=persona_key)
    print(f"\nDone. Wrote transcript: {out}", file=sys.stderr)
    print(f"Final status: {final['status'].value}", file=sys.stderr)
    if final.get("termination_reason"):
        print(f"  Termination reason: {final['termination_reason']}", file=sys.stderr)
    return final


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a sample advisor conversation.")
    parser.add_argument(
        "--persona",
        choices=sorted(PERSONA_FILES),
        default="david",
        help="Which persona to run (default: david).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all three personas in sequence.",
    )
    args = parser.parse_args(argv)

    keys = list(PERSONA_FILES) if args.all else [args.persona]
    for key in keys:
        final = run(key)
        if final["status"] is not ConversationStatus.RESOLVED:
            print(f"WARN: persona {key} did not resolve ({final['status'].value}).", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
