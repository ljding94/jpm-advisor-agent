"""End-to-end integration tests with mocked LLM. Runs each persona to RESOLVED."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.agents.advisor import AdvisorAgent
from src.agents.analyst import AnalystAgent
from src.agents.client import ClientAgent
from src.agents.reviewer import ReviewerAgent
from src.graph.builder import build_graph
from src.graph.state import ConversationStatus, initial_state
from src.schemas import AgentRole, ClientProfile
from src.tools.knowledge_store import KnowledgeStore
from src.tools.web_search import FakeWebSearchProvider, WebResult


def _persona(key: str) -> ClientProfile:
    files = {
        "margaret": "margaret_conservative.json",
        "david": "david_moderate.json",
        "priya": "priya_aggressive.json",
    }
    data = json.loads((Path("data/personas") / files[key]).read_text())
    return ClientProfile(**data)


def _build_kb(tmp_path: Path, fake_embedder) -> KnowledgeStore:
    docs = tmp_path / "kb"
    docs.mkdir()
    (docs / "alloc.md").write_text(
        "Asset allocation across stocks bonds and cash explains the bulk of portfolio variance. "
        "Conservative profiles tilt to bonds; aggressive profiles tilt to equities."
    )
    (docs / "diversification.md").write_text(
        "Diversification is the only free lunch. Spread risk across asset classes and geographies."
    )
    store = KnowledgeStore(persist_path=tmp_path / "chroma", embedder=fake_embedder)
    store.reset()
    store.ingest_directory(docs)
    return store


def _scripted_llm(persona_key: str):
    """Return a FakeLLM scripted to drive the graph to RESOLVED for any persona."""
    from tests.conftest import FakeLLM

    return FakeLLM(
        marker_responses={
            # Advisor decisions — keyed by phrases unique to the decide prompt.
            "decide the advisor's next action": json.dumps({
                "next_action": "dispatch_analyst",
                "target": "analyst",
                "message": f"Research a sensible portfolio for the {persona_key} persona.",
            }),
            # Analyst synthesis — generic but on-topic.
            "research task from the advisor": (
                "A balanced allocation appropriate to the client's risk tolerance "
                "is well supported by standard planning principles."
            ),
            # Client confirmation — always confirm.
            "reply on a single line starting with [confirm] or [reject]": (
                "[CONFIRM] this matches my goals and time horizon."
            ),
            # Client question answering — generic in-character reply.
            "answer in 1–4 sentences in first person": (
                "That sounds right — please go ahead and recommend something."
            ),
        },
        default=json.dumps({"next_action": "draft_advice", "target": "client", "message": ""}),
    )


@pytest.mark.parametrize("persona_key", ["margaret", "david", "priya"])
def test_graph_runs_to_resolved(persona_key, tmp_path, fake_embedder):
    """Acceptance criterion #4: each persona produces all-three-agent output and ends RESOLVED."""
    profile = _persona(persona_key)
    llm = _scripted_llm(persona_key)
    kb = _build_kb(tmp_path, fake_embedder)
    web = FakeWebSearchProvider(canned=[
        WebResult(title="Vanguard primer", url="https://x", snippet="general primer")
    ])

    client = ClientAgent(profile=profile, llm=llm)
    advisor = AdvisorAgent(llm=llm)
    analyst = AnalystAgent(llm=llm, knowledge_store=kb, web_search=web)
    reviewer = ReviewerAgent(llm=llm)

    state = initial_state(profile)
    state = client.open_conversation(state)

    graph = build_graph(client=client, advisor=advisor, analyst=analyst, reviewer=reviewer)
    final = graph.invoke(state, config={"recursion_limit": 50})

    assert final["status"] is ConversationStatus.RESOLVED, (
        f"persona {persona_key} did not resolve: {final.get('status')}, "
        f"{final.get('termination_reason')}"
    )

    senders = {m.sender for m in final["conversation_history"]}
    assert AgentRole.CLIENT in senders
    assert AgentRole.ADVISOR in senders
    assert AgentRole.ANALYST in senders, (
        f"analyst never produced output for {persona_key}"
    )
    # Reviewer must always be present — all advisor→client traffic flows through it.
    assert AgentRole.REVIEWER in senders, (
        f"reviewer never spoke for {persona_key}"
    )

    # Final confirmation message metadata flag.
    last = final["conversation_history"][-1]
    assert last.sender is AgentRole.CLIENT
    assert last.metadata.get("confirmed") is True


def test_graph_terminates_when_max_turns_breached(tmp_path, fake_embedder):
    """Hard-limit termination path: a never-ending loop trips MAX_TURNS."""
    profile = _persona("david")
    # LLM that always sends the advisor back to ask the client another question.
    from tests.conftest import FakeLLM

    llm = FakeLLM(
        marker_responses={
            "decide the advisor's next action": json.dumps({
                "next_action": "ask_client",
                "target": "client",
                "message": "Tell me more.",
            }),
            "answer in 1–4 sentences in first person": "I keep talking.",
        },
        default=json.dumps({"next_action": "ask_client", "target": "client", "message": "more?"}),
    )
    client = ClientAgent(profile=profile, llm=llm)
    advisor = AdvisorAgent(llm=llm)
    analyst = AnalystAgent(
        llm=llm,
        knowledge_store=_build_kb(tmp_path, fake_embedder),
        web_search=FakeWebSearchProvider(),
    )
    reviewer = ReviewerAgent(llm=llm)
    state = initial_state(profile)
    state = client.open_conversation(state)
    graph = build_graph(client=client, advisor=advisor, analyst=analyst, reviewer=reviewer)
    final = graph.invoke(state, config={"recursion_limit": 80})
    assert final["status"] is ConversationStatus.TERMINATED
    assert final.get("termination_reason")
