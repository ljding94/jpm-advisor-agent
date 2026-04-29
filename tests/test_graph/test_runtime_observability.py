"""Verify TurnLogger is wired and cost limit can trip during a run."""
from __future__ import annotations

import json
from pathlib import Path

from src.agents.advisor import AdvisorAgent
from src.agents.analyst import AnalystAgent
from src.agents.client import ClientAgent
from src.graph.builder import build_graph
from src.graph.state import ConversationStatus, initial_state
from src.guardrails.limits import MAX_TOTAL_COST_USD
from src.observability.logger import TurnLogger
from src.tools.web_search import FakeWebSearchProvider


def _scripted_llm(persona_key: str, *, cost_per_call: float = 0.001):
    from tests.conftest import FakeLLM

    return FakeLLM(
        marker_responses={
            "decide the advisor's next action": json.dumps({
                "next_action": "dispatch_analyst",
                "target": "analyst",
                "message": f"Research portfolio for {persona_key}.",
            }),
            "research task from the advisor": "Balanced allocation.",
            "reply on a single line starting with [confirm] or [reject]": "[CONFIRM] sounds good.",
            "answer in 1-4 sentences in first person": "Please go ahead.",
        },
        default=json.dumps({"next_action": "draft_advice", "target": "client", "message": ""}),
        cost_per_call_usd=cost_per_call,
    )


def test_turn_logger_records_each_node_call(david_profile):
    llm = _scripted_llm("david")
    client = ClientAgent(profile=david_profile, llm=llm)
    advisor = AdvisorAgent(llm=llm)
    analyst = AnalystAgent(llm=llm, knowledge_store=None, web_search=FakeWebSearchProvider())

    log = TurnLogger()
    state = initial_state(david_profile)
    state = client.open_conversation(state)
    graph = build_graph(client=client, advisor=advisor, analyst=analyst, turn_logger=log)
    final = graph.invoke(state, config={"recursion_limit": 50})

    assert final["status"] is ConversationStatus.RESOLVED
    assert log.records, "TurnLogger should have recorded at least one turn"
    # Every record has the structured-JSON fields the spec required.
    for r in log.records:
        assert r.agent in {"advisor", "analyst", "client"}
        assert r.action  # non-empty
        assert r.duration_ms >= 0
        assert r.input_tokens >= 0
        assert r.output_tokens >= 0
        assert r.cost_usd >= 0
    summary = log.summary()
    assert summary["turns"] == len(log.records)
    assert summary["total_cost_usd"] > 0


def test_jsonl_export_round_trips(tmp_path: Path, david_profile):
    llm = _scripted_llm("david")
    client = ClientAgent(profile=david_profile, llm=llm)
    advisor = AdvisorAgent(llm=llm)
    analyst = AnalystAgent(llm=llm, knowledge_store=None, web_search=FakeWebSearchProvider())
    log = TurnLogger()
    graph = build_graph(client=client, advisor=advisor, analyst=analyst, turn_logger=log)
    state = client.open_conversation(initial_state(david_profile))
    graph.invoke(state, config={"recursion_limit": 50})

    out = log.write_jsonl(tmp_path / "log.jsonl")
    text = out.read_text()
    parsed = [json.loads(line) for line in text.strip().splitlines()]
    assert len(parsed) == len(log.records)
    assert all("turn" in p and "agent" in p for p in parsed)


def test_cost_limit_trips_termination(david_profile):
    """If a single call costs more than MAX_TOTAL_COST_USD, the next node trips."""
    expensive_llm = _scripted_llm("david", cost_per_call=MAX_TOTAL_COST_USD + 0.5)
    client = ClientAgent(profile=david_profile, llm=expensive_llm)
    advisor = AdvisorAgent(llm=expensive_llm)
    analyst = AnalystAgent(
        llm=expensive_llm, knowledge_store=None, web_search=FakeWebSearchProvider()
    )

    log = TurnLogger()
    state = client.open_conversation(initial_state(david_profile))
    graph = build_graph(client=client, advisor=advisor, analyst=analyst, turn_logger=log)
    final = graph.invoke(state, config={"recursion_limit": 50})

    assert final["status"] is ConversationStatus.TERMINATED
    assert "MAX_TOTAL_COST_USD" in (final.get("termination_reason") or "")
    assert log.total_cost_usd >= MAX_TOTAL_COST_USD


def test_runtime_runs_without_logger(david_profile):
    """Backwards-compat: build_graph(..., turn_logger=None) still works."""
    llm = _scripted_llm("david")
    client = ClientAgent(profile=david_profile, llm=llm)
    advisor = AdvisorAgent(llm=llm)
    analyst = AnalystAgent(llm=llm, knowledge_store=None, web_search=FakeWebSearchProvider())
    state = client.open_conversation(initial_state(david_profile))
    graph = build_graph(client=client, advisor=advisor, analyst=analyst)
    final = graph.invoke(state, config={"recursion_limit": 50})
    assert final["status"] is ConversationStatus.RESOLVED
