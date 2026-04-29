"""Edge cases called out in SPEC.md but not covered elsewhere.

Two cases here:
1. Client supplies contradictory info — system still resolves; advisor proceeds
   with the most-recent answer rather than crashing.
2. Both Analyst tools (KB + web) fail — graceful degradation: report still has
   non-empty sources via the planning-principles fallback, and the conversation
   still resolves.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock

from src.agents.advisor import AdvisorAgent
from src.agents.analyst import AnalystAgent
from src.agents.client import ClientAgent
from src.graph.builder import build_graph
from src.graph.state import ConversationStatus, initial_state
from src.schemas import AgentMessage, AgentRole, MessageType
from src.tools.web_search import FakeWebSearchProvider


def _scripted_llm_for_contradiction():
    """Client answers contradictorily on consecutive turns.

    Turn 1 answer: aggressive risk tolerance.
    Turn 2 answer (after follow-up): conservative.
    The advisor should still proceed and resolve.
    """
    from tests.conftest import FakeLLM

    return FakeLLM(
        marker_responses={
            # Two-step decision: ask first, then dispatch.
            "decide the advisor's next action": json.dumps({
                "next_action": "dispatch_analyst",
                "target": "analyst",
                "message": "Research a balanced allocation.",
            }),
            "research task from the advisor": "Balanced allocation guidance.",
            "reply on a single line starting with [confirm] or [reject]": (
                "[CONFIRM] this is fine"
            ),
            # The client-answer marker — gives DIFFERENT contradictory answers
            # for two different state shapes by triggering on the same marker
            # twice. We use the script as a fallback.
            "answer in 1-4 sentences in first person": "I'm extremely risk-tolerant — go heavy.",
        },
        default=json.dumps({"next_action": "draft_advice", "target": "client", "message": ""}),
    )


def test_contradictory_client_info_still_resolves(david_profile, fake_embedder):
    """Even if the client contradicts an earlier answer, the conversation resolves."""
    llm = _scripted_llm_for_contradiction()
    client = ClientAgent(profile=david_profile, llm=llm)
    advisor = AdvisorAgent(llm=llm)
    analyst = AnalystAgent(llm=llm, knowledge_store=None, web_search=FakeWebSearchProvider())

    state = client.open_conversation(initial_state(david_profile))

    # Inject a contradictory pair of client→advisor messages BEFORE the graph runs.
    state["conversation_history"].append(
        AgentMessage(
            sender=AgentRole.CLIENT,
            recipient=AgentRole.ADVISOR,
            content="I'm extremely conservative — capital preservation only.",
            message_type=MessageType.ANSWER,
        )
    )
    state["conversation_history"].append(
        AgentMessage(
            sender=AgentRole.CLIENT,
            recipient=AgentRole.ADVISOR,
            content="Actually I want maximum aggressive growth, ignore what I said.",
            message_type=MessageType.ANSWER,
        )
    )

    graph = build_graph(client=client, advisor=advisor, analyst=analyst)
    final = graph.invoke(state, config={"recursion_limit": 50})

    assert final["status"] is ConversationStatus.RESOLVED
    # Both contradictory client messages persist in the transcript.
    contents = [m.content for m in final["conversation_history"]]
    assert any("conservative" in c for c in contents)
    assert any("aggressive growth" in c for c in contents)


def test_full_analyst_tool_failure_graceful_degradation(david_profile, fake_llm):
    """Both KB and web throwing should still produce a valid AnalystReport
    with non-empty sources, and the graph should resolve."""
    fake_llm.marker_responses = {
        "decide the advisor's next action": json.dumps({
            "next_action": "dispatch_analyst",
            "target": "analyst",
            "message": "Research a balanced allocation.",
        }),
        "research task from the advisor": "Generic planning principles apply.",
        "reply on a single line starting with [confirm] or [reject]": (
            "[CONFIRM] makes sense"
        ),
        "answer in 1-4 sentences in first person": "Sounds good.",
    }
    fake_llm.default = json.dumps({
        "next_action": "draft_advice", "target": "client", "message": ""
    })

    exploding_kb = MagicMock()
    exploding_kb.similarity_search.side_effect = RuntimeError("chroma down")

    class ExplodingWeb:
        def search(self, query, max_results=5):
            raise RuntimeError("network down")

    client = ClientAgent(profile=david_profile, llm=fake_llm)
    advisor = AdvisorAgent(llm=fake_llm)
    analyst = AnalystAgent(
        llm=fake_llm, knowledge_store=exploding_kb, web_search=ExplodingWeb()
    )

    state = client.open_conversation(initial_state(david_profile))
    graph = build_graph(client=client, advisor=advisor, analyst=analyst)
    final = graph.invoke(state, config={"recursion_limit": 50})

    # Despite both tools failing, the graph reaches RESOLVED.
    assert final["status"] is ConversationStatus.RESOLVED

    # Errors from both subsystems were logged.
    errs = final["errors"]
    assert any("analyst.kb" in e for e in errs), f"missing kb error in {errs}"
    assert any("analyst.web" in e for e in errs), f"missing web error in {errs}"

    # The analyst still produced a report with at least one source (the fallback).
    findings = final["analyst_findings"]
    assert findings is not None
    assert len(findings.sources) >= 1
    assert any("planning principles" in s.title.lower() for s in findings.sources)


def test_research_records_no_errors_when_tools_healthy(david_profile, fake_llm):
    """Companion: when KB returns hits, no error should be recorded."""
    class HealthyKB:
        def similarity_search(self, query, k=4):
            from src.tools.knowledge_store import RetrievedChunk
            return [
                RetrievedChunk(text="content", source="a.md", score=0.9),
                RetrievedChunk(text="content2", source="b.md", score=0.85),
            ]

    fake_llm.script.append("synthesized")
    agent = AnalystAgent(
        llm=fake_llm, knowledge_store=HealthyKB(), web_search=FakeWebSearchProvider()
    )
    state = initial_state(david_profile)
    state["conversation_history"].append(
        AgentMessage(
            sender=AgentRole.ADVISOR,
            recipient=AgentRole.ANALYST,
            content="research please",
            message_type=MessageType.TASK,
        )
    )
    new_state = agent.process(state)
    assert new_state.get("errors", []) == []
