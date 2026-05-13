"""Verify state.errors gets populated at the documented sites."""
from __future__ import annotations

from unittest.mock import MagicMock

from src.agents.advisor import AdvisorAgent
from src.agents.analyst import AnalystAgent
from src.agents.client import ClientAgent
from src.agents.reviewer import ReviewerAgent
from src.graph.builder import build_graph
from src.graph.state import AdvisorState, ConversationStatus, append_error, initial_state
from src.guardrails.limits import MAX_TOTAL_COST_USD
from src.observability.logger import TurnLogger
from src.schemas import AgentMessage, AgentRole, MessageType
from src.tools.web_search import FakeWebSearchProvider


def test_append_error_format(david_profile):
    state = initial_state(david_profile)
    append_error(state, source="advisor", detail="something went wrong")
    assert len(state["errors"]) == 1
    line = state["errors"][0]
    assert "[advisor]" in line
    assert "something went wrong" in line


def test_advisor_logs_when_llm_returns_garbage(david_profile, fake_llm):
    fake_llm.script.append("not a json")
    advisor = AdvisorAgent(llm=fake_llm)
    state = initial_state(david_profile)
    state["conversation_history"].append(
        AgentMessage(
            sender=AgentRole.CLIENT,
            recipient=AgentRole.ADVISOR,
            content="hi",
            message_type=MessageType.QUESTION,
        )
    )
    new_state = advisor.process(state)
    assert any("[advisor]" in e for e in new_state["errors"])


def test_analyst_logs_kb_failure(david_profile, fake_llm):
    fake_llm.script.append("findings text")
    bad_kb = MagicMock()
    bad_kb.similarity_search.side_effect = RuntimeError("chroma corrupt")
    agent = AnalystAgent(llm=fake_llm, knowledge_store=bad_kb, web_search=FakeWebSearchProvider())
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
    assert any("analyst.kb" in e and "chroma corrupt" in e for e in new_state["errors"])


def test_analyst_logs_web_failure(david_profile, fake_llm):
    fake_llm.script.append("findings text")

    class ExplodingWeb:
        def search(self, query, max_results=5):
            raise RuntimeError("ddg outage")

    bad_kb = MagicMock()
    bad_kb.similarity_search.return_value = []  # weak KB → triggers web fallback
    agent = AnalystAgent(llm=fake_llm, knowledge_store=bad_kb, web_search=ExplodingWeb())
    state = initial_state(david_profile)
    state["conversation_history"].append(
        AgentMessage(
            sender=AgentRole.ADVISOR,
            recipient=AgentRole.ANALYST,
            content="anything",
            message_type=MessageType.TASK,
        )
    )
    new_state = agent.process(state)
    assert any("analyst.web" in e and "ddg outage" in e for e in new_state["errors"])


def test_limit_breach_logs_to_state_errors(david_profile):
    """When the wrapper trips MAX_COST, it should also append to state.errors."""
    from tests.conftest import FakeLLM
    import json

    expensive = FakeLLM(
        marker_responses={
            "decide the advisor's next action": json.dumps({
                "next_action": "ask_client", "target": "client", "message": "hi",
            }),
            "answer in 1-4 sentences in first person": "ok",
        },
        default=json.dumps({"next_action": "ask_client", "target": "client", "message": "hi"}),
        cost_per_call_usd=MAX_TOTAL_COST_USD + 0.5,
    )
    client = ClientAgent(profile=david_profile, llm=expensive)
    advisor = AdvisorAgent(llm=expensive)
    analyst = AnalystAgent(llm=expensive, knowledge_store=None, web_search=FakeWebSearchProvider())
    reviewer = ReviewerAgent(llm=expensive)
    log = TurnLogger()
    state = client.open_conversation(initial_state(david_profile))
    graph = build_graph(client=client, advisor=advisor, analyst=analyst, reviewer=reviewer, turn_logger=log)
    final = graph.invoke(state, config={"recursion_limit": 30})

    assert final["status"] is ConversationStatus.TERMINATED
    assert any("[limits]" in e for e in final["errors"])
