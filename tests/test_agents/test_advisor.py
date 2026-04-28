"""AdvisorAgent tests."""
from __future__ import annotations

import json

from src.agents.advisor import AdvisorAgent
from src.graph.state import ConversationStatus, initial_state
from src.schemas import (
    AgentMessage,
    AgentRole,
    AnalystReport,
    MessageType,
    Source,
)


def test_advisor_decide_returns_valid_action(david_profile, fake_llm):
    fake_llm.script.append(json.dumps({
        "next_action": "ask_client",
        "target": "client",
        "message": "What is your time horizon?",
    }))
    advisor = AdvisorAgent(llm=fake_llm)
    state = initial_state(david_profile)
    state["conversation_history"].append(
        AgentMessage(
            sender=AgentRole.CLIENT,
            recipient=AgentRole.ADVISOR,
            content="Hi, I'd like advice.",
            message_type=MessageType.QUESTION,
        )
    )
    new_state = advisor.process(state)
    last = new_state["conversation_history"][-1]
    assert last.sender is AgentRole.ADVISOR
    assert last.recipient is AgentRole.CLIENT
    assert last.message_type is MessageType.QUESTION
    assert "time horizon" in last.content


def test_advisor_dispatches_to_analyst(david_profile, fake_llm):
    fake_llm.script.append(json.dumps({
        "next_action": "dispatch_analyst",
        "target": "analyst",
        "message": "Research moderate allocations for an 18-year horizon.",
    }))
    advisor = AdvisorAgent(llm=fake_llm)
    state = initial_state(david_profile)
    state["conversation_history"].append(
        AgentMessage(
            sender=AgentRole.CLIENT,
            recipient=AgentRole.ADVISOR,
            content="What allocation should I have?",
            message_type=MessageType.QUESTION,
        )
    )
    new_state = advisor.process(state)
    last = new_state["conversation_history"][-1]
    assert last.recipient is AgentRole.ANALYST
    assert last.message_type is MessageType.TASK
    assert new_state["status"] is ConversationStatus.ANALYZE
    assert new_state["current_advisor_query"]


def test_advisor_drafts_advice(david_profile, fake_llm):
    fake_llm.script.append(json.dumps({
        "next_action": "draft_advice",
        "target": "client",
        "message": "synthesize advice now",
    }))
    advisor = AdvisorAgent(llm=fake_llm)
    state = initial_state(david_profile)
    state["analyst_findings"] = AnalystReport(
        query="moderate allocation",
        findings="A 60–70% equity allocation is standard for moderate risk.",
        sources=[Source(title="kb::01_asset_allocation.md")],
        confidence=0.8,
    )
    state["conversation_history"].append(
        AgentMessage(
            sender=AgentRole.CLIENT,
            recipient=AgentRole.ADVISOR,
            content="What's your recommendation?",
            message_type=MessageType.QUESTION,
        )
    )
    new_state = advisor.process(state)
    advice = new_state["draft_advice"]
    assert advice is not None
    assert advice.recommendations
    assert any("not financial advice" in d.lower() for d in advice.disclaimers)
    last = new_state["conversation_history"][-1]
    assert last.recipient is AgentRole.CLIENT
    assert last.message_type is MessageType.ADVICE
    assert new_state["status"] is ConversationStatus.CONFIRM


def test_advisor_after_report_drafts_advice(david_profile, fake_llm):
    advisor = AdvisorAgent(llm=fake_llm)
    state = initial_state(david_profile)
    state["status"] = ConversationStatus.ANALYZE
    state["analyst_findings"] = AnalystReport(
        query="q",
        findings="balanced is fine",
        sources=[Source(title="kb::doc.md")],
        confidence=0.7,
    )
    state["conversation_history"].append(
        AgentMessage(
            sender=AgentRole.ANALYST,
            recipient=AgentRole.ADVISOR,
            content="findings",
            message_type=MessageType.REPORT,
        )
    )
    new_state = advisor.process(state)
    # After receiving a REPORT, advisor immediately drafts advice for the client.
    assert new_state["status"] is ConversationStatus.CONFIRM
    assert new_state["draft_advice"] is not None
    last = new_state["conversation_history"][-1]
    assert last.recipient is AgentRole.CLIENT
    assert last.message_type is MessageType.ADVICE


def test_advisor_decision_falls_back_when_llm_returns_garbage(david_profile, fake_llm):
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
    last = new_state["conversation_history"][-1]
    assert last.sender is AgentRole.ADVISOR
    # default fallback during GATHER_PROFILE → ask_client
    assert last.recipient is AgentRole.CLIENT


def test_advisor_no_op_when_not_addressed(david_profile, fake_llm):
    advisor = AdvisorAgent(llm=fake_llm)
    state = initial_state(david_profile)
    state["conversation_history"].append(
        AgentMessage(
            sender=AgentRole.ADVISOR,
            recipient=AgentRole.CLIENT,
            content="...",
            message_type=MessageType.QUESTION,
        )
    )
    new_state = advisor.process(state)
    assert new_state == state


def test_synthesize_advice_uses_strategy(david_profile, fake_llm):
    advisor = AdvisorAgent(llm=fake_llm)
    state = initial_state(david_profile)
    advice = advisor.synthesize_advice(state)
    assert any("60% equities" in r or "65% equities" in r for r in advice.recommendations)


def test_advisor_factory_via_agent_factory(fake_llm):
    """Factory pattern: AgentFactory builds an AdvisorAgent."""
    from src.factories.agent_factory import AgentFactory
    advisor = AgentFactory.create(AgentRole.ADVISOR, llm=fake_llm)
    assert isinstance(advisor, AdvisorAgent)
