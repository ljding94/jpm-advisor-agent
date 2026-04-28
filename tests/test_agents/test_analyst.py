"""AnalystAgent tests with FakeLLM, fake web search, and a tmp KnowledgeStore."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from src.agents.analyst import AnalystAgent
from src.graph.state import ConversationStatus, initial_state
from src.schemas import AgentMessage, AgentRole, AnalystReport, MessageType
from src.tools.knowledge_store import KnowledgeStore
from src.tools.web_search import FakeWebSearchProvider, WebResult


def _make_kb(tmp_path: Path, fake_embedder) -> KnowledgeStore:
    docs = tmp_path / "kb"
    docs.mkdir()
    (docs / "alloc.md").write_text(
        "Asset allocation is the primary driver of portfolio variance. "
        "Conservative portfolios hold 30-40% equities and 50-60% bonds."
    )
    (docs / "diversification.md").write_text(
        "Diversification reduces unsystematic risk without lowering expected return."
    )
    store = KnowledgeStore(persist_path=tmp_path / "chroma", embedder=fake_embedder)
    store.reset()
    store.ingest_directory(docs)
    return store


def test_research_returns_report_with_sources(tmp_path, fake_embedder, fake_llm):
    fake_llm.script.append("Conservative allocations emphasize bonds and cash for capital preservation.")
    kb = _make_kb(tmp_path, fake_embedder)
    web = FakeWebSearchProvider()
    agent = AnalystAgent(llm=fake_llm, knowledge_store=kb, web_search=web)
    report = agent.research("What's a good conservative allocation?")
    assert isinstance(report, AnalystReport)
    assert len(report.sources) >= 1
    assert "Conservative" in report.findings or "conservative" in report.findings


def test_process_appends_report_message_and_advances(tmp_path, fake_embedder, fake_llm, david_profile):
    fake_llm.script.append("Generic finding.")
    kb = _make_kb(tmp_path, fake_embedder)
    agent = AnalystAgent(llm=fake_llm, knowledge_store=kb, web_search=FakeWebSearchProvider())

    state = initial_state(david_profile)
    state["conversation_history"].append(
        AgentMessage(
            sender=AgentRole.ADVISOR,
            recipient=AgentRole.ANALYST,
            content="Research target equity allocation for a moderate 18-yr horizon.",
            message_type=MessageType.TASK,
        )
    )
    new_state = agent.process(state)
    last = new_state["conversation_history"][-1]
    assert last.sender is AgentRole.ANALYST
    assert last.recipient is AgentRole.ADVISOR
    assert last.message_type is MessageType.REPORT
    assert new_state["status"] is ConversationStatus.ADVISE
    assert new_state["analyst_findings"] is not None


def test_process_noop_when_not_addressed_to_analyst(david_profile, fake_llm):
    agent = AnalystAgent(llm=fake_llm)
    state = initial_state(david_profile)
    state["conversation_history"].append(
        AgentMessage(
            sender=AgentRole.CLIENT,
            recipient=AgentRole.ADVISOR,
            content="hi advisor",
            message_type=MessageType.QUESTION,
        )
    )
    new_state = agent.process(state)
    assert new_state == state


def test_research_falls_back_to_web_when_kb_weak(fake_llm):
    fake_llm.script.append("Synthesized.")
    weak_kb = MagicMock()
    weak_kb.similarity_search.return_value = []  # empty → triggers web fallback
    web = FakeWebSearchProvider(
        canned=[WebResult(title="Vanguard primer", url="https://x", snippet="primer text")]
    )
    agent = AnalystAgent(llm=fake_llm, knowledge_store=weak_kb, web_search=web)
    report = agent.research("retirement glide path")
    assert any("Vanguard" in s.title for s in report.sources)
    assert web.calls == ["retirement glide path"]


def test_research_skips_web_when_kb_strong(tmp_path, fake_embedder, fake_llm):
    fake_llm.script.append("Strong KB synthesis.")

    # Build a KB that returns artificially strong scores for any query.
    class StrongKB:
        def similarity_search(self, query, k=4):
            from src.tools.knowledge_store import RetrievedChunk

            return [
                RetrievedChunk(text="strong content one", source="a.md", score=0.9),
                RetrievedChunk(text="strong content two", source="b.md", score=0.85),
            ]

    web = FakeWebSearchProvider(canned=[WebResult(title="should_not_use", url="", snippet="")])
    agent = AnalystAgent(llm=fake_llm, knowledge_store=StrongKB(), web_search=web)
    report = agent.research("anything")
    assert web.calls == []  # web search skipped
    assert all("should_not_use" not in s.title for s in report.sources)


def test_research_emergency_fallback_source(fake_llm):
    fake_llm.script.append("findings")
    agent = AnalystAgent(llm=fake_llm, knowledge_store=None, web_search=None)
    report = agent.research("deserted island")
    assert len(report.sources) >= 1


def test_no_analyst_to_client_route_attempted(tmp_path, fake_embedder, fake_llm, david_profile):
    """Acceptance criterion: Analyst output is always addressed to Advisor."""
    fake_llm.script.append("findings")
    kb = _make_kb(tmp_path, fake_embedder)
    agent = AnalystAgent(llm=fake_llm, knowledge_store=kb, web_search=FakeWebSearchProvider())
    state = initial_state(david_profile)
    state["conversation_history"].append(
        AgentMessage(
            sender=AgentRole.ADVISOR,
            recipient=AgentRole.ANALYST,
            content="please research",
            message_type=MessageType.TASK,
        )
    )
    new_state = agent.process(state)
    for msg in new_state["conversation_history"]:
        if msg.sender is AgentRole.ANALYST:
            assert msg.recipient is AgentRole.ADVISOR
