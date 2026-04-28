"""AnalystAgent — fetches information for the Advisor. Never talks to Client."""
from __future__ import annotations

import json

from src.agents.base import BaseAgent
from src.graph.state import AdvisorState, ConversationStatus
from src.providers.llm import LLMProvider
from src.schemas import (
    AgentMessage,
    AgentRole,
    AnalystReport,
    MessageType,
    Source,
)
from src.tools.knowledge_store import KnowledgeStore, RetrievedChunk
from src.tools.web_search import WebResult, WebSearchProvider

ANALYST_SYSTEM_PROMPT = """You are a meticulous financial research analyst.
You receive research tasks from a financial advisor and return concise, sourced findings.

Rules:
- Always cite sources. Never invent URLs or titles.
- Stay general — do not name individual tickers or recommend specific securities.
- Be concrete about numbers (rules of thumb, percentages) where the source supports it.
- Reply in plain prose (3–6 sentences). Do not include the raw chunks.
"""


class AnalystAgent(BaseAgent):
    role = AgentRole.ANALYST

    def __init__(
        self,
        llm: LLMProvider,
        knowledge_store: KnowledgeStore | None = None,
        web_search: WebSearchProvider | None = None,
        name: str = "Analyst",
        system_prompt: str | None = None,
        kb_top_k: int = 4,
        web_top_k: int = 3,
        kb_score_threshold: float = 0.4,
    ) -> None:
        super().__init__(
            name=name,
            system_prompt=system_prompt or ANALYST_SYSTEM_PROMPT,
            llm=llm,
            tools={"knowledge_store": knowledge_store, "web_search": web_search},
        )
        self.knowledge_store = knowledge_store
        self.web_search = web_search
        self.kb_top_k = kb_top_k
        self.web_top_k = web_top_k
        self.kb_score_threshold = kb_score_threshold

    # -------- main loop entry --------

    def process(self, state: AdvisorState) -> AdvisorState:
        history = list(state.get("conversation_history", []))
        last = history[-1] if history else None
        if last is None or last.recipient is not AgentRole.ANALYST:
            return state

        query = last.content.strip()
        report = self.research(query)

        msg = AgentMessage(
            sender=AgentRole.ANALYST,
            recipient=AgentRole.ADVISOR,
            content=self._render_report_content(report),
            message_type=MessageType.REPORT,
            metadata={"report": json.loads(report.model_dump_json())},
        )
        history.append(msg)
        new_state: AdvisorState = {
            **state,
            "conversation_history": history,
            "analyst_findings": report,
            "status": ConversationStatus.ADVISE,
            "turn_count": state.get("turn_count", 0) + 1,
        }
        return new_state

    # -------- research pipeline --------

    def research(self, query: str) -> AnalystReport:
        kb_chunks = self._kb_search(query)
        web_results = self._web_search_if_needed(query, kb_chunks)
        sources = self._merge_sources(kb_chunks, web_results)
        if not sources:
            # Hard guarantee — AnalystReport requires non-empty sources.
            sources = [
                Source(
                    title="general financial planning principles",
                    snippet="Fallback: no specific source retrieved.",
                )
            ]
        findings = self._synthesize(query=query, kb_chunks=kb_chunks, web_results=web_results)
        confidence = self._estimate_confidence(kb_chunks, web_results)
        return AnalystReport(
            query=query,
            findings=findings,
            sources=sources,
            confidence=confidence,
        )

    def _kb_search(self, query: str) -> list[RetrievedChunk]:
        if self.knowledge_store is None:
            return []
        try:
            return self.knowledge_store.similarity_search(query, k=self.kb_top_k)
        except Exception:  # pragma: no cover - defensive
            return []

    def _web_search_if_needed(
        self, query: str, kb_chunks: list[RetrievedChunk]
    ) -> list[WebResult]:
        if self.web_search is None:
            return []
        # Only fall back to the web if the KB hits are weak or empty.
        strong_kb = [c for c in kb_chunks if c.score >= self.kb_score_threshold]
        if strong_kb and len(strong_kb) >= 2:
            return []
        try:
            return self.web_search.search(query, max_results=self.web_top_k)
        except Exception:  # pragma: no cover - degrade gracefully
            return []

    def _merge_sources(
        self, kb_chunks: list[RetrievedChunk], web_results: list[WebResult]
    ) -> list[Source]:
        sources: list[Source] = []
        seen: set[str] = set()
        for c in kb_chunks:
            key = f"kb::{c.source}"
            if key in seen:
                continue
            seen.add(key)
            sources.append(
                Source(title=f"knowledge_base::{c.source}", snippet=c.text[:240])
            )
        for w in web_results:
            key = f"web::{w.url or w.title}"
            if key in seen:
                continue
            seen.add(key)
            sources.append(Source(title=w.title, url=w.url, snippet=w.snippet[:240]))
        return sources

    def _synthesize(
        self,
        query: str,
        kb_chunks: list[RetrievedChunk],
        web_results: list[WebResult],
    ) -> str:
        kb_text = "\n\n".join(f"[KB: {c.source}] {c.text}" for c in kb_chunks)
        web_text = "\n\n".join(
            f"[WEB: {w.title}] {w.snippet}" for w in web_results
        )
        prompt = (
            f"Research task from the advisor:\n{query}\n\n"
            f"Knowledge base context:\n{kb_text or '(none)'}\n\n"
            f"Web search context:\n{web_text or '(none)'}\n\n"
            "Write a 3–6 sentence finding the advisor can use. "
            "Do not invent specifics; rely on the context."
        )
        out = self._call_llm(prompt, max_tokens=600).strip()
        if not out:
            out = (
                "No high-confidence material was retrieved. The advisor should rely "
                "on standard prudent planning guidance and confirm with the client."
            )
        return out

    def _estimate_confidence(
        self,
        kb_chunks: list[RetrievedChunk],
        web_results: list[WebResult],
    ) -> float:
        if kb_chunks:
            avg = sum(c.score for c in kb_chunks) / len(kb_chunks)
            base = max(0.4, min(0.95, avg))
        else:
            base = 0.4
        if web_results:
            base = min(0.95, base + 0.05)
        return round(base, 2)

    @staticmethod
    def _render_report_content(report: AnalystReport) -> str:
        sources_md = "\n".join(
            f"- {s.title}" + (f" ({s.url})" if s.url else "") for s in report.sources
        )
        return (
            f"Findings (confidence {report.confidence:.2f}):\n{report.findings}\n\n"
            f"Sources:\n{sources_md}"
        )


__all__ = ["AnalystAgent", "ANALYST_SYSTEM_PROMPT"]
