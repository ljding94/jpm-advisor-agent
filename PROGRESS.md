# PROGRESS.md

Build trajectory for the LangGraph multi-agent financial advisor. Newest entries at the bottom.

## 2026-04-28 — Step 1: scaffold
- Created directory tree (`src/`, `tests/`, `data/{knowledge_base,personas}`, `examples/`).
- Wrote `requirements.txt` (pinned), `pyproject.toml`, `.env.example`.
- Initialized git repo and pushed to `github.com/ljding94/jpm-advisor` (private).

## 2026-04-28 — Step 2: schemas
- `src/schemas/{messages,client_profile,advice}.py` with Pydantic v2 models.
- Routing constraint: `AgentMessage` rejects illegal (sender, recipient) pairs (Analyst↔Client blocked).
- `AdviceOutput` force-appends standard disclaimer if missing.
- 17/17 schema tests pass (`tests/test_schemas.py`).

## 2026-04-28 — Step 3: knowledge base + personas + ingestion
- 6 finance markdown docs in `data/knowledge_base/` (asset allocation, risk tolerance, retirement, diversification, tax-advantaged accounts, emergency fund + insurance).
- 3 persona JSONs in `data/personas/` (Margaret/David/Priya) — load and validate as `ClientProfile`.
- `EmbeddingProvider` interface with `OpenRouterEmbeddings` and `LocalEmbeddings` (sentence-transformers).
- `KnowledgeStore` (Chroma persistent) with chunker, `ingest_directory`, `similarity_search`.
- CLI: `python -m src.tools.ingest`.
- 28/28 tests pass (schemas + personas + knowledge store with FakeEmbedder).

## 2026-04-28 — Steps 4–5: web search + LLM/embedding providers
- `WebSearchProvider` interface; `DDGSearchProvider` (DuckDuckGo) with retry, rate-limit, mockable; `FakeWebSearchProvider` for tests.
- `LLMProvider` interface; `OpenRouterLLM` (OpenAI SDK pointed at OpenRouter, retried).
- `EmbeddingProvider` already wired in step 3; added explicit get/fallback tests.
- 6 web-search tests + 9 provider tests pass (43 total so far).

## 2026-04-28 — Steps 6–11: agents, factory, strategies, guardrails
- `BaseAgent` (abstract) + `AgentFactory` with lazy concrete imports.
- `ClientAgent` (loads persona, answers questions, [CONFIRM]/[REJECT] advice → drives state machine).
- `AnalystAgent` (KB → web fallback → AnalystReport with non-empty sources, never speaks to Client).
- `AdvisorAgent` (Mediator: ask_client/dispatch_analyst/draft_advice/finalize; synthesizes via RiskStrategy).
- `RiskStrategy` (Strategy pattern): Conservative/Moderate/Aggressive with allocations + headline advice.
- Guardrails: PII redaction (SSN/CC/account/email/phone), output filter (banned phrases, tickers, disclaimer), limits (turns/cost).
- 79 tests pass total (schemas 17, personas 4, knowledge 7, web 6, providers 9, agents 21, strategies 11, guardrails 25, ingestion impl).

## 2026-04-28 — Steps 12–13: graph + integration
- `src/graph/state.py`: `AdvisorState` TypedDict + `ConversationStatus` enum + `initial_state`.
- `src/graph/routing.py`: `route_next(state)` dispatches to client/advisor/analyst by last-message recipient; ends on RESOLVED/TERMINATED.
- `src/graph/builder.py`: `build_graph()` wires LangGraph `StateGraph` with three nodes; per-node wrapper enforces hard limits and marks `status=TERMINATED`.
- Bug found and fixed: when Advisor receives an analyst REPORT, it now immediately drafts advice in the same turn (previously left the conversation in an infinite advisor→advisor loop).
- Bug found and fixed: limit enforcement moved entirely to the node wrapper; routing only checks status. Without this, hitting MAX_TURNS exited but didn't mark TERMINATED.
- Integration tests: all three personas (Margaret/David/Priya) drive the graph to RESOLVED; MAX_TURNS termination path verified.
- 111/111 tests pass.
