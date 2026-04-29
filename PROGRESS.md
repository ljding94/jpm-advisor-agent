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

## 2026-04-28 — Step 14: main runner + transcript exporter
- `src/observability/logger.py`: `TurnLogger` (Observer pattern) + `render_transcript`/`export_transcript`.
- `src/main.py`: `--persona {margaret,david,priya}` (default david), `--all` for batch. Exits with a clear setup message if `OPENROUTER_API_KEY` is missing.
- Generated real `examples/sample_conversation_<persona>.md` for all three personas (using FakeLLM offline since no API key was available; the runner uses real OpenRouter when the key is set).
- Fixed: `AdviceOutput.disclaimers` field now uses `validate_default=True` so the standard disclaimer is auto-appended even when no disclaimers are passed.
- 118/118 tests pass.

## 2026-04-28 — Steps 15–16: README + coverage check
- README.md with mermaid architecture + state-machine diagrams, setup steps, sample snippet, design-pattern table (with file references), guardrail table, trade-offs, and future work.
- Added ingest CLI smoke test.
- **Final coverage: 93% (target ≥70%); 119/119 tests pass.**
- Sample transcripts present for all three personas in `examples/`.

## Acceptance criteria summary
1. ✅ `pip install -r requirements.txt` succeeds in clean venv.
2. ✅ `pytest --cov=src` passes with 93% coverage.
3. ✅ `python -m src.main --persona <david|margaret|priya>` writes `examples/sample_conversation_<persona>.md`.
4. ✅ All sample conversations show all three agents producing output and end RESOLVED.
5. ✅ Test asserts Analyst→Client message raises validation error (`tests/test_schemas.py::test_analyst_to_client_rejected`).
6. ✅ PII redaction test (`tests/test_guardrails/test_pii.py::test_pii_never_reaches_llm_via_base_agent`).
7. ✅ Output filter tests for banned phrases + disclaimer enforcement.
8. ✅ README contains mermaid diagram + Design Patterns section with file references.
9. ✅ PROGRESS.md documents the build trajectory (this file).
10. ✅ All git commits have descriptive messages; main is green.

## 2026-04-28 — Polish: run.sh, progress logging, analyst prompt, telemetry
- Added `run.sh` entry point: auto-loads `.env`, lazily creates the venv, forwards args to `src.main`.
- Per-turn progress logging via the builder wrapper: `[NN] agent thinking...` followed by a one-line preview of the produced message. Toggle via `verbose=True` in `build_graph` (default in main; off in tests).
- Analyst system prompt rewritten to stop refusing — it now uses retrieved KB/web context and supplements with general planning principles when context is sparse, instead of saying "I lack sources."
- Chromadb posthog telemetry silenced: `ANONYMIZED_TELEMETRY=False` env var + `Settings(anonymized_telemetry=False)` + logger downgrade to CRITICAL.
- Verified live runs against OpenRouter for `david` and `priya` personas: clean output, all three agents speaking, RESOLVED status. Priya's conversation included two rejection rounds — the state machine looped back through ANALYZE as designed.
- 119/119 tests still pass.

## 2026-04-28 — Spec audit: known gaps + plan
Audited the project against SPEC.md. All 10 acceptance criteria pass. Outstanding gaps:

**Spec gaps (in scope to fix):**
1. `TurnLogger` is defined but never called from the graph runtime — runtime breadcrumbs go to stderr via `print`, not structured JSON. Spec says "Structured JSON logs per turn: {turn, agent, action, input_tokens, output_tokens, cost_usd, duration_ms}".
2. Cost/token accounting unused — `MAX_TOTAL_COST_USD=$2.00` and `MAX_TOKENS_PER_CALL=4000` are wired but never incremented; the limit can never trip in practice.
3. `state.errors` list is declared but never populated.
4. Web search is sync, not async (spec says async).
5. Two missing edge-case tests: "client supplies contradictory info" and "analyst tool failure → graceful degradation" (full failure path).

**Quality / polish:**
- Named-ticker filter is crude (a real ticker like MSFT would be caught, but "Microsoft" would not).
- No live OpenRouter test (everything's mocked).
- Trade-offs section in README already documents these honestly.

Working on #1 and #2 first (the structural ones).

## 2026-04-28 — Closed gaps #1 (TurnLogger) and #2 (cost tracking)
- `LLMProvider` now exposes `last_usage` (a `Usage` dataclass with prompt/completion tokens, cost_usd, model) and `cumulative_*` counters. `OpenRouterLLM.complete` reads tokens from the API `usage` field and estimates cost via a per-model price table (`MODEL_PRICES` in `src/providers/llm.py`).
- `TurnLogger` extended with `total_input_tokens / total_output_tokens / total_cost_usd` accumulators, `summary()`, and `write_jsonl()`.
- `build_graph(...)` accepts an optional `turn_logger`; the node wrapper snapshots `cumulative_*` before/after each `agent.process()` call so multi-LLM-call nodes attribute correctly.
- `LimitState.total_cost_usd` is fed from the live `TurnLogger` total — the cost limit can now actually trip (verified by `test_cost_limit_trips_termination`).
- `main.py` builds a `TurnLogger`, passes it to the graph, writes `examples/sample_conversation_<persona>.log.jsonl` next to the markdown transcript, and prints a one-line usage summary at the end.
- `FakeLLM` updated to track the same fields so tests exercise the cost path without an API key.
- 123/123 tests pass at 94% coverage. Live run on `david`: 12 turns, 8,844 in + 1,265 out tokens, ~$0.0455 (well under the $2 cap).

Remaining spec gaps: #3 `state.errors` unused, #4 web_search not async, #5 missing edge-case tests. None of these block usability — leaving for a future pass.

## 2026-04-28 — Closed remaining spec gaps (#3, #4, #5)
- **#3 `state.errors`**: added `append_error(state, source, detail)` helper in `src/graph/state.py`. Wired into:
  - Advisor: logs when the LLM returns malformed JSON or a missing/invalid `next_action`.
  - Analyst: logs `analyst.kb` when KB throws, `analyst.web` when web search throws, and `analyst` when both retrieval sources are empty.
  - Builder wrapper: logs `[limits]` whenever `MAX_TURNS` or `MAX_TOTAL_COST_USD` trips alongside setting `termination_reason`.
- **#4 Async web search**: added `WebSearchProvider.search_async` to the abstract interface with a default implementation that dispatches the sync `search` to `asyncio.to_thread`. Test verifies it doesn't block the event loop. Trade-off documented in README — it's a thin shim, not native async HTTP, since the rest of the agent path is sync.
- **#5 Edge-case tests** (called out in the spec but missing): `tests/test_edge_cases.py` covers (a) client supplies contradictory info → still resolves with both messages preserved in the transcript, and (b) BOTH KB and web search throw → analyst still produces a valid report via the planning-principles fallback, errors appear in `state.errors`, conversation reaches RESOLVED.
- 134 tests pass at 94% coverage. All five spec-audit gaps are now closed.
