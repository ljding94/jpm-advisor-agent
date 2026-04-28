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
