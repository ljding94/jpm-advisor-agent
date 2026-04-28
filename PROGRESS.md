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
