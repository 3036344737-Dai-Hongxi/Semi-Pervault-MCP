# Contributing to Semi-Pervault MCP

Thanks for your interest! Semi-Pervault MCP is a local-first memory MCP server. This guide covers setup, conventions, and how to submit changes.

## Project layout

- `apps/mcp_host/` — thin MCP server (`httpx` + `mcp`); forwards tool calls to the daemon.
- `backend/` — resident FastAPI memory daemon; the only writer of `data.db`.
- `packages/memory_core/` — framework-free memory kernel (no web-framework imports).

## Prerequisites

- Python ≥ 3.11
- [uv](https://docs.astral.sh/uv/getting-started/)

## Run it locally

```bash
cd backend
cp .env.example .env        # fill keys for enrichment; optional for basic storage
uv run python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Then point your MCP client at `apps/mcp_host` (see the README "Quick start").

## Tests (run the relevant suite before opening a PR)

```bash
cd packages/memory_core && uv run python -m pytest -q                                # kernel (largest suite + import-boundary guard)
cd backend            && uv run python -m pytest tests/ -q                           # daemon
cd apps/mcp_host      && uv run python -m pytest -q --ignore=tests/e2e_roundtrip.py  # MCP bridge
```

## Conventions

- **Verify with tests.** Every change ends with the relevant suite passing. Don't claim "done" without a green run.
- **Small, focused PRs.** One cohesive change per PR; avoid drive-by refactors.
- **Kernel stays framework-free.** `packages/memory_core/` must not import `fastapi` / `slowapi` / web stuff (an import-boundary test enforces this).
- **One writer.** Only the daemon writes `data.db`; the MCP host stays a thin forwarder.
- **Local-first.** No Redis / Celery / external DB — SQLite is the source of truth.

## Submitting a change

1. Fork and create a branch (`fix/...`, `feat/...`).
2. Make your change; add or update tests when behavior changes.
3. Run the relevant test suite(s) — all green.
4. Open a PR describing what changed, why, and the test output.

## Good first issues

Check issues labeled `good first issue`. Starter ideas: example client configs (Cursor, etc.), a Dockerfile for the daemon, and docs improvements.
