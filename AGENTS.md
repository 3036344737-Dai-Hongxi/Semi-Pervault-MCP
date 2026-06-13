# Pervault AGENTS.md

This repository uses `AGENTS.md` as the primary instruction format for coding agents.
Use this root file for repository-wide rules, then read the nearest nested `AGENTS.md` for subproject-specific guidance.

## Project shape

- MCP product line carved out of the Pervault main project, three areas:
  - `apps/mcp_host/`: thin MCP server bridging to the daemon (httpx + mcp only)
  - `backend/`: resident memory daemon (FastAPI + SQLite); the sole writer of `data.db`
  - `packages/memory_core/`: reusable memory kernel (no web-framework deps)
- Primary flow:
  - tool/text input -> memory storage -> hybrid retrieval -> graph/persona/reflection side effects
- The kernel already has real memory, graph, retrieval, persona, reflection, and sleep-agent behavior. Inspect current code before assuming any subsystem is a stub.
- Current roadmap reality:
  - the four-memory architecture core is already implemented
  - the next likely engineering theme is orchestration, reliability, and operability rather than inventing a new memory layer from scratch

## Source of truth

- Code beats docs when they disagree.
- Treat these files as orientation, not guaranteed truth:
  - `docs/codebase.md`
- Before changing behavior, inspect the current implementation in code.
- For architecture and roadmap context, prefer:
  - `docs/derivative/01-架构改造方案.md`
  - `docs/plan/four-memory-architecture-integration-plan.md`

## Workspace safety

- This repo may be dirty. Never revert unrelated changes.
- Limit edits to files needed for the task.
- Avoid broad cleanup, opportunistic refactors, or style-only churn unless explicitly requested.
- If you notice conflicting user changes in files you need, stop and explain the conflict instead of overwriting them.

## How to work

- Read first, then implement.
- Prefer the smallest complete change that preserves existing behavior.
- Preserve current API contracts unless the task explicitly changes them.
- Keep Chinese user-facing copy consistent with nearby UI text unless the task says otherwise.
- Add or update tests when behavior changes.
- Mention assumptions and residual risks in the final handoff.
- Before proposing a new subsystem, check whether the repo already has an in-place version of it. This codebase often has a working V1 already.

## Validation

- Run the narrowest relevant checks first.
- Typical checks (run from each package dir):
  - memory_core: `env PYTHONPYCACHEPREFIX=/tmp/pervault-pycache uv run python -m pytest tests/ -q`
  - backend: `env PYTHONPYCACHEPREFIX=/tmp/pervault-pycache uv run python -m pytest tests/ -q`
  - mcp_host: `uv run python -m pytest -q --ignore=tests/e2e_roundtrip.py`
- If a task only touches one package, do not default to running unrelated full-suite checks.

## Editing map

- For work under `backend/`, also read `backend/AGENTS.md`.
- If the task is to generate prompts for this repo, also read `AGENT_PROMPT_GUIDE.md`.

## Avoid stale assumptions

- Do not assume auth uses bearer-token-only semantics; current implementation uses session cookies.
- Do not assume retrieval is a single-file subsystem; it is already split into multiple modules.
