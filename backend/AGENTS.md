# Backend AGENTS.md

Follow the root `AGENTS.md` first, then these backend-specific rules.

## Stack and entry points

- FastAPI app entry: `backend/main.py`
- Database and schema: `backend/database.py`
- Routers live in `backend/routers/`
- Core domain logic lives in `backend/services/`
- Tests live in `backend/tests/`
- Current high-touch routers:
  - `routers/chat.py`
  - `routers/memory.py`
  - `routers/graph.py`
  - `routers/auth.py`

## Architecture guardrails

- This backend is SQLite-first and local-first. Do not introduce Redis, Celery, or external infrastructure unless the task explicitly requires it.
- Prefer existing application patterns over inventing a parallel subsystem.
- Before changing retrieval, inspect the split retrieval modules rather than assuming a single legacy retrieval file:
  - `services/retrieval_constants.py`
  - `services/retrieval_intent.py`
  - `services/retrieval_primitives.py`
  - `services/retrieval_context.py`
  - `services/retrieval_boot.py`
- Authentication is session/cookie based. Inspect `routers/auth.py` and `main.py` before changing auth behavior.
- `main.py` already starts lifecycle-managed background schedulers for consolidation, weight decay, and sleep agent. Check existing scheduler behavior before adding more startup loops.

## Database rules

- Inspect current connection usage before adding new DB access patterns.
- Preserve SQLite compatibility and migration safety for existing local databases.
- Favor short transactions.
- Do not hold long-running external calls inside transactions.
- Avoid `SELECT *` in new code unless there is a strong reason.
- When adding schema, update the database initialization and any compatibility/migration helpers together.
- The app currently relies on shared SQLite access for request paths. If you change orchestration or background execution, reason explicitly about request-path access vs long-lived worker access.

## Background work

- The codebase already contains multiple async side effects around memory creation, chat persistence, consolidation, decay, and sleep-agent behavior.
- Keep task ordering and idempotency in mind before adding new side effects.
- Memory creation side effects are currently triggered from both `routers/memory.py` and `routers/chat.py`. If you change write-path orchestration, inspect both places and keep them aligned.
- If changing background execution behavior, verify how it interacts with:
  - memory kind correction
  - admission scoring
  - persona extraction
  - graph extraction
  - embedding indexing
  - sleep-agent and scheduled jobs

## API conventions

- Business routes live under `/api/`.
- Preserve response shape compatibility unless the task explicitly changes the contract.
- For new operational or internal endpoints, require auth and keep the scope minimal.
- Export behavior is already security-sensitive and audited. Treat `memory/export` changes as high-risk.

## Testing

- Default backend validation:
  - `env PYTHONPYCACHEPREFIX=/tmp/pervault-pycache uv run python -m pytest tests/ -q`
- When changing a narrow module, run the most relevant targeted tests first, then broaden if needed.
- Add regression tests for bug fixes and state-machine changes.
- When touching retrieval, preference revision, admission, persona, or sleep-agent logic, prefer adding focused tests near the corresponding existing backend test modules rather than relying only on broad regression.

## Practical reading order

For most backend tasks, start here:
1. `backend/main.py`
2. relevant router in `backend/routers/`
3. corresponding service in `backend/services/`
4. `backend/database.py`
5. existing tests in `backend/tests/`

## Current-reality reminders

- Chat sessions currently load from `chat_messages`; do not assume pagination already exists.
- Memory export includes memories, facts, persona, reflections, revision log, and graph data.
- Background side effects currently include graph extraction, embedding indexing, kind correction, emotion scoring, importance scoring, and admission scoring.
