<p align="center">
  <img src="assets/readme/pixel-memory.svg" alt="Semi-Pervault MCP pixel memory banner" width="100%" />
</p>

<h1 align="center">Semi-Pervault MCP</h1>

<p align="center">
  <strong>Local-first long-term memory for AI agents.</strong>
  <br />
  A private memory layer for Claude Desktop, Cursor, and any MCP-compatible client.
</p>

<p align="center">
  <a href="https://modelcontextprotocol.io/"><img alt="MCP" src="https://img.shields.io/badge/MCP-ready-38bdf8?style=for-the-badge" /></a>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.11%2B-34d399?style=for-the-badge" />
  <img alt="Storage" src="https://img.shields.io/badge/SQLite-local--first-fbbf24?style=for-the-badge" />
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/License-MIT-a78bfa?style=for-the-badge" /></a>
</p>

<p align="center">
  <a href="#quick-start">Quick start</a>
  &nbsp;&middot;&nbsp;
  <a href="#use-cases">Use cases</a>
  &nbsp;&middot;&nbsp;
  <a href="#mcp-tools">MCP tools</a>
  &nbsp;&middot;&nbsp;
  <a href="#architecture">Architecture</a>
  &nbsp;&middot;&nbsp;
  <a href="docs/launch-kit.md">Launch kit</a>
  &nbsp;&middot;&nbsp;
  <a href="#development">Development</a>
</p>

<br />

Semi-Pervault MCP is a local memory server built around Pervault's four-layer memory architecture. It stores important facts, preferences, decisions, project context, and relationship notes in a local SQLite database, then exposes that memory as MCP tools for agentic workflows.

The goal is simple: let your AI tools remember useful context across sessions without sending your private memory store to a hosted service.

<br />

## Why It Exists

Most AI tools are brilliant in the moment and forgetful by design. Semi-Pervault MCP gives them a durable, inspectable memory surface:

| Need | What Semi-Pervault Provides |
|---|---|
| Remember me across sessions | Long-term local storage for facts, preferences, decisions, and progress |
| Recall the right context | Hybrid retrieval across facts, full text, embeddings, and graph context |
| Keep memory explainable | Evidence trails, admission scores, and revision audit history |
| Stay private by default | SQLite on your machine, with a loopback-only daemon |
| Avoid MCP process fragility | A thin MCP bridge forwards to a resident daemon that owns background work |

<br />

## What It Does

- **Stores long-term memory** for facts, preferences, decisions, events, relationships, and project state.
- **Retrieves with hybrid search** across structured facts, full-text search, embeddings, and graph context.
- **Builds stable persona context** from repeated preferences and high-confidence traits.
- **Extracts knowledge graph links** between people, projects, topics, and events.
- **Generates reflections** through a background consolidation loop.
- **Explains beliefs** with supporting memories, confidence, admission scores, and audit trail.
- **Keeps data local-first** with SQLite storage and a loopback daemon.

<br />

## Use Cases

Semi-Pervault MCP is most useful when your AI assistant needs continuity:

| Use case | Example memory |
|---|---|
| Personal assistant context | "I prefer concise technical explanations with concrete next steps." |
| Project continuity | "Semi-Pervault MCP uses a thin MCP bridge and a resident FastAPI daemon." |
| Research and study notes | "I am studying data science and engineering at HKU." |
| Decision tracking | "We chose SQLite-first storage to keep memory local and inspectable." |
| Relationship and collaboration context | "Alice owns backend reliability decisions for Project X." |

Try asking your MCP client:

```text
Remember that I am building a local-first memory MCP server called Semi-Pervault.
```

Then later:

```text
What do you remember about my memory MCP project, and why?
```

The second question can use `memory_search`, `persona_get`, or `memory_why` depending on the context.

<br />

## Architecture

This repository is the MCP product slice of the larger Pervault memory system. The important design choice is that MCP is only a bridge. The resident backend daemon owns the database, the write pipeline, and all background intelligence.

| Area | Role |
|---|---|
| `apps/mcp_host/` | Thin MCP bridge. It exposes tools over stdio and forwards calls to the local daemon. |
| `backend/` | Resident FastAPI daemon. It owns HTTP APIs, auth, background jobs, and all writes to `data.db`. |
| `packages/memory_core/` | Framework-free memory kernel: schema, models, write pipeline, retrieval, graph, persona, reflections, and consolidation. |

```mermaid
flowchart TB
  subgraph Clients["AI clients and local runtimes"]
    Claude["Claude Desktop / Codex / Cursor<br/>model account or API configured"]
    Local["OpenClaw / Hermes local runtime<br/>local model already running"]
    Browser["Future browser extension"]
    Web["Optional local web UI"]
  end

  subgraph Adapters["Thin adapters"]
    MCP["apps/mcp_host/server.py<br/>MCP stdio bridge<br/>no DB, no background loops"]
    HTTP["Browser or web HTTP clients"]
  end

  subgraph Daemon["backend resident memory daemon"]
    FastAPI["backend/main.py<br/>FastAPI lifespan"]
    CoreAPI["/core/* loopback API<br/>127.0.0.1 + X-Pervault-Token"]
    CookieAPI["/api/* browser API<br/>session cookie auth"]
    Runtime["MemoryRuntime<br/>background_jobs + consolidation<br/>weight_decay + sleep_agent"]
  end

  subgraph Kernel["packages/memory_core memory kernel"]
    Store["memory_service<br/>store, update, facts, admission jobs"]
    Retrieval["retrieval_context<br/>intent routing + hybrid retrieval"]
    Graph["graph_pipeline + graph_retrieval<br/>entities and relationships"]
    Persona["persona_service + sleep_agent<br/>stable user traits"]
    Reflection["sleep_agent + provenance<br/>reflections and why explanations"]
  end

  subgraph Storage["Local SQLite: ~/.pervault/data.db"]
    Items["memory_items<br/>raw episodic memories"]
    Facts["structured_facts<br/>normalized facts"]
    PersonaTable["user_persona<br/>stable traits"]
    ReflectionTable["memory_reflection<br/>higher-level insights"]
    GraphTables["graph_nodes + graph_edges<br/>knowledge graph"]
    Search["memory_fts + vec_items<br/>FTS5 and vector index"]
    Jobs["background_jobs + scheduler logs<br/>async enrichment"]
  end

  Providers["Optional OpenAI-compatible LLM and embedding providers<br/>enrichment, routing, embeddings"]

  Claude --> MCP
  Local --> MCP
  Browser --> HTTP
  Web --> HTTP
  MCP --> CoreAPI
  HTTP --> CoreAPI
  HTTP --> CookieAPI
  CoreAPI --> FastAPI
  CookieAPI --> FastAPI
  FastAPI --> Runtime
  FastAPI --> Store
  FastAPI --> Retrieval
  FastAPI --> Graph
  FastAPI --> Persona
  FastAPI --> Reflection
  Runtime --> Store
  Runtime --> Graph
  Runtime --> Persona
  Runtime --> Reflection
  Store --> Items
  Store --> Facts
  Store --> Jobs
  Retrieval --> Items
  Retrieval --> Facts
  Retrieval --> Search
  Retrieval --> GraphTables
  Graph --> GraphTables
  Persona --> PersonaTable
  Reflection --> ReflectionTable
  Store -.-> Providers
  Runtime -.-> Providers
```

### Architecture invariants

| Invariant | Why it matters |
|---|---|
| One database writer | The backend daemon is the only process that writes to SQLite, so MCP client restarts cannot corrupt background work. |
| Thin MCP bridge | `apps/mcp_host/server.py` only reads `~/.pervault/core_token` and calls `127.0.0.1:8000/core/*`. |
| Runtime owns background work | `MemoryRuntime` starts the job worker, consolidation loop, weight decay, and sleep agent from one lifecycle point. |
| Local auth is split by surface | MCP and local adapters use `X-Pervault-Token`; browser-facing `/api/*` routes use session cookies. |
| Provider keys are optional | Basic memory storage works locally; LLM and embedding keys unlock enrichment, vector recall, persona extraction, graph extraction, and stronger reflections. |

### Core data model

| Layer | SQLite tables | Purpose |
|---|---|---|
| Layer 1: episodic memory | `memory_items` | Raw memories, source tags, weights, admission scores, revision state. |
| Layer 2: structured facts | `structured_facts` | Normalized facts extracted from durable memories. |
| Layer 3: persona | `user_persona` | Stable user traits backed by source memory evidence. |
| Layer 4: reflections | `memory_reflection` | Higher-level insights generated by the sleep agent. |
| Graph context | `graph_nodes`, `graph_edges` | Entity and relationship memory for people, projects, topics, and events. |
| Retrieval indexes | `memory_fts`, `vec_items` | Full-text and vector search surfaces for hybrid recall. |
| Evidence and audit | `memory_admission_log`, `preference_revision_log` | Admission evidence, confidence history, and correction trail. |
| Async orchestration | `background_jobs`, scheduler logs | Persistent queue and run history for enrichment, consolidation, and sleep-agent work. |

The write path is intentionally boring: tools and UI call the daemon, the daemon calls `memory_core`, and `memory_core` persists to local SQLite. That separation keeps the MCP process replaceable while the memory engine stays alive.

<br />

## MCP Tools

| Tool | Purpose |
|---|---|
| `memory_store` | Store a long-term memory item. |
| `memory_search` | Retrieve relevant memories with hybrid search. |
| `memory_graph` | Query graph context for a topic. |
| `memory_update` | Revise an existing memory. |
| `persona_get` | Read stable user persona traits. |
| `reflections_list` | Read higher-level reflections generated by the background agent. |
| `memory_why` | Explain a belief with supporting memories, scores, and audit trail. |
| `memory_stats` | Show memory counts and admission statistics. |

<br />

## Quick Start

### Important: model/API setup

Semi-Pervault MCP is a memory layer, not a model provider. You still need an AI client or runtime that can call MCP tools.

There are two separate API layers:

| Layer | What you need |
|---|---|
| **AI client / model runtime** | Claude Desktop, Codex, and similar cloud-model clients must be configured with their own account or API access before they can use this MCP server. OpenClaw / Hermes-style local runtimes do not need a cloud API for the client side if the local model is already installed and running. |
| **Semi-Pervault enrichment** | `OPENAI_API_KEY` and embedding keys are optional for basic storage, but needed for LLM enrichment, embeddings, stronger hybrid retrieval, and some background intelligence. Without them, the daemon still stores memories and falls back to non-vector retrieval paths. |

If the MCP client cannot call a model yet, Semi-Pervault will not fix that part. Start your client/runtime first, then connect this memory server.

### Requirements

- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/getting-started/)

### 1. Start the local daemon

From the project root, double-click `RUN.command`, or run:

```bash
cd backend
cp .env.example .env
uv run python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

The daemon binds to `127.0.0.1` only.

On first use it creates a local pairing token at:

```text
~/.pervault/core_token
```

The default database path is:

```text
~/.pervault/data.db
```

Set `PERVAULT_DB_PATH` if you want to store the database somewhere else.

### 2. Connect an MCP client

Claude Desktop example:

```json
{
  "mcpServers": {
    "pervault-memory": {
      "command": "<absolute path to uv>",
      "args": [
        "--directory",
        "<absolute project path>/apps/mcp_host",
        "run",
        "python",
        "server.py"
      ]
    }
  }
}
```

Find your `uv` path with:

```bash
which uv
```

Restart your MCP client after editing its config.

Bridge-specific details live in [`apps/mcp_host/README.md`](apps/mcp_host/README.md).

<br />

## Configuration

The backend config template is [`backend/.env.example`](backend/.env.example).

For a first local test, you can leave provider keys empty and use the basic memory path. Add LLM and embedding keys when you want enrichment, vector retrieval, persona extraction, graph extraction, and stronger reflections.

| Variable | Description |
|---|---|
| `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `LLM_MODEL` | LLM settings used by enrichment and routing. |
| `GEMINI_API_KEY`, `EMBEDDING_MODEL`, `EMBEDDING_BASE_URL`, `EMBEDDING_DIM` | Embedding settings for vector retrieval. |
| `PERVAULT_DB_PATH` | SQLite database location. Defaults to `~/.pervault/data.db` in daemon usage. |
| `AUTH_PASSWORD`, `API_KEY`, `SESSION_SECRET` | Backend auth/session settings for HTTP routes. |
| `CONSOLIDATION_SCHEDULER_ENABLED`, `WEIGHT_DECAY_SCHEDULER_ENABLED`, `SLEEP_AGENT_ENABLED` | Background maintenance toggles. |

If embedding credentials are not configured, retrieval falls back to non-vector search paths.

<br />

## Project Map

```text
.
├── apps/mcp_host/          # MCP stdio bridge
├── backend/                # FastAPI daemon and HTTP routes
├── packages/memory_core/   # Framework-free memory kernel
├── docs/                   # Architecture and planning notes
├── RUN.command             # macOS helper to start the daemon
└── README.md
```

<br />

## Development

Run checks from each package directory:

```bash
# memory kernel
cd packages/memory_core
env PYTHONPYCACHEPREFIX=/tmp/pervault-pycache uv run python -m pytest tests/ -q

# backend daemon
cd backend
env PYTHONPYCACHEPREFIX=/tmp/pervault-pycache uv run python -m pytest tests/ -q

# MCP bridge
cd apps/mcp_host
uv run python -m pytest -q --ignore=tests/e2e_roundtrip.py
```

<br />

## Design Notes

- The daemon is local-first and SQLite-first.
- The MCP bridge is deliberately thin and stateless.
- `memory_core` is framework-free and guarded against web-framework imports.
- Long-running side effects are handled by the daemon, not the MCP stdio process.
- Data is private by default, but external LLM or embedding providers may receive text you send to them through configured API keys.

Further reading:

- [`docs/derivative/01-架构改造方案.md`](docs/derivative/01-架构改造方案.md)
- [`docs/plan/four-memory-architecture-integration-plan.md`](docs/plan/four-memory-architecture-integration-plan.md)
- [`packages/memory_core/README.md`](packages/memory_core/README.md)
- [`docs/launch-kit.md`](docs/launch-kit.md)

<br />

## License

MIT. See [`LICENSE`](LICENSE).
