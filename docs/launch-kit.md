# Semi-Pervault MCP Launch Kit

This document collects the pieces that make the repository easier to understand, search, and share.

## Repository Description

Use this for the GitHub About description:

```text
Local-first long-term memory MCP server for AI agents, with SQLite storage, hybrid retrieval, persona extraction, graph context, and explainable memory.
```

Shorter version:

```text
Local-first long-term memory MCP server for AI agents.
```

## Setup Caveat To Mention

Use this whenever you share the project, because it prevents the most common setup confusion:

```text
Semi-Pervault MCP is the memory layer, not the model runtime. Claude Desktop / Codex-style clients still need their own account or API setup. OpenClaw / Hermes-style local runtimes do not need a cloud API on the client side if the local model is already running. Semi-Pervault's own LLM and embedding keys are optional for basic storage, but recommended for enrichment, vector retrieval, persona extraction, and graph context.
```

## Suggested Topics

Add these in the repository About panel:

```text
mcp
ai-agents
memory
local-first
sqlite
knowledge-graph
claude
cursor
python
fastapi
long-term-memory
personal-ai
```

## Profile README Template

Create a public repository named exactly like your GitHub username, then put this in its `README.md`.

```markdown
# Hi, I am Henry

Data science and engineering student at HKU, building local-first AI memory tools.

## Featured Project

### Semi-Pervault MCP

Local-first long-term memory MCP server for AI agents.

- SQLite-backed private memory
- MCP bridge for Claude Desktop / Cursor
- Hybrid retrieval, persona extraction, graph context, and explainable memory

Repo: https://github.com/3036344737-Dai-Hongxi/Semi-Pervault-MCP

## Interests

- AI agents
- Local-first software
- Personal memory systems
- Retrieval and knowledge graphs
- Python backend engineering
```

## Username Notes

Current username:

```text
3036344737-Dai-Hongxi
```

It is usable, but it is long and hard to say out loud. For a public portfolio, a shorter username is easier to remember, search, and put on resumes.

Better directions:

```text
daihongxi
henrydai
henry-dai
daihongxi-ai
henrydai-ai
```

Before changing a GitHub username, remember:

- Old profile and repository URLs usually redirect, but it is still better to update links you control.
- Local git remotes may need `git remote set-url`.
- Package names, website links, badges, and README links should be checked after the rename.
- If the current username is already used in school or project submissions, change it only when you are ready to update those links.

## Share Copy

### One-liner

```text
I built Semi-Pervault MCP: a local-first long-term memory server for AI agents, with SQLite storage, hybrid retrieval, persona extraction, graph context, and explainable memory.
```

### Longer post

```text
I just published Semi-Pervault MCP, a local-first memory layer for MCP-compatible AI clients like Claude Desktop and Cursor.

It stores long-term memory in local SQLite, exposes memory tools over MCP, and supports hybrid retrieval, persona extraction, graph context, background reflections, and evidence tracing.

Important setup note: it is the memory layer, not the model runtime. Claude Desktop / Codex-style clients still need their own account or API setup; OpenClaw / Hermes-style local runtimes can run without a cloud API on the client side if the local model is already running.

The main design choice is keeping the MCP process thin: it forwards tool calls to a resident local daemon that owns storage and background work.

Repo: https://github.com/3036344737-Dai-Hongxi/Semi-Pervault-MCP
```

## Where To Share

Start with communities that already care about AI agents, local-first tools, or MCP:

- GitHub profile pinned repositories
- LinkedIn portfolio post
- X / Twitter build log
- Reddit communities such as LocalLLaMA, ClaudeAI, Python, or self-hosted AI groups
- Discord communities around Claude, Cursor, MCP, local AI, and agent tooling
- HKU project portfolio, resume, or personal website

## Repository Polish Checklist

- [ ] Add repository topics.
- [ ] Add a social preview image in repository settings.
- [ ] Pin the repository on the GitHub profile.
- [ ] Create a profile README.
- [ ] Add one short demo GIF or screenshot if possible.
- [ ] Add a `good first issue` or roadmap item for outside contributors.
- [ ] Publish one short post explaining the problem and the design choice.
- [ ] Link the repo from resume, portfolio, and LinkedIn.

## MCP Directories To Submit To

These are where people actively look for MCP servers — the highest-ROI distribution for this project.

- `punkpeye/awesome-mcp-servers` (GitHub) — open a PR adding the entry below
- Glama — https://glama.ai/mcp (indexes public MCP repos)
- Smithery — https://smithery.ai
- PulseMCP — https://www.pulsemcp.com
- mcp.so
- cursor.directory (MCP section)
- Official servers list at https://modelcontextprotocol.io

### Ready entry for awesome-mcp-servers

```text
- [Semi-Pervault MCP](https://github.com/3036344737-Dai-Hongxi/Semi-Pervault-MCP) 🐍 🏠 - Local-first long-term memory for AI agents: SQLite storage, hybrid retrieval, persona extraction, knowledge-graph context, and explainable memory.
```

(🐍 = Python, 🏠 = local service, per that list's legend.)

## Launch Posts

### Show HN

Title:

```text
Show HN: Semi-Pervault – local-first long-term memory MCP server
```

Body:

```text
Semi-Pervault is a local-first memory layer for MCP clients (Claude Desktop, Cursor, etc.). It stores long-term memory in local SQLite and exposes it as MCP tools, so your AI tools remember context across sessions without shipping your memory to a hosted service.

The design choice I care about: the MCP process is a thin bridge. It forwards tool calls to a resident local daemon that owns storage and background work (enrichment, consolidation, decay, a sleep-agent). So memory survives MCP clients starting and stopping, and there is a single writer to the DB.

Memory is four layers — raw episodic, structured facts, stable persona, higher-level reflections — with hybrid retrieval (facts + FTS + vectors + graph) and a memory_why tool that returns the evidence trail behind a belief.

LLM/embedding keys are optional (used only for enrichment/vector recall); basic storage works without them.

Repo: https://github.com/3036344737-Dai-Hongxi/Semi-Pervault-MCP
Feedback welcome, especially on the daemon/bridge split and the retrieval design.
```

### Reddit (r/LocalLLaMA, r/mcp, r/ClaudeAI)

Title:

```text
I built a local-first long-term memory MCP server (SQLite, hybrid retrieval, explainable memory)
```

Body: same gist as the Show HN post, plus this setup caveat near the end:

```text
Setup note: Semi-Pervault is the memory layer, not the model runtime. Claude Desktop / Cursor still use their own accounts; local runtimes do not need a cloud API on the client side. Semi-Pervault's own LLM/embedding keys are optional.
```

## v0.1.0 Release Notes (draft)

```text
Semi-Pervault MCP v0.1.0 — first public release

Local-first long-term memory MCP server for AI agents.

Highlights
- 8 MCP tools: memory_store, memory_search, memory_graph, memory_update,
  persona_get, reflections_list, memory_why, memory_stats
- Four-layer memory: raw episodic, structured facts, persona, reflections
- Hybrid retrieval: structured facts + full-text + vectors + graph context
- Explainable memory: memory_why returns evidence, confidence, admission scores
- Resident daemon owns storage + background jobs; MCP host is a thin bridge
- Local-first: SQLite on your machine, loopback-only daemon; provider keys optional

Install: see README "Quick start".
```

## Demo GIF Storyboard (10–15s)

Record inside your MCP client (e.g. Claude Desktop); keep it short:

1. (1s) Chat open.
2. (3s) Type: "Remember that I'm building a local-first memory MCP server called Semi-Pervault." → `memory_store` fires.
3. (2s) Show the stored confirmation.
4. (1s) Start a new chat / fresh session.
5. (4s) Type: "What do you remember about my project, and why?" → `memory_search` / `memory_why` returns it with the evidence trail.
6. (2s) End on the recalled answer.

Capture: macOS screen recording → convert with `gifski` or `ffmpeg` → save to `assets/readme/demo.gif` → embed at the top of the README.

## Good First Issues (drafts to file on GitHub)

1. **Example MCP client configs beyond Claude Desktop** — add JSON for Cursor and other clients to `apps/mcp_host/README.md`. Labels: `good first issue`, `documentation`.
2. **Dockerfile for the daemon** — containerize `backend` so users can run the memory daemon without a local Python setup. Labels: `good first issue`, `enhancement`.
3. **Offline smoke script** — one command that starts the daemon on a temp DB, hits `/core/stats` + store + recall, and prints PASS/FAIL. Labels: `good first issue`, `testing`.
