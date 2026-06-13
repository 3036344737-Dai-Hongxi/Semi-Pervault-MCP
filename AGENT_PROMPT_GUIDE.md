# Pervault Agent Prompt Guide

This file explains how to use this repository's `AGENTS.md` rules to generate high-quality prompts for coding agents.

Use this guide when you open a new chat window and want the agent to produce prompts that already fit this project.

## Goal

This repository uses two layers:

- `AGENTS.md` files:
  - long-lived repository rules
  - codebase reality
  - architecture boundaries
  - validation expectations
- task prompt:
  - what to do right now
  - scope, constraints, deliverables, and validation for the current task

The best results come from combining both:

- `AGENTS.md` tells the agent how to behave in this repo
- the task prompt tells the agent what this specific task is

## Files to read first

When generating a prompt for this repo, read these files first:

1. `AGENTS.md`
2. `backend/AGENTS.md` if the task touches backend
3. `frontend/AGENTS.md` if the task touches frontend

For planning or roadmap-sensitive work, also read:

4. `docs/future_plan.md`
5. `docs/plan/four-memory-architecture-integration-plan.md`

## Prompt-writing rules for this repo

When writing a prompt for an execution agent in Pervault, make sure the prompt does all of the following:

- tells the agent to read current code before acting
- tells the agent to treat code as the source of truth when docs conflict
- reminds the agent that this repo is already beyond prototype stage
- avoids telling the agent to invent a parallel subsystem without checking existing code
- keeps scope narrow
- asks for minimal complete validation
- asks for a final report with changed files, design choices, validation, and remaining risks

## Important project realities to preserve in prompts

These are easy for agents to get wrong, so prompts should reinforce them when relevant:

- auth is session/cookie based, not bearer-token-only
- frontend pages are connected to real APIs
- retrieval is already split across multiple modules
- memory-related side effects currently exist in both `backend/routers/memory.py` and `backend/routers/chat.py`
- the project already has importance, admission, persona, sleep-agent, and PAHF foundations
- likely next-phase work is around orchestration, reliability, operability, and productization, not inventing an entirely new memory architecture

## Prompt generation workflow

When using a new chat window, ask it to do prompt generation in this order:

1. read the relevant `AGENTS.md` files
2. inspect the most relevant code files for the task
3. summarize current reality
4. generate a task prompt that matches the current codebase, not stale docs

## Standard prompt skeleton

Use this structure for most execution prompts:

1. role
2. project context
3. task scope
4. required reading
5. implementation constraints
6. validation requirements
7. final output format

## Reusable meta-prompt for generating execution prompts

Copy and adapt this when you want a new chat window to write a project-fit prompt instead of directly coding:

```text
You are not implementing the task yet. First generate a high-quality execution prompt for another coding agent.

Before writing the prompt, read:
- AGENTS.md
- backend/AGENTS.md if backend is involved
- frontend/AGENTS.md if frontend is involved

Also inspect the relevant code files for the task so the prompt matches current implementation rather than stale assumptions.

Your job is to output a prompt that:
- fits this repository's actual architecture
- respects the AGENTS.md rules
- keeps scope tight
- tells the execution agent what files to inspect first
- requires minimal complete validation
- asks for a concise final report with changed files, validation, and residual risks

Task to generate a prompt for:
[PASTE TASK HERE]
```

## Reusable execution prompt template

Copy and adapt this when you already know the task and want another coding agent to execute it:

```text
You are the execution agent for this repository. Follow the repository rules before making changes.

Read first:
- AGENTS.md
- backend/AGENTS.md if this task touches backend
- frontend/AGENTS.md if this task touches frontend

Then inspect the relevant code before implementing anything.

Rules:
- Use current code as source of truth when docs conflict
- Keep scope limited to the requested task
- Do not do unrelated cleanup or refactors
- Preserve existing API contracts unless the task explicitly changes them
- Add or update focused tests when behavior changes
- Run the narrowest relevant validation
- In the final report, include: what changed, why, files changed, validation run, residual risks

Task:
[PASTE TASK HERE]

Constraints:
[PASTE CONSTRAINTS HERE]
```

## Reusable planning prompt template

Use this when you want an agent to plan before coding:

```text
You are doing planning only for this repository. Do not modify code yet.

Read first:
- AGENTS.md
- backend/AGENTS.md if backend is involved
- frontend/AGENTS.md if frontend is involved
- relevant code files

Use code as the source of truth.

Output a reality-based execution plan that includes:
- current state
- gaps and risks
- recommended next step
- files/modules likely involved
- validation plan
- residual risks

Task to plan:
[PASTE TASK HERE]
```

## Reusable review prompt template

Use this when you want an agent to review an implementation:

```text
Review this work against the repository rules and current code reality.

Read first:
- AGENTS.md
- backend/AGENTS.md if backend is involved
- frontend/AGENTS.md if frontend is involved
- the changed files

Review focus:
- correctness
- scope control
- contract compatibility
- code quality
- validation adequacy
- hidden risks

Return:
- findings first, ordered by severity
- then assumptions/open questions
- then a short summary
```

## Good prompt habits for this repo

- Say exactly which files the execution agent should inspect first
- Name what is out of scope
- Tell the agent whether you want planning only or direct implementation
- Tell the agent whether the task is frontend-only, backend-only, or cross-stack
- Tell the agent what verification is required

## Bad prompt habits for this repo

- "Refactor this area" without scope
- "Improve the architecture" without constraints
- "Fix this properly" without naming files, risks, or acceptance criteria
- prompts that assume the repo is still in mock/prototype state
- prompts that ask the agent to follow old docs without checking code

## Recommended usage pattern

For best results, use a two-window workflow:

- Window A:
  - strategy
  - planning
  - prompt writing
  - review of another agent's output
- Window B:
  - execution
  - code changes
  - validation

In Window A, ask the agent to read this file plus the relevant `AGENTS.md` files before writing a prompt.

## Short command for future chats

If you want a new chat window to immediately behave correctly, start with:

```text
Read AGENT_PROMPT_GUIDE.md and the relevant AGENTS.md files, then generate a project-fit prompt for this task:
[PASTE TASK HERE]
```

