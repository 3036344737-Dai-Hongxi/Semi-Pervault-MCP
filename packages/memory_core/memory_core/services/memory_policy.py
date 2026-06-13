"""Shared memory-system policy: constants, whitelists, text utilities, pure judgments.

This module is the single source of truth for policy rules shared between
the store path (routers/memory.py) and the offline consolidation path
(services/consolidation.py).

Contents are limited to:
  - Pure constants and whitelists
  - Stateless text utility functions
  - Stateless judgment predicates

This module must NOT contain:
  - Async I/O or DB access
  - Orchestration logic or control flow
  - Result dataclasses (ConsolidationDecision, FactPlan, etc.)
  - LLM prompts or API calls
"""

import re

# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------


def contains_any(text: str, patterns: tuple[str, ...]) -> bool:
    """Check whether *text* contains any of the given *patterns* (case-insensitive)."""
    normalized = text.strip().lower()
    return any(pattern.lower() in normalized for pattern in patterns)


def contains_cjk(text: str) -> bool:
    return bool(re.search(r"[\u3400-\u9fff]", text))


def normalize_fact_text(text: str) -> str:
    """Strip whitespace and trailing CJK/ASCII punctuation.

    Used by the store-path structured-fact extractor.
    """
    return text.strip().strip("，,。.!！?？：:；; ")


def normalize_query_key(text: str) -> str:
    """Remove all punctuation and whitespace, then lowercase.

    Used as a dedup/comparison key for query strings and memory content.
    Shared between chat.py and retrieval.py.
    """
    return re.sub(r"[，。！？、,.!?\s]", "", text.strip().lower())


def normalize_text(text: str) -> str:
    """Collapse whitespace, then strip punctuation.

    Stricter than normalize_fact_text — also merges consecutive whitespace.
    Used by consolidation-path fact normalization.
    """
    return re.sub(r"\s+", " ", text.strip()).strip("，,。.!！?？：:；; ")


# ---------------------------------------------------------------------------
# Fact layer (Layer 2) — supported kinds
# ---------------------------------------------------------------------------

_BASE_FACT_KINDS: frozenset[str] = frozenset({
    "project_update",
    "preference",
    "relationship_event",
    "fact",
})


def fact_supported_kinds(*, include_task: bool = False) -> set[str]:
    """Return the set of memory kinds eligible for structured_facts consideration.

    ``include_task=True`` does NOT mean task memories bypass the task guard
    and enter Layer 2 directly.  It only means the consolidation pipeline is
    allowed to *inspect* task memories.  A task still must pass the long-term
    stability gate (``_apply_task_route_guard`` + ``task_fact_is_stable_long_term``)
    before being promoted — and the promoted fact kind must be ``preference``
    or ``fact``, never ``task`` itself.

    Store path should always call with ``include_task=False``.
    """
    base = set(_BASE_FACT_KINDS)
    if include_task:
        base.add("task")
    return base


def should_update_memory_kind(original_kind: str, new_kind: str) -> bool:
    """Return True when the consolidation path should backwrite memory_items.kind.

    Rules (all must hold):
    1. ``new_kind`` is not the same as ``original_kind`` — no-op when equal.
    2. ``new_kind`` is not ``"task"`` — the task kind is never written back from
       the consolidation path; task memories that pass the stability gate are
       promoted to ``preference`` or ``fact`` in structured_facts, but the
       memory_items row should not be re-labelled ``task`` by consolidation.
    3. ``new_kind`` is in ``_BASE_FACT_KINDS`` — rejects any illegal value
       (e.g. ``"other"``, ``"unknown"``, arbitrary LLM hallucinations).

    This is a pure, stateless predicate — safe to call from any context.
    """
    if new_kind == original_kind:
        return False
    if new_kind == "task":
        return False
    if new_kind not in _BASE_FACT_KINDS:
        return False
    return True


# ---------------------------------------------------------------------------
# Graph layer (Layer 3) — whitelists
# ---------------------------------------------------------------------------

STORE_PATH_NODE_TYPES: frozenset[str] = frozenset({"person", "project", "task", "idea"})

CONSOLIDATION_NODE_TYPES: frozenset[str] = frozenset({"person", "project", "event"})

ALL_QUERY_NODE_TYPES: frozenset[str] = STORE_PATH_NODE_TYPES | CONSOLIDATION_NODE_TYPES

ALLOWED_RELATIONS: frozenset[str] = frozenset({"related_to", "belongs_to", "mentioned_with"})

GRAPH_ELIGIBLE_MEMORY_KINDS: frozenset[str] = frozenset({
    "project_update",
    "relationship_event",
})


def is_graph_eligible_kind(kind: str) -> bool:
    """Whether a memory kind should trigger graph extraction on the store path."""
    return kind in GRAPH_ELIGIBLE_MEMORY_KINDS


# ---------------------------------------------------------------------------
# Task guard policy
# ---------------------------------------------------------------------------

TASK_LONG_TERM_POSITIVE_PATTERNS: tuple[str, ...] = (
    "一直",
    "长期",
    "长期坚持",
    "一贯",
    "习惯",
    "长期处于",
    "长期在",
    "持续",
    "坚持",
)

TASK_SHORT_TERM_PATTERNS: tuple[str, ...] = (
    "开始",
    "下周",
    "明天",
    "后天",
    "我要",
    "打算",
    "计划",
    "准备",
    "试着",
    "先",
    "少吃点",
)

TASK_FACT_ALLOWED_KINDS: frozenset[str] = frozenset({"preference", "fact"})


def task_fact_is_stable_long_term(content: str, fact_kind: str) -> bool:
    """Check whether a task memory qualifies as a stable long-term signal.

    Used by consolidation's ``_apply_task_route_guard`` to decide whether a
    task-classified memory may be promoted to Layer 2.  A task passes only
    when ALL of the following hold:

    1. The promoted fact kind is ``preference`` or ``fact`` (never ``task``).
    2. Content contains no short-term signal patterns.
    3. Content contains at least one long-term positive pattern.
    """
    normalized = normalize_text(content)
    if not normalized:
        return False
    if fact_kind not in TASK_FACT_ALLOWED_KINDS:
        return False
    if contains_any(normalized, TASK_SHORT_TERM_PATTERNS):
        return False
    if not contains_any(normalized, TASK_LONG_TERM_POSITIVE_PATTERNS):
        return False
    return True
