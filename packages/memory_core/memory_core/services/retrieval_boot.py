"""Retrieval layer — boot context (startup memory priming).

At session start, Pervault injects a small set of high-signal memories into
the LLM context so the AI already "knows" the user before any message is sent.
This module owns that boot-context assembly logic.

Boot context is assembled in four phases:
  Phase 0 (persona): high-confidence long-term user traits.
  Phase 1 (reflection): high-importance long-term insights.
  Phase 2 (facts): structured facts from accepted fact records, ordered by
    consolidation status then recency.
  Phase 3 (kind): recent memories by kind (project_update, preference,
    relationship_event, task), each capped at its per-kind limit.

Public API:
  get_boot_context(db, *, exclude_ids, exclude_content_keys) -> list[dict]

This module must NOT contain:
  - Intent detection (→ retrieval_intent.py)
  - Non-boot retrieval pipelines (→ retrieval_context.py)
  - Raw primitive queries unrelated to boot context (→ retrieval_primitives.py)
"""

import logging
import sqlite3
from typing import Any

from memory_core.services.retrieval_constants import (
    BOOT_CONTEXT_FACT_LIMIT,
    BOOT_CONTEXT_KIND_LIMITS,
    BOOT_CONTEXT_MAX_RESULTS,
    BOOT_CONTEXT_PERSONA_LIMIT,
    BOOT_CONTEXT_REFLECTION_LIMIT,
)
from memory_core.services.memory_policy import normalize_query_key
from memory_core.services.retrieval_primitives import (
    _build_admission_visibility_clause,
    _build_task_visibility_clause,
    _retrieve_persona_traits,
    _row_to_brief,
    _source_composition,
    _tag_source,
)

logger = logging.getLogger("uvicorn.error")


# ---------------------------------------------------------------------------
# Boot-specific DB retrieval helpers
# ---------------------------------------------------------------------------


async def _retrieve_recent_kind_boot_items(
    db, kind: str, *, limit: int
) -> list[dict[str, Any]]:
    """Fetch the most recent *limit* memory items of a given kind for boot context."""
    task_filter = _build_task_visibility_clause(
        only_open_tasks=kind == "task",
        exclude_closed_tasks=kind != "task",
    )
    admission_filter = _build_admission_visibility_clause()
    try:
        cursor = await db.execute(
            f"""SELECT id, content, created_at, kind
                FROM memory_items
                WHERE kind = ?{task_filter}{admission_filter}
                ORDER BY created_at DESC
                LIMIT ?""",
            (kind, limit),
        )
        rows = await cursor.fetchall()
        return [_row_to_brief(row, include_kind=True) for row in rows]
    except sqlite3.OperationalError:
        logger.warning("Boot context kind retrieval unavailable before migration")
        return []


async def _retrieve_structured_fact_boot_items(
    db, *, limit: int = BOOT_CONTEXT_FACT_LIMIT
) -> list[dict[str, Any]]:
    """Fetch accepted structured facts for the boot context fact phase."""
    kinds = tuple(kind for kind, _ in BOOT_CONTEXT_KIND_LIMITS if kind != "task")
    if not kinds:
        return []

    placeholders = ",".join("?" for _ in kinds)
    admission_filter = _build_admission_visibility_clause(table_alias="m")
    try:
        cursor = await db.execute(
            f"""SELECT DISTINCT m.id, m.content, m.created_at, f.kind AS kind
                FROM structured_facts f
                JOIN memory_items m ON m.id = f.memory_id
                WHERE f.status = 'accepted'
                  AND f.kind IN ({placeholders})
                  {admission_filter}
                ORDER BY m.consolidated DESC, f.created_at DESC
                LIMIT ?""",
            kinds + (limit,),
        )
        rows = await cursor.fetchall()
        return [_row_to_brief(row, include_kind=True) for row in rows]
    except sqlite3.OperationalError:
        logger.warning("Structured fact boot items unavailable before migration")
        return []


async def _retrieve_reflection_boot_items(
    db, *, limit: int = BOOT_CONTEXT_REFLECTION_LIMIT
) -> list[dict[str, Any]]:
    """Fetch high-importance reflections for boot context."""
    try:
        cursor = await db.execute(
            """SELECT id, insight, source_memory_ids, importance, created_at
               FROM memory_reflection
               ORDER BY importance DESC, created_at DESC
               LIMIT ?""",
            (limit,),
        )
        rows = await cursor.fetchall()
    except sqlite3.OperationalError:
        logger.warning("Reflection boot items unavailable before migration")
        return []

    return [
        {
            "id": f"reflection:{row['id']}",
            "content": f"长期洞察：{row['insight']}",
            "created_at": row["created_at"],
            "kind": "reflection",
        }
        for row in rows
    ]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


async def get_boot_context(
    db,
    *,
    exclude_ids: set[str] | None = None,
    exclude_content_keys: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Assemble the boot-context memory set for a new chat session.

    Returns up to BOOT_CONTEXT_MAX_RESULTS items, deduplicated by id and
    normalised content key.  Callers may pass ``exclude_ids`` and
    ``exclude_content_keys`` to prevent overlap with items already injected
    into the prompt by other means.
    """
    selected: list[dict[str, Any]] = []
    seen_ids = set(exclude_ids or set())
    seen_content_keys = set(exclude_content_keys or set())

    # Phase 0: high-confidence persona
    persona_items = _tag_source(
        await _retrieve_persona_traits(
            "",
            db,
            limit=BOOT_CONTEXT_PERSONA_LIMIT,
            min_confidence=0.7,
        ),
        "persona",
    )
    for item in persona_items:
        item_id = item["id"]
        content_key = normalize_query_key(item["content"])
        if item_id in seen_ids:
            continue
        if content_key and content_key in seen_content_keys:
            continue

        selected.append(item)
        seen_ids.add(item_id)
        if content_key:
            seen_content_keys.add(content_key)

        if len(selected) >= BOOT_CONTEXT_MAX_RESULTS:
            logger.info(
                "boot context reached max results (persona phase) selected_ids=%s composition=%s",
                [entry["id"] for entry in selected],
                _source_composition(selected),
            )
            return selected

    # Phase 1: high-importance reflections
    reflection_items = _tag_source(
        await _retrieve_reflection_boot_items(db), "boot_reflection"
    )
    for item in reflection_items:
        item_id = item["id"]
        content_key = normalize_query_key(item["content"])
        if item_id in seen_ids:
            continue
        if content_key and content_key in seen_content_keys:
            continue

        selected.append(item)
        seen_ids.add(item_id)
        if content_key:
            seen_content_keys.add(content_key)

        if len(selected) >= BOOT_CONTEXT_MAX_RESULTS:
            logger.info(
                "boot context reached max results (reflection phase) selected_ids=%s composition=%s",
                [entry["id"] for entry in selected],
                _source_composition(selected),
            )
            return selected

    # Phase 2: structured facts
    fact_items = _tag_source(
        await _retrieve_structured_fact_boot_items(db), "boot_fact"
    )
    for item in fact_items:
        item_id = item["id"]
        content_key = normalize_query_key(item["content"])
        if item_id in seen_ids:
            continue
        if content_key and content_key in seen_content_keys:
            continue

        selected.append(item)
        seen_ids.add(item_id)
        if content_key:
            seen_content_keys.add(content_key)

        if len(selected) >= BOOT_CONTEXT_MAX_RESULTS:
            logger.info(
                "boot context reached max results (fact phase) selected_ids=%s composition=%s",
                [entry["id"] for entry in selected],
                _source_composition(selected),
            )
            return selected

    # Phase 3: recent items by kind
    for kind, limit in BOOT_CONTEXT_KIND_LIMITS:
        items = _tag_source(
            await _retrieve_recent_kind_boot_items(db, kind, limit=limit),
            "boot_kind",
        )
        for item in items:
            item_id = item["id"]
            content_key = normalize_query_key(item["content"])
            if item_id in seen_ids:
                continue
            if content_key and content_key in seen_content_keys:
                continue

            selected.append(item)
            seen_ids.add(item_id)
            if content_key:
                seen_content_keys.add(content_key)

            if len(selected) >= BOOT_CONTEXT_MAX_RESULTS:
                logger.info(
                    "boot context reached max results selected_ids=%s composition=%s",
                    [entry["id"] for entry in selected],
                    _source_composition(selected),
                )
                return selected

    logger.info(
        "boot context selected_ids=%s selected_kinds=%s composition=%s",
        [item["id"] for item in selected],
        [item["kind"] for item in selected],
        _source_composition(selected),
    )
    return selected
