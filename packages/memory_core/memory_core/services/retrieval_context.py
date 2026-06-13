"""Retrieval layer — intent-specific retrieval pipelines.

This module orchestrates primitive retrieval functions from retrieval_primitives.py
into complete retrieval responses for each query intent.

Each ``_retrieve_<intent>_context`` function follows the same three-phase pattern:
  Phase 1 (primary): structured facts + hybrid search
  Phase 2 (fallback): graph/pattern matching when Phase 1 is insufficient
  Phase 3 (generic): unconstrained hybrid as a last resort

The project and preference pipelines share this pattern via
``_retrieve_layered_context`` + ``LayeredRetrievalConfig``.

The public entry point is ``retrieve_context(query, db)``, which detects intent
and dispatches to the appropriate pipeline.

This module must NOT contain:
  - Raw DB query logic (→ retrieval_primitives.py)
  - Intent detection (→ retrieval_intent.py)
  - Boot context logic (→ retrieval_boot.py)
  - Constants (→ retrieval_constants.py)
"""

import logging
from dataclasses import dataclass
from typing import Any, Callable

from memory_core.services.logging_policy import text_fingerprint
from memory_core.services.retrieval_constants import (
    MAX_LAYER_RESULTS,
    MAX_TOTAL_RESULTS,
    PEOPLE_MEMORY_PATTERNS,
    PREFERENCE_FALLBACK_MAX_ADDITIONS,
    PREFERENCE_GENERIC_MAX_ADDITIONS,
    PREFERENCE_MEMORY_PATTERNS,
    PROJECT_FALLBACK_MAX_ADDITIONS,
    PROJECT_GENERIC_MAX_ADDITIONS,
    PROJECT_MEMORY_PATTERNS,
    QueryIntent,
    SUMMARY_STRUCTURED_FACT_KINDS,
)
from memory_core.services.retrieval_intent import detect_query_intent
from memory_core.services.retrieval_primitives import (
    _extract_structured_fact_query_terms,
    _filter_preference_fallback_candidates,
    _filter_preference_generic_candidates,
    _filter_project_fallback_candidates,
    _filter_project_generic_candidates,
    _merge_prioritized_results,
    _retrieve_graph_matches,
    _retrieve_hybrid_context,
    _retrieve_memories_by_graph_type,
    _retrieve_persona_traits,
    _retrieve_recent_high_value_memories,
    _retrieve_recent_kind_memories,
    _retrieve_recent_pattern_memories,
    _retrieve_structured_fact_memories,
    _source_composition,
    _tag_source,
)

logger = logging.getLogger("uvicorn.error")


# ---------------------------------------------------------------------------
# Layered retrieval template (shared by project + preference pipelines)
# ---------------------------------------------------------------------------


@dataclass
class LayeredRetrievalConfig:
    """Configuration that parameterises the three-phase layered retrieval pattern.

    Both the project and preference pipelines follow the same structure:
      Phase 1 — structured facts + hybrid search → primary_matches
      Phase 2 — graph/pattern fallback when primary < MAX_LAYER_RESULTS
      Phase 3 — unconstrained generic hybrid as last resort

    All differences between the two pipelines are captured here.
    """

    label: str
    """Short name used in hybrid search labels and log messages ("project" / "preference")."""

    kinds: tuple[str, ...]
    """Memory kinds passed to structured-fact and hybrid queries."""

    fallback_filter: Callable
    """Filter function for Phase-2 fallback candidates."""

    generic_filter: Callable
    """Filter function for Phase-3 generic candidates."""

    fallback_max: int
    """Maximum number of items to add from Phase-2 fallback."""

    generic_max: int
    """Maximum number of items to add from Phase-3 generic fallback."""

    fallback_patterns: tuple[str, ...] | None = None
    """Pattern list for recent-pattern fallback fetch; None disables pattern fallback."""

    graph_type: str | None = None
    """Graph node type for graph-based fallback fetch; None disables graph fallback."""

    require_query_match: bool = False
    """When True, structured-fact retrieval requires query-term overlap."""


async def _retrieve_layered_context(
    query: str,
    db,
    cfg: LayeredRetrievalConfig,
) -> list[dict[str, Any]]:
    """Execute the three-phase layered retrieval pipeline parameterised by *cfg*.

    Phase 1 — primary:
        Merge structured-fact matches with hybrid matches.

    Phase 2 — fallback (runs only when primary < MAX_LAYER_RESULTS):
        Optionally fetch graph-type matches and/or recent-pattern matches,
        filter each with cfg.fallback_filter, then merge into primary.

    Phase 3 — generic (runs only when merged < MAX_LAYER_RESULTS):
        Fetch an unconstrained hybrid result, filter with cfg.generic_filter,
        then merge into the running result.
    """
    # --- Phase 1: primary ---
    fact_matches = _tag_source(
        await _retrieve_structured_fact_memories(
            db, cfg.kinds, query=query,
            require_query_match=cfg.require_query_match,
        ),
        "structured_fact",
    )
    if len(fact_matches) < MAX_LAYER_RESULTS:
        logger.info(
            "structured facts fallback kind=%s fact_count=%s query_fp=%s query_len=%d",
            cfg.kinds[0], len(fact_matches),
            text_fingerprint(query),
            len(query.strip()),
        )
    hybrid_matches = _tag_source(
        await _retrieve_hybrid_context(query, db, label=cfg.label, kinds=cfg.kinds),
        "hybrid",
    )
    primary_matches = _merge_prioritized_results(
        fact_matches, hybrid_matches, limit=MAX_LAYER_RESULTS
    )

    # --- Phase 2: fallback ---
    if len(primary_matches) < MAX_LAYER_RESULTS:
        fallback_limit = min(
            MAX_LAYER_RESULTS - len(primary_matches),
            cfg.fallback_max,
        )
        existing = primary_matches

        filtered_graph: list[dict[str, Any]] = []
        raw_graph: list[dict[str, Any]] = []
        if cfg.graph_type:
            raw_graph = _tag_source(
                await _retrieve_memories_by_graph_type(cfg.graph_type, db), "graph"
            )
            filtered_graph = cfg.fallback_filter(
                query, raw_graph, existing,
                source="graph", limit=fallback_limit,
            )
            existing = existing + filtered_graph
            fallback_limit = max(fallback_limit - len(filtered_graph), 0)

        filtered_pattern: list[dict[str, Any]] = []
        raw_pattern: list[dict[str, Any]] = []
        if cfg.fallback_patterns:
            raw_pattern = _tag_source(
                await _retrieve_recent_pattern_memories(db, cfg.fallback_patterns),
                "pattern",
            )
            filtered_pattern = cfg.fallback_filter(
                query, raw_pattern, existing,
                source="pattern", limit=fallback_limit,
            )

        merged = _merge_prioritized_results(
            primary_matches, filtered_graph, filtered_pattern,
            limit=MAX_LAYER_RESULTS,
        )
        logger.info(
            "%s retrieval fact_ids=%s hybrid_ids=%s primary_ids=%s "
            "graph_ids=%s pattern_ids=%s filtered_graph_ids=%s filtered_pattern_ids=%s "
            "query_fp=%s query_len=%d",
            cfg.label,
            [item["id"] for item in fact_matches],
            [item["id"] for item in hybrid_matches],
            [item["id"] for item in primary_matches],
            [item["id"] for item in raw_graph],
            [item["id"] for item in raw_pattern],
            [item["id"] for item in filtered_graph],
            [item["id"] for item in filtered_pattern],
            text_fingerprint(query),
            len(query.strip()),
        )
    else:
        merged = primary_matches
        logger.info(
            "%s retrieval fact_ids=%s hybrid_ids=%s primary_ids=%s "
            "fallback=skipped reason=primary_sufficient query_fp=%s query_len=%d",
            cfg.label,
            [item["id"] for item in fact_matches],
            [item["id"] for item in hybrid_matches],
            [item["id"] for item in primary_matches],
            text_fingerprint(query),
            len(query.strip()),
        )

    # --- Phase 3: generic fallback ---
    if len(merged) < MAX_LAYER_RESULTS:
        generic_hybrid = _tag_source(
            await _retrieve_hybrid_context(query, db, label=f"{cfg.label}_generic"),
            "hybrid",
        )
        filtered_generic = cfg.generic_filter(
            query, generic_hybrid, merged,
            source="generic_hybrid",
            limit=min(MAX_LAYER_RESULTS - len(merged), cfg.generic_max),
        )
        merged = _merge_prioritized_results(merged, filtered_generic)
        logger.info(
            "%s retrieval generic_hybrid_ids=%s filtered_generic_ids=%s "
            "merged_ids=%s composition=%s",
            cfg.label,
            [item["id"] for item in generic_hybrid],
            [item["id"] for item in filtered_generic],
            [item["id"] for item in merged],
            _source_composition(merged),
        )
    else:
        logger.info(
            "%s retrieval merged_ids=%s composition=%s generic=skipped",
            cfg.label,
            [item["id"] for item in merged],
            _source_composition(merged),
        )

    return merged


# ---------------------------------------------------------------------------
# Summary pipeline
# ---------------------------------------------------------------------------


async def _retrieve_summary_context(query: str, db) -> list[dict[str, Any]]:
    summary_fact_matches = _tag_source(
        await _retrieve_structured_fact_memories(
            db,
            SUMMARY_STRUCTURED_FACT_KINDS,
            query=query,
            limit=MAX_LAYER_RESULTS,
        ),
        "structured_fact",
    )
    recent_high_value = _tag_source(
        await _retrieve_recent_high_value_memories(query, db),
        "recent",
    )
    merged = _merge_prioritized_results(
        summary_fact_matches,
        recent_high_value,
        limit=MAX_TOTAL_RESULTS,
    )
    logger.info(
        "summary context query_fp=%s fact_ids=%s high_value_ids=%s merged_ids=%s composition=%s",
        text_fingerprint(query),
        [item["id"] for item in summary_fact_matches],
        [item["id"] for item in recent_high_value],
        [item["id"] for item in merged],
        _source_composition(merged),
    )
    return merged


# ---------------------------------------------------------------------------
# Task pipeline
# ---------------------------------------------------------------------------


async def _retrieve_task_context(query: str, db) -> list[dict[str, Any]]:
    task_fact_matches = _tag_source(
        await _retrieve_structured_fact_memories(
            db,
            ("task",),
            query=query,
            only_open_tasks=True,
            exclude_closed_tasks=False,
        ),
        "structured_fact",
    )
    task_hybrid_matches = _tag_source(
        await _retrieve_hybrid_context(
            query, db,
            label="task",
            kinds=("task",),
            only_open_tasks=True,
            exclude_closed_tasks=False,
        ),
        "hybrid",
    )
    task_recent_matches = _tag_source(
        await _retrieve_recent_kind_memories(
            db,
            ("task",),
            limit=MAX_LAYER_RESULTS,
            only_open_tasks=True,
            exclude_closed_tasks=False,
        ),
        "recent",
    )
    merged = _merge_prioritized_results(
        task_fact_matches,
        task_hybrid_matches,
        task_recent_matches,
        limit=MAX_LAYER_RESULTS,
    )
    logger.info(
        "task retrieval query_fp=%s fact_ids=%s hybrid_ids=%s recent_ids=%s merged_ids=%s composition=%s",
        text_fingerprint(query),
        [item["id"] for item in task_fact_matches],
        [item["id"] for item in task_hybrid_matches],
        [item["id"] for item in task_recent_matches],
        [item["id"] for item in merged],
        _source_composition(merged),
    )
    return merged


# ---------------------------------------------------------------------------
# Project pipeline
# ---------------------------------------------------------------------------

_PROJECT_CFG = LayeredRetrievalConfig(
    label="project",
    kinds=("project_update",),
    fallback_filter=_filter_project_fallback_candidates,
    generic_filter=_filter_project_generic_candidates,
    fallback_max=PROJECT_FALLBACK_MAX_ADDITIONS,
    generic_max=PROJECT_GENERIC_MAX_ADDITIONS,
    graph_type="project",
    fallback_patterns=PROJECT_MEMORY_PATTERNS,
)


async def _retrieve_project_context(query: str, db) -> list[dict[str, Any]]:
    return await _retrieve_layered_context(query, db, _PROJECT_CFG)


# ---------------------------------------------------------------------------
# Preference pipeline
# ---------------------------------------------------------------------------


async def _retrieve_preference_context(query: str, db) -> list[dict[str, Any]]:
    query_terms = _extract_structured_fact_query_terms(query)
    cfg = LayeredRetrievalConfig(
        label="preference",
        kinds=("preference",),
        fallback_filter=_filter_preference_fallback_candidates,
        generic_filter=_filter_preference_generic_candidates,
        fallback_max=PREFERENCE_FALLBACK_MAX_ADDITIONS,
        generic_max=PREFERENCE_GENERIC_MAX_ADDITIONS,
        graph_type=None,
        fallback_patterns=PREFERENCE_MEMORY_PATTERNS,
        require_query_match=bool(query_terms),
    )
    return await _retrieve_layered_context(query, db, cfg)


# ---------------------------------------------------------------------------
# Persona pipeline
# ---------------------------------------------------------------------------


async def _retrieve_persona_context(query: str, db) -> list[dict[str, Any]]:
    persona_matches = _tag_source(
        await _retrieve_persona_traits(query, db),
        "persona",
    )
    fallback_items: list[dict[str, Any]] = []
    if len(persona_matches) < MAX_LAYER_RESULTS:
        preference_items = await _retrieve_preference_context(query, db)
        summary_items = await _retrieve_summary_context(query, db)
        fallback_items = _merge_prioritized_results(
            preference_items,
            summary_items,
            limit=MAX_LAYER_RESULTS - len(persona_matches),
        )

    merged = _merge_prioritized_results(
        persona_matches,
        fallback_items,
        limit=MAX_TOTAL_RESULTS,
    )
    logger.info(
        "persona retrieval query_fp=%s persona_ids=%s fallback_ids=%s merged_ids=%s composition=%s",
        text_fingerprint(query),
        [item["id"] for item in persona_matches],
        [item["id"] for item in fallback_items],
        [item["id"] for item in merged],
        _source_composition(merged),
    )
    return merged


# ---------------------------------------------------------------------------
# People pipeline
# ---------------------------------------------------------------------------


async def _retrieve_people_context(query: str, db) -> list[dict[str, Any]]:
    people_fact_matches = _tag_source(
        await _retrieve_structured_fact_memories(
            db, ("relationship_event",), query=query
        ),
        "structured_fact",
    )
    if len(people_fact_matches) < MAX_LAYER_RESULTS:
        logger.info(
            "structured facts fallback kind=%s fact_count=%s query_fp=%s query_len=%d",
            "relationship_event", len(people_fact_matches),
            text_fingerprint(query),
            len(query.strip()),
        )
    people_hybrid_matches = _tag_source(
        await _retrieve_hybrid_context(
            query, db, label="people", kinds=("relationship_event",)
        ),
        "hybrid",
    )
    people_graph_matches = _tag_source(
        await _retrieve_memories_by_graph_type("person", db), "graph"
    )
    people_pattern_matches = _tag_source(
        await _retrieve_recent_pattern_memories(db, PEOPLE_MEMORY_PATTERNS),
        "pattern",
    )
    merged = _merge_prioritized_results(
        people_fact_matches,
        people_hybrid_matches,
        people_graph_matches,
        people_pattern_matches,
    )

    generic_hybrid_matches: list[dict[str, Any]] = []
    if len(merged) < MAX_LAYER_RESULTS:
        generic_hybrid_matches = _tag_source(
            await _retrieve_hybrid_context(query, db, label="people_generic"),
            "hybrid",
        )
        merged = _merge_prioritized_results(merged, generic_hybrid_matches)

    logger.info(
        "people retrieval query_fp=%s fact_ids=%s hybrid_ids=%s graph_ids=%s "
        "pattern_ids=%s generic_hybrid_ids=%s merged_ids=%s composition=%s",
        text_fingerprint(query),
        [item["id"] for item in people_fact_matches],
        [item["id"] for item in people_hybrid_matches],
        [item["id"] for item in people_graph_matches],
        [item["id"] for item in people_pattern_matches],
        [item["id"] for item in generic_hybrid_matches],
        [item["id"] for item in merged],
        _source_composition(merged),
    )
    return merged


# ---------------------------------------------------------------------------
# Generic pipeline
# ---------------------------------------------------------------------------


async def _retrieve_generic_context(query: str, db) -> list[dict[str, Any]]:
    hybrid_matches = _tag_source(
        await _retrieve_hybrid_context(query, db, label="generic"), "hybrid"
    )
    graph_matches = _tag_source(await _retrieve_graph_matches(query, db), "graph")
    merged = _merge_prioritized_results(hybrid_matches, graph_matches)

    logger.info(
        "generic retrieval query_fp=%s hybrid_ids=%s graph_ids=%s merged_ids=%s composition=%s",
        text_fingerprint(query),
        [item["id"] for item in hybrid_matches],
        [item["id"] for item in graph_matches],
        [item["id"] for item in merged],
        _source_composition(merged),
    )
    return merged


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


async def retrieve_context(
    query: str,
    db,
    *,
    intent: QueryIntent | None = None,
) -> list[dict[str, Any]]:
    """Detect query intent and dispatch to the appropriate retrieval pipeline.

    This is the main entry point called by routers/chat.py.
    If *intent* is provided it is used directly, skipping a redundant LLM call.
    """
    if intent is None:
        intent = await detect_query_intent(query)
    logger.info(
        "query router intent=%s query_fp=%s query_len=%d",
        intent,
        text_fingerprint(query),
        len(query.strip()),
    )

    if intent == "task_query":
        results = await _retrieve_task_context(query, db)
    elif intent == "persona_query":
        results = await _retrieve_persona_context(query, db)
    elif intent == "summary_query":
        results = await _retrieve_summary_context(query, db)
    elif intent == "project_query":
        results = await _retrieve_project_context(query, db)
    elif intent == "preference_query":
        results = await _retrieve_preference_context(query, db)
    elif intent == "people_query":
        results = await _retrieve_people_context(query, db)
    else:
        results = await _retrieve_generic_context(query, db)

    logger.info(
        "retrieval result intent=%s query_fp=%s total=%s composition=%s",
        intent,
        text_fingerprint(query),
        len(results),
        _source_composition(results),
    )
    return results
