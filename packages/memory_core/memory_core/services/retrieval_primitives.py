"""Retrieval layer — primitive DB query functions and shared utilities.

Contains all low-level, single-purpose building blocks used by the higher-level
retrieval pipelines in retrieval_context.py and retrieval_boot.py.

Responsibilities:
  - Row → dict adapters (_row_to_brief)
  - SQL clause builders (_build_task_visibility_clause, _build_kind_clause)
  - Query term extractors (_extract_hybrid_query_terms, etc.)
  - Candidate filter helpers (_filter_*_candidates)
  - Merge utility (_merge_prioritized_results)
  - All individual DB retrieval functions (_retrieve_recent_*, _retrieve_vector_*,
    _retrieve_hybrid_*, _retrieve_structured_fact_memories, etc.)

Graph context retrieval lives in graph_retrieval.py.

This module must NOT contain:
  - Intent detection logic (→ retrieval_intent.py)
  - Multi-step retrieval pipelines that combine primitives by intent
    (→ retrieval_context.py)
  - Boot-context orchestration (→ retrieval_boot.py)
  - Graph context retrieval (→ graph_retrieval.py)
"""

import logging
import math
import re
import sqlite3
from datetime import datetime, timezone
from typing import Any

from sqlite_vec import serialize_float32

from memory_core.services.llm import embed_text
from memory_core.services.logging_policy import text_fingerprint
from memory_core.services.memory_policy import contains_any, contains_cjk, normalize_query_key
from memory_core.services.retrieval_constants import (
    HYBRID_KEYWORD_SCAN_LIMIT,
    HYBRID_SCORE_LOG_LIMIT,
    HYBRID_VECTOR_K,
    INTENT_MEMORY_SCAN_LIMIT,
    KEYWORD_SCORE_WEIGHT,
    MAX_LAYER_RESULTS,
    MAX_TOTAL_RESULTS,
    PREFERENCE_MEMORY_PATTERNS,
    PROJECT_FALLBACK_STRONG_PATTERNS,
    SEMANTIC_SCORE_WEIGHT,
    STRUCTURED_FACT_QUERY_STOPWORDS,
    STRUCTURED_FACT_SCAN_LIMIT,
    SUMMARY_MEMORY_LIMIT,
    SUMMARY_MEMORY_SCAN_LIMIT,
    SourceTag,
)
from memory_core.services.retrieval_intent import is_summary_candidate

logger = logging.getLogger("uvicorn.error")


# ---------------------------------------------------------------------------
# Row → dict adapters
# ---------------------------------------------------------------------------


def _row_to_brief(row, *, include_kind: bool = False) -> dict[str, Any]:
    """Convert a DB row to a minimal dict.

    ``include_kind=True`` adds the ``kind`` field; used by hybrid retrieval
    and boot context where kind-based filtering is needed downstream.
    """
    result: dict[str, Any] = {
        "id": row["id"],
        "content": row["content"],
        "created_at": row["created_at"],
    }
    if include_kind:
        result["kind"] = row["kind"]
    row_keys = set(row.keys())
    if "importance" in row_keys:
        result["importance"] = float(row["importance"] or 5.0)
    return result


def _parse_created_at(value: str | None) -> datetime | None:
    if not value:
        return None
    normalized = value.strip().replace("Z", "+00:00")
    if " " in normalized and "T" not in normalized:
        normalized = normalized.replace(" ", "T", 1)
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _recency_score(created_at: str | None, *, now: datetime | None = None) -> float:
    parsed = _parse_created_at(created_at)
    if parsed is None:
        return 0.5
    reference_time = now or datetime.now(timezone.utc)
    if reference_time.tzinfo is None:
        reference_time = reference_time.replace(tzinfo=timezone.utc)
    age_days = max((reference_time - parsed).total_seconds() / 86400, 0.0)
    return math.exp(-math.log(2) * age_days / 30.0)


def _generative_score(item: dict[str, Any], *, now: datetime | None = None) -> float:
    recency = _recency_score(item.get("created_at"), now=now)
    importance = max(1.0, min(10.0, float(item.get("importance") or 5.0))) / 10.0
    relevance = float(item.get("final_score") or 0.0)
    return round(0.3 * recency + 0.4 * importance + 0.3 * relevance, 4)


def _summary_memory_score(item: dict[str, Any], *, now: datetime | None = None) -> float:
    recency = _recency_score(item.get("created_at"), now=now)
    importance = max(1.0, min(10.0, float(item.get("importance") or 5.0))) / 10.0
    return round(0.45 * recency + 0.55 * importance, 4)




# ---------------------------------------------------------------------------
# Source tagging and composition helpers
# ---------------------------------------------------------------------------


def _tag_source(items: list[dict[str, Any]], tag: SourceTag) -> list[dict[str, Any]]:
    for item in items:
        item.setdefault("_source", tag)
    return items


def _source_composition(items: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        tag = item.get("_source", "unknown")
        counts[tag] = counts.get(tag, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# SQL clause builders
# ---------------------------------------------------------------------------


def _qualify_column(table_alias: str, column_name: str) -> str:
    return f"{table_alias}.{column_name}" if table_alias else column_name


def _build_task_visibility_clause(
    *,
    table_alias: str = "",
    only_open_tasks: bool = False,
    exclude_closed_tasks: bool = True,
) -> str:
    """Return a SQL WHERE fragment that controls task row visibility.

    The returned string always starts with ' AND' so callers can safely
    concatenate it after any existing WHERE predicate.
    """
    kind_column = _qualify_column(table_alias, "kind")
    status_column = _qualify_column(table_alias, "task_status")

    if only_open_tasks:
        return (
            f" AND {kind_column} = 'task'"
            f" AND COALESCE({status_column}, 'open') = 'open'"
        )
    if not exclude_closed_tasks:
        return ""
    return (
        f" AND NOT ({kind_column} = 'task'"
        f" AND COALESCE({status_column}, 'open') IN ('done', 'expired'))"
    )


def _build_admission_visibility_clause(*, table_alias: str = "") -> str:
    tier_column = _qualify_column(table_alias, "admission_tier")
    return f" AND COALESCE({tier_column}, 'standard') = 'standard'"


def _build_kind_clause(kinds: tuple[str, ...]) -> tuple[str, tuple]:
    """Return (SQL fragment, params) for an optional ``kind IN (...)`` filter.

    The SQL fragment is either an empty string (no filter) or a string starting
    with ' AND', ready to be interpolated into an f-string after an existing
    WHERE predicate.
    """
    if not kinds:
        return "", ()
    placeholders = ",".join("?" for _ in kinds)
    return f" AND kind IN ({placeholders})", kinds


# ---------------------------------------------------------------------------
# Pattern and query-term matching helpers
# ---------------------------------------------------------------------------


def _matching_patterns(text: str, patterns: tuple[str, ...]) -> list[str]:
    normalized = text.strip().lower()
    return [pattern for pattern in patterns if pattern.lower() in normalized]


def _matching_query_terms(text: str, query_terms: list[str]) -> list[str]:
    content_key = normalize_query_key(text)
    matched_terms: list[str] = []

    for term in query_terms:
        if not term:
            continue
        min_length = 3 if re.search(r"[A-Za-z]", term) else 2
        if len(term) < min_length:
            continue
        if term in content_key and term not in matched_terms:
            matched_terms.append(term)

    return matched_terms


_PERSONA_TOPIC_HINTS: dict[str, tuple[str, ...]] = {
    "communication_style": (
        "沟通风格",
        "沟通方式",
        "表达方式",
        "说话方式",
        "沟通",
        "表达",
        "说话",
        "直接",
        "委婉",
        "communication",
        "direct",
    ),
    "preference": (
        "长期偏好",
        "偏好",
        "喜欢",
        "不喜欢",
        "更喜欢",
        "口味",
        "倾向",
        "preference",
    ),
    "goal": (
        "长期目标",
        "目标",
        "计划",
        "未来",
        "想达成",
        "goal",
    ),
    "habit": (
        "习惯",
        "通常",
        "经常",
        "habit",
    ),
    "identity": (
        "什么样的人",
        "怎样的人",
        "个人情况",
        "个人资料",
        "身份",
        "背景",
        "profile",
        "identity",
    ),
    "work_style": (
        "工作方式",
        "做事方式",
        "协作方式",
        "工作节奏",
        "工作",
        "做事",
        "协作",
        "work",
    ),
    "health": (
        "健康",
        "运动",
        "跑步",
        "睡眠",
        "health",
        "running",
    ),
}


_PERSONA_TRAIT_KEYWORD_HINTS: dict[str, tuple[str, ...]] = {
    "communication": ("沟通", "表达", "说话"),
    "style": ("风格", "方式"),
    "preference": ("偏好", "喜欢", "不喜欢", "更喜欢"),
    "goal": ("目标", "计划", "未来", "长期"),
    "habit": ("习惯", "通常", "经常"),
    "identity": ("身份", "背景", "个人"),
    "profile": ("什么样的人", "怎样的人", "个人情况"),
    "work": ("工作", "做事", "协作", "项目"),
    "health": ("健康", "运动", "跑步", "睡眠"),
    "direct": ("直接", "清晰"),
    "running": ("跑步", "运动"),
}


def _match_persona_topics(text: str) -> set[str]:
    normalized = normalize_query_key(text)
    if not normalized:
        return set()

    matches: set[str] = set()
    for topic, hints in _PERSONA_TOPIC_HINTS.items():
        if any(normalize_query_key(hint) in normalized for hint in hints):
            matches.add(topic)
    return matches


def _persona_trait_terms(trait_key: str, trait_value: str) -> set[str]:
    terms: set[str] = set()

    for raw_part in re.split(r"[._-]+", trait_key.lower()):
        if len(raw_part) >= 2:
            terms.add(raw_part)
        for hint in _PERSONA_TRAIT_KEYWORD_HINTS.get(raw_part, ()):
            terms.add(hint)

    return terms


def _persona_query_relevance(
    query: str,
    *,
    trait_key: str,
    trait_value: str,
) -> float:
    query_key = normalize_query_key(query)
    if not query_key:
        return 0.0

    trait_terms = _persona_trait_terms(trait_key, trait_value)
    trait_text = " ".join([trait_key, trait_value, *sorted(trait_terms)])
    query_topics = _match_persona_topics(query)
    trait_topics = _match_persona_topics(trait_text)

    score = 0.0
    score += 3.0 * len(query_topics & trait_topics)

    trait_value_key = normalize_query_key(trait_value)
    if trait_value_key:
        if trait_value_key in query_key:
            score += 4.0
        elif len(query_key) >= 4 and query_key in trait_value_key:
            score += 2.0

    score += sum(
        1.0
        for term in trait_terms
        if (term_key := normalize_query_key(term)) and term_key in query_key
    )
    return round(score, 4)


# ---------------------------------------------------------------------------
# Query term extractors
# ---------------------------------------------------------------------------


def _extract_project_anchor_terms(items: list[dict[str, Any]]) -> list[str]:
    anchors: list[str] = []
    seen: set[str] = set()

    for item in items:
        content_key = normalize_query_key(item.get("content", ""))
        for anchor in re.findall(
            r"([a-z][a-z0-9._-]{1,31}|[\u4e00-\u9fff]{2,12})项目",
            content_key,
        ):
            if anchor in seen:
                continue
            seen.add(anchor)
            anchors.append(anchor)

    return anchors


def _extract_structured_fact_query_terms(query: str) -> list[str]:
    from memory_core.services.retrieval_constants import (
        PEOPLE_QUERY_PATTERNS,
        PREFERENCE_QUERY_PATTERNS,
        PROJECT_QUERY_PATTERNS,
        SUMMARY_QUERY_PATTERNS,
    )

    cleaned = query.strip()
    for pattern in (
        PROJECT_QUERY_PATTERNS
        + PREFERENCE_QUERY_PATTERNS
        + PEOPLE_QUERY_PATTERNS
        + SUMMARY_QUERY_PATTERNS
    ):
        cleaned = cleaned.replace(pattern, " ")

    raw_terms = re.findall(r"[A-Za-z][A-Za-z0-9._-]{1,31}|[\u4e00-\u9fff]{2,8}", cleaned)
    terms: list[str] = []
    seen: set[str] = set()
    for raw_term in raw_terms:
        term = normalize_query_key(raw_term)
        for prefix in ("我的", "你的", "我", "你", "的"):
            if term.startswith(prefix) and len(term) > len(prefix) + 1:
                term = term[len(prefix):]
        if term.endswith("的") and len(term) > 2:
            term = term[:-1]
        if not term or term in STRUCTURED_FACT_QUERY_STOPWORDS:
            continue
        if term not in seen:
            seen.add(term)
            terms.append(term)

        # Short substrings help match object-level facts in longer residues.
        if contains_cjk(term) and len(term) > 4:
            for length in range(2, 5):
                for index in range(len(term) - length + 1):
                    subterm = term[index : index + length]
                    if subterm in STRUCTURED_FACT_QUERY_STOPWORDS:
                        continue
                    if subterm not in seen:
                        seen.add(subterm)
                        terms.append(subterm)
    return terms


def _extract_hybrid_query_terms(query: str) -> list[str]:
    cleaned = query.strip()
    raw_terms = re.findall(r"[A-Za-z][A-Za-z0-9._-]{1,31}|[\u4e00-\u9fff]{2,10}", cleaned)
    terms: list[str] = []
    seen: set[str] = set()

    for raw_term in raw_terms:
        term = normalize_query_key(raw_term)
        if not term or term in STRUCTURED_FACT_QUERY_STOPWORDS:
            continue

        if term not in seen:
            seen.add(term)
            terms.append(term)

        if contains_cjk(term) and len(term) > 4:
            for length in range(2, 5):
                for index in range(len(term) - length + 1):
                    subterm = term[index : index + length]
                    if subterm in STRUCTURED_FACT_QUERY_STOPWORDS:
                        continue
                    if subterm not in seen:
                        seen.add(subterm)
                        terms.append(subterm)

    return terms


# ---------------------------------------------------------------------------
# Merge and filter utilities
# ---------------------------------------------------------------------------


def _merge_prioritized_results(
    *groups: list[dict[str, Any]], limit: int = MAX_TOTAL_RESULTS
) -> list[dict[str, Any]]:
    """Merge result groups in priority order, deduplicating by id."""
    ordered: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for group in groups:
        for item in group:
            item_id = item["id"]
            if item_id in seen_ids:
                continue
            seen_ids.add(item_id)
            ordered.append(item)
            if len(ordered) >= limit:
                return ordered

    return ordered


def _filter_retrieval_candidates(
    query: str,
    candidates: list[dict[str, Any]],
    existing_items: list[dict[str, Any]],
    *,
    source: str,
    limit: int,
    required_patterns: tuple[str, ...],
    weak_signal_reason: str,
    anchor_terms: list[str] | None = None,
    exclude_if_only_cross_patterns: tuple[str, ...] | None = None,
    log_prefix: str = "retrieval",
) -> list[dict[str, Any]]:
    """Generic candidate filter used by all project/preference fallback paths.

    Filters out:
      - Duplicate ids or content already in existing_items
      - Items lacking the required domain signal (patterns or query terms)
      - Items that match cross-domain patterns without any domain-specific signal

    Replaces the four near-identical _filter_*_candidates functions.
    """
    if limit <= 0:
        return []

    query_terms = _extract_structured_fact_query_terms(query)
    seen_ids = {item["id"] for item in existing_items}
    seen_content_keys = {
        normalize_query_key(item["content"])
        for item in existing_items
        if item.get("content")
    }
    selected: list[dict[str, Any]] = []

    for item in candidates:
        if len(selected) >= limit:
            logger.info(
                "%s filtered source=%s id=%s reasons=%s",
                log_prefix,
                source,
                item["id"],
                ["limit_trimmed"],
            )
            continue

        reasons: list[str] = []
        content = item.get("content", "")
        content_key = normalize_query_key(content)
        matched_query_terms = _matching_query_terms(content, query_terms)
        matched_required = _matching_patterns(content, required_patterns)
        matched_anchors = (
            _matching_query_terms(content, anchor_terms) if anchor_terms else []
        )
        matched_cross = (
            _matching_patterns(content, exclude_if_only_cross_patterns)
            if exclude_if_only_cross_patterns
            else []
        )

        if item["id"] in seen_ids:
            reasons.append("duplicate_id")
        if content_key and content_key in seen_content_keys:
            reasons.append("duplicate_content")

        has_domain_signal = bool(matched_required or matched_query_terms or matched_anchors)
        if not has_domain_signal:
            reasons.append(weak_signal_reason)

        if (
            matched_cross
            and not matched_required
            and not matched_anchors
        ):
            reasons.append("weak_intent_signal")

        if reasons:
            logger.info(
                "%s filtered source=%s id=%s reasons=%s matched_query_terms=%s matched_required=%s",
                log_prefix,
                source,
                item["id"],
                reasons,
                matched_query_terms,
                matched_required,
            )
            continue

        selected.append(item)
        seen_ids.add(item["id"])
        if content_key:
            seen_content_keys.add(content_key)

    return selected


# ---------------------------------------------------------------------------
# Convenience wrappers — map domain-specific call-sites to the generic filter.
# ---------------------------------------------------------------------------

def _filter_project_fallback_candidates(
    query: str,
    candidates: list[dict[str, Any]],
    existing_items: list[dict[str, Any]],
    *,
    source: str,
    limit: int,
) -> list[dict[str, Any]]:
    return _filter_retrieval_candidates(
        query, candidates, existing_items,
        source=source, limit=limit,
        required_patterns=PROJECT_FALLBACK_STRONG_PATTERNS,
        weak_signal_reason="weak_project_signal",
        log_prefix="project fallback",
    )


def _filter_preference_fallback_candidates(
    query: str,
    candidates: list[dict[str, Any]],
    existing_items: list[dict[str, Any]],
    *,
    source: str,
    limit: int,
) -> list[dict[str, Any]]:
    return _filter_retrieval_candidates(
        query, candidates, existing_items,
        source=source, limit=limit,
        required_patterns=PREFERENCE_MEMORY_PATTERNS,
        weak_signal_reason="missing_preference_signal",
        log_prefix="preference fallback",
    )


def _filter_project_generic_candidates(
    query: str,
    candidates: list[dict[str, Any]],
    existing_items: list[dict[str, Any]],
    *,
    source: str,
    limit: int,
) -> list[dict[str, Any]]:
    anchor_terms = _extract_project_anchor_terms(existing_items)
    return _filter_retrieval_candidates(
        query, candidates, existing_items,
        source=source, limit=limit,
        required_patterns=PROJECT_FALLBACK_STRONG_PATTERNS,
        weak_signal_reason="weak_project_signal",
        anchor_terms=anchor_terms,
        exclude_if_only_cross_patterns=PREFERENCE_MEMORY_PATTERNS,
        log_prefix="project generic",
    )


def _filter_preference_generic_candidates(
    query: str,
    candidates: list[dict[str, Any]],
    existing_items: list[dict[str, Any]],
    *,
    source: str,
    limit: int,
) -> list[dict[str, Any]]:
    return _filter_retrieval_candidates(
        query, candidates, existing_items,
        source=source, limit=limit,
        required_patterns=PREFERENCE_MEMORY_PATTERNS,
        weak_signal_reason="weak_preference_signal",
        log_prefix="preference generic",
    )


# ---------------------------------------------------------------------------
# Primitive DB retrieval functions
# ---------------------------------------------------------------------------


async def _retrieve_memory_matches(query: str, db) -> list[dict[str, Any]]:
    search_term = query.strip()
    if not search_term:
        return []

    if contains_cjk(search_term):
        like_term = f"%{search_term}%"
        admission_filter = _build_admission_visibility_clause()
        cursor = await db.execute(
            """SELECT id, content, created_at
               FROM memory_items
               WHERE (content LIKE ? OR tags LIKE ?)
                 {admission_filter}
               ORDER BY created_at DESC
               LIMIT ?""".format(admission_filter=admission_filter),
            (like_term, like_term, MAX_LAYER_RESULTS),
        )
        rows = await cursor.fetchall()
        return [_row_to_brief(row) for row in rows]

    try:
        admission_filter = _build_admission_visibility_clause(table_alias="m")
        cursor = await db.execute(
            f"""SELECT m.id, m.content, m.created_at
               FROM memory_items m
               JOIN memory_fts f ON m.rowid = f.rowid
               WHERE memory_fts MATCH ?
                 {admission_filter}
               ORDER BY m.created_at DESC
               LIMIT ?""",
            (search_term, MAX_LAYER_RESULTS),
        )
        rows = await cursor.fetchall()
        return [_row_to_brief(row) for row in rows]
    except sqlite3.OperationalError:
        like_term = f"%{search_term}%"
        admission_filter = _build_admission_visibility_clause()
        cursor = await db.execute(
            f"""SELECT id, content, created_at
               FROM memory_items
               WHERE (content LIKE ? OR tags LIKE ?)
                 {admission_filter}
               ORDER BY created_at DESC
               LIMIT ?""",
            (like_term, like_term, MAX_LAYER_RESULTS),
        )
        rows = await cursor.fetchall()
        return [_row_to_brief(row) for row in rows]


async def _retrieve_recent_high_value_memories(query: str, db) -> list[dict[str, Any]]:
    task_filter = _build_task_visibility_clause()
    admission_filter = _build_admission_visibility_clause()
    cursor = await db.execute(
        f"""SELECT id, content, created_at, importance
            FROM memory_items
            WHERE 1 = 1{task_filter}{admission_filter}
            ORDER BY created_at DESC
            LIMIT ?""",
        (SUMMARY_MEMORY_SCAN_LIMIT,),
    )
    rows = await cursor.fetchall()

    candidates = [
        _row_to_brief(row)
        for row in rows
        if is_summary_candidate(row["content"])
    ]
    filtered = sorted(
        candidates,
        key=lambda item: (
            _summary_memory_score(item),
            item.get("created_at") or "",
        ),
        reverse=True,
    )[:SUMMARY_MEMORY_LIMIT]

    logger.info(
        "summary memory retrieval query_fp=%s candidate_count=%s selected=%s",
        text_fingerprint(query),
        len(rows),
        [
            (item["id"], _summary_memory_score(item), float(item.get("importance") or 5.0))
            for item in filtered
        ],
    )
    return filtered


async def _retrieve_recent_pattern_memories(
    db,
    patterns: tuple[str, ...],
    *,
    limit: int = MAX_LAYER_RESULTS,
    only_open_tasks: bool = False,
    exclude_closed_tasks: bool = True,
) -> list[dict[str, Any]]:
    task_filter = _build_task_visibility_clause(
        only_open_tasks=only_open_tasks,
        exclude_closed_tasks=exclude_closed_tasks,
    )
    admission_filter = _build_admission_visibility_clause()
    cursor = await db.execute(
        f"""SELECT id, content, created_at
            FROM memory_items
            WHERE 1 = 1{task_filter}{admission_filter}
            ORDER BY created_at DESC
            LIMIT ?""",
        (INTENT_MEMORY_SCAN_LIMIT,),
    )
    rows = await cursor.fetchall()

    filtered: list[dict[str, Any]] = []
    for row in rows:
        content = row["content"] or ""
        if not contains_any(content, patterns):
            continue
        filtered.append(_row_to_brief(row))
        if len(filtered) >= limit:
            break

    return filtered


async def _retrieve_recent_kind_memories(
    db,
    kinds: tuple[str, ...],
    *,
    limit: int = MAX_LAYER_RESULTS,
    only_open_tasks: bool = False,
    exclude_closed_tasks: bool = True,
) -> list[dict[str, Any]]:
    if not kinds:
        return []

    placeholders = ",".join("?" for _ in kinds)
    task_filter = _build_task_visibility_clause(
        only_open_tasks=only_open_tasks,
        exclude_closed_tasks=exclude_closed_tasks,
    )
    admission_filter = _build_admission_visibility_clause()
    try:
        cursor = await db.execute(
            f"""SELECT id, content, created_at
                FROM memory_items
                WHERE kind IN ({placeholders})
                {task_filter}
                {admission_filter}
                ORDER BY created_at DESC
                LIMIT ?""",
            kinds + (limit,),
        )
        rows = await cursor.fetchall()
        return [_row_to_brief(row) for row in rows]
    except sqlite3.OperationalError:
        logger.warning("Kind-based retrieval unavailable before migration")
        return []


async def _retrieve_structured_fact_memories(
    db,
    kinds: tuple[str, ...],
    *,
    query: str = "",
    limit: int = MAX_LAYER_RESULTS,
    require_query_match: bool = False,
    only_open_tasks: bool = False,
    exclude_closed_tasks: bool = True,
) -> list[dict[str, Any]]:
    """Fetch structured facts that match *kinds*, optionally prioritised by query relevance.

    This function intentionally bundles three concerns in a single pass over the
    result set to avoid a second round-trip to the DB:

      1. **DB query** — fetch accepted structured facts joined to memory_items.
      2. **Query-term extraction** — tokenise *query* into searchable terms.
      3. **In-memory filter/rank** — split results into *matched* (query terms
         found in subject/predicate/object/content) and *candidates* (everything
         else), then merge in priority order.

    The ``require_query_match`` flag controls which results are returned:
      - ``False`` (default): matched items first, then remaining candidates.
        Suitable for broad retrieval where any relevant fact is useful.
      - ``True``: only items that contain at least one query term.
        Used by the preference pipeline when the query is specific enough
        (query_terms non-empty) that unrelated preferences would be noise.

    If *query* is empty, query_terms will be empty and all fetched items are
    treated as candidates (no filtering beyond the kind/status SQL predicates).
    """
    if not kinds:
        return []

    placeholders = ",".join("?" for _ in kinds)
    task_filter = _build_task_visibility_clause(
        table_alias="m",
        only_open_tasks=only_open_tasks,
        exclude_closed_tasks=exclude_closed_tasks,
    )
    admission_filter = _build_admission_visibility_clause(table_alias="m")
    try:
        cursor = await db.execute(
            f"""SELECT m.id,
                       m.content,
                       m.created_at,
                       m.consolidated,
                       f.kind AS fact_kind,
                       f.subject,
                       f.predicate,
                       f.object
                FROM structured_facts f
                JOIN memory_items m ON m.id = f.memory_id
                WHERE f.status = 'accepted'
                  AND f.kind IN ({placeholders})
                  {task_filter}
                  {admission_filter}
                ORDER BY m.consolidated DESC, f.created_at DESC
                LIMIT ?""",
            kinds + (STRUCTURED_FACT_SCAN_LIMIT,),
        )
        rows = await cursor.fetchall()
    except sqlite3.OperationalError:
        logger.warning("Structured fact retrieval unavailable before migration")
        return []

    query_terms = _extract_structured_fact_query_terms(query)
    matched: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []

    for row in rows:
        item = _row_to_brief(row)
        candidates.append(item)
        if not query_terms:
            continue
        haystack = normalize_query_key(
            " ".join(
                filter(
                    None,
                    [row["subject"], row["predicate"], row["object"], row["content"]],
                )
            )
        )
        if any(term in haystack for term in query_terms):
            matched.append(item)

    if require_query_match:
        selected = _merge_prioritized_results(matched, limit=limit)
    else:
        selected = _merge_prioritized_results(matched, candidates, limit=limit)

    logger.info(
        "structured fact retrieval query_fp=%s kinds=%s query_term_count=%s count=%s memory_ids=%s",
        text_fingerprint(query),
        list(kinds),
        len(query_terms),
        len(selected),
        [item["id"] for item in selected],
    )
    return selected


async def _retrieve_persona_traits(
    query: str,
    db,
    *,
    limit: int = MAX_LAYER_RESULTS,
    min_confidence: float = 0.4,
) -> list[dict[str, Any]]:
    try:
        cursor = await db.execute(
            """SELECT id, trait_key, trait_value, confidence, evidence_count, last_updated
               FROM user_persona
               WHERE confidence >= ?
               ORDER BY confidence DESC, evidence_count DESC, last_updated DESC""",
            (min_confidence,),
        )
        rows = await cursor.fetchall()
    except sqlite3.OperationalError:
        logger.warning(
            "persona retrieval unavailable before migration query_fp=%s",
            text_fingerprint(query),
        )
        return []

    scored_rows = sorted(
        rows,
        key=lambda row: (
            _persona_query_relevance(
                query,
                trait_key=str(row["trait_key"] or ""),
                trait_value=str(row["trait_value"] or ""),
            ),
            float(row["confidence"] or 0.0),
            int(row["evidence_count"] or 0),
            row["last_updated"] or "",
        ),
        reverse=True,
    )[:limit]

    logger.info(
        "persona trait retrieval query_fp=%s candidate_count=%s selected=%s",
        text_fingerprint(query),
        len(rows),
        [
            (
                row["id"],
                _persona_query_relevance(
                    query,
                    trait_key=str(row["trait_key"] or ""),
                    trait_value=str(row["trait_value"] or ""),
                ),
            )
            for row in scored_rows
        ],
    )

    return [
        {
            "id": f"persona:{row['id']}",
            "content": f"用户画像：{row['trait_key']} = {row['trait_value']}",
            "created_at": row["last_updated"],
            "kind": "persona",
        }
        for row in scored_rows
    ]


async def _retrieve_graph_matches(
    query: str,
    db,
    *,
    only_open_tasks: bool = False,
    exclude_closed_tasks: bool = True,
) -> list[dict[str, Any]]:
    search_term = query.strip()
    if not search_term:
        return []

    cursor = await db.execute(
        """SELECT id
           FROM graph_nodes
           WHERE label LIKE ?
             AND status = 'confirmed'
           ORDER BY last_seen_at DESC
           LIMIT ?""",
        (f"%{search_term}%", MAX_LAYER_RESULTS),
    )
    node_rows = await cursor.fetchall()
    if not node_rows:
        return []

    node_ids = [row["id"] for row in node_rows]
    placeholders = ",".join("?" for _ in node_ids)

    edge_cursor = await db.execute(
        f"""SELECT DISTINCT e.source_memory_id
            FROM graph_edges e
            JOIN graph_nodes s ON s.id = e.source_id AND s.status = 'confirmed'
            JOIN graph_nodes t ON t.id = e.target_id AND t.status = 'confirmed'
            WHERE (e.source_id IN ({placeholders}) OR e.target_id IN ({placeholders}))
              AND e.source_memory_id IS NOT NULL
            LIMIT ?""",
        node_ids + node_ids + [MAX_LAYER_RESULTS * 3],
    )
    edge_rows = await edge_cursor.fetchall()
    memory_ids = [row["source_memory_id"] for row in edge_rows if row["source_memory_id"]]
    if not memory_ids:
        return []

    unique_memory_ids = list(dict.fromkeys(memory_ids))[:MAX_LAYER_RESULTS]
    memory_placeholders = ",".join("?" for _ in unique_memory_ids)
    task_filter = _build_task_visibility_clause(
        only_open_tasks=only_open_tasks,
        exclude_closed_tasks=exclude_closed_tasks,
    )
    admission_filter = _build_admission_visibility_clause()
    memory_cursor = await db.execute(
        f"""SELECT id, content, created_at
            FROM memory_items
            WHERE id IN ({memory_placeholders})
            {task_filter}
            {admission_filter}
            ORDER BY created_at DESC
            LIMIT ?""",
        unique_memory_ids + [MAX_LAYER_RESULTS],
    )
    memory_rows = await memory_cursor.fetchall()
    return [_row_to_brief(row) for row in memory_rows]


async def _retrieve_memories_by_graph_type(
    node_type: str,
    db,
    *,
    limit: int = MAX_LAYER_RESULTS,
    only_open_tasks: bool = False,
    exclude_closed_tasks: bool = True,
) -> list[dict[str, Any]]:
    cursor = await db.execute(
        """SELECT DISTINCT e.source_memory_id
           FROM graph_edges e
           JOIN graph_nodes s ON s.id = e.source_id AND s.status = 'confirmed'
           JOIN graph_nodes t ON t.id = e.target_id AND t.status = 'confirmed'
           WHERE e.source_memory_id IS NOT NULL
             AND (s.type = ? OR t.type = ?)
           ORDER BY e.created_at DESC
           LIMIT ?""",
        (node_type, node_type, limit * 3),
    )
    rows = await cursor.fetchall()
    memory_ids = [row["source_memory_id"] for row in rows if row["source_memory_id"]]
    if not memory_ids:
        return []

    unique_memory_ids = list(dict.fromkeys(memory_ids))[:limit]
    placeholders = ",".join("?" for _ in unique_memory_ids)
    task_filter = _build_task_visibility_clause(
        only_open_tasks=only_open_tasks,
        exclude_closed_tasks=exclude_closed_tasks,
    )
    admission_filter = _build_admission_visibility_clause()
    memory_cursor = await db.execute(
        f"""SELECT id, content, created_at
            FROM memory_items
            WHERE id IN ({placeholders})
            {task_filter}
            {admission_filter}
            ORDER BY created_at DESC
            LIMIT ?""",
        unique_memory_ids + [limit],
    )
    memory_rows = await memory_cursor.fetchall()
    return [_row_to_brief(row) for row in memory_rows]


async def _retrieve_vector_matches(query: str, db) -> list[dict[str, Any]]:
    if not getattr(db, "sqlite_vec_loaded", False):
        return []

    search_term = query.strip()
    if not search_term:
        return []

    try:
        query_embedding = await embed_text(search_term)
        cursor = await db.execute(
            """SELECT ref_id, distance
               FROM vec_items
               WHERE ref_type = ?
                 AND embedding MATCH ?
                 AND k = ?
               ORDER BY distance""",
            ("memory", serialize_float32(query_embedding), MAX_LAYER_RESULTS),
        )
        rows = await cursor.fetchall()
        memory_ids = [row["ref_id"] for row in rows]
        if not memory_ids:
            return []

        placeholders = ",".join("?" for _ in memory_ids)
        admission_filter = _build_admission_visibility_clause()
        memory_cursor = await db.execute(
            f"""SELECT id, content, created_at
                FROM memory_items
                WHERE id IN ({placeholders})
                {admission_filter}
                ORDER BY created_at DESC
                LIMIT ?""",
            memory_ids + [MAX_LAYER_RESULTS],
        )
        memory_rows = await memory_cursor.fetchall()
        return [_row_to_brief(row) for row in memory_rows]
    except Exception:
        logger.exception("Vector retrieval failed query_fp=%s", text_fingerprint(query))
        return []


async def _retrieve_hybrid_keyword_candidates(
    query: str,
    db,
    *,
    kinds: tuple[str, ...] = (),
    only_open_tasks: bool = False,
    exclude_closed_tasks: bool = True,
) -> dict[str, dict[str, Any]]:
    query_terms = _extract_hybrid_query_terms(query)
    if not query_terms:
        fallback_term = normalize_query_key(query)
        if fallback_term:
            query_terms = [fallback_term]
    if not query_terms:
        return {}

    candidates: dict[str, dict[str, Any]] = {}
    whole_query_key = normalize_query_key(query)
    task_filter = _build_task_visibility_clause(
        only_open_tasks=only_open_tasks,
        exclude_closed_tasks=exclude_closed_tasks,
    )
    kind_clause, kind_params = _build_kind_clause(kinds)
    admission_filter = _build_admission_visibility_clause()

    for term in query_terms:
        like_term = f"%{term}%"
        cursor = await db.execute(
            f"""SELECT id, content, created_at, kind, importance
                FROM memory_items
                WHERE (content LIKE ? OR tags LIKE ?){kind_clause}{task_filter}{admission_filter}
                ORDER BY created_at DESC
                LIMIT ?""",
            (like_term, like_term) + kind_params + (HYBRID_KEYWORD_SCAN_LIMIT,),
        )
        rows = await cursor.fetchall()
        for row in rows:
            item_id = row["id"]
            entry = candidates.setdefault(
                item_id,
                {
                    **_row_to_brief(row, include_kind=True),
                    "matched_terms": set(),
                },
            )
            entry["matched_terms"].add(term)

    total_terms = max(len(query_terms), 1)
    for entry in candidates.values():
        content_key = normalize_query_key(entry["content"])
        matched_terms = entry["matched_terms"]
        coverage_score = len(matched_terms) / total_terms
        max_term_length = max((len(term) for term in matched_terms), default=0)
        query_term_length = max((len(term) for term in query_terms), default=1)
        precision_score = max_term_length / query_term_length
        exact_match_bonus = 1.0 if whole_query_key and whole_query_key in content_key else 0.0
        keyword_score = min(
            1.0,
            0.6 * coverage_score + 0.3 * precision_score + 0.1 * exact_match_bonus,
        )
        entry["keyword_score"] = round(keyword_score, 4)
        entry["matched_terms"] = sorted(matched_terms)

    return candidates


async def _retrieve_hybrid_semantic_candidates(
    query: str,
    db,
    *,
    kinds: tuple[str, ...] = (),
    only_open_tasks: bool = False,
    exclude_closed_tasks: bool = True,
) -> dict[str, dict[str, Any]]:
    if not getattr(db, "sqlite_vec_loaded", False):
        return {}

    search_term = query.strip()
    if not search_term:
        return {}

    try:
        query_embedding = await embed_text(search_term)
        cursor = await db.execute(
            """SELECT ref_id, distance
               FROM vec_items
               WHERE ref_type = ?
                 AND embedding MATCH ?
                 AND k = ?
               ORDER BY distance""",
            ("memory", serialize_float32(query_embedding), HYBRID_VECTOR_K),
        )
        rows = await cursor.fetchall()
    except Exception:
        logger.exception(
            "Hybrid semantic retrieval failed query_fp=%s",
            text_fingerprint(query),
        )
        return {}

    distance_map = {row["ref_id"]: row["distance"] for row in rows}
    if not distance_map:
        return {}

    memory_ids = list(distance_map.keys())
    placeholders = ",".join("?" for _ in memory_ids)
    task_filter = _build_task_visibility_clause(
        only_open_tasks=only_open_tasks,
        exclude_closed_tasks=exclude_closed_tasks,
    )
    kind_clause, kind_params = _build_kind_clause(kinds)
    admission_filter = _build_admission_visibility_clause()

    cursor = await db.execute(
        f"""SELECT id, content, created_at, kind, importance
            FROM memory_items
            WHERE id IN ({placeholders}){kind_clause}{task_filter}{admission_filter}""",
        tuple(memory_ids) + kind_params,
    )
    rows = await cursor.fetchall()

    candidates: dict[str, dict[str, Any]] = {}
    for row in rows:
        distance = float(distance_map[row["id"]])
        semantic_score = 1.0 / (1.0 + max(distance, 0.0))
        candidates[row["id"]] = {
            **_row_to_brief(row, include_kind=True),
            "semantic_score": round(semantic_score, 4),
            "distance": round(distance, 4),
        }

    return candidates


async def _retrieve_hybrid_context(
    query: str,
    db,
    *,
    label: str,
    kinds: tuple[str, ...] = (),
    limit: int = MAX_LAYER_RESULTS,
    only_open_tasks: bool = False,
    exclude_closed_tasks: bool = True,
) -> list[dict[str, Any]]:
    import asyncio

    keyword_candidates, semantic_candidates = await asyncio.gather(
        _retrieve_hybrid_keyword_candidates(
            query, db,
            kinds=kinds,
            only_open_tasks=only_open_tasks,
            exclude_closed_tasks=exclude_closed_tasks,
        ),
        _retrieve_hybrid_semantic_candidates(
            query, db,
            kinds=kinds,
            only_open_tasks=only_open_tasks,
            exclude_closed_tasks=exclude_closed_tasks,
        ),
    )

    merged_candidates: dict[str, dict[str, Any]] = {}
    for source in (keyword_candidates, semantic_candidates):
        for item_id, candidate in source.items():
            entry = merged_candidates.setdefault(
                item_id,
                {
                    "id": candidate["id"],
                    "content": candidate["content"],
                    "created_at": candidate["created_at"],
                    "kind": candidate.get("kind", ""),
                    "importance": float(candidate.get("importance") or 5.0),
                    "semantic_score": 0.0,
                    "keyword_score": 0.0,
                    "matched_terms": [],
                },
            )
            entry["semantic_score"] = max(
                entry["semantic_score"],
                float(candidate.get("semantic_score", 0.0)),
            )
            entry["keyword_score"] = max(
                entry["keyword_score"],
                float(candidate.get("keyword_score", 0.0)),
            )
            if candidate.get("matched_terms"):
                entry["matched_terms"] = candidate["matched_terms"]

    scored_candidates = []
    for candidate in merged_candidates.values():
        scored = {
            **candidate,
            "final_score": round(
                SEMANTIC_SCORE_WEIGHT * candidate["semantic_score"]
                + KEYWORD_SCORE_WEIGHT * candidate["keyword_score"],
                4,
            ),
        }
        scored["generative_score"] = _generative_score(scored)
        scored_candidates.append(scored)

    scored_candidates.sort(
        key=lambda item: (
            item["generative_score"],
            item["final_score"],
            item["keyword_score"],
            item["created_at"] or "",
        ),
        reverse=True,
    )

    logger.info(
        "hybrid retrieval query_fp=%s label=%s kinds=%s candidates=%s",
        text_fingerprint(query),
        label,
        list(kinds),
        len(scored_candidates),
    )
    for candidate in scored_candidates[:HYBRID_SCORE_LOG_LIMIT]:
        logger.info(
            "candidate id=%s semantic_score=%.4f keyword_score=%.4f final_score=%.4f generative_score=%.4f importance=%.2f label=%s matched_terms=%s",
            candidate["id"],
            candidate["semantic_score"],
            candidate["keyword_score"],
            candidate["final_score"],
            candidate["generative_score"],
            candidate.get("importance", 5.0),
            label,
            candidate.get("matched_terms", []),
        )

    top_candidates = scored_candidates[:limit]
    logger.info(
        "hybrid retrieval top_ids=%s label=%s",
        [candidate["id"] for candidate in top_candidates],
        label,
    )
    return [_row_to_brief(candidate) for candidate in top_candidates]
