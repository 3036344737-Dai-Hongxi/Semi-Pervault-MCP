"""Admission scoring for long-term memory retrieval.

This module decides whether a stored memory should participate in generative
retrieval. Storage remains optimistic; admission scoring runs after the write
and only updates the memory tier when scoring succeeds.
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher

from memory_core.services.llm import score_admission_with_llm
from memory_core.services.memory_policy import normalize_query_key

ADMISSION_SCAN_LIMIT = 200
LOW_VALUE_THRESHOLD = 0.45

TYPE_PRIOR_SCORES: dict[str, float] = {
    "project_update": 0.95,
    "preference": 0.9,
    "relationship_event": 0.85,
    "fact": 0.85,
    "task": 0.8,
    "other": 0.35,
}


@dataclass(frozen=True)
class AdmissionScore:
    utility: float
    confidence: float
    novelty: float
    recency: float
    type_prior: float
    total: float
    tier: str


def _clamp_unit(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _score_type_prior(kind: str) -> float:
    return TYPE_PRIOR_SCORES.get(kind or "other", TYPE_PRIOR_SCORES["other"])


def _similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right).ratio()


async def _score_novelty(
    content: str,
    db,
    *,
    exclude_memory_id: str | None = None,
) -> float:
    content_key = normalize_query_key(content)
    if not content_key:
        return 0.0

    params: list[object] = []
    exclude_clause = ""
    if exclude_memory_id:
        exclude_clause = " AND id != ?"
        params.append(exclude_memory_id)

    cursor = await db.execute(
        f"""SELECT id, content
            FROM memory_items
            WHERE content IS NOT NULL
              AND TRIM(content) != ''
              {exclude_clause}
            ORDER BY created_at DESC
            LIMIT ?""",
        tuple(params + [ADMISSION_SCAN_LIMIT]),
    )
    rows = await cursor.fetchall()

    highest_similarity = 0.0
    for row in rows:
        existing_key = normalize_query_key(row["content"] or "")
        if not existing_key:
            continue
        if existing_key == content_key:
            return 0.0
        highest_similarity = max(
            highest_similarity,
            _similarity(content_key, existing_key),
        )

    if highest_similarity >= 0.92:
        return 0.15
    if highest_similarity >= 0.82:
        return 0.35
    if highest_similarity >= 0.70:
        return 0.65
    return 1.0


def _tier_for_total(total: float) -> str:
    return "low_value" if total < LOW_VALUE_THRESHOLD else "standard"


async def compute_admission_score(
    content: str,
    kind: str,
    db,
    *,
    exclude_memory_id: str | None = None,
) -> AdmissionScore:
    llm_score = await score_admission_with_llm(content, kind)
    utility = _clamp_unit(llm_score["utility"])
    confidence = _clamp_unit(llm_score["confidence"])
    novelty = await _score_novelty(
        content,
        db,
        exclude_memory_id=exclude_memory_id,
    )
    recency = 1.0
    type_prior = _score_type_prior(kind)
    total = round(
        0.45 * utility
        + 0.20 * confidence
        + 0.15 * novelty
        + 0.05 * recency
        + 0.15 * type_prior,
        4,
    )
    return AdmissionScore(
        utility=utility,
        confidence=confidence,
        novelty=novelty,
        recency=recency,
        type_prior=type_prior,
        total=total,
        tier=_tier_for_total(total),
    )
