"""Retrieval layer — query intent detection.

Responsible for deciding *what kind of information* the user is asking for so
that retrieve_context() in retrieval_context.py can route to the correct
retrieval pipeline.

Two detection strategies are layered:
  1. LLM-based (primary): calls classify_query_intent() in services/llm.py.
     More accurate, especially for ambiguous or mixed-intent queries.
  2. Keyword-based (fallback): pure synchronous pattern matching against
     pattern lists defined in retrieval_constants.py.
     Used when the LLM call fails, times out, or returns an invalid label.

Public API:
  detect_query_intent(query)        async, LLM + fallback
  is_low_value_content(text)        sync, filter trivial messages
  is_summary_query(query)           sync, keyword-only (intentionally cheap)
  is_summary_candidate(text)        sync, filter non-summarisable messages
  _detect_query_intent_keyword(q)   sync, internal + exported for tests
"""

import logging

from memory_core.services.llm import classify_query_intent
from memory_core.services.logging_policy import text_fingerprint
from memory_core.services.memory_policy import contains_any
from memory_core.services.retrieval_constants import (
    CORRECTION_QUERY_PATTERNS,
    EXPLICIT_CORRECTION_QUERY_PATTERNS,
    LOW_VALUE_MESSAGE_KEYS,
    PEOPLE_QUERY_PATTERNS,
    PERSONA_QUERY_PATTERNS,
    PREFERENCE_QUERY_PATTERNS,
    PROJECT_QUERY_PATTERNS,
    QueryIntent,
    SUMMARY_EXCLUDED_PATTERNS,
    SUMMARY_QUERY_PATTERNS,
    TASK_QUERY_PATTERNS,
)

logger = logging.getLogger("uvicorn.error")

# ---------------------------------------------------------------------------
# LLM label → QueryIntent mapping
# ---------------------------------------------------------------------------

_INTENT_LABEL_TO_QUERY_INTENT: dict[str, QueryIntent] = {
    "correction": "correction_intent",
    "project": "project_query",
    "persona": "persona_query",
    "preference": "preference_query",
    "people": "people_query",
    "task": "task_query",
    "summary": "summary_query",
    "generic": "generic",
}


# ---------------------------------------------------------------------------
# Low-value content filter
# ---------------------------------------------------------------------------


def is_low_value_content(text: str) -> bool:
    """Return True when *text* is too trivial to store or retrieve against."""
    normalized = _normalize_key(text)
    if not normalized:
        return True
    return normalized in LOW_VALUE_MESSAGE_KEYS


def _normalize_key(text: str) -> str:
    """Strip punctuation and whitespace, then lowercase — used internally."""
    import re
    return re.sub(r"[，。！？、,.!?\s]", "", text.strip().lower())


_QUESTION_TOKENS: tuple[str, ...] = (
    "?",
    "？",
    "什么",
    "哪种",
    "哪个",
    "哪些",
    "吗",
    "嘛",
    "么",
    "呢",
)


def _has_explicit_correction_framing(query: str) -> bool:
    return contains_any(query, EXPLICIT_CORRECTION_QUERY_PATTERNS)


def _looks_like_preference_question(query: str) -> bool:
    if not contains_any(query, PREFERENCE_QUERY_PATTERNS):
        return False
    return any(token in query for token in _QUESTION_TOKENS)


# ---------------------------------------------------------------------------
# Keyword-based intent detection (sync, no I/O)
# ---------------------------------------------------------------------------


def _detect_query_intent_keyword(query: str) -> QueryIntent:
    """Keyword-based intent detection. Used as fallback and for sync-only paths.

    Priority order (highest to lowest):
      correction → task → persona → project → preference → people → summary → generic
    """
    normalized = query.strip()
    if not normalized:
        return "generic"

    if _has_explicit_correction_framing(normalized):
        return "correction_intent"
    if contains_any(normalized, TASK_QUERY_PATTERNS):
        return "task_query"
    if contains_any(normalized, PERSONA_QUERY_PATTERNS):
        return "persona_query"
    if _looks_like_preference_question(normalized):
        return "preference_query"
    if contains_any(normalized, CORRECTION_QUERY_PATTERNS):
        return "correction_intent"
    if contains_any(normalized, PROJECT_QUERY_PATTERNS):
        return "project_query"
    if contains_any(normalized, PREFERENCE_QUERY_PATTERNS):
        return "preference_query"
    if contains_any(normalized, PEOPLE_QUERY_PATTERNS):
        return "people_query"
    if contains_any(normalized, SUMMARY_QUERY_PATTERNS):
        return "summary_query"
    return "generic"


# ---------------------------------------------------------------------------
# LLM-based intent detection (async, falls back to keyword on any error)
# ---------------------------------------------------------------------------


async def detect_query_intent(query: str) -> QueryIntent:
    """Keyword-first intent detection with LLM fallback for generic queries.

    Fast path: keyword classifier handles most common intents with no I/O.
    Slow path: LLM is only called when keyword returns 'generic', covering
    ambiguous queries that patterns cannot resolve.
    """
    # Fast path: keyword classifier covers most common intents
    keyword_intent = _detect_query_intent_keyword(query)
    if keyword_intent != "generic":
        logger.info(
            "intent classification keyword intent=%s query_fp=%s query_len=%d",
            keyword_intent,
            text_fingerprint(query),
            len(query.strip()),
        )
        return keyword_intent

    # Slow path: only call LLM when keyword returns generic
    try:
        result = await classify_query_intent(query)
        intent_label = result["intent"]
        intent: QueryIntent = _INTENT_LABEL_TO_QUERY_INTENT[intent_label]
        logger.info(
            "intent classification llm intent=%s confidence=%.2f reason=%s query_fp=%s query_len=%d",
            intent,
            result["confidence"],
            result["reason"],
            text_fingerprint(query),
            len(query.strip()),
        )
        return intent
    except Exception as exc:
        logger.warning(
            "intent classification fallback exc_type=%s exc=%s fallback_intent=%s query_fp=%s query_len=%d",
            type(exc).__name__,
            exc,
            keyword_intent,
            text_fingerprint(query),
            len(query.strip()),
        )
        return keyword_intent


# ---------------------------------------------------------------------------
# Helper predicates (sync, keyword-only by design)
# ---------------------------------------------------------------------------


def is_summary_query(query: str) -> bool:
    """Return True when the query asks for a general memory summary.

    Intentionally uses the keyword path (not LLM) because this predicate is
    called inside a tight filter loop over candidate memories in
    _retrieve_recent_high_value_memories().  Calling the LLM per-candidate
    would be prohibitively expensive.
    """
    return _detect_query_intent_keyword(query) == "summary_query"


def is_summary_candidate(text: str) -> bool:
    """Return True when *text* is suitable content for a summary response.

    Filters out low-value messages, questions, and explicit record commands
    that should not be surfaced as summarised memory items.
    """
    stripped = text.strip()
    if is_low_value_content(stripped):
        return False
    if is_summary_query(stripped):
        return False
    if any(pattern in stripped for pattern in SUMMARY_EXCLUDED_PATTERNS):
        return False
    if "?" in stripped or "？" in stripped:
        return False
    if stripped.endswith(("吗", "么", "嘛")):
        return False
    return True
