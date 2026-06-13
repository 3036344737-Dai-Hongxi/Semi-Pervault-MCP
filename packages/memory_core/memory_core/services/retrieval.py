"""Retrieval service — public API and backward-compatibility shim.

The retrieval layer has been split into focused sub-modules:

  retrieval_constants.py   — pattern lists, stopword sets, numeric limits
  retrieval_intent.py      — query intent detection (LLM + keyword fallback)
  retrieval_primitives.py  — low-level DB query functions, filters, helpers
  retrieval_context.py     — intent-specific retrieval pipelines
  retrieval_boot.py        — boot-context (session startup) assembly
  graph_retrieval.py       — graph-context retrieval (query → triples → string)

This file re-exports the complete public surface so that all existing callers
(routers/chat.py, scripts/, tests/) continue to work without changes.

New code should import directly from the relevant sub-module.
"""

# Re-export everything callers currently import from memory_core.services.retrieval
from memory_core.services.graph_retrieval import (
    _extract_graph_terms as _extract_graph_terms,
    retrieve_graph_context as retrieve_graph_context,
)
from memory_core.services.retrieval_boot import get_boot_context as get_boot_context
from memory_core.services.retrieval_constants import (
    BOOT_CONTEXT_FACT_LIMIT as BOOT_CONTEXT_FACT_LIMIT,
    BOOT_CONTEXT_KIND_LIMITS as BOOT_CONTEXT_KIND_LIMITS,
    BOOT_CONTEXT_MAX_RESULTS as BOOT_CONTEXT_MAX_RESULTS,
    BOOT_CONTEXT_PERSONA_LIMIT as BOOT_CONTEXT_PERSONA_LIMIT,
    GRAPH_STOPWORDS as GRAPH_STOPWORDS,
    HYBRID_KEYWORD_SCAN_LIMIT as HYBRID_KEYWORD_SCAN_LIMIT,
    HYBRID_SCORE_LOG_LIMIT as HYBRID_SCORE_LOG_LIMIT,
    HYBRID_VECTOR_K as HYBRID_VECTOR_K,
    INTENT_MEMORY_SCAN_LIMIT as INTENT_MEMORY_SCAN_LIMIT,
    KEYWORD_SCORE_WEIGHT as KEYWORD_SCORE_WEIGHT,
    LOW_VALUE_MESSAGE_KEYS as LOW_VALUE_MESSAGE_KEYS,
    MAX_GRAPH_CONTEXT_EDGES as MAX_GRAPH_CONTEXT_EDGES,
    MAX_LAYER_RESULTS as MAX_LAYER_RESULTS,
    MAX_TOTAL_RESULTS as MAX_TOTAL_RESULTS,
    PEOPLE_MEMORY_PATTERNS as PEOPLE_MEMORY_PATTERNS,
    PEOPLE_QUERY_PATTERNS as PEOPLE_QUERY_PATTERNS,
    PERSONA_QUERY_PATTERNS as PERSONA_QUERY_PATTERNS,
    PREFERENCE_FALLBACK_MAX_ADDITIONS as PREFERENCE_FALLBACK_MAX_ADDITIONS,
    PREFERENCE_GENERIC_MAX_ADDITIONS as PREFERENCE_GENERIC_MAX_ADDITIONS,
    PREFERENCE_MEMORY_PATTERNS as PREFERENCE_MEMORY_PATTERNS,
    PREFERENCE_QUERY_PATTERNS as PREFERENCE_QUERY_PATTERNS,
    PROJECT_FALLBACK_MAX_ADDITIONS as PROJECT_FALLBACK_MAX_ADDITIONS,
    PROJECT_FALLBACK_STRONG_PATTERNS as PROJECT_FALLBACK_STRONG_PATTERNS,
    PROJECT_GENERIC_MAX_ADDITIONS as PROJECT_GENERIC_MAX_ADDITIONS,
    PROJECT_MEMORY_PATTERNS as PROJECT_MEMORY_PATTERNS,
    PROJECT_QUERY_PATTERNS as PROJECT_QUERY_PATTERNS,
    QueryIntent as QueryIntent,
    SEMANTIC_SCORE_WEIGHT as SEMANTIC_SCORE_WEIGHT,
    SourceTag as SourceTag,
    STRUCTURED_FACT_QUERY_STOPWORDS as STRUCTURED_FACT_QUERY_STOPWORDS,
    STRUCTURED_FACT_SCAN_LIMIT as STRUCTURED_FACT_SCAN_LIMIT,
    SUMMARY_EXCLUDED_PATTERNS as SUMMARY_EXCLUDED_PATTERNS,
    SUMMARY_MEMORY_LIMIT as SUMMARY_MEMORY_LIMIT,
    SUMMARY_MEMORY_SCAN_LIMIT as SUMMARY_MEMORY_SCAN_LIMIT,
    SUMMARY_QUERY_PATTERNS as SUMMARY_QUERY_PATTERNS,
    SUMMARY_STRUCTURED_FACT_KINDS as SUMMARY_STRUCTURED_FACT_KINDS,
    TASK_QUERY_PATTERNS as TASK_QUERY_PATTERNS,
)
from memory_core.services.retrieval_context import retrieve_context as retrieve_context
from memory_core.services.retrieval_intent import (
    _detect_query_intent_keyword as _detect_query_intent_keyword,
    detect_query_intent as detect_query_intent,
    is_low_value_content as is_low_value_content,
    is_summary_candidate as is_summary_candidate,
    is_summary_query as is_summary_query,
)
from memory_core.services.retrieval_primitives import (
    _build_kind_clause as _build_kind_clause,
    _build_task_visibility_clause as _build_task_visibility_clause,
    _extract_hybrid_query_terms as _extract_hybrid_query_terms,
    _extract_project_anchor_terms as _extract_project_anchor_terms,
    _extract_structured_fact_query_terms as _extract_structured_fact_query_terms,
    _filter_preference_fallback_candidates as _filter_preference_fallback_candidates,
    _filter_preference_generic_candidates as _filter_preference_generic_candidates,
    _filter_project_fallback_candidates as _filter_project_fallback_candidates,
    _filter_project_generic_candidates as _filter_project_generic_candidates,
    _matching_patterns as _matching_patterns,
    _matching_query_terms as _matching_query_terms,
    _merge_prioritized_results as _merge_prioritized_results,
    _retrieve_graph_matches as _retrieve_graph_matches,
    _retrieve_hybrid_context as _retrieve_hybrid_context,
    _retrieve_hybrid_keyword_candidates as _retrieve_hybrid_keyword_candidates,
    _retrieve_hybrid_semantic_candidates as _retrieve_hybrid_semantic_candidates,
    _retrieve_memories_by_graph_type as _retrieve_memories_by_graph_type,
    _retrieve_memory_matches as _retrieve_memory_matches,
    _retrieve_persona_traits as _retrieve_persona_traits,
    _retrieve_recent_high_value_memories as _retrieve_recent_high_value_memories,
    _retrieve_recent_kind_memories as _retrieve_recent_kind_memories,
    _retrieve_recent_pattern_memories as _retrieve_recent_pattern_memories,
    _retrieve_structured_fact_memories as _retrieve_structured_fact_memories,
    _retrieve_vector_matches as _retrieve_vector_matches,
    _row_to_brief as _row_to_brief,
    _source_composition as _source_composition,
    _tag_source as _tag_source,
)
