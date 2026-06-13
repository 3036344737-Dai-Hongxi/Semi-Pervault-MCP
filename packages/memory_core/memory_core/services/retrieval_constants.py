"""Retrieval layer — shared constants, pattern lists, and type aliases.

This module is the single source of truth for all pattern lists, stopword
sets, numeric limits, and type aliases used across the retrieval sub-package.

Contents are limited to:
  - Numeric tuning constants (limits, weights)
  - Pattern tuples used for keyword matching
  - Stopword sets used for query normalisation
  - Typed aliases (QueryIntent, SourceTag)

This module must NOT contain:
  - Any I/O or DB access
  - Any function definitions
  - Any imports from other services/* modules
"""

from typing import Literal

# ---------------------------------------------------------------------------
# Numeric limits and weights
# ---------------------------------------------------------------------------

MAX_LAYER_RESULTS = 5
MAX_TOTAL_RESULTS = 10
MAX_GRAPH_CONTEXT_EDGES = 12
SUMMARY_MEMORY_LIMIT = 8
SUMMARY_MEMORY_SCAN_LIMIT = 30
INTENT_MEMORY_SCAN_LIMIT = 40
STRUCTURED_FACT_SCAN_LIMIT = 40
HYBRID_VECTOR_K = 12
HYBRID_KEYWORD_SCAN_LIMIT = 20
HYBRID_SCORE_LOG_LIMIT = 10
SEMANTIC_SCORE_WEIGHT = 0.7
KEYWORD_SCORE_WEIGHT = 0.3
BOOT_CONTEXT_MAX_RESULTS = 8
BOOT_CONTEXT_FACT_LIMIT = 4
BOOT_CONTEXT_PERSONA_LIMIT = 3
BOOT_CONTEXT_REFLECTION_LIMIT = 2
PROJECT_FALLBACK_MAX_ADDITIONS = 2
PREFERENCE_FALLBACK_MAX_ADDITIONS = 2
PROJECT_GENERIC_MAX_ADDITIONS = 1
PREFERENCE_GENERIC_MAX_ADDITIONS = 1

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

QueryIntent = Literal[
    "correction_intent",
    "task_query",
    "persona_query",
    "project_query",
    "preference_query",
    "people_query",
    "summary_query",
    "generic",
]

SourceTag = Literal[
    "persona",
    "structured_fact",
    "hybrid",
    "graph",
    "pattern",
    "recent",
    "boot_fact",
    "boot_kind",
    "boot_reflection",
]

# ---------------------------------------------------------------------------
# Boot context kind limits: (kind, per-kind max)
# ---------------------------------------------------------------------------

BOOT_CONTEXT_KIND_LIMITS: tuple[tuple[str, int], ...] = (
    ("project_update", 2),
    ("preference", 2),
    ("relationship_event", 2),
    ("task", 2),
)

SUMMARY_STRUCTURED_FACT_KINDS: tuple[str, ...] = (
    "project_update",
    "preference",
    "relationship_event",
)

# ---------------------------------------------------------------------------
# Low-value message keys (normalised, no punctuation/whitespace)
# ---------------------------------------------------------------------------

LOW_VALUE_MESSAGE_KEYS: set[str] = {
    "哈哈",
    "你好",
    "哈咯",
    "hi",
    "保存",
    "在吗",
    "ok",
    "嗯",
    "哦",
}

# ---------------------------------------------------------------------------
# Query intent patterns
# Used by retrieval_intent.py to detect what the user is asking about.
# These are *query-side* patterns (what the user asks); the corresponding
# *store-side* patterns (what the user said) live in memory_service.py.
# ---------------------------------------------------------------------------

SUMMARY_QUERY_PATTERNS: tuple[str, ...] = (
    "我最近都干什么了",
    "我最近在做什么",
    "我是谁",
    "你记得我说过什么",
)

CORRECTION_QUERY_PATTERNS: tuple[str, ...] = (
    "不对",
    "你记错了",
    "不是这样的",
    "我改变主意了",
    "以后不要这样记",
    "纠正一下",
    "不是，我",
    "其实我",
    "我不喜欢",
    "我更喜欢",
)

EXPLICIT_CORRECTION_QUERY_PATTERNS: tuple[str, ...] = (
    "不对",
    "你记错了",
    "不是这样的",
    "我改变主意了",
    "以后不要这样记",
    "纠正一下",
    "不是，我",
)

PERSONA_QUERY_PATTERNS: tuple[str, ...] = (
    "我的习惯",
    "我的风格",
    "我的沟通风格",
    "我的工作方式",
    "我通常",
    "我是一个什么样的人",
    "你觉得我是怎样的人",
    "我的长期偏好",
)

PROJECT_QUERY_PATTERNS: tuple[str, ...] = (
    "什么项目",
    "哪些项目",
    "在做什么项目",
    "做什么项目",
    "项目进展",
    "项目推进",
    "项目",
)

PREFERENCE_QUERY_PATTERNS: tuple[str, ...] = (
    "喜欢什么",
    "喜欢吃什么",
    "喜欢喝什么",
    "喜欢穿什么",
    "爱吃什么",
    "想吃什么",
    "什么口味",
    "什么偏好",
    "偏好",
    "偏向",
    "更喜欢",
    "想买什么",
    "想喝什么",
)

PEOPLE_QUERY_PATTERNS: tuple[str, ...] = (
    "和谁",
    "跟谁",
    "与谁",
    "谁有交集",
    "联系过谁",
    "见了谁",
    "聊过谁",
    "接触过谁",
)

TASK_QUERY_PATTERNS: tuple[str, ...] = (
    "任务",
    "待办",
    "todo",
    "TODO",
    "还要做什么",
    "还有什么要做",
    "要做什么",
    "没完成",
    "还没完成",
    "进行中的任务",
)

# Patterns that, if present, indicate the user is *recording* rather than
# querying — used by is_summary_candidate to exclude such messages.
SUMMARY_EXCLUDED_PATTERNS: tuple[str, ...] = (
    "帮我记一下",
    "帮我记录一下",
    "帮我记录下来",
    "记一下",
    "记录一下",
    "记住这个",
    "你能记得我发啥了吗",
    "去知识图谱里看",
)

# ---------------------------------------------------------------------------
# Memory content matching patterns
# Used when scanning recent memories to check relevance to a query intent.
# ---------------------------------------------------------------------------

PROJECT_MEMORY_PATTERNS: tuple[str, ...] = (
    "项目",
    "开发",
    "推进",
    "迭代",
    "版本",
    "上线",
    "stage",
)

PROJECT_FALLBACK_STRONG_PATTERNS: tuple[str, ...] = (
    "项目",
    "开发",
    "推进",
    "迭代",
    "stage",
)

PREFERENCE_MEMORY_PATTERNS: tuple[str, ...] = (
    "喜欢",
    "爱吃",
    "想吃",
    "口味",
    "偏好",
    "偏向",
    "更喜欢",
    "想买",
    "想喝",
    "不喜欢",
)

PEOPLE_MEMORY_PATTERNS: tuple[str, ...] = (
    "一起",
    "见了",
    "聊了",
    "讨论了",
    "联系了",
)

# ---------------------------------------------------------------------------
# Stopword sets
# ---------------------------------------------------------------------------

# Stopwords for graph-term extraction (query → graph node label search).
GRAPH_STOPWORDS: set[str] = {
    "什么",
    "怎么",
    "为什么",
    "最近",
    "一下",
    "帮我",
    "请问",
    "知道",
    "可以",
    "一下子",
    "这个",
    "那个",
    "哪些",
    "多少",
    "是谁",
    "谁是",
}

# Stopwords for structured-fact and hybrid query term extraction.
STRUCTURED_FACT_QUERY_STOPWORDS: set[str] = {
    "最近",
    "什么",
    "项目",
    "口味",
    "偏好",
    "喜欢",
    "想吃",
    "想喝",
    "和谁",
    "跟谁",
    "与谁",
    "交集",
    "联系",
    "见了",
    "聊过",
    "接触",
    "我的",
    "你的",
    "一下",
}
