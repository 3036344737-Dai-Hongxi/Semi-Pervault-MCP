"""Golden query definitions for retrieval regression checks.

Each entry specifies:
- query:           the user input
- expected_intent: what detect_query_intent should return
- checks:          list of (check_name, check_config) pairs applied when
                   sources are non-empty.  When sources are empty the
                   check is automatically SKIP.

Supported check types
---------------------
intent_match        Always runs.  Verifies detected intent == expected.
fact_present        PASS if source_composition contains "structured_fact".
fact_priority       PASS if the first source has _source == "structured_fact".
no_closed_tasks     PASS if no returned source has task_status in
                    ("done", "expired").  Requires a DB lookup on
                    memory_items.task_status.
"""

from typing import Any

GoldenQuery = dict[str, Any]

GOLDEN_QUERIES: list[GoldenQuery] = [
    # ── project_query ──────────────────────────────────────
    {
        "query": "我在做什么项目",
        "expected_intent": "project_query",
        "checks": ["intent_match", "fact_priority"],
    },
    {
        "query": "项目进展怎么样",
        "expected_intent": "project_query",
        "checks": ["intent_match", "fact_priority"],
    },
    {
        "query": "哪些项目在推进",
        "expected_intent": "project_query",
        "checks": ["intent_match", "fact_priority"],
    },
    # ── preference_query ───────────────────────────────────
    {
        "query": "我喜欢什么",
        "expected_intent": "preference_query",
        "checks": ["intent_match", "fact_present"],
    },
    {
        "query": "我爱吃什么",
        "expected_intent": "preference_query",
        "checks": ["intent_match", "fact_present"],
    },
    {
        "query": "我的口味偏好",
        "expected_intent": "preference_query",
        "checks": ["intent_match", "fact_present"],
    },
    # ── summary_query ──────────────────────────────────────
    {
        "query": "我最近都干什么了",
        "expected_intent": "summary_query",
        "checks": ["intent_match", "fact_present"],
    },
    {
        "query": "我最近在做什么",
        "expected_intent": "summary_query",
        "checks": ["intent_match", "fact_present"],
    },
    # ── task_query ─────────────────────────────────────────
    {
        "query": "我还有什么任务没完成",
        "expected_intent": "task_query",
        "checks": ["intent_match", "no_closed_tasks"],
    },
    {
        "query": "我的待办是什么",
        "expected_intent": "task_query",
        "checks": ["intent_match", "no_closed_tasks"],
    },
]
