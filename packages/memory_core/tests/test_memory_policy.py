"""Unit tests for services/memory_policy.py.

All functions are pure / stateless — zero I/O, zero mocking needed.
"""

import pytest

from memory_core.services.memory_policy import (
    GRAPH_ELIGIBLE_MEMORY_KINDS,
    contains_any,
    contains_cjk,
    fact_supported_kinds,
    is_graph_eligible_kind,
    normalize_fact_text,
    normalize_query_key,
    normalize_text,
    task_fact_is_stable_long_term,
)


# ---------------------------------------------------------------------------
# contains_any
# ---------------------------------------------------------------------------


class TestContainsAny:
    def test_returns_true_when_pattern_found(self):
        assert contains_any("我喜欢吃火锅", ("我喜欢",)) is True

    def test_returns_false_when_no_pattern_matches(self):
        assert contains_any("今天天气不错", ("项目", "任务")) is False

    def test_case_insensitive_match(self):
        assert contains_any("I love Pervault", ("pervault",)) is True

    def test_empty_text_returns_false(self):
        assert contains_any("", ("项目",)) is False

    def test_whitespace_only_text_returns_false(self):
        assert contains_any("   ", ("项目",)) is False

    def test_empty_patterns_tuple_returns_false(self):
        assert contains_any("有内容的文本", ()) is False

    def test_partial_match_counts(self):
        # "项目" is a substring of "我在做项目推进"
        assert contains_any("我在做项目推进", ("项目",)) is True


# ---------------------------------------------------------------------------
# contains_cjk
# ---------------------------------------------------------------------------


class TestContainsCjk:
    def test_detects_common_cjk(self):
        assert contains_cjk("你好世界") is True

    def test_pure_ascii_returns_false(self):
        assert contains_cjk("hello world") is False

    def test_mixed_text_returns_true(self):
        assert contains_cjk("Pervault 项目") is True

    def test_numbers_only_returns_false(self):
        assert contains_cjk("12345") is False

    def test_empty_string_returns_false(self):
        assert contains_cjk("") is False

    def test_cjk_range_boundary(self):
        # U+4E00 is the canonical start of CJK Unified Ideographs — in range
        assert contains_cjk("\u4e00") is True

    def test_hiragana_not_detected(self):
        # Hiragana (U+3040-U+309F) is below the U+3400 threshold
        assert contains_cjk("\u3041") is False


# ---------------------------------------------------------------------------
# normalize_fact_text
# ---------------------------------------------------------------------------


class TestNormalizeFactText:
    def test_strips_leading_trailing_whitespace(self):
        assert normalize_fact_text("  hello  ") == "hello"

    def test_strips_cjk_trailing_punctuation(self):
        assert normalize_fact_text("我喜欢吃火锅。") == "我喜欢吃火锅"

    def test_strips_ascii_trailing_punctuation(self):
        assert normalize_fact_text("Pervault!") == "Pervault"

    def test_strips_multiple_trailing_punctuation(self):
        assert normalize_fact_text("内容，。！") == "内容"

    def test_preserves_inner_content(self):
        assert normalize_fact_text("项目，进展") == "项目，进展"

    def test_empty_string_stays_empty(self):
        assert normalize_fact_text("") == ""

    def test_only_punctuation_becomes_empty(self):
        assert normalize_fact_text("，。！") == ""

    def test_strips_colon(self):
        assert normalize_fact_text("结论：") == "结论"


# ---------------------------------------------------------------------------
# normalize_query_key
# ---------------------------------------------------------------------------


class TestNormalizeQueryKey:
    def test_removes_cjk_punctuation(self):
        assert normalize_query_key("你好，世界。") == "你好世界"

    def test_removes_whitespace(self):
        assert normalize_query_key("per vault") == "pervault"

    def test_lowercases_result(self):
        assert normalize_query_key("Pervault") == "pervault"

    def test_removes_question_marks(self):
        assert normalize_query_key("你在哪？") == "你在哪"

    def test_empty_string_stays_empty(self):
        assert normalize_query_key("") == ""

    def test_only_punctuation_becomes_empty(self):
        assert normalize_query_key("，。！？") == ""

    def test_mixed_content(self):
        result = normalize_query_key("  Hello，世界！  ")
        assert result == "hello世界"


# ---------------------------------------------------------------------------
# normalize_text
# ---------------------------------------------------------------------------


class TestNormalizeText:
    def test_collapses_internal_whitespace(self):
        assert normalize_text("hello   world") == "hello world"

    def test_strips_trailing_punctuation(self):
        assert normalize_text("内容。") == "内容"

    def test_strips_leading_whitespace(self):
        assert normalize_text("  前置空格") == "前置空格"

    def test_preserves_single_internal_space(self):
        assert normalize_text("a b") == "a b"

    def test_empty_string_stays_empty(self):
        assert normalize_text("") == ""

    def test_differs_from_normalize_fact_text_on_internal_space(self):
        # normalize_text collapses internal spaces; normalize_fact_text does not
        result_nt = normalize_text("a    b")
        result_nf = normalize_fact_text("a    b")
        assert result_nt == "a b"
        assert result_nf == "a    b"


# ---------------------------------------------------------------------------
# fact_supported_kinds
# ---------------------------------------------------------------------------


class TestFactSupportedKinds:
    def test_base_kinds_present(self):
        kinds = fact_supported_kinds()
        assert "project_update" in kinds
        assert "preference" in kinds
        assert "relationship_event" in kinds
        assert "fact" in kinds

    def test_task_excluded_by_default(self):
        kinds = fact_supported_kinds()
        assert "task" not in kinds

    def test_task_included_with_flag(self):
        kinds = fact_supported_kinds(include_task=True)
        assert "task" in kinds

    def test_base_kinds_still_present_with_task(self):
        kinds = fact_supported_kinds(include_task=True)
        assert "project_update" in kinds
        assert "preference" in kinds

    def test_returns_mutable_set_not_frozenset(self):
        kinds = fact_supported_kinds()
        assert isinstance(kinds, set)

    def test_other_not_included(self):
        kinds = fact_supported_kinds(include_task=True)
        assert "other" not in kinds


# ---------------------------------------------------------------------------
# is_graph_eligible_kind
# ---------------------------------------------------------------------------


class TestIsGraphEligibleKind:
    def test_project_update_is_eligible(self):
        assert is_graph_eligible_kind("project_update") is True

    def test_relationship_event_is_eligible(self):
        assert is_graph_eligible_kind("relationship_event") is True

    def test_preference_is_not_eligible(self):
        assert is_graph_eligible_kind("preference") is False

    def test_task_is_not_eligible(self):
        assert is_graph_eligible_kind("task") is False

    def test_fact_is_not_eligible(self):
        assert is_graph_eligible_kind("fact") is False

    def test_other_is_not_eligible(self):
        assert is_graph_eligible_kind("other") is False

    def test_unknown_kind_is_not_eligible(self):
        assert is_graph_eligible_kind("nonexistent_kind") is False

    def test_eligible_kinds_match_constant(self):
        # Confirm test coverage stays in sync with the constant
        for kind in GRAPH_ELIGIBLE_MEMORY_KINDS:
            assert is_graph_eligible_kind(kind) is True


# ---------------------------------------------------------------------------
# task_fact_is_stable_long_term
# ---------------------------------------------------------------------------


class TestTaskFactIsStableLongTerm:
    def test_returns_true_long_term_preference(self):
        assert task_fact_is_stable_long_term("我一直习惯早起", "preference") is True

    def test_returns_true_long_term_fact(self):
        assert task_fact_is_stable_long_term("我长期在远程工作", "fact") is True

    def test_returns_true_with_persist_pattern(self):
        assert task_fact_is_stable_long_term("我坚持每天跑步", "preference") is True

    def test_returns_false_disallowed_fact_kind_task(self):
        # "task" is never a valid promoted kind
        assert task_fact_is_stable_long_term("我一直习惯早起", "task") is False

    def test_returns_false_disallowed_fact_kind_other(self):
        assert task_fact_is_stable_long_term("我长期在远程工作", "other") is False

    def test_returns_false_has_short_term_signal_woyao(self):
        # "我要" is a short-term signal
        assert task_fact_is_stable_long_term("我要开始坚持早起", "preference") is False

    def test_returns_false_has_short_term_signal_tomorrow(self):
        assert task_fact_is_stable_long_term("明天开始长期坚持跑步", "fact") is False

    def test_returns_false_no_long_term_signal(self):
        # Valid kind, no short-term, but also no long-term marker
        assert task_fact_is_stable_long_term("我喜欢吃火锅", "preference") is False

    def test_returns_false_empty_content(self):
        assert task_fact_is_stable_long_term("", "preference") is False

    def test_returns_false_whitespace_only_content(self):
        assert task_fact_is_stable_long_term("   ", "fact") is False

    def test_short_term_overrides_long_term_signal(self):
        # Both signals present — short-term check comes first → False
        assert task_fact_is_stable_long_term("打算长期坚持跑步", "preference") is False


# ---------------------------------------------------------------------------
# should_update_memory_kind
# ---------------------------------------------------------------------------


class TestShouldUpdateMemoryKind:
    """Pure predicate — zero I/O, zero mocking needed."""

    def test_other_to_preference_should_update(self):
        from memory_core.services.memory_policy import should_update_memory_kind
        assert should_update_memory_kind("other", "preference") is True

    def test_other_to_project_update_should_update(self):
        from memory_core.services.memory_policy import should_update_memory_kind
        assert should_update_memory_kind("other", "project_update") is True

    def test_other_to_fact_should_update(self):
        from memory_core.services.memory_policy import should_update_memory_kind
        assert should_update_memory_kind("other", "fact") is True

    def test_other_to_relationship_event_should_update(self):
        from memory_core.services.memory_policy import should_update_memory_kind
        assert should_update_memory_kind("other", "relationship_event") is True

    def test_preference_to_fact_should_update(self):
        # Keyword classifier and LLM disagree — LLM wins
        from memory_core.services.memory_policy import should_update_memory_kind
        assert should_update_memory_kind("preference", "fact") is True

    def test_same_kind_no_update(self):
        from memory_core.services.memory_policy import should_update_memory_kind
        assert should_update_memory_kind("preference", "preference") is False

    def test_any_to_task_no_update(self):
        # task must never be backwritten from consolidation
        from memory_core.services.memory_policy import should_update_memory_kind
        assert should_update_memory_kind("other", "task") is False

    def test_preference_to_task_no_update(self):
        from memory_core.services.memory_policy import should_update_memory_kind
        assert should_update_memory_kind("preference", "task") is False

    def test_new_kind_other_no_update(self):
        # "other" is not in _BASE_FACT_KINDS — LLM should never return it but guard anyway
        from memory_core.services.memory_policy import should_update_memory_kind
        assert should_update_memory_kind("preference", "other") is False

    def test_new_kind_illegal_no_update(self):
        # Arbitrary LLM hallucination value
        from memory_core.services.memory_policy import should_update_memory_kind
        assert should_update_memory_kind("other", "unknown_type") is False

    def test_new_kind_empty_no_update(self):
        from memory_core.services.memory_policy import should_update_memory_kind
        assert should_update_memory_kind("other", "") is False

    def test_original_task_to_preference_should_update(self):
        # A task memory whose LLM-promoted fact kind is preference → backwrite is fine
        from memory_core.services.memory_policy import should_update_memory_kind
        assert should_update_memory_kind("task", "preference") is True

    def test_original_task_to_fact_should_update(self):
        from memory_core.services.memory_policy import should_update_memory_kind
        assert should_update_memory_kind("task", "fact") is True


# ---------------------------------------------------------------------------
# classify_memory_kind — regression + positive cases
# ---------------------------------------------------------------------------


class TestClassifyMemoryKind:
    """Tests for services.memory_service.classify_memory_kind.

    Two regression cases guard against previously reported misclassifications.
    Positive cases ensure existing correct classifications are not broken.
    """

    # ── regression: previously misclassified ──────────────────────────────

    def test_woyaoqu_not_task(self):
        # "我要去找小王讨论项目" — "我要" was too broad a trigger for task;
        # after tightening to "我要做", this should NOT be classified as task.
        from memory_core.services.memory_service import classify_memory_kind
        assert classify_memory_kind("我要去找小王讨论项目") != "task"

    def test_project_update_with_xihuan_not_preference(self):
        # "今天推进了，我喜欢这种进度" — bare "我喜欢" should not override
        # the stronger project signal "推进了".
        from memory_core.services.memory_service import classify_memory_kind
        assert classify_memory_kind("今天推进了，我喜欢这种进度") == "project_update"

    # ── positive: correct classifications must remain intact ──────────────

    def test_specific_task_still_task(self):
        # "我要做" is the tightened pattern — must still match.
        from memory_core.services.memory_service import classify_memory_kind
        assert classify_memory_kind("我要做一个演示文稿") == "task"

    def test_clear_preference(self):
        from memory_core.services.memory_service import classify_memory_kind
        assert classify_memory_kind("我更喜欢吃辣的") == "preference"

    def test_clear_relationship_event(self):
        from memory_core.services.memory_service import classify_memory_kind
        assert classify_memory_kind("今天和小王聊了项目进展") == "relationship_event"

    def test_clear_project_update(self):
        from memory_core.services.memory_service import classify_memory_kind
        assert classify_memory_kind("今天推进了Pervault的图谱功能") == "project_update"
