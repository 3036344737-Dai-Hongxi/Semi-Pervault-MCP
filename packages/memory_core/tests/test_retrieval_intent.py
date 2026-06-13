"""Unit tests for retrieval intent detection.

Two layers:
  Layer A — _detect_query_intent_keyword: pure sync, no mock needed.
  Layer B — detect_query_intent: async, mocks classify_query_intent.
"""

from unittest.mock import AsyncMock, patch

import pytest

from memory_core.services.retrieval import (
    _detect_query_intent_keyword,
    detect_query_intent,
)


# ---------------------------------------------------------------------------
# Layer A: _detect_query_intent_keyword (sync, no mock)
# ---------------------------------------------------------------------------


class TestDetectQueryIntentKeyword:
    def test_correction_query_memory_wrong(self):
        assert _detect_query_intent_keyword("你记错了，我不喜欢被催") == "correction_intent"

    def test_correction_query_direct_preference(self):
        assert _detect_query_intent_keyword("不对，我更喜欢直接一点") == "correction_intent"

    def test_correction_query_changed_food_preference(self):
        assert _detect_query_intent_keyword("其实我不喜欢吃辣了") == "correction_intent"

    def test_preference_question_more_like_is_not_swallowed_by_correction(self):
        assert _detect_query_intent_keyword("我更喜欢什么口味？") == "preference_query"

    def test_preference_question_dislike_is_not_swallowed_by_correction(self):
        assert _detect_query_intent_keyword("我不喜欢什么口味？") == "preference_query"

    def test_project_query(self):
        assert _detect_query_intent_keyword("我现在在做什么项目") == "project_query"

    def test_project_query_progress(self):
        assert _detect_query_intent_keyword("项目进展怎么样了") == "project_query"

    def test_preference_query_likes(self):
        # PREFERENCE_QUERY_PATTERNS includes "喜欢什么" — the 吃 in "喜欢吃什么"
        # breaks the substring match, so use "我喜欢什么" which matches directly.
        # NOTE: "我喜欢吃什么" returning generic is a known keyword-pattern gap
        # (see production defect noted in test file footer).
        assert _detect_query_intent_keyword("我喜欢什么") == "preference_query"

    def test_preference_query_wants(self):
        assert _detect_query_intent_keyword("我想喝什么") == "preference_query"

    def test_preference_query_tendency(self):
        assert _detect_query_intent_keyword("我有什么偏好") == "preference_query"

    def test_people_query_with_whom(self):
        assert _detect_query_intent_keyword("我最近和谁见过面") == "people_query"

    def test_people_query_contacted(self):
        assert _detect_query_intent_keyword("我联系过谁") == "people_query"

    def test_persona_query_communication_style(self):
        assert _detect_query_intent_keyword("我的沟通风格是什么") == "persona_query"

    def test_persona_query_long_term_preference(self):
        assert _detect_query_intent_keyword("我的长期偏好是什么") == "persona_query"

    def test_task_query_todo(self):
        assert _detect_query_intent_keyword("我有什么待办任务") == "task_query"

    def test_task_query_uppercase_todo(self):
        # TASK_QUERY_PATTERNS contains "TODO" (uppercase)
        assert _detect_query_intent_keyword("还有什么TODO没完成") == "task_query"

    def test_task_query_not_done(self):
        assert _detect_query_intent_keyword("我还有什么没完成的") == "task_query"

    def test_summary_query(self):
        assert _detect_query_intent_keyword("我最近都干什么了") == "summary_query"

    def test_summary_query_who_am_i(self):
        assert _detect_query_intent_keyword("我是谁") == "summary_query"

    def test_generic_fallback_unrelated(self):
        assert _detect_query_intent_keyword("今天天气不错") == "generic"

    def test_generic_fallback_empty_string(self):
        assert _detect_query_intent_keyword("") == "generic"

    def test_generic_fallback_whitespace_only(self):
        assert _detect_query_intent_keyword("   ") == "generic"

    def test_task_takes_priority_over_project(self):
        # "任务" triggers task_query; "项目" alone would trigger project_query
        # task patterns are checked before project patterns in the function
        assert _detect_query_intent_keyword("我的项目任务有哪些") == "task_query"

    def test_task_takes_priority_over_summary(self):
        assert _detect_query_intent_keyword("我最近的待办任务") == "task_query"

    def test_preference_query_likes_eat(self):
        assert _detect_query_intent_keyword("我喜欢吃什么") == "preference_query"

    def test_preference_query_likes_drink(self):
        assert _detect_query_intent_keyword("我喜欢喝什么") == "preference_query"

    def test_preference_query_likes_wear(self):
        assert _detect_query_intent_keyword("我喜欢穿什么") == "preference_query"


# ---------------------------------------------------------------------------
# Layer B: detect_query_intent (async, mocks classify_query_intent)
# ---------------------------------------------------------------------------

_MOCK_TARGET = "memory_core.services.retrieval_intent.classify_query_intent"


class TestDetectQueryIntentAsync:
    async def test_llm_correction_label_returns_correction_intent(self):
        mock = AsyncMock(return_value={"intent": "correction", "confidence": 0.9, "reason": "ok"})
        with patch(_MOCK_TARGET, mock):
            result = await detect_query_intent("不是这个意思")
        assert result == "correction_intent"

    async def test_keyword_preference_boundary_skips_llm(self):
        mock = AsyncMock(return_value={"intent": "correction", "confidence": 0.9, "reason": "ok"})
        with patch(_MOCK_TARGET, mock):
            result = await detect_query_intent("我更喜欢什么口味？")
        assert result == "preference_query"
        mock.assert_not_awaited()

    async def test_llm_project_label_returns_project_query(self):
        mock = AsyncMock(return_value={"intent": "project", "confidence": 0.9, "reason": "ok"})
        with patch(_MOCK_TARGET, mock):
            result = await detect_query_intent("我在做什么项目")
        assert result == "project_query"

    async def test_llm_preference_label_returns_preference_query(self):
        mock = AsyncMock(return_value={"intent": "preference", "confidence": 0.85, "reason": ""})
        with patch(_MOCK_TARGET, mock):
            result = await detect_query_intent("我喜欢吃什么")
        assert result == "preference_query"

    async def test_llm_persona_label_returns_persona_query(self):
        mock = AsyncMock(return_value={"intent": "persona", "confidence": 0.86, "reason": ""})
        with patch(_MOCK_TARGET, mock):
            result = await detect_query_intent("形容一下我")
        assert result == "persona_query"

    async def test_llm_people_label_returns_people_query(self):
        mock = AsyncMock(return_value={"intent": "people", "confidence": 0.8, "reason": ""})
        with patch(_MOCK_TARGET, mock):
            result = await detect_query_intent("和谁见面了")
        assert result == "people_query"

    async def test_llm_task_label_returns_task_query(self):
        mock = AsyncMock(return_value={"intent": "task", "confidence": 0.88, "reason": ""})
        with patch(_MOCK_TARGET, mock):
            result = await detect_query_intent("待办有哪些")
        assert result == "task_query"

    async def test_llm_summary_label_returns_summary_query(self):
        mock = AsyncMock(return_value={"intent": "summary", "confidence": 0.92, "reason": ""})
        with patch(_MOCK_TARGET, mock):
            result = await detect_query_intent("最近都干了啥")
        assert result == "summary_query"

    async def test_llm_generic_label_returns_generic(self):
        mock = AsyncMock(return_value={"intent": "generic", "confidence": 0.6, "reason": ""})
        with patch(_MOCK_TARGET, mock):
            result = await detect_query_intent("今天天气不错")
        assert result == "generic"

    async def test_llm_exception_falls_back_to_keyword(self):
        mock = AsyncMock(side_effect=Exception("network error"))
        with patch(_MOCK_TARGET, mock):
            result = await detect_query_intent("我在做什么项目")
        # Keyword fallback should classify this as project_query
        assert result == "project_query"

    async def test_llm_invalid_label_keyerror_falls_back(self):
        # Return a label not in _INTENT_LABEL_TO_QUERY_INTENT → KeyError in caller
        mock = AsyncMock(return_value={"intent": "unknown_label", "confidence": 0.5, "reason": ""})
        with patch(_MOCK_TARGET, mock):
            result = await detect_query_intent("今天天气不错")
        # KeyError from the dict lookup triggers fallback; "今天天气不错" → generic
        assert result == "generic"

    async def test_llm_network_timeout_falls_back(self):
        import asyncio
        mock = AsyncMock(side_effect=asyncio.TimeoutError())
        with patch(_MOCK_TARGET, mock):
            result = await detect_query_intent("我有什么待办")
        assert result == "task_query"

    async def test_llm_called_with_original_query(self):
        mock = AsyncMock(return_value={"intent": "generic", "confidence": 0.5, "reason": ""})
        with patch(_MOCK_TARGET, mock):
            await detect_query_intent("测试查询内容")
        mock.assert_called_once_with("测试查询内容")
