"""Tests for LLM-based emotion scoring.

Two layers:
  Layer A — score_emotion_with_llm: async, mocks LLM client.
  Layer B — create_memory_item: async, verifies the new await path is hit.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from memory_core.services.llm import (
    score_admission_with_llm,
    score_emotion_with_llm,
    score_importance_with_llm,
)
from memory_core.services.memory_service import estimate_emotion_score

_MOCK_TARGET = "memory_core.services.llm.get_client"


def _make_llm_response(payload: dict | str) -> MagicMock:
    """Build a mock that mimics openai ChatCompletion response."""
    if isinstance(payload, dict):
        content = json.dumps(payload)
    else:
        content = payload

    choice = MagicMock()
    choice.message.content = content
    resp = MagicMock()
    resp.choices = [choice]

    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=resp)
    return client


# ---------------------------------------------------------------------------
# Layer A: score_emotion_with_llm
# ---------------------------------------------------------------------------


class TestScoreEmotionWithLlm:
    async def test_positive_content_returns_llm_score(self):
        client = _make_llm_response({"emotion_score": 0.8})
        with patch(_MOCK_TARGET, return_value=client):
            result = await score_emotion_with_llm("今天项目顺利上线了")
        assert result == pytest.approx(0.8)

    async def test_neutral_content_returns_zero(self):
        client = _make_llm_response({"emotion_score": 0.0})
        with patch(_MOCK_TARGET, return_value=client):
            result = await score_emotion_with_llm("我今天开了个会")
        assert result == pytest.approx(0.0)

    async def test_negative_content_returns_negative_score(self):
        client = _make_llm_response({"emotion_score": -0.7})
        with patch(_MOCK_TARGET, return_value=client):
            result = await score_emotion_with_llm("感觉压力很大很焦虑")
        assert result == pytest.approx(-0.7)

    async def test_value_clamped_above_one(self):
        client = _make_llm_response({"emotion_score": 2.5})
        with patch(_MOCK_TARGET, return_value=client):
            result = await score_emotion_with_llm("极度开心")
        assert result == pytest.approx(1.0)

    async def test_value_clamped_below_minus_one(self):
        client = _make_llm_response({"emotion_score": -1.5})
        with patch(_MOCK_TARGET, return_value=client):
            result = await score_emotion_with_llm("极度痛苦")
        assert result == pytest.approx(-1.0)

    async def test_empty_string_short_circuits_no_llm_call(self):
        client = MagicMock()
        client.chat.completions.create = AsyncMock()
        with patch(_MOCK_TARGET, return_value=client):
            result = await score_emotion_with_llm("")
        assert result == pytest.approx(0.0)
        client.chat.completions.create.assert_not_called()

    async def test_whitespace_only_short_circuits_no_llm_call(self):
        client = MagicMock()
        client.chat.completions.create = AsyncMock()
        with patch(_MOCK_TARGET, return_value=client):
            result = await score_emotion_with_llm("   ")
        assert result == pytest.approx(0.0)
        client.chat.completions.create.assert_not_called()

    async def test_llm_exception_fallback_to_keyword(self):
        client = MagicMock()
        client.chat.completions.create = AsyncMock(side_effect=Exception("network error"))
        with patch(_MOCK_TARGET, return_value=client):
            result = await score_emotion_with_llm("开心高兴")
        # keyword fallback: "开心" + "高兴" → 2 positive hits → 0.7
        assert result == pytest.approx(estimate_emotion_score("开心高兴"))

    async def test_invalid_json_fallback(self):
        client = _make_llm_response("this is not json")
        with patch(_MOCK_TARGET, return_value=client):
            result = await score_emotion_with_llm("今天很不错")
        # fallback: "不错" → 1 positive hit → 0.6
        assert result == pytest.approx(estimate_emotion_score("今天很不错"))

    async def test_missing_field_fallback(self):
        client = _make_llm_response({"other_key": 99})
        with patch(_MOCK_TARGET, return_value=client):
            result = await score_emotion_with_llm("心情一般")
        # missing "emotion_score" key → KeyError → fallback → 0.0
        assert result == pytest.approx(estimate_emotion_score("心情一般"))

    async def test_wrong_type_fallback(self):
        client = _make_llm_response({"emotion_score": "high"})
        with patch(_MOCK_TARGET, return_value=client):
            result = await score_emotion_with_llm("心情不错")
        assert result == pytest.approx(estimate_emotion_score("心情不错"))

    async def test_timeout_fallback(self):
        import asyncio
        client = MagicMock()
        client.chat.completions.create = AsyncMock(side_effect=asyncio.TimeoutError())
        with patch(_MOCK_TARGET, return_value=client):
            result = await score_emotion_with_llm("今天很崩溃")
        # fallback: "崩溃" → 1 negative hit → -0.6
        assert result == pytest.approx(estimate_emotion_score("今天很崩溃"))


# ---------------------------------------------------------------------------
# Layer A2: score_importance_with_llm
# ---------------------------------------------------------------------------


class TestScoreImportanceWithLlm:
    async def test_returns_llm_importance(self):
        client = _make_llm_response({"importance": 8.5})
        with patch(_MOCK_TARGET, return_value=client):
            result = await score_importance_with_llm("我每天六点起床跑步")
        assert result == pytest.approx(8.5)

    async def test_value_clamped_above_ten(self):
        client = _make_llm_response({"importance": 99})
        with patch(_MOCK_TARGET, return_value=client):
            result = await score_importance_with_llm("非常重要的长期事实")
        assert result == pytest.approx(10.0)

    async def test_value_clamped_below_one(self):
        client = _make_llm_response({"importance": -2})
        with patch(_MOCK_TARGET, return_value=client):
            result = await score_importance_with_llm("普通闲聊")
        assert result == pytest.approx(1.0)

    async def test_empty_string_short_circuits_to_default(self):
        client = MagicMock()
        client.chat.completions.create = AsyncMock()
        with patch(_MOCK_TARGET, return_value=client):
            result = await score_importance_with_llm("")
        assert result == pytest.approx(5.0)
        client.chat.completions.create.assert_not_called()

    async def test_invalid_json_fallback_to_default(self):
        client = _make_llm_response("not json")
        with patch(_MOCK_TARGET, return_value=client):
            result = await score_importance_with_llm("今天开了会")
        assert result == pytest.approx(5.0)

    async def test_missing_field_fallback_to_default(self):
        client = _make_llm_response({"other_key": 9})
        with patch(_MOCK_TARGET, return_value=client):
            result = await score_importance_with_llm("今天开了会")
        assert result == pytest.approx(5.0)

    async def test_wrong_type_fallback_to_default(self):
        client = _make_llm_response({"importance": "high"})
        with patch(_MOCK_TARGET, return_value=client):
            result = await score_importance_with_llm("今天开了会")
        assert result == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Layer A3: score_admission_with_llm
# ---------------------------------------------------------------------------


class TestScoreAdmissionWithLlm:
    async def test_returns_llm_admission_scores(self):
        client = _make_llm_response({"utility": 0.8, "confidence": 0.7})
        with patch(_MOCK_TARGET, return_value=client):
            result = await score_admission_with_llm("我长期喜欢吃辣", "preference")
        assert result == {"utility": pytest.approx(0.8), "confidence": pytest.approx(0.7)}

    async def test_values_are_clamped(self):
        client = _make_llm_response({"utility": 2.0, "confidence": -1.0})
        with patch(_MOCK_TARGET, return_value=client):
            result = await score_admission_with_llm("今天推进了 Pervault", "project_update")
        assert result == {"utility": pytest.approx(1.0), "confidence": pytest.approx(0.0)}

    async def test_empty_string_raises_without_llm_call(self):
        client = MagicMock()
        client.chat.completions.create = AsyncMock()
        with patch(_MOCK_TARGET, return_value=client):
            with pytest.raises(ValueError):
                await score_admission_with_llm("", "other")
        client.chat.completions.create.assert_not_called()

    async def test_invalid_json_raises(self):
        client = _make_llm_response("not json")
        with patch(_MOCK_TARGET, return_value=client):
            with pytest.raises(ValueError):
                await score_admission_with_llm("今天开了会", "other")

    async def test_missing_field_raises(self):
        client = _make_llm_response({"utility": 0.5})
        with patch(_MOCK_TARGET, return_value=client):
            with pytest.raises(ValueError):
                await score_admission_with_llm("今天开了会", "other")

    async def test_wrong_type_raises(self):
        client = _make_llm_response({"utility": "high", "confidence": 0.5})
        with patch(_MOCK_TARGET, return_value=client):
            with pytest.raises(ValueError):
                await score_admission_with_llm("今天开了会", "other")


# ---------------------------------------------------------------------------
# Layer B: create_memory_item — verify keyword estimate is used (no LLM call)
# ---------------------------------------------------------------------------


def _make_db_mock(emotion_score: float = 0.0) -> MagicMock:
    """Build a minimal async aiosqlite connection mock."""
    row = {
        "id": "test-id",
        "voice_record_id": None,
        "content": "今天很开心",
        "tags": "[]",
        "kind": "other",
        "task_status": None,
        "emotion_score": emotion_score,
        "consolidated": 0,
        "weight": 1.0,
        "last_referenced_at": None,
        "created_at": "2024-01-01T00:00:00",
    }
    row_mock = MagicMock()
    row_mock.__getitem__ = lambda self, k: row[k]
    row_mock.keys = lambda: row.keys()

    cursor = AsyncMock()
    cursor.rowcount = 1
    cursor.fetchone = AsyncMock(return_value=row_mock)

    db = AsyncMock()
    db.sqlite_vec_loaded = False
    db.execute = AsyncMock(return_value=cursor)
    db.commit = AsyncMock()
    db.close = AsyncMock()
    db.rollback = AsyncMock()
    return db


class TestCreateMemoryItemEmotionScorePath:
    async def test_create_memory_item_uses_keyword_estimate_not_llm(self):
        """create_memory_item must NOT await score_emotion_with_llm.

        Emotion scoring was moved to _update_emotion_score_in_background so that
        the store endpoint returns immediately without waiting for an LLM call.
        The keyword estimate is used for the initial DB write.
        """
        db = _make_db_mock()
        with patch(
            "memory_core.services.memory_service.score_emotion_with_llm",
            new=AsyncMock(return_value=0.99),
        ) as mock_score, patch(
            "memory_core.services.memory_service.get_db",
            return_value=db,
        ):
            from memory_core.services.memory_service import create_memory_item
            await create_memory_item("今天很开心")

        # LLM must NOT have been called — that's the background job's responsibility now.
        mock_score.assert_not_awaited()
        db.close.assert_awaited_once()

    async def test_create_memory_item_emotion_score_is_keyword_based(self):
        """The emotion_score on the returned item reflects the keyword estimate."""
        from memory_core.services.memory_service import estimate_emotion_score

        content = "今天开心高兴"
        expected = estimate_emotion_score(content)
        db = _make_db_mock(emotion_score=expected)

        with patch(
            "memory_core.services.memory_service.score_emotion_with_llm",
            new=AsyncMock(return_value=0.99),
        ), patch(
            "memory_core.services.memory_service.get_db",
            return_value=db,
        ):
            from memory_core.services.memory_service import create_memory_item
            item = await create_memory_item(content)

        assert item.emotion_score == pytest.approx(expected)
        db.close.assert_awaited_once()


# ---------------------------------------------------------------------------
# Layer C: _update_emotion_score_in_background — LLM runs and writes to DB
# ---------------------------------------------------------------------------


class TestUpdateEmotionScoreInBackground:
    async def test_calls_llm_and_writes_to_db(self):
        """_update_emotion_score_in_background calls LLM and updates the DB row."""
        db = _make_db_mock()
        with patch(
            "memory_core.services.memory_service.score_emotion_with_llm",
            new=AsyncMock(return_value=0.75),
        ) as mock_score, patch(
            "memory_core.services.memory_service.get_db",
            return_value=db,
        ):
            from memory_core.services.memory_service import _update_emotion_score_in_background
            await _update_emotion_score_in_background("test-id", "今天很开心")

        mock_score.assert_awaited_once_with("今天很开心")
        db.execute.assert_called()
        db.commit.assert_awaited()

    async def test_timeout_skips_db_write(self):
        """A timeout from score_emotion_with_llm must not touch the DB."""
        import asyncio as _asyncio

        db = _make_db_mock()
        with patch(
            "memory_core.services.memory_service.score_emotion_with_llm",
            new=AsyncMock(side_effect=_asyncio.TimeoutError()),
        ), patch(
            "memory_core.services.memory_service.get_db",
            return_value=db,
        ):
            from memory_core.services.memory_service import _update_emotion_score_in_background
            await _update_emotion_score_in_background("test-id", "今天很开心")

        db.execute.assert_not_called()
        db.commit.assert_not_awaited()


# ---------------------------------------------------------------------------
# Layer D: _update_importance_in_background — LLM runs and writes to DB
# ---------------------------------------------------------------------------


class TestUpdateImportanceInBackground:
    async def test_calls_llm_and_writes_to_db(self):
        db = _make_db_mock()
        with patch(
            "memory_core.services.memory_service.score_importance_with_llm",
            new=AsyncMock(return_value=8.0),
        ) as mock_score, patch(
            "memory_core.services.memory_service.get_db",
            return_value=db,
        ):
            from memory_core.services.memory_service import _update_importance_in_background
            await _update_importance_in_background("test-id", "我长期坚持跑步")

        mock_score.assert_awaited_once_with("我长期坚持跑步")
        db.execute.assert_called()
        assert db.execute.call_args.args[1] == (8.0, "test-id")
        db.commit.assert_awaited()

    async def test_timeout_skips_db_write(self):
        import asyncio as _asyncio

        db = _make_db_mock()
        with patch(
            "memory_core.services.memory_service.score_importance_with_llm",
            new=AsyncMock(side_effect=_asyncio.TimeoutError()),
        ), patch(
            "memory_core.services.memory_service.get_db",
            return_value=db,
        ):
            from memory_core.services.memory_service import _update_importance_in_background
            await _update_importance_in_background("test-id", "我长期坚持跑步")

        db.execute.assert_not_called()
        db.commit.assert_not_awaited()

    async def test_db_error_rolls_back(self):
        db = _make_db_mock()
        db.execute = AsyncMock(side_effect=Exception("db error"))
        with patch(
            "memory_core.services.memory_service.score_importance_with_llm",
            new=AsyncMock(return_value=8.0),
        ), patch(
            "memory_core.services.memory_service.get_db",
            return_value=db,
        ):
            from memory_core.services.memory_service import _update_importance_in_background
            await _update_importance_in_background("test-id", "我长期坚持跑步")

        db.rollback.assert_awaited()


# ---------------------------------------------------------------------------
# Layer D2: _score_memory_admission_in_background — score + log write
# ---------------------------------------------------------------------------


class TestScoreMemoryAdmissionInBackground:
    async def test_success_updates_memory_and_inserts_log(self):
        from memory_core.services.memory_admission import AdmissionScore

        db = _make_db_mock()
        score = AdmissionScore(
            utility=0.8,
            confidence=0.7,
            novelty=1.0,
            recency=1.0,
            type_prior=0.9,
            total=0.825,
            tier="standard",
        )
        with patch(
            "memory_core.services.memory_service.compute_admission_score",
            new=AsyncMock(return_value=score),
        ) as mock_score, patch(
            "memory_core.services.memory_service.get_db",
            return_value=db,
        ), patch(
            "memory_core.services.memory_service._extract_persona_in_background",
            new=AsyncMock(),
        ) as mock_persona:
            from memory_core.services.memory_service import _score_memory_admission_in_background
            await _score_memory_admission_in_background(
                "test-id", "我长期喜欢吃辣", "preference"
            )

        mock_score.assert_awaited_once_with(
            "我长期喜欢吃辣",
            "preference",
            db,
            exclude_memory_id="test-id",
        )
        assert db.execute.await_count == 2
        update_args = db.execute.await_args_list[0].args[1]
        insert_args = db.execute.await_args_list[1].args[1]
        assert update_args == (0.825, "standard", "test-id")
        assert insert_args[1:3] == ("test-id", "我长期喜欢吃辣")
        assert insert_args[-2:] == (1, "standard")
        db.commit.assert_awaited()
        mock_persona.assert_awaited_once_with("test-id", "我长期喜欢吃辣", "preference")

    async def test_low_value_does_not_trigger_persona_extraction(self):
        from memory_core.services.memory_admission import AdmissionScore

        db = _make_db_mock()
        score = AdmissionScore(
            utility=0.1,
            confidence=0.2,
            novelty=1.0,
            recency=1.0,
            type_prior=0.35,
            total=0.3375,
            tier="low_value",
        )
        with patch(
            "memory_core.services.memory_service.compute_admission_score",
            new=AsyncMock(return_value=score),
        ), patch(
            "memory_core.services.memory_service.get_db",
            return_value=db,
        ), patch(
            "memory_core.services.memory_service._extract_persona_in_background",
            new=AsyncMock(),
        ) as mock_persona:
            from memory_core.services.memory_service import _score_memory_admission_in_background
            await _score_memory_admission_in_background("test-id", "哈哈", "other")

        mock_persona.assert_not_awaited()

    async def test_timeout_skips_db_write(self):
        import asyncio as _asyncio

        db = _make_db_mock()
        with patch(
            "memory_core.services.memory_service.compute_admission_score",
            new=AsyncMock(side_effect=_asyncio.TimeoutError()),
        ), patch(
            "memory_core.services.memory_service.get_db",
            return_value=db,
        ):
            from memory_core.services.memory_service import _score_memory_admission_in_background
            await _score_memory_admission_in_background("test-id", "哈哈", "other")

        db.execute.assert_not_called()
        db.commit.assert_not_awaited()

    async def test_db_error_rolls_back(self):
        from memory_core.services.memory_admission import AdmissionScore

        db = _make_db_mock()
        db.execute = AsyncMock(side_effect=Exception("db error"))
        score = AdmissionScore(
            utility=0.1,
            confidence=0.2,
            novelty=1.0,
            recency=1.0,
            type_prior=0.35,
            total=0.3375,
            tier="low_value",
        )
        with patch(
            "memory_core.services.memory_service.compute_admission_score",
            new=AsyncMock(return_value=score),
        ), patch(
            "memory_core.services.memory_service.get_db",
            return_value=db,
        ):
            from memory_core.services.memory_service import _score_memory_admission_in_background
            await _score_memory_admission_in_background("test-id", "哈哈", "other")

        db.rollback.assert_awaited()


# ---------------------------------------------------------------------------
# Layer E: row_to_item schema compatibility
# ---------------------------------------------------------------------------


class TestRowToItemSchemaCompatibility:
    def test_old_row_without_importance_or_admission_fields_uses_defaults(self):
        from memory_core.services.memory_service import row_to_item

        row = {
            "id": "old-id",
            "voice_record_id": None,
            "content": "旧数据",
            "tags": "[]",
            "kind": "other",
            "task_status": None,
            "emotion_score": 0.0,
            "consolidated": 0,
            "weight": 1.0,
            "last_referenced_at": None,
            "created_at": "2024-01-01T00:00:00",
        }
        row_mock = MagicMock()
        row_mock.__getitem__ = lambda self, k: row[k]
        row_mock.keys = lambda: row.keys()

        item = row_to_item(row_mock)

        assert item.importance == pytest.approx(5.0)
        assert item.admission_score is None
        assert item.admission_tier == "standard"
