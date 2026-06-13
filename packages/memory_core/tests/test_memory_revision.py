import json
from unittest.mock import AsyncMock, MagicMock, patch

import aiosqlite
import pytest

from memory_core.database import ensure_preference_revision_schema, ensure_user_persona_schema
from memory_core.services.memory_revision import (
    PersonaRevisionDraft,
    extract_persona_revision_with_llm,
    generate_persona_clarification,
    get_low_confidence_personas,
    handle_persona_revision_message,
    is_persona_correction_message,
    revise_persona,
)


def _make_llm_response(payload: dict | str) -> MagicMock:
    content = json.dumps(payload) if isinstance(payload, dict) else payload
    choice = MagicMock()
    choice.message.content = content
    resp = MagicMock()
    resp.choices = [choice]
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=resp)
    return client


async def _revision_db():
    db = await aiosqlite.connect(":memory:")
    db.row_factory = aiosqlite.Row
    await ensure_user_persona_schema(db)
    await ensure_preference_revision_schema(db)
    await db.commit()
    return db


async def _insert_persona(
    db,
    persona_id: str = "persona-1",
    *,
    trait_key: str = "communication_style.direct",
    trait_value: str = "用户喜欢被提醒",
    confidence: float = 0.5,
    evidence_count: int = 1,
):
    await db.execute(
        """INSERT INTO user_persona
           (id, trait_key, trait_value, confidence, evidence_count, source_memory_ids)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (persona_id, trait_key, trait_value, confidence, evidence_count, "[]"),
    )


class TestPersonaCorrectionDetection:
    def test_correction_keywords_are_detected(self):
        assert is_persona_correction_message("你记错了，我不喜欢被催")
        assert is_persona_correction_message("不对，我更喜欢直接一点")

    def test_non_correction_message_is_ignored(self):
        assert not is_persona_correction_message("我的沟通风格是什么")


class TestRevisionExtraction:
    async def test_valid_json_parses_revision_draft(self):
        client = _make_llm_response(
            {
                "is_revision": True,
                "persona_id": "persona-1",
                "trait_key": "communication_style.direct",
                "new_value": "用户不喜欢被催促",
                "confidence": 1.5,
                "trigger": "用户明确纠正",
            }
        )
        candidates = [{"id": "persona-1", "trait_key": "communication_style.direct"}]

        with patch("memory_core.services.memory_revision.get_client", return_value=client):
            draft = await extract_persona_revision_with_llm(
                "你记错了，我不喜欢被催",
                candidates,
            )

        assert draft == PersonaRevisionDraft(
            is_revision=True,
            persona_id="persona-1",
            trait_key="communication_style.direct",
            old_value=None,
            new_value="用户不喜欢被催促",
            confidence=pytest.approx(1.0),
            trigger="用户明确纠正",
        )

    async def test_unknown_persona_id_is_treated_as_new_or_trait_match(self):
        client = _make_llm_response(
            {
                "is_revision": True,
                "persona_id": "missing",
                "trait_key": "communication_style.direct",
                "new_value": "用户不喜欢被催促",
                "confidence": 0.9,
                "trigger": "用户明确纠正",
            }
        )

        with patch("memory_core.services.memory_revision.get_client", return_value=client):
            draft = await extract_persona_revision_with_llm(
                "你记错了，我不喜欢被催",
                [{"id": "persona-1"}],
            )

        assert draft is not None
        assert draft.persona_id is None

    @pytest.mark.parametrize(
        "payload",
        [
            "not json",
            {"is_revision": False},
            {"is_revision": True, "trait_key": "Bad Key!", "new_value": "x"},
            {"is_revision": True, "trait_key": "habit.running", "new_value": ""},
        ],
    )
    async def test_invalid_payload_returns_none(self, payload):
        client = _make_llm_response(payload)

        with patch("memory_core.services.memory_revision.get_client", return_value=client):
            draft = await extract_persona_revision_with_llm(
                "你记错了，我不喜欢被催",
                [{"id": "persona-1"}],
            )

        assert draft is None

    async def test_llm_exception_returns_none(self):
        client = MagicMock()
        client.chat.completions.create = AsyncMock(side_effect=Exception("network"))

        with patch("memory_core.services.memory_revision.get_client", return_value=client):
            draft = await extract_persona_revision_with_llm(
                "你记错了，我不喜欢被催",
                [{"id": "persona-1"}],
            )

        assert draft is None


class TestRevisePersona:
    async def test_revises_existing_persona_and_writes_log(self):
        db = await _revision_db()
        try:
            await _insert_persona(db, confidence=0.4, evidence_count=2)
            await db.commit()
            result = await revise_persona(
                PersonaRevisionDraft(
                    is_revision=True,
                    persona_id="persona-1",
                    trait_key="communication_style.direct",
                    old_value=None,
                    new_value="用户不喜欢被催促",
                    confidence=0.9,
                    trigger="用户明确纠正",
                ),
                db,
            )
            persona = await (
                await db.execute(
                    "SELECT trait_value, confidence, evidence_count FROM user_persona WHERE id = ?",
                    ("persona-1",),
                )
            ).fetchone()
            log = await (
                await db.execute(
                    "SELECT persona_id, old_value, new_value, trigger FROM preference_revision_log"
                )
            ).fetchone()
        finally:
            await db.close()

        assert result.applied is True
        assert result.old_value == "用户喜欢被提醒"
        assert persona["trait_value"] == "用户不喜欢被催促"
        assert persona["confidence"] == pytest.approx(0.9)
        assert persona["evidence_count"] == 3
        assert log["persona_id"] == "persona-1"
        assert log["old_value"] == "用户喜欢被提醒"
        assert log["new_value"] == "用户不喜欢被催促"
        assert log["trigger"] == "用户明确纠正"

    async def test_creates_new_persona_and_writes_log(self):
        db = await _revision_db()
        try:
            result = await revise_persona(
                PersonaRevisionDraft(
                    is_revision=True,
                    persona_id=None,
                    trait_key="preference.food.spicy",
                    old_value=None,
                    new_value="用户现在不喜欢吃辣",
                    confidence=0.7,
                    trigger="用户纠正偏好",
                ),
                db,
            )
            persona = await (
                await db.execute(
                    "SELECT id, trait_key, trait_value, confidence, evidence_count FROM user_persona"
                )
            ).fetchone()
            log = await (
                await db.execute("SELECT persona_id, old_value, new_value FROM preference_revision_log")
            ).fetchone()
        finally:
            await db.close()

        assert result.applied is True
        assert persona["trait_key"] == "preference.food.spicy"
        assert persona["trait_value"] == "用户现在不喜欢吃辣"
        assert persona["confidence"] == pytest.approx(0.85)
        assert persona["evidence_count"] == 1
        assert log["persona_id"] == persona["id"]
        assert log["old_value"] is None

    async def test_existing_trait_key_wins_over_different_persona_id(self):
        db = await _revision_db()
        try:
            await _insert_persona(
                db,
                "persona-1",
                trait_key="communication_style.direct",
                trait_value="用户偏好直接沟通",
            )
            await _insert_persona(
                db,
                "persona-2",
                trait_key="preference.food.spicy",
                trait_value="用户喜欢吃辣",
            )
            await db.commit()
            result = await revise_persona(
                PersonaRevisionDraft(
                    is_revision=True,
                    persona_id="persona-1",
                    trait_key="preference.food.spicy",
                    old_value=None,
                    new_value="用户不喜欢吃辣",
                    confidence=0.9,
                    trigger="用户纠正偏好",
                ),
                db,
            )
            rows = await (
                await db.execute(
                    "SELECT id, trait_key, trait_value FROM user_persona ORDER BY id"
                )
            ).fetchall()
        finally:
            await db.close()

        assert result.applied is True
        assert result.persona_id == "persona-2"
        assert len(rows) == 2
        assert rows[1]["trait_value"] == "用户不喜欢吃辣"

    async def test_db_error_rolls_back(self):
        db = await _revision_db()
        try:
            await _insert_persona(db)
            await db.commit()
            await db.execute("DROP TABLE preference_revision_log")
            result = await revise_persona(
                PersonaRevisionDraft(
                    is_revision=True,
                    persona_id="persona-1",
                    trait_key="communication_style.direct",
                    old_value=None,
                    new_value="用户不喜欢被催促",
                    confidence=0.9,
                    trigger="用户明确纠正",
                ),
                db,
            )
            persona = await (
                await db.execute(
                    "SELECT trait_value FROM user_persona WHERE id = ?",
                    ("persona-1",),
                )
            ).fetchone()
        finally:
            await db.close()

        assert result.applied is False
        assert persona["trait_value"] == "用户喜欢被提醒"


class TestClarification:
    async def test_low_confidence_personas_filter_by_query_relevance(self):
        db = await _revision_db()
        try:
            await _insert_persona(
                db,
                "persona-low-match",
                trait_key="communication_style.direct",
                trait_value="用户偏好直接沟通",
                confidence=0.4,
            )
            await _insert_persona(
                db,
                "persona-low-miss",
                trait_key="health.habit.running",
                trait_value="用户长期跑步",
                confidence=0.4,
            )
            await _insert_persona(
                db,
                "persona-high",
                trait_key="communication_style.formal",
                trait_value="用户偏好正式沟通",
                confidence=0.9,
            )
            await db.commit()
            results = await get_low_confidence_personas("我的沟通风格是什么", db)
        finally:
            await db.close()

        assert [row["id"] for row in results] == ["persona-low-match"]

    async def test_clarification_fallback_when_llm_fails(self):
        client = MagicMock()
        client.chat.completions.create = AsyncMock(side_effect=Exception("network"))

        with patch("memory_core.services.memory_revision.get_client", return_value=client):
            question = await generate_persona_clarification(
                "我的沟通风格是什么",
                [{"trait_value": "用户偏好直接沟通", "confidence": 0.4}],
            )

        assert question == "我对这条关于你的理解还不太确定：“用户偏好直接沟通”。这是准确的吗？"

    async def test_handle_revision_uncertain_returns_clarification(self):
        db = await _revision_db()
        try:
            with patch(
                "memory_core.services.memory_revision.extract_persona_revision_with_llm",
                new=AsyncMock(return_value=None),
            ):
                result = await handle_persona_revision_message("你记错了", db)
        finally:
            await db.close()

        assert result.applied is False
        assert result.needs_clarification is True
        assert result.clarification_question
