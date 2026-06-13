import json
from unittest.mock import AsyncMock, MagicMock, patch

import aiosqlite
import pytest

from memory_core.database import ensure_user_persona_schema
from memory_core.services.persona_service import (
    PersonaTraitCandidate,
    extract_persona_traits_with_llm,
    upsert_persona_traits,
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


async def _persona_db():
    db = await aiosqlite.connect(":memory:")
    db.row_factory = aiosqlite.Row
    await ensure_user_persona_schema(db)
    await db.commit()
    return db


class TestPersonaSchema:
    async def test_ensure_user_persona_schema_creates_table_and_indexes(self):
        db = await _persona_db()
        try:
            table = await (
                await db.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='user_persona'"
                )
            ).fetchone()
            unique_index = await (
                await db.execute(
                    "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_user_persona_key'"
                )
            ).fetchone()
            cursor = await db.execute(
                """INSERT INTO user_persona
                   (id, trait_key, trait_value)
                   VALUES (?, ?, ?)""",
                ("id-1", "habit.morning", "用户习惯早起"),
            )
            assert cursor.rowcount == 1
            row = await (
                await db.execute(
                    "SELECT source_memory_ids FROM user_persona WHERE id = ?",
                    ("id-1",),
                )
            ).fetchone()
        finally:
            await db.close()

        assert table is not None
        assert unique_index is not None
        assert row["source_memory_ids"] == "[]"

    async def test_ensure_user_persona_schema_is_idempotent(self):
        db = await _persona_db()
        try:
            await ensure_user_persona_schema(db)
            await ensure_user_persona_schema(db)
            await db.commit()
        finally:
            await db.close()


class TestExtractPersonaTraitsWithLlm:
    async def test_parses_valid_traits(self):
        client = _make_llm_response(
            {
                "traits": [
                    {
                        "trait_key": "communication_style.direct",
                        "trait_value": "用户偏好直接清晰的沟通",
                        "confidence": 0.83,
                    }
                ]
            }
        )
        with patch("memory_core.services.persona_service.get_client", return_value=client):
            traits = await extract_persona_traits_with_llm(
                "我长期喜欢直接一点的沟通方式",
                "preference",
            )

        assert traits == [
            PersonaTraitCandidate(
                trait_key="communication_style.direct",
                trait_value="用户偏好直接清晰的沟通",
                confidence=pytest.approx(0.83),
            )
        ]

    async def test_invalid_json_returns_empty(self):
        client = _make_llm_response("not json")
        with patch("memory_core.services.persona_service.get_client", return_value=client):
            traits = await extract_persona_traits_with_llm(
                "我长期喜欢直接一点的沟通方式",
                "preference",
            )
        assert traits == []

    async def test_invalid_key_and_low_confidence_are_skipped(self):
        client = _make_llm_response(
            {
                "traits": [
                    {
                        "trait_key": "Bad Key!",
                        "trait_value": "非法 key",
                        "confidence": 0.9,
                    },
                    {
                        "trait_key": "habit.running",
                        "trait_value": "用户习惯跑步",
                        "confidence": 0.6,
                    },
                ]
            }
        )
        with patch("memory_core.services.persona_service.get_client", return_value=client):
            traits = await extract_persona_traits_with_llm("我长期跑步", "fact")
        assert traits == []

    async def test_confidence_is_clamped(self):
        client = _make_llm_response(
            {
                "traits": [
                    {
                        "trait_key": "habit.running",
                        "trait_value": "用户习惯跑步",
                        "confidence": 2.0,
                    }
                ]
            }
        )
        with patch("memory_core.services.persona_service.get_client", return_value=client):
            traits = await extract_persona_traits_with_llm("我长期跑步", "fact")
        assert traits[0].confidence == pytest.approx(1.0)

    async def test_empty_content_does_not_call_llm(self):
        client = MagicMock()
        client.chat.completions.create = AsyncMock()
        with patch("memory_core.services.persona_service.get_client", return_value=client):
            traits = await extract_persona_traits_with_llm("", "fact")
        assert traits == []
        client.chat.completions.create.assert_not_called()

    async def test_ineligible_kind_does_not_call_llm(self):
        client = MagicMock()
        client.chat.completions.create = AsyncMock()
        with patch("memory_core.services.persona_service.get_client", return_value=client):
            traits = await extract_persona_traits_with_llm("普通聊天", "other")
        assert traits == []
        client.chat.completions.create.assert_not_called()


class TestUpsertPersonaTraits:
    async def test_inserts_new_trait(self):
        db = await _persona_db()
        try:
            count = await upsert_persona_traits(
                "mem-1",
                [
                    PersonaTraitCandidate(
                        "habit.running",
                        "用户长期跑步",
                        0.8,
                    )
                ],
                db,
            )
            await db.commit()
            row = await (
                await db.execute(
                    "SELECT trait_key, trait_value, confidence, evidence_count, source_memory_ids FROM user_persona"
                )
            ).fetchone()
        finally:
            await db.close()

        assert count == 1
        assert row["trait_key"] == "habit.running"
        assert row["trait_value"] == "用户长期跑步"
        assert row["confidence"] == pytest.approx(0.8)
        assert row["evidence_count"] == 1
        assert json.loads(row["source_memory_ids"]) == ["mem-1"]

    async def test_merges_same_key_and_value(self):
        db = await _persona_db()
        try:
            await upsert_persona_traits(
                "mem-1",
                [PersonaTraitCandidate("habit.running", "用户长期跑步", 0.8)],
                db,
            )
            await upsert_persona_traits(
                "mem-2",
                [PersonaTraitCandidate("habit.running", "用户长期跑步", 0.9)],
                db,
            )
            await db.commit()
            row = await (
                await db.execute(
                    "SELECT confidence, evidence_count, source_memory_ids FROM user_persona"
                )
            ).fetchone()
        finally:
            await db.close()

        assert row["confidence"] == pytest.approx(0.95)
        assert row["evidence_count"] == 2
        assert json.loads(row["source_memory_ids"]) == ["mem-1", "mem-2"]

    async def test_duplicate_source_memory_does_not_increment_evidence(self):
        db = await _persona_db()
        try:
            await upsert_persona_traits(
                "mem-1",
                [PersonaTraitCandidate("habit.running", "用户长期跑步", 0.8)],
                db,
            )
            count = await upsert_persona_traits(
                "mem-1",
                [PersonaTraitCandidate("habit.running", "用户长期跑步", 0.9)],
                db,
            )
            await db.commit()
            row = await (
                await db.execute(
                    "SELECT confidence, evidence_count, source_memory_ids FROM user_persona"
                )
            ).fetchone()
        finally:
            await db.close()

        assert count == 0
        assert row["confidence"] == pytest.approx(0.8)
        assert row["evidence_count"] == 1
        assert json.loads(row["source_memory_ids"]) == ["mem-1"]

    async def test_conflicting_value_does_not_overwrite(self):
        db = await _persona_db()
        try:
            await upsert_persona_traits(
                "mem-1",
                [PersonaTraitCandidate("preference.food", "用户喜欢吃辣", 0.8)],
                db,
            )
            count = await upsert_persona_traits(
                "mem-2",
                [PersonaTraitCandidate("preference.food", "用户喜欢吃甜", 0.9)],
                db,
            )
            await db.commit()
            row = await (
                await db.execute(
                    "SELECT trait_value, evidence_count, source_memory_ids FROM user_persona"
                )
            ).fetchone()
        finally:
            await db.close()

        assert count == 0
        assert row["trait_value"] == "用户喜欢吃辣"
        assert row["evidence_count"] == 1
        assert json.loads(row["source_memory_ids"]) == ["mem-1"]

    async def test_conflicting_value_can_lower_confidence_for_sleep_refresh(self):
        db = await _persona_db()
        try:
            await upsert_persona_traits(
                "mem-1",
                [PersonaTraitCandidate("preference.food", "用户喜欢吃辣", 0.8)],
                db,
            )
            count = await upsert_persona_traits(
                "mem-2",
                [PersonaTraitCandidate("preference.food", "用户喜欢吃甜", 0.9)],
                db,
                conflict_strategy="lower_confidence",
            )
            await db.commit()
            row = await (
                await db.execute(
                    "SELECT trait_value, confidence, evidence_count, source_memory_ids FROM user_persona"
                )
            ).fetchone()
        finally:
            await db.close()

        assert count == 0
        assert row["trait_value"] == "用户喜欢吃辣"
        assert row["confidence"] == pytest.approx(0.75)
        assert row["evidence_count"] == 1
        assert json.loads(row["source_memory_ids"]) == ["mem-1"]

    async def test_source_memory_ids_are_deduped_and_capped(self):
        db = await _persona_db()
        try:
            for index in range(25):
                await upsert_persona_traits(
                    f"mem-{index}",
                    [PersonaTraitCandidate("habit.running", "用户长期跑步", 0.8)],
                    db,
                )
            await upsert_persona_traits(
                "mem-24",
                [PersonaTraitCandidate("habit.running", "用户长期跑步", 0.8)],
                db,
            )
            await db.commit()
            row = await (
                await db.execute("SELECT source_memory_ids FROM user_persona")
            ).fetchone()
        finally:
            await db.close()

        source_ids = json.loads(row["source_memory_ids"])
        assert len(source_ids) == 20
        assert source_ids[-1] == "mem-24"
        assert len(source_ids) == len(set(source_ids))
