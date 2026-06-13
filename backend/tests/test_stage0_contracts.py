import json
import sys
import types
from unittest.mock import AsyncMock, patch

import aiosqlite
import pytest
from starlette.requests import Request

from memory_core.database import (
    ensure_auth_sessions_schema,
    ensure_chat_messages_schema,
    ensure_data_export_log_schema,
    ensure_memory_reflection_schema,
    ensure_preference_revision_schema,
    ensure_sleep_agent_checkpoint_schema,
)
from memory_core.models import ChatResponse, MemoryExportRequest, MemoryReflection, PreferenceRevisionLog


async def _schema_db():
    db = await aiosqlite.connect(":memory:")
    db.row_factory = aiosqlite.Row
    await db.execute(
        """CREATE TABLE user_persona (
               id TEXT PRIMARY KEY,
               trait_key TEXT NOT NULL,
               trait_value TEXT NOT NULL
           )"""
    )
    await db.commit()
    return db


class TestStage0SchemaContracts:
    async def test_ensure_memory_reflection_schema_creates_table_and_index(self):
        db = await _schema_db()
        try:
            await ensure_memory_reflection_schema(db)
            await ensure_memory_reflection_schema(db)
            await db.commit()
            table = await (
                await db.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='memory_reflection'"
                )
            ).fetchone()
            index = await (
                await db.execute(
                    "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_memory_reflection_importance_created'"
                )
            ).fetchone()
            insight_key_index = await (
                await db.execute(
                    "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_memory_reflection_insight_dedupe_key'"
                )
            ).fetchone()
            source_fingerprint_index = await (
                await db.execute(
                    "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_memory_reflection_source_fingerprint'"
                )
            ).fetchone()
            cursor = await db.execute(
                """INSERT INTO memory_reflection (id, insight)
                   VALUES (?, ?)""",
                ("reflection-1", "用户最近持续关注健康"),
            )
            row = await (
                await db.execute(
                    """SELECT source_memory_ids, insight_dedupe_key,
                              source_memory_fingerprint, importance
                       FROM memory_reflection
                       WHERE id = ?""",
                    ("reflection-1",),
                )
            ).fetchone()
        finally:
            await db.close()

        assert table is not None
        assert index is not None
        assert insight_key_index is not None
        assert source_fingerprint_index is not None
        assert cursor.rowcount == 1
        assert row["source_memory_ids"] == "[]"
        assert row["insight_dedupe_key"] == ""
        assert row["source_memory_fingerprint"] == "[]"
        assert row["importance"] == pytest.approx(8.0)

    async def test_ensure_memory_reflection_schema_backfills_dedup_columns_for_existing_table(self):
        db = await _schema_db()
        try:
            await db.execute(
                """CREATE TABLE memory_reflection (
                       id TEXT PRIMARY KEY,
                       insight TEXT NOT NULL,
                       source_memory_ids TEXT DEFAULT '[]',
                       importance REAL DEFAULT 8.0,
                       created_at DATETIME DEFAULT (datetime('now'))
                   )"""
            )
            await db.execute(
                """INSERT INTO memory_reflection
                   (id, insight, source_memory_ids)
                   VALUES (?, ?, ?)""",
                ("reflection-1", "用户最近持续关注健康。", json.dumps(["mem-2", "mem-1"])),
            )
            await db.commit()

            await ensure_memory_reflection_schema(db)
            await ensure_memory_reflection_schema(db)
            await db.commit()

            row = await (
                await db.execute(
                    """SELECT insight_dedupe_key, source_memory_fingerprint
                       FROM memory_reflection
                       WHERE id = ?""",
                    ("reflection-1",),
                )
            ).fetchone()
        finally:
            await db.close()

        assert row["insight_dedupe_key"] == "用户最近持续关注健康"
        assert row["source_memory_fingerprint"] == '["mem-1","mem-2"]'

    async def test_ensure_preference_revision_schema_creates_table_and_index(self):
        db = await _schema_db()
        try:
            await ensure_preference_revision_schema(db)
            await ensure_preference_revision_schema(db)
            await db.commit()
            table = await (
                await db.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='preference_revision_log'"
                )
            ).fetchone()
            index = await (
                await db.execute(
                    "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_preference_revision_log_persona_created'"
                )
            ).fetchone()
            cursor = await db.execute(
                """INSERT INTO preference_revision_log
                   (id, persona_id, old_value, new_value, trigger)
                   VALUES (?, ?, ?, ?, ?)""",
                ("revision-1", None, "旧值", "新值", "manual_correction"),
            )
        finally:
            await db.close()

        assert table is not None
        assert index is not None
        assert cursor.rowcount == 1

    async def test_ensure_auth_session_and_export_log_schema_create_tables(self):
        db = await _schema_db()
        try:
            await ensure_auth_sessions_schema(db)
            await ensure_data_export_log_schema(db)
            await db.commit()
            auth_table = await (
                await db.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='auth_sessions'"
                )
            ).fetchone()
            export_table = await (
                await db.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='data_export_log'"
                )
            ).fetchone()
        finally:
            await db.close()

        assert auth_table is not None
        assert export_table is not None

    async def test_ensure_chat_messages_schema_adds_clarification_columns(self):
        db = await _schema_db()
        try:
            await ensure_chat_messages_schema(db)
            await ensure_chat_messages_schema(db)
            await db.commit()
            cursor = await db.execute(
                """INSERT INTO chat_messages (id, session_id, role, content)
                   VALUES (?, ?, ?, ?)""",
                ("msg-1", "session-1", "assistant", "你好"),
            )
            row = await (
                await db.execute(
                    """SELECT needs_clarification, clarification_question
                       FROM chat_messages
                       WHERE id = ?""",
                    ("msg-1",),
                )
            ).fetchone()
        finally:
            await db.close()

        assert cursor.rowcount == 1
        assert row["needs_clarification"] == 0
        assert row["clarification_question"] is None

    async def test_ensure_sleep_agent_checkpoint_schema_creates_table(self):
        db = await _schema_db()
        try:
            await ensure_sleep_agent_checkpoint_schema(db)
            await ensure_sleep_agent_checkpoint_schema(db)
            await db.commit()
            table = await (
                await db.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='sleep_agent_checkpoint'"
                )
            ).fetchone()
            cursor = await db.execute(
                """INSERT INTO sleep_agent_checkpoint
                   (stage_name, checkpoint_created_at, last_run_id)
                   VALUES (?, ?, ?)""",
                ("topic_regroup", "2026-04-17 10:00:00", "run-1"),
            )
        finally:
            await db.close()

        assert table is not None
        assert cursor.rowcount == 1


class TestStage0ModelContracts:
    def test_chat_response_defaults_needs_clarification_false(self):
        response = ChatResponse(reply="ok", sources=[])

        assert response.needs_clarification is False
        assert response.clarification_question is None

    def test_memory_reflection_model_constructs(self):
        reflection = MemoryReflection(
            id="reflection-1",
            insight="用户近期持续关注健康",
            source_memory_ids=["mem-1"],
        )

        assert reflection.importance == pytest.approx(8.0)
        assert reflection.source_memory_ids == ["mem-1"]

    def test_preference_revision_log_model_constructs(self):
        revision = PreferenceRevisionLog(
            id="revision-1",
            persona_id="persona-1",
            old_value="喜欢被提醒",
            new_value="不喜欢被催",
            trigger="user_correction",
        )

        assert revision.persona_id == "persona-1"
        assert revision.trigger == "user_correction"


class _Cursor:
    def __init__(self, rows):
        self._rows = rows

    async def fetchall(self):
        return self._rows

    async def fetchone(self):
        return self._rows[0] if self._rows else None


class _ExportDb:
    def __init__(self):
        self.queries = []
        self.committed = False

    async def execute(self, sql, params=()):
        self.queries.append(sql)
        return _Cursor([])

    async def commit(self):
        self.committed = True


def _request() -> Request:
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/api/memory/export",
        "headers": [(b"user-agent", b"pytest")],
        "client": ("testclient", 50000),
        "server": ("testserver", 80),
        "scheme": "http",
        "state": {"auth_session_id": "session-1"},
    }
    return Request(scope)


class TestStage0ExportContracts:
    async def test_export_includes_reflection_and_revision_arrays(self):
        limiter = types.SimpleNamespace(limit=lambda _rule: (lambda func: func))
        sys.modules.setdefault("main", types.SimpleNamespace(limiter=limiter))
        from routers.memory import export_memories

        db = _ExportDb()
        with patch("routers.memory.get_shared_db", new=AsyncMock(return_value=db)):
            response = await export_memories(_request(), MemoryExportRequest(confirm_export=True))

        payload = json.loads(response.body)

        assert payload["memory_reflection"] == []
        assert payload["preference_revision_log"] == []
        assert "memories" in payload
        assert "structured_facts" in payload
        assert "graph" in payload
        assert db.committed is True
