from unittest.mock import AsyncMock, MagicMock, patch

import aiosqlite
import pytest

from memory_core.services.graph_retrieval import retrieve_graph_context
from memory_core.services.retrieval_primitives import (
    _retrieve_hybrid_keyword_candidates,
    _retrieve_hybrid_semantic_candidates,
    _retrieve_recent_high_value_memories,
    _retrieve_structured_fact_memories,
)


async def _memory_db():
    db = await aiosqlite.connect(":memory:")
    db.row_factory = aiosqlite.Row
    await db.executescript(
        """
        CREATE TABLE memory_items (
            id TEXT PRIMARY KEY,
            content TEXT,
            tags TEXT DEFAULT '[]',
            kind TEXT DEFAULT 'other',
            task_status TEXT,
            consolidated INTEGER DEFAULT 0,
            importance REAL DEFAULT 5.0,
            admission_tier TEXT DEFAULT 'standard',
            created_at TEXT
        );
        CREATE TABLE structured_facts (
            id TEXT PRIMARY KEY,
            memory_id TEXT NOT NULL,
            kind TEXT NOT NULL,
            subject TEXT NOT NULL DEFAULT '',
            predicate TEXT NOT NULL DEFAULT '',
            object TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL DEFAULT 'accepted',
            created_at TEXT
        );
        """
    )
    await db.executemany(
        """INSERT INTO memory_items
           (id, content, tags, kind, task_status, consolidated, importance, admission_tier, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        [
            (
                "std",
                "我长期喜欢吃辣",
                "[]",
                "preference",
                None,
                1,
                8.0,
                "standard",
                "2026-04-14 10:00:00",
            ),
            (
                "low",
                "我长期喜欢吃甜",
                "[]",
                "preference",
                None,
                1,
                8.0,
                "low_value",
                "2026-04-14 11:00:00",
            ),
        ],
    )
    await db.executemany(
        """INSERT INTO structured_facts
           (id, memory_id, kind, subject, predicate, object, status, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        [
            ("fact-std", "std", "preference", "user", "likes", "辣", "accepted", "2026-04-14 10:00:00"),
            ("fact-low", "low", "preference", "user", "likes", "甜", "accepted", "2026-04-14 11:00:00"),
        ],
    )
    await db.commit()
    return db


async def _summary_memory_db():
    db = await aiosqlite.connect(":memory:")
    db.row_factory = aiosqlite.Row
    await db.executescript(
        """
        CREATE TABLE memory_items (
            id TEXT PRIMARY KEY,
            content TEXT,
            tags TEXT DEFAULT '[]',
            kind TEXT DEFAULT 'other',
            task_status TEXT,
            consolidated INTEGER DEFAULT 0,
            importance REAL DEFAULT 5.0,
            admission_tier TEXT DEFAULT 'standard',
            created_at TEXT
        );
        """
    )
    await db.executemany(
        """INSERT INTO memory_items
           (id, content, tags, kind, task_status, consolidated, importance, admission_tier, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        [
            (
                "new-low",
                "我昨天补完了一个很小的样式改动",
                "[]",
                "other",
                None,
                0,
                2.0,
                "standard",
                "2026-04-16 10:00:00",
            ),
            (
                "old-high",
                "我完成了四记忆架构的关键整合，长期影响比较大",
                "[]",
                "other",
                None,
                0,
                9.0,
                "standard",
                "2026-04-10 10:00:00",
            ),
            (
                "new-default",
                "我整理了本周的工作优先级",
                "[]",
                "other",
                None,
                0,
                None,
                "standard",
                "2026-04-15 10:00:00",
            ),
            (
                "old-default",
                "我回顾了上周的工作优先级",
                "[]",
                "other",
                None,
                0,
                None,
                "standard",
                "2026-04-11 10:00:00",
            ),
            (
                "question",
                "我喜欢什么口味？",
                "[]",
                "other",
                None,
                0,
                10.0,
                "standard",
                "2026-04-17 10:00:00",
            ),
        ],
    )
    await db.commit()
    return db


class TestAdmissionRetrievalVisibility:
    async def test_hybrid_keyword_excludes_low_value_memories(self):
        db = await _memory_db()
        try:
            results = await _retrieve_hybrid_keyword_candidates("长期喜欢", db)
        finally:
            await db.close()

        assert set(results) == {"std"}

    async def test_recent_high_value_excludes_low_value_memories(self):
        db = await _memory_db()
        try:
            results = await _retrieve_recent_high_value_memories("我最近都干什么了", db)
        finally:
            await db.close()

        assert [item["id"] for item in results] == ["std"]

    async def test_recent_high_value_prefers_higher_importance_over_mere_recency(self):
        db = await _summary_memory_db()
        try:
            results = await _retrieve_recent_high_value_memories("我最近都干什么了", db)
        finally:
            await db.close()

        ids = [item["id"] for item in results]
        assert ids.index("old-high") < ids.index("new-low")

    async def test_recent_high_value_keeps_summary_candidate_filter(self):
        db = await _summary_memory_db()
        try:
            results = await _retrieve_recent_high_value_memories("我最近都干什么了", db)
        finally:
            await db.close()

        assert "question" not in [item["id"] for item in results]

    async def test_recent_high_value_falls_back_to_recency_when_importance_is_equal_or_missing(self):
        db = await _summary_memory_db()
        try:
            results = await _retrieve_recent_high_value_memories("我最近都干什么了", db)
        finally:
            await db.close()

        ids = [item["id"] for item in results]
        assert ids.index("new-default") < ids.index("old-default")

    async def test_structured_fact_retrieval_excludes_low_value_source_memory(self):
        db = await _memory_db()
        try:
            results = await _retrieve_structured_fact_memories(
                db,
                ("preference",),
                query="喜欢什么",
            )
        finally:
            await db.close()

        assert [item["id"] for item in results] == ["std"]

    async def test_hybrid_semantic_query_includes_admission_filter(self):
        vec_cursor = MagicMock()
        vec_cursor.fetchall = AsyncMock(
            return_value=[
                {"ref_id": "std", "distance": 0.1},
                {"ref_id": "low", "distance": 0.1},
            ]
        )
        memory_cursor = MagicMock()
        memory_cursor.fetchall = AsyncMock(
            return_value=[
                {
                    "id": "std",
                    "content": "我长期喜欢吃辣",
                    "created_at": "2026-04-14 10:00:00",
                    "kind": "preference",
                    "importance": 8.0,
                }
            ]
        )
        db = MagicMock()
        db.sqlite_vec_loaded = True
        db.execute = AsyncMock(side_effect=[vec_cursor, memory_cursor])

        with patch(
            "memory_core.services.retrieval_primitives.embed_text",
            new=AsyncMock(return_value=[0.1, 0.2]),
        ), patch(
            "memory_core.services.retrieval_primitives.serialize_float32",
            return_value=b"vector",
        ):
            results = await _retrieve_hybrid_semantic_candidates("喜欢什么", db)

        memory_sql = db.execute.await_args_list[1].args[0]
        assert "COALESCE(admission_tier, 'standard') = 'standard'" in memory_sql
        assert set(results) == {"std"}


class TestAdmissionGraphContextVisibility:
    async def test_graph_context_excludes_low_value_source_memory(self):
        db = await aiosqlite.connect(":memory:")
        db.row_factory = aiosqlite.Row
        await db.executescript(
            """
            CREATE TABLE memory_items (
                id TEXT PRIMARY KEY,
                admission_tier TEXT DEFAULT 'standard'
            );
            CREATE TABLE graph_nodes (
                id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                label TEXT NOT NULL,
                status TEXT DEFAULT 'confirmed',
                last_seen_at TEXT DEFAULT '2026-04-14 10:00:00'
            );
            CREATE TABLE graph_edges (
                id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                relation TEXT NOT NULL,
                source_memory_id TEXT,
                created_at TEXT DEFAULT '2026-04-14 10:00:00'
            );
            """
        )
        await db.executemany(
            "INSERT INTO memory_items (id, admission_tier) VALUES (?, ?)",
            [("std", "standard"), ("low", "low_value")],
        )
        await db.executemany(
            "INSERT INTO graph_nodes (id, type, label, status) VALUES (?, ?, ?, ?)",
            [
                ("person", "person", "小王", "confirmed"),
                ("project", "project", "Pervault", "confirmed"),
                ("noise", "project", "无关项目", "confirmed"),
            ],
        )
        await db.executemany(
            """INSERT INTO graph_edges
               (id, source_id, target_id, relation, source_memory_id)
               VALUES (?, ?, ?, ?, ?)""",
            [
                ("edge-std", "person", "project", "discussed", "std"),
                ("edge-low", "person", "noise", "mentioned", "low"),
            ],
        )
        await db.commit()
        try:
            context = await retrieve_graph_context("小王", db)
        finally:
            await db.close()

        assert "Pervault" in context
        assert "无关项目" not in context
