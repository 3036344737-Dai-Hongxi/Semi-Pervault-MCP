import aiosqlite

from memory_core.services.retrieval_boot import get_boot_context
from memory_core.services.retrieval_context import _retrieve_persona_context
from memory_core.services.retrieval_primitives import _retrieve_persona_traits


async def _db_with_persona():
    db = await aiosqlite.connect(":memory:")
    db.row_factory = aiosqlite.Row
    await db.executescript(
        """
        CREATE TABLE user_persona (
            id TEXT PRIMARY KEY,
            trait_key TEXT NOT NULL,
            trait_value TEXT NOT NULL,
            confidence REAL DEFAULT 0.8,
            evidence_count INTEGER DEFAULT 1,
            source_memory_ids TEXT DEFAULT '[]',
            last_updated TEXT DEFAULT '2026-04-15 10:00:00',
            created_at TEXT DEFAULT '2026-04-15 10:00:00'
        );
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
        CREATE TABLE memory_reflection (
            id TEXT PRIMARY KEY,
            insight TEXT NOT NULL,
            source_memory_ids TEXT DEFAULT '[]',
            importance REAL DEFAULT 8.0,
            created_at TEXT
        );
        """
    )
    await db.executemany(
        """INSERT INTO user_persona
           (id, trait_key, trait_value, confidence, evidence_count, last_updated)
           VALUES (?, ?, ?, ?, ?, ?)""",
        [
            (
                "persona-1",
                "communication_style.direct",
                "用户偏好直接清晰的沟通",
                0.9,
                3,
                "2026-04-15 10:00:00",
            ),
            (
                "persona-2",
                "habit.running",
                "用户长期跑步",
                0.75,
                2,
                "2026-04-15 09:00:00",
            ),
            (
                "persona-3",
                "goal.health",
                "用户重视长期健康目标",
                0.6,
                1,
                "2026-04-15 08:30:00",
            ),
            (
                "persona-4",
                "preference.food",
                "用户更喜欢清淡口味",
                0.55,
                1,
                "2026-04-15 08:00:00",
            ),
            (
                "persona-5",
                "profile.identity",
                "用户是偏独立思考的人",
                0.5,
                1,
                "2026-04-15 07:30:00",
            ),
            (
                "persona-low",
                "temporary.mood",
                "用户今天很累",
                0.35,
                1,
                "2026-04-15 08:00:00",
            ),
        ],
    )
    await db.commit()
    return db


class TestPersonaRetrieval:
    async def test_retrieve_persona_traits_prefers_matching_communication_trait(self):
        db = await _db_with_persona()
        try:
            results = await _retrieve_persona_traits("我的沟通风格是什么", db)
        finally:
            await db.close()

        assert results[0]["id"] == "persona:persona-1"
        assert results[0]["kind"] == "persona"
        assert results[0]["content"] == "用户画像：communication_style.direct = 用户偏好直接清晰的沟通"

    async def test_retrieve_persona_traits_prefers_goal_trait_for_goal_question(self):
        db = await _db_with_persona()
        try:
            results = await _retrieve_persona_traits("我的长期目标是什么", db)
        finally:
            await db.close()

        assert results[0]["id"] == "persona:persona-3"

    async def test_retrieve_persona_traits_prefers_preference_trait_for_preference_question(self):
        db = await _db_with_persona()
        try:
            results = await _retrieve_persona_traits("我的偏好是什么", db)
        finally:
            await db.close()

        assert results[0]["id"] == "persona:persona-4"

    async def test_retrieve_persona_traits_falls_back_to_confidence_when_query_not_specific(self):
        db = await _db_with_persona()
        try:
            results = await _retrieve_persona_traits("今天天气不错", db)
        finally:
            await db.close()

        assert [item["id"] for item in results[:3]] == [
            "persona:persona-1",
            "persona:persona-2",
            "persona:persona-3",
        ]

    async def test_retrieve_persona_context_falls_back_when_persona_empty(self):
        db = await aiosqlite.connect(":memory:")
        db.row_factory = aiosqlite.Row
        await db.executescript(
            """
            CREATE TABLE user_persona (
                id TEXT PRIMARY KEY,
                trait_key TEXT NOT NULL,
                trait_value TEXT NOT NULL,
                confidence REAL DEFAULT 0.8,
                evidence_count INTEGER DEFAULT 1,
                source_memory_ids TEXT DEFAULT '[]',
                last_updated TEXT DEFAULT '2026-04-15 10:00:00',
                created_at TEXT DEFAULT '2026-04-15 10:00:00'
            );
            CREATE TABLE memory_items (
                id TEXT PRIMARY KEY,
                content TEXT,
                tags TEXT DEFAULT '[]',
                kind TEXT DEFAULT 'preference',
                task_status TEXT,
                consolidated INTEGER DEFAULT 1,
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
        await db.execute(
            """INSERT INTO memory_items
               (id, content, kind, created_at)
               VALUES (?, ?, ?, ?)""",
            ("mem-1", "我长期喜欢直接沟通", "preference", "2026-04-15 10:00:00"),
        )
        await db.execute(
            """INSERT INTO structured_facts
               (id, memory_id, kind, subject, predicate, object, status, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                "fact-1",
                "mem-1",
                "preference",
                "user",
                "likes",
                "直接沟通",
                "accepted",
                "2026-04-15 10:00:00",
            ),
        )
        await db.commit()
        try:
            results = await _retrieve_persona_context("我的沟通风格是什么", db)
        finally:
            await db.close()

        assert results
        assert results[0]["id"] == "mem-1"

    async def test_boot_context_includes_at_most_three_high_confidence_persona(self):
        db = await _db_with_persona()
        try:
            await db.executemany(
                """INSERT INTO user_persona
                   (id, trait_key, trait_value, confidence, evidence_count, last_updated)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                [
                    ("persona-6", "work.style", "用户偏好深度工作", 0.8, 1, "2026-04-15 08:00:00"),
                    ("persona-7", "goal.health", "用户重视健康", 0.8, 1, "2026-04-15 07:00:00"),
                ],
            )
            await db.commit()
            results = await get_boot_context(db)
        finally:
            await db.close()

        persona_items = [item for item in results if item.get("kind") == "persona"]
        assert len(persona_items) == 3
        assert all(item["_source"] == "persona" for item in persona_items)

    async def test_boot_context_includes_at_most_two_reflections_after_persona(self):
        db = await _db_with_persona()
        try:
            await db.executemany(
                """INSERT INTO memory_reflection
                   (id, insight, source_memory_ids, importance, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                [
                    ("reflection-1", "用户持续推进长期记忆系统", '["mem-1"]', 9.0, "2026-04-15 10:00:00"),
                    ("reflection-2", "用户偏好分阶段验证复杂功能", '["mem-2"]', 8.5, "2026-04-15 09:00:00"),
                    ("reflection-3", "用户重视清晰计划", '["mem-3"]', 8.0, "2026-04-15 08:00:00"),
                ],
            )
            await db.commit()
            results = await get_boot_context(db)
        finally:
            await db.close()

        reflection_items = [item for item in results if item.get("kind") == "reflection"]
        assert [item["id"] for item in reflection_items] == [
            "reflection:reflection-1",
            "reflection:reflection-2",
        ]
        assert all(item["_source"] == "boot_reflection" for item in reflection_items)
        assert reflection_items[0]["content"] == "长期洞察：用户持续推进长期记忆系统"
