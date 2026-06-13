import sys
import types
from unittest.mock import AsyncMock, patch

import aiosqlite


async def _memory_search_db():
    db = await aiosqlite.connect(":memory:")
    db.row_factory = aiosqlite.Row
    await db.executescript(
        """
        CREATE TABLE memory_items (
            id TEXT PRIMARY KEY,
            voice_record_id TEXT,
            content TEXT,
            tags TEXT DEFAULT '[]',
            kind TEXT DEFAULT 'other',
            task_status TEXT,
            emotion_score REAL DEFAULT 0.0,
            consolidated INTEGER DEFAULT 0,
            importance REAL DEFAULT 5.0,
            admission_score REAL DEFAULT NULL,
            admission_tier TEXT DEFAULT 'standard',
            weight REAL DEFAULT 1.0,
            last_referenced_at TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        );
        """
    )
    await db.commit()
    return db


class TestMemorySearchAdmissionTierFilter:
    async def test_search_memories_filters_by_admission_tier(self):
        sys.modules.setdefault(
            "main",
            types.SimpleNamespace(
                limiter=types.SimpleNamespace(limit=lambda _rule: (lambda func: func))
            ),
        )
        from routers.memory import search_memories

        db = await _memory_search_db()
        try:
            await db.executemany(
                """INSERT INTO memory_items
                   (id, content, kind, admission_tier, weight, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                [
                    ("mem-low-1", "低价值记忆 1", "other", "low_value", 1.0, "2026-04-17 10:00:00"),
                    ("mem-low-2", "低价值记忆 2", "task", "low_value", 1.0, "2026-04-17 09:00:00"),
                    ("mem-std-1", "标准记忆", "other", "standard", 1.0, "2026-04-17 08:00:00"),
                ],
            )
            await db.commit()

            with patch("routers.memory.get_db", new=AsyncMock(return_value=db)):
                result = await search_memories(
                    q="",
                    kind="",
                    admission_tier="low_value",
                    limit=50,
                    offset=0,
                )
        finally:
            await db.close()

        assert result.total == 2
        assert [item.id for item in result.items] == ["mem-low-1", "mem-low-2"]

    async def test_search_memories_combines_kind_and_admission_tier_filters(self):
        sys.modules.setdefault(
            "main",
            types.SimpleNamespace(
                limiter=types.SimpleNamespace(limit=lambda _rule: (lambda func: func))
            ),
        )
        from routers.memory import search_memories

        db = await _memory_search_db()
        try:
            await db.executemany(
                """INSERT INTO memory_items
                   (id, content, kind, admission_tier, weight, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                [
                    ("mem-low-task", "低价值任务", "task", "low_value", 1.0, "2026-04-17 10:00:00"),
                    ("mem-low-note", "低价值笔记", "other", "low_value", 1.0, "2026-04-17 09:00:00"),
                    ("mem-std-task", "标准任务", "task", "standard", 1.0, "2026-04-17 08:00:00"),
                ],
            )
            await db.commit()

            with patch("routers.memory.get_db", new=AsyncMock(return_value=db)):
                result = await search_memories(
                    q="",
                    kind="task",
                    admission_tier="low_value",
                    limit=50,
                    offset=0,
                )
        finally:
            await db.close()

        assert result.total == 1
        assert [item.id for item in result.items] == ["mem-low-task"]
