import sys
import types
from unittest.mock import AsyncMock, patch

import aiosqlite


async def _layers_db():
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
            last_updated TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE memory_reflection (
            id TEXT PRIMARY KEY,
            insight TEXT NOT NULL,
            source_memory_ids TEXT DEFAULT '[]',
            importance REAL DEFAULT 8.0,
            created_at TEXT DEFAULT (datetime('now'))
        );
        """
    )
    await db.commit()
    return db


class TestLongTermLayersRoute:
    async def test_returns_persona_and_reflection_items(self):
        sys.modules.setdefault(
            "main",
            types.SimpleNamespace(
                limiter=types.SimpleNamespace(limit=lambda _rule: (lambda func: func))
            ),
        )
        from routers.memory import get_long_term_layers

        db = await _layers_db()
        try:
            await db.executemany(
                """INSERT INTO user_persona
                   (id, trait_key, trait_value, confidence, evidence_count, source_memory_ids, last_updated)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                [
                    (
                        "persona-1",
                        "communication_style.direct",
                        "用户偏好直接沟通",
                        0.92,
                        3,
                        '["mem-1","mem-2"]',
                        "2026-04-17 10:00:00",
                    ),
                    (
                        "persona-2",
                        "habit.running",
                        "用户长期跑步",
                        0.55,
                        1,
                        "not-json",
                        "2026-04-17 09:00:00",
                    ),
                ],
            )
            await db.executemany(
                """INSERT INTO memory_reflection
                   (id, insight, source_memory_ids, importance, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                [
                    (
                        "reflection-1",
                        "用户在项目推进中更接受明确节奏。",
                        '["mem-1","mem-3","mem-4"]',
                        9.1,
                        "2026-04-17 11:00:00",
                    ),
                    (
                        "reflection-2",
                        "用户会长期坚持运动习惯。",
                        "not-json",
                        7.4,
                        "2026-04-17 08:30:00",
                    ),
                ],
            )
            await db.commit()

            with patch("routers.memory.get_db", new=AsyncMock(return_value=db)):
                response = await get_long_term_layers()
        finally:
            await db.close()

        assert [item.id for item in response.persona_items] == ["persona-1", "persona-2"]
        assert response.persona_items[0].source_memory_ids == ["mem-1", "mem-2"]
        assert response.persona_items[1].source_memory_ids == []

        assert [item.id for item in response.reflection_items] == [
            "reflection-1",
            "reflection-2",
        ]
        assert response.reflection_items[0].source_memory_count == 3
        assert response.reflection_items[1].source_memory_ids == []
        assert response.reflection_items[1].source_memory_count == 0

    async def test_returns_empty_lists_when_no_long_term_items_exist(self):
        sys.modules.setdefault(
            "main",
            types.SimpleNamespace(
                limiter=types.SimpleNamespace(limit=lambda _rule: (lambda func: func))
            ),
        )
        from routers.memory import get_long_term_layers

        db = await _layers_db()
        try:
            with patch("routers.memory.get_shared_db", new=AsyncMock(return_value=db)):
                response = await get_long_term_layers()
        finally:
            await db.close()

        assert response.persona_items == []
        assert response.reflection_items == []
