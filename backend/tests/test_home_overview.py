import sys
import types
from unittest.mock import AsyncMock, patch

import aiosqlite


async def _overview_db():
    db = await aiosqlite.connect(":memory:")
    db.row_factory = aiosqlite.Row
    await db.executescript(
        """
        CREATE TABLE user_persona (
            id TEXT PRIMARY KEY
        );
        CREATE TABLE memory_reflection (
            id TEXT PRIMARY KEY
        );
        CREATE TABLE graph_nodes (
            id TEXT PRIMARY KEY,
            status TEXT DEFAULT 'confirmed'
        );
        CREATE TABLE memory_items (
            id TEXT PRIMARY KEY,
            admission_tier TEXT DEFAULT 'standard'
        );
        """
    )
    await db.commit()
    return db


class TestHomeLongTermOverviewRoute:
    async def test_returns_long_term_layer_counts(self):
        sys.modules.setdefault(
            "main",
            types.SimpleNamespace(
                limiter=types.SimpleNamespace(limit=lambda _rule: (lambda func: func))
            ),
        )
        from routers.memory import get_long_term_layer_overview

        db = await _overview_db()
        try:
            await db.executemany(
                "INSERT INTO user_persona (id) VALUES (?)",
                [("persona-1",), ("persona-2",)],
            )
            await db.executemany(
                "INSERT INTO memory_reflection (id) VALUES (?)",
                [("reflection-1",), ("reflection-2",), ("reflection-3",)],
            )
            await db.executemany(
                "INSERT INTO graph_nodes (id, status) VALUES (?, ?)",
                [
                    ("node-1", "pending"),
                    ("node-2", "pending"),
                    ("node-3", "confirmed"),
                ],
            )
            await db.executemany(
                "INSERT INTO memory_items (id, admission_tier) VALUES (?, ?)",
                [
                    ("memory-1", "low_value"),
                    ("memory-2", "low_value"),
                    ("memory-3", "standard"),
                ],
            )
            await db.commit()

            with patch("routers.memory.get_db", new=AsyncMock(return_value=db)):
                response = await get_long_term_layer_overview()
        finally:
            await db.close()

        assert response.persona_count == 2
        assert response.reflection_count == 3
        assert response.pending_graph_node_count == 2
        assert response.low_value_memory_count == 2

    async def test_returns_zeros_when_tables_are_empty(self):
        sys.modules.setdefault(
            "main",
            types.SimpleNamespace(
                limiter=types.SimpleNamespace(limit=lambda _rule: (lambda func: func))
            ),
        )
        from routers.memory import get_long_term_layer_overview

        db = await _overview_db()
        try:
            with patch("routers.memory.get_shared_db", new=AsyncMock(return_value=db)):
                response = await get_long_term_layer_overview()
        finally:
            await db.close()

        assert response.persona_count == 0
        assert response.reflection_count == 0
        assert response.pending_graph_node_count == 0
        assert response.low_value_memory_count == 0
