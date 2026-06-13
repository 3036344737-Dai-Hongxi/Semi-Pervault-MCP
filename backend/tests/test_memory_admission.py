import sys
import types
from unittest.mock import AsyncMock, MagicMock, patch

import aiosqlite
import pytest

from memory_core.services.memory_admission import (
    AdmissionScore,
    _score_novelty,
    compute_admission_score,
)


class _Cursor:
    def __init__(self, rows):
        self._rows = rows

    async def fetchall(self):
        return self._rows


class _Db:
    def __init__(self, rows):
        self.rows = rows
        self.execute_calls = []

    async def execute(self, sql, params=()):
        self.execute_calls.append((sql, params))
        return _Cursor(self.rows)


def _row(memory_id: str, content: str):
    row = MagicMock()
    row.__getitem__ = lambda self, key: {"id": memory_id, "content": content}[key]
    return row


class TestMemoryAdmissionService:
    async def test_high_value_preference_is_standard(self):
        db = _Db([])
        with patch(
            "memory_core.services.memory_admission.score_admission_with_llm",
            new=AsyncMock(return_value={"utility": 0.9, "confidence": 0.85}),
        ):
            score = await compute_admission_score(
                "我长期喜欢吃辣",
                "preference",
                db,
                exclude_memory_id="current-id",
            )

        assert isinstance(score, AdmissionScore)
        assert score.type_prior == pytest.approx(0.9)
        assert score.novelty == pytest.approx(1.0)
        assert score.total == pytest.approx(0.91)
        assert score.tier == "standard"

    async def test_low_utility_noise_is_low_value(self):
        db = _Db([])
        with patch(
            "memory_core.services.memory_admission.score_admission_with_llm",
            new=AsyncMock(return_value={"utility": 0.1, "confidence": 0.2}),
        ):
            score = await compute_admission_score("哈哈", "other", db)

        assert score.total == pytest.approx(0.3375)
        assert score.tier == "low_value"

    async def test_exact_duplicate_has_zero_novelty(self):
        db = _Db([_row("old-id", "我长期喜欢吃辣")])

        novelty = await _score_novelty(
            "我长期喜欢吃辣",
            db,
            exclude_memory_id="current-id",
        )

        assert novelty == pytest.approx(0.0)

    async def test_near_duplicate_lowers_novelty(self):
        db = _Db([_row("old-id", "我长期喜欢吃辣的食物")])

        novelty = await _score_novelty(
            "我长期喜欢吃辣食物",
            db,
            exclude_memory_id="current-id",
        )

        assert novelty < 1.0

    async def test_excludes_current_memory_from_novelty_scan(self):
        db = _Db([_row("other-id", "我长期喜欢吃辣")])

        await _score_novelty("我长期喜欢吃辣", db, exclude_memory_id="current-id")

        sql, params = db.execute_calls[0]
        assert "id != ?" in sql
        assert params[0] == "current-id"


async def _admission_explanation_db():
    db = await aiosqlite.connect(":memory:")
    db.row_factory = aiosqlite.Row
    await db.executescript(
        """
        CREATE TABLE memory_admission_log (
            id TEXT PRIMARY KEY,
            memory_id TEXT,
            raw_content TEXT NOT NULL,
            score_utility REAL,
            score_confidence REAL,
            score_novelty REAL,
            score_recency REAL,
            score_type_prior REAL,
            total_score REAL,
            admitted INTEGER NOT NULL DEFAULT 1,
            tier TEXT NOT NULL DEFAULT 'standard',
            created_at TEXT DEFAULT (datetime('now'))
        );
        """
    )
    await db.commit()
    return db


class TestMemoryAdmissionExplanationRoute:
    async def test_returns_latest_admission_explanation_for_memory(self):
        sys.modules.setdefault(
            "main",
            types.SimpleNamespace(limiter=types.SimpleNamespace(limit=lambda _rule: (lambda func: func))),
        )
        from routers.memory import get_memory_admission_explanation

        db = await _admission_explanation_db()
        try:
            await db.execute(
                """INSERT INTO memory_admission_log
                   (id, memory_id, raw_content, score_utility, score_confidence,
                    score_novelty, score_recency, score_type_prior, total_score,
                    admitted, tier, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    "log-1",
                    "mem-1",
                    "旧内容",
                    0.2,
                    0.3,
                    0.4,
                    1.0,
                    0.35,
                    0.31,
                    0,
                    "low_value",
                    "2026-04-17 09:00:00",
                ),
            )
            await db.execute(
                """INSERT INTO memory_admission_log
                   (id, memory_id, raw_content, score_utility, score_confidence,
                    score_novelty, score_recency, score_type_prior, total_score,
                    admitted, tier, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    "log-2",
                    "mem-1",
                    "新内容",
                    0.8,
                    0.7,
                    0.6,
                    1.0,
                    0.9,
                    0.765,
                    1,
                    "standard",
                    "2026-04-17 10:00:00",
                ),
            )
            await db.commit()

            with patch("routers.memory.get_db", new=AsyncMock(return_value=db)):
                response = await get_memory_admission_explanation("mem-1")
        finally:
            await db.close()

        assert response.memory_id == "mem-1"
        assert response.explanation is not None
        assert response.explanation.utility == pytest.approx(0.8)
        assert response.explanation.confidence == pytest.approx(0.7)
        assert response.explanation.novelty == pytest.approx(0.6)
        assert response.explanation.recency == pytest.approx(1.0)
        assert response.explanation.type_prior == pytest.approx(0.9)
        assert response.explanation.total_score == pytest.approx(0.765)
        assert response.explanation.tier == "standard"
        assert response.explanation.created_at == "2026-04-17 10:00:00"

    async def test_returns_null_explanation_when_memory_has_no_admission_log(self):
        sys.modules.setdefault(
            "main",
            types.SimpleNamespace(limiter=types.SimpleNamespace(limit=lambda _rule: (lambda func: func))),
        )
        from routers.memory import get_memory_admission_explanation

        db = await _admission_explanation_db()
        try:
            with patch("routers.memory.get_shared_db", new=AsyncMock(return_value=db)):
                response = await get_memory_admission_explanation("mem-missing")
        finally:
            await db.close()

        assert response.memory_id == "mem-missing"
        assert response.explanation is None
