import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import aiosqlite
import pytest

from memory_core.services.persona_service import PersonaTraitCandidate
from memory_core.services.sleep_agent import (
    SleepAgentResult,
    SleepMemory,
    SleepTopic,
    _ReflectionCandidate,
    _cluster_topics_with_llm,
    _generate_reflections,
    _load_persona_refresh_memories,
    _load_topic_memories,
    _persona_refresh,
    _reflection_insight_dedupe_key,
    _reflection_source_fingerprint,
    _topic_regroup,
    run_sleep_agent_once,
    run_sleep_agent_periodically,
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


async def _sleep_db():
    db = await aiosqlite.connect(":memory:")
    db.row_factory = aiosqlite.Row
    await db.executescript(
        """
        CREATE TABLE memory_items (
            id TEXT PRIMARY KEY,
            content TEXT,
            kind TEXT DEFAULT 'other',
            importance REAL DEFAULT 5.0,
            admission_tier TEXT DEFAULT 'standard',
            created_at TEXT
        );
        CREATE TABLE user_persona (
            id TEXT PRIMARY KEY,
            trait_key TEXT NOT NULL,
            trait_value TEXT NOT NULL,
            confidence REAL DEFAULT 0.8,
            evidence_count INTEGER DEFAULT 1,
            source_memory_ids TEXT DEFAULT '[]',
            last_updated TEXT DEFAULT (datetime('now')),
            created_at TEXT DEFAULT (datetime('now'))
        );
        CREATE UNIQUE INDEX idx_user_persona_key ON user_persona(trait_key);
        CREATE TABLE memory_reflection (
            id TEXT PRIMARY KEY,
            insight TEXT NOT NULL,
            source_memory_ids TEXT DEFAULT '[]',
            insight_dedupe_key TEXT DEFAULT '',
            source_memory_fingerprint TEXT DEFAULT '[]',
            importance REAL DEFAULT 8.0,
            created_at TEXT DEFAULT (datetime('now'))
        );
        CREATE TABLE scheduler_run_log (
            id TEXT PRIMARY KEY,
            scheduler_name TEXT NOT NULL,
            status TEXT NOT NULL,
            started_at TEXT DEFAULT (datetime('now')),
            finished_at TEXT,
            summary_json TEXT,
            error_count INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE sleep_agent_checkpoint (
            stage_name TEXT PRIMARY KEY,
            checkpoint_created_at TEXT,
            last_run_id TEXT,
            updated_at TEXT DEFAULT (datetime('now'))
        );
        """
    )
    await db.commit()
    return db


async def _insert_memory(
    db,
    memory_id: str,
    *,
    content: str = "我长期坚持跑步",
    kind: str = "fact",
    importance: float = 8.0,
    admission_tier: str = "standard",
    created_at_sql: str = "datetime('now')",
):
    await db.execute(
        f"""INSERT INTO memory_items
            (id, content, kind, importance, admission_tier, created_at)
            VALUES (?, ?, ?, ?, ?, {created_at_sql})""",
        (memory_id, content, kind, importance, admission_tier),
    )


async def _insert_reflection(
    db,
    reflection_id: str,
    *,
    insight: str,
    source_memory_ids: list[str],
    importance: float = 8.0,
    created_at_sql: str = "datetime('now')",
):
    await db.execute(
        f"""INSERT INTO memory_reflection
            (
                id,
                insight,
                source_memory_ids,
                insight_dedupe_key,
                source_memory_fingerprint,
                importance,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, {created_at_sql})""",
        (
            reflection_id,
            insight,
            json.dumps(source_memory_ids, ensure_ascii=False),
            _reflection_insight_dedupe_key(insight),
            _reflection_source_fingerprint(source_memory_ids),
            importance,
        ),
    )


class TestSleepAgentLoaders:
    async def test_load_topic_memories_filters_recent_standard_high_importance(self):
        db = await _sleep_db()
        try:
            await _insert_memory(db, "good", importance=6.5)
            await _insert_memory(db, "low-importance", importance=5.9)
            await _insert_memory(db, "low-tier", importance=8.0, admission_tier="low_value")
            await _insert_memory(db, "old", importance=9.0, created_at_sql="datetime('now', '-2 days')")
            await db.commit()

            memories = await _load_topic_memories(db)
        finally:
            await db.close()

        assert [memory.id for memory in memories] == ["good"]

    async def test_load_persona_refresh_memories_filters_window_tier_importance_and_kind(self):
        db = await _sleep_db()
        try:
            await _insert_memory(db, "good", kind="preference", importance=7.5)
            await _insert_memory(db, "other-kind", kind="other", importance=9.0)
            await _insert_memory(db, "low-importance", kind="fact", importance=6.9)
            await _insert_memory(db, "low-tier", kind="fact", importance=9.0, admission_tier="low_value")
            await _insert_memory(db, "old", kind="fact", importance=9.0, created_at_sql="datetime('now', '-8 days')")
            await db.commit()

            memories = await _load_persona_refresh_memories(db)
        finally:
            await db.close()

        assert [memory.id for memory in memories] == ["good"]


class TestSleepAgentTopicClustering:
    async def test_cluster_topics_parses_json_and_drops_unknown_ids(self):
        memories = [
            SleepMemory("m1", "推进 Pervault", "project_update", 8.0, "2026-04-15"),
            SleepMemory("m2", "整理长期记忆", "fact", 7.0, "2026-04-15"),
        ]
        client = _make_llm_response(
            {
                "topics": [
                    {
                        "title": "长期记忆",
                        "summary": "用户在推进长期记忆架构",
                        "source_memory_ids": ["m1", "unknown"],
                    },
                    {
                        "title": "无效主题",
                        "summary": "只引用未知 id",
                        "source_memory_ids": ["missing"],
                    },
                ]
            }
        )

        with patch("memory_core.services.sleep_agent.get_client", return_value=client):
            topics = await _cluster_topics_with_llm(memories)

        assert topics == [
            SleepTopic(
                title="长期记忆",
                summary="用户在推进长期记忆架构",
                source_memory_ids=["m1"],
            )
        ]

    async def test_cluster_topics_invalid_json_returns_empty(self):
        memories = [SleepMemory("m1", "推进 Pervault", "project_update", 8.0, "2026-04-15")]
        client = _make_llm_response("not json")

        with patch("memory_core.services.sleep_agent.get_client", return_value=client):
            topics = await _cluster_topics_with_llm(memories)

        assert topics == []

    async def test_cluster_topics_llm_exception_returns_empty(self):
        memories = [SleepMemory("m1", "推进 Pervault", "project_update", 8.0, "2026-04-15")]
        client = MagicMock()
        client.chat.completions.create = AsyncMock(side_effect=Exception("network"))

        with patch("memory_core.services.sleep_agent.get_client", return_value=client):
            topics = await _cluster_topics_with_llm(memories)

        assert topics == []


class TestSleepAgentPersonaRefresh:
    async def test_persona_refresh_upserts_traits_and_continues_after_llm_error(self):
        db = await _sleep_db()
        try:
            await _insert_memory(db, "good", content="我长期喜欢直接沟通", kind="preference", importance=8.0)
            await _insert_memory(db, "bad", content="这条会失败", kind="fact", importance=8.0)
            await db.commit()

            async def fake_extract(content: str, kind: str):
                if "失败" in content:
                    raise Exception("llm failed")
                return [
                    PersonaTraitCandidate(
                        "communication_style.direct",
                        "用户偏好直接沟通",
                        0.85,
                    )
                ]

            result = SleepAgentResult()
            with patch(
                "memory_core.services.sleep_agent.extract_persona_traits_with_llm",
                new=AsyncMock(side_effect=fake_extract),
            ):
                await _persona_refresh(db, result)

            row = await (
                await db.execute(
                    "SELECT trait_key, trait_value, evidence_count FROM user_persona"
                )
            ).fetchone()
        finally:
            await db.close()

        assert result.persona_memory_count == 2
        assert result.persona_traits_upserted == 1
        assert result.errors == ["persona:bad:Exception"]
        assert row["trait_key"] == "communication_style.direct"
        assert row["trait_value"] == "用户偏好直接沟通"
        assert row["evidence_count"] == 1


class TestSleepAgentReflections:
    async def test_generate_reflections_skips_when_total_importance_is_low(self):
        db = await _sleep_db()
        try:
            await _insert_memory(db, "m1", importance=8.0)
            await db.commit()
            result = SleepAgentResult()
            with patch(
                "memory_core.services.sleep_agent._generate_reflections_with_llm",
                new=AsyncMock(return_value=[]),
            ) as mock_generate:
                await _generate_reflections(db, result)
        finally:
            await db.close()

        assert result.skipped_reason == "insufficient_total_importance"
        mock_generate.assert_not_awaited()

    async def test_generate_reflections_writes_candidates_when_threshold_met(self):
        db = await _sleep_db()
        try:
            for index in range(7):
                await _insert_memory(
                    db,
                    f"m{index}",
                    content=f"长期记忆阶段 {index}",
                    importance=8.0,
                )
            await db.commit()
            result = SleepAgentResult()
            candidate = _ReflectionCandidate(
                insight="用户最近持续推进长期记忆系统，并偏好分阶段验证。",
                importance=8.5,
                source_memory_ids=["m0", "m1"],
            )
            with patch(
                "memory_core.services.sleep_agent._generate_reflections_with_llm",
                new=AsyncMock(return_value=[candidate]),
            ):
                await _generate_reflections(db, result)
            rows = await (
                await db.execute(
                    "SELECT insight, source_memory_ids, importance FROM memory_reflection"
                )
            ).fetchall()
        finally:
            await db.close()

        assert result.reflections_created == 1
        assert len(rows) == 1
        assert rows[0]["insight"] == candidate.insight
        assert json.loads(rows[0]["source_memory_ids"]) == ["m0", "m1"]
        assert rows[0]["importance"] == pytest.approx(8.5)

    async def test_generate_reflections_skips_duplicate_insight(self):
        db = await _sleep_db()
        try:
            for index in range(7):
                await _insert_memory(db, f"m{index}", importance=8.0)
            await _insert_reflection(
                db,
                "existing",
                insight="用户最近持续推进长期记忆系统，并偏好分阶段验证。",
                source_memory_ids=["m0", "m1"],
                importance=8.5,
            )
            await db.commit()
            result = SleepAgentResult()
            candidate = _ReflectionCandidate(
                insight="用户最近持续推进长期记忆系统，并偏好分阶段验证。",
                importance=9.0,
                source_memory_ids=["m0", "m1"],
            )
            with patch(
                "memory_core.services.sleep_agent._generate_reflections_with_llm",
                new=AsyncMock(return_value=[candidate]),
            ):
                await _generate_reflections(db, result)
            count = await (
                await db.execute("SELECT COUNT(*) AS cnt FROM memory_reflection")
            ).fetchone()
        finally:
            await db.close()

        assert result.reflections_created == 0
        assert count["cnt"] == 1

    async def test_generate_reflections_skips_exact_duplicate_even_when_older_than_previous_scan_window(self):
        db = await _sleep_db()
        try:
            for index in range(7):
                await _insert_memory(db, f"m{index}", importance=8.0)
            await _insert_reflection(
                db,
                "existing-old",
                insight="用户最近持续推进长期记忆系统，并偏好分阶段验证。",
                source_memory_ids=["m0", "m1"],
                importance=8.5,
                created_at_sql="'2026-04-01 09:00:00'",
            )
            for index in range(205):
                await _insert_reflection(
                    db,
                    f"filler-{index}",
                    insight=f"其他长期洞察 {index}",
                    source_memory_ids=[f"other-{index}"],
                    importance=7.5,
                )
            await db.commit()

            result = SleepAgentResult()
            candidate = _ReflectionCandidate(
                insight="用户最近持续推进长期记忆系统，并偏好分阶段验证。",
                importance=9.0,
                source_memory_ids=["m0", "m1"],
            )
            with patch(
                "memory_core.services.sleep_agent._generate_reflections_with_llm",
                new=AsyncMock(return_value=[candidate]),
            ):
                await _generate_reflections(db, result)
            count = await (
                await db.execute("SELECT COUNT(*) AS cnt FROM memory_reflection")
            ).fetchone()
        finally:
            await db.close()

        assert result.reflections_created == 0
        assert count["cnt"] == 206

    async def test_generate_reflections_skips_same_source_high_similarity_duplicate(self):
        db = await _sleep_db()
        try:
            for index in range(7):
                await _insert_memory(db, f"m{index}", importance=8.0)
            await _insert_reflection(
                db,
                "existing",
                insight="用户最近持续推进长期记忆系统，并偏好分阶段验证。",
                source_memory_ids=["m0", "m1"],
                importance=8.5,
            )
            await db.commit()

            result = SleepAgentResult()
            candidate = _ReflectionCandidate(
                insight="用户最近持续推进长期记忆系统，也偏好分阶段验证。",
                importance=8.8,
                source_memory_ids=["m1", "m0"],
            )
            with patch(
                "memory_core.services.sleep_agent._generate_reflections_with_llm",
                new=AsyncMock(return_value=[candidate]),
            ):
                await _generate_reflections(db, result)
            count = await (
                await db.execute("SELECT COUNT(*) AS cnt FROM memory_reflection")
            ).fetchone()
        finally:
            await db.close()

        assert result.reflections_created == 0
        assert count["cnt"] == 1

    async def test_generate_reflections_allows_distinct_reflection_with_same_sources(self):
        db = await _sleep_db()
        try:
            for index in range(7):
                await _insert_memory(db, f"m{index}", importance=8.0)
            await _insert_reflection(
                db,
                "existing",
                insight="用户最近持续推进长期记忆系统，并偏好分阶段验证。",
                source_memory_ids=["m0", "m1"],
                importance=8.5,
            )
            await db.commit()

            result = SleepAgentResult()
            candidate = _ReflectionCandidate(
                insight="用户这段时间更关注交付节奏，并主动压缩验证风险。",
                importance=8.6,
                source_memory_ids=["m1", "m0"],
            )
            with patch(
                "memory_core.services.sleep_agent._generate_reflections_with_llm",
                new=AsyncMock(return_value=[candidate]),
            ):
                await _generate_reflections(db, result)
            rows = await (
                await db.execute(
                    """SELECT insight, source_memory_ids
                       FROM memory_reflection
                       ORDER BY created_at ASC"""
                )
            ).fetchall()
        finally:
            await db.close()

        assert result.reflections_created == 1
        assert [row["insight"] for row in rows] == [
            "用户最近持续推进长期记忆系统，并偏好分阶段验证。",
            "用户这段时间更关注交付节奏，并主动压缩验证风险。",
        ]


class TestSleepAgentEntrypoints:
    async def test_run_sleep_agent_once_executes_all_stages(self):
        db = await _sleep_db()
        original_close = db.close
        topics = [SleepTopic("主题", "摘要", ["m1"])]

        try:
            db.close = AsyncMock()
            with patch("memory_core.services.sleep_agent.get_db", new=AsyncMock(return_value=db)), patch(
                "memory_core.services.sleep_agent._topic_regroup",
                new=AsyncMock(return_value=(topics, "2026-04-17 10:00:00")),
            ) as mock_topic, patch(
                "memory_core.services.sleep_agent._persona_refresh",
                new=AsyncMock(return_value="2026-04-17 10:05:00"),
            ) as mock_persona, patch(
                "memory_core.services.sleep_agent._generate_reflections",
                new=AsyncMock(return_value="2026-04-17 10:10:00"),
            ) as mock_reflection:
                returned = await run_sleep_agent_once()

            row = await (
                await db.execute(
                    "SELECT scheduler_name, status, error_count FROM scheduler_run_log"
                )
            ).fetchone()
            checkpoint_rows = await (
                await db.execute(
                    """SELECT stage_name, checkpoint_created_at, last_run_id
                       FROM sleep_agent_checkpoint
                       ORDER BY stage_name ASC"""
                )
            ).fetchall()
        finally:
            await original_close()

        assert isinstance(returned, SleepAgentResult)
        mock_topic.assert_awaited_once()
        mock_persona.assert_awaited_once()
        mock_reflection.assert_awaited_once()
        assert returned.error_count == 0
        assert row["scheduler_name"] == "sleep_agent"
        assert row["status"] == "completed"
        assert row["error_count"] == 0
        assert [(row["stage_name"], row["checkpoint_created_at"]) for row in checkpoint_rows] == [
            ("persona_refresh", "2026-04-17 10:05:00"),
            ("reflection_generation", "2026-04-17 10:10:00"),
            ("topic_regroup", "2026-04-17 10:00:00"),
        ]
        assert all(row["last_run_id"] for row in checkpoint_rows)
        db.close.assert_awaited_once()

    async def test_run_sleep_agent_once_does_not_advance_checkpoint_for_failed_stage(self):
        db = await _sleep_db()
        original_close = db.close
        try:
            await db.execute(
                """INSERT INTO sleep_agent_checkpoint
                   (stage_name, checkpoint_created_at, last_run_id)
                   VALUES (?, ?, ?)""",
                ("topic_regroup", "2026-04-17 09:00:00", "run-old"),
            )
            await db.commit()

            db.close = AsyncMock()
            with patch("memory_core.services.sleep_agent.get_db", new=AsyncMock(return_value=db)), patch(
                "memory_core.services.sleep_agent._topic_regroup",
                new=AsyncMock(side_effect=Exception("boom")),
            ), patch(
                "memory_core.services.sleep_agent._persona_refresh",
                new=AsyncMock(return_value=None),
            ), patch(
                "memory_core.services.sleep_agent._generate_reflections",
                new=AsyncMock(return_value=None),
            ):
                returned = await run_sleep_agent_once()

            checkpoint_row = await (
                await db.execute(
                    """SELECT checkpoint_created_at, last_run_id
                       FROM sleep_agent_checkpoint
                       WHERE stage_name = ?""",
                    ("topic_regroup",),
                )
            ).fetchone()
        finally:
            await original_close()

        assert returned.error_count == 1
        assert checkpoint_row["checkpoint_created_at"] == "2026-04-17 09:00:00"
        assert checkpoint_row["last_run_id"] == "run-old"
        db.close.assert_awaited_once()

    async def test_topic_regroup_uses_persisted_checkpoint_to_reduce_rescan(self):
        db = await _sleep_db()
        try:
            await _insert_memory(
                db,
                "before-checkpoint",
                importance=8.0,
                created_at_sql="'2026-04-17 09:00:00'",
            )
            await _insert_memory(
                db,
                "after-checkpoint",
                importance=8.0,
                created_at_sql="'2026-04-17 11:00:00'",
            )
            await db.execute(
                """INSERT INTO sleep_agent_checkpoint
                   (stage_name, checkpoint_created_at, last_run_id)
                   VALUES (?, ?, ?)""",
                ("topic_regroup", "2026-04-17 10:00:00", "run-old"),
            )
            await db.commit()

            result = SleepAgentResult()
            with patch(
                "memory_core.services.sleep_agent._cluster_topics_with_llm",
                new=AsyncMock(return_value=[]),
            ):
                topics, checkpoint_candidate = await _topic_regroup(db, result)
        finally:
            await db.close()

        assert topics == []
        assert result.topic_memory_count == 1
        assert result.topic_count == 0
        assert checkpoint_candidate == "2026-04-17 11:00:00"

    async def test_run_sleep_agent_periodically_can_be_cancelled(self):
        with patch(
            "memory_core.services.sleep_agent.run_sleep_agent_once",
            new=AsyncMock(return_value=SleepAgentResult()),
        ) as mock_run:
            task = asyncio.create_task(
                run_sleep_agent_periodically(
                    interval_seconds=3600,
                    startup_delay_seconds=0,
                )
            )
            await asyncio.sleep(0)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        mock_run.assert_awaited()
