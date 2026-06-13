import json
import sys
import types
from unittest.mock import AsyncMock, patch

import aiosqlite
from fastapi import HTTPException


async def _pipeline_trace_db():
    db = await aiosqlite.connect(":memory:")
    db.row_factory = aiosqlite.Row
    await db.executescript(
        """
        CREATE TABLE memory_items (
            id TEXT PRIMARY KEY,
            content TEXT,
            content_version INTEGER NOT NULL DEFAULT 1,
            created_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE background_jobs (
            id TEXT PRIMARY KEY,
            job_type TEXT NOT NULL,
            status TEXT NOT NULL,
            origin TEXT NOT NULL DEFAULT 'pipeline',
            origin_run_id TEXT,
            payload_json TEXT NOT NULL,
            dedupe_key TEXT NOT NULL,
            attempt_count INTEGER NOT NULL DEFAULT 0,
            max_attempts INTEGER NOT NULL DEFAULT 3,
            available_at TEXT,
            started_at TEXT,
            finished_at TEXT,
            last_error TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now')),
            lease_expires_at TEXT,
            heartbeat_at TEXT,
            lease_token TEXT,
            terminal_reason TEXT
        );
        """
    )
    await db.commit()
    return db


def _job_payload(memory_id: str, subject_version: int) -> str:
    return json.dumps(
        {
            "memory_id": memory_id,
            "subject_version": subject_version,
            "content_hash": f"hash:{memory_id}:{subject_version}",
            "pipeline_version": "memory_pipeline_v1",
        },
        ensure_ascii=False,
    )


class TestMemoryPipelineTraceRoute:
    async def test_returns_current_version_jobs_and_hides_older_versions(self):
        sys.modules.setdefault(
            "main",
            types.SimpleNamespace(
                limiter=types.SimpleNamespace(limit=lambda _rule: (lambda func: func))
            ),
        )
        from routers.memory import get_memory_pipeline_trace

        db = await _pipeline_trace_db()
        try:
            await db.execute(
                """INSERT INTO memory_items (id, content, content_version)
                   VALUES (?, ?, ?)""",
                ("mem-1", "当前记忆内容", 2),
            )
            await db.executemany(
                """INSERT INTO background_jobs
                   (id, job_type, status, origin, origin_run_id, payload_json, dedupe_key,
                    attempt_count, created_at, updated_at, finished_at, terminal_reason, last_error)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    (
                        "job-current-1",
                        "kind_correction",
                        "completed",
                        "pipeline",
                        None,
                        _job_payload("mem-1", 2),
                        "dedupe-current-1",
                        1,
                        "2026-04-17 10:00:00",
                        "2026-04-17 10:00:05",
                        "2026-04-17 10:00:05",
                        "completed",
                        None,
                    ),
                    (
                        "job-current-2",
                        "graph_extract",
                        "failed",
                        "manual_reprocess",
                        "run-456",
                        _job_payload("mem-1", 2),
                        "dedupe-current-2",
                        2,
                        "2026-04-17 10:00:06",
                        "2026-04-17 10:01:00",
                        "2026-04-17 10:01:00",
                        "failed",
                        "graph timeout",
                    ),
                    (
                        "job-old-1",
                        "embedding_index",
                        "completed",
                        "pipeline",
                        None,
                        _job_payload("mem-1", 1),
                        "dedupe-old-1",
                        1,
                        "2026-04-17 09:00:00",
                        "2026-04-17 09:00:03",
                        "2026-04-17 09:00:03",
                        "obsolete",
                        None,
                    ),
                ],
            )
            await db.commit()

            with patch("routers.memory.get_db", new=AsyncMock(return_value=db)):
                response = await get_memory_pipeline_trace("mem-1")
        finally:
            await db.close()

        assert response.memory_id == "mem-1"
        assert response.content_version == 2
        assert response.hidden_job_count == 1
        assert [job.job_type for job in response.jobs] == [
            "kind_correction",
            "graph_extract",
        ]
        assert len(response.runs) == 2
        assert response.runs[0].origin == "manual_reprocess"
        assert response.runs[0].origin_run_id == "run-456"
        assert response.runs[0].subject_version == 2
        assert response.runs[0].job_count == 1
        assert response.runs[0].status_counts == {"failed": 1}
        assert response.runs[0].started_at == "2026-04-17 10:00:06"
        assert response.runs[0].updated_at == "2026-04-17 10:01:00"
        assert response.runs[0].finished_at == "2026-04-17 10:01:00"
        assert response.runs[0].is_current_version is True
        assert [job.job_type for job in response.runs[0].jobs] == ["graph_extract"]
        assert response.runs[1].origin == "pipeline"
        assert response.runs[1].origin_run_id is None
        assert response.runs[1].job_count == 1
        assert response.runs[1].status_counts == {"completed": 1}
        assert response.jobs[0].subject_version == 2
        assert response.jobs[1].origin == "manual_reprocess"
        assert response.jobs[1].origin_run_id == "run-456"
        assert response.jobs[1].attempt_count == 2
        assert response.jobs[1].terminal_reason == "failed"
        assert response.jobs[1].last_error == "graph timeout"

    async def test_returns_empty_jobs_when_memory_has_no_pipeline_rows(self):
        sys.modules.setdefault(
            "main",
            types.SimpleNamespace(
                limiter=types.SimpleNamespace(limit=lambda _rule: (lambda func: func))
            ),
        )
        from routers.memory import get_memory_pipeline_trace

        db = await _pipeline_trace_db()
        try:
            await db.execute(
                """INSERT INTO memory_items (id, content, content_version)
                   VALUES (?, ?, ?)""",
                ("mem-empty", "无作业记忆", 1),
            )
            await db.commit()

            with patch("routers.memory.get_db", new=AsyncMock(return_value=db)):
                response = await get_memory_pipeline_trace("mem-empty")
        finally:
            await db.close()

        assert response.memory_id == "mem-empty"
        assert response.content_version == 1
        assert response.hidden_job_count == 0
        assert response.runs == []
        assert response.jobs == []

    async def test_groups_same_origin_without_run_id_into_one_run_summary(self):
        sys.modules.setdefault(
            "main",
            types.SimpleNamespace(
                limiter=types.SimpleNamespace(limit=lambda _rule: (lambda func: func))
            ),
        )
        from routers.memory import get_memory_pipeline_trace

        db = await _pipeline_trace_db()
        try:
            await db.execute(
                """INSERT INTO memory_items (id, content, content_version)
                   VALUES (?, ?, ?)""",
                ("mem-grouped", "分组记忆", 3),
            )
            await db.executemany(
                """INSERT INTO background_jobs
                   (id, job_type, status, origin, origin_run_id, payload_json, dedupe_key,
                    attempt_count, created_at, updated_at, finished_at, terminal_reason, last_error)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    (
                        "job-grouped-1",
                        "kind_correction",
                        "completed",
                        "pipeline",
                        None,
                        _job_payload("mem-grouped", 3),
                        "dedupe-grouped-1",
                        1,
                        "2026-04-17 11:00:00",
                        "2026-04-17 11:00:10",
                        "2026-04-17 11:00:10",
                        "completed",
                        None,
                    ),
                    (
                        "job-grouped-2",
                        "embedding_index",
                        "running",
                        "pipeline",
                        None,
                        _job_payload("mem-grouped", 3),
                        "dedupe-grouped-2",
                        1,
                        "2026-04-17 11:00:05",
                        "2026-04-17 11:01:00",
                        None,
                        None,
                        None,
                    ),
                ],
            )
            await db.commit()

            with patch("routers.memory.get_db", new=AsyncMock(return_value=db)):
                response = await get_memory_pipeline_trace("mem-grouped")
        finally:
            await db.close()

        assert len(response.runs) == 1
        run = response.runs[0]
        assert run.origin == "pipeline"
        assert run.origin_run_id is None
        assert run.subject_version == 3
        assert run.job_count == 2
        assert run.status_counts == {"completed": 1, "running": 1}
        assert run.started_at == "2026-04-17 11:00:00"
        assert run.updated_at == "2026-04-17 11:01:00"
        assert run.finished_at is None
        assert [job.job_type for job in run.jobs] == [
            "kind_correction",
            "embedding_index",
        ]

    async def test_raises_404_when_memory_is_missing(self):
        sys.modules.setdefault(
            "main",
            types.SimpleNamespace(
                limiter=types.SimpleNamespace(limit=lambda _rule: (lambda func: func))
            ),
        )
        from routers.memory import get_memory_pipeline_trace

        db = await _pipeline_trace_db()
        try:
            with patch("routers.memory.get_shared_db", new=AsyncMock(return_value=db)):
                try:
                    await get_memory_pipeline_trace("missing-memory")
                except HTTPException as error:
                    response_error = error
                else:
                    raise AssertionError("expected HTTPException")
        finally:
            await db.close()

        assert response_error.status_code == 404
        assert response_error.detail == "记忆不存在"
