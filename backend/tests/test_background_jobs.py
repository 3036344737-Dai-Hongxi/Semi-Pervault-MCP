import asyncio
from collections import Counter
import json
import sqlite3
import time
from pathlib import Path
import sys
from unittest.mock import AsyncMock, patch

import aiosqlite
from fastapi.testclient import TestClient
import pytest

import memory_core.database as database
from memory_core.database import (
    ensure_auth_sessions_schema,
    ensure_background_jobs_schema,
    ensure_scheduler_run_log_schema,
    ensure_sleep_agent_checkpoint_schema,
)
from memory_core.services import background_jobs
from memory_core.services.memory_admission import AdmissionScore
from services.rate_limit import limiter


async def _jobs_db():
    db = await aiosqlite.connect(":memory:")
    db.row_factory = aiosqlite.Row
    await ensure_background_jobs_schema(db)
    await ensure_scheduler_run_log_schema(db)
    await ensure_sleep_agent_checkpoint_schema(db)
    await ensure_auth_sessions_schema(db)
    await db.commit()
    return db


def _wait_for(predicate, *, timeout: float = 3.0, interval: float = 0.05):
    deadline = time.time() + timeout
    while time.time() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(interval)
    return None


def _fetch_job_type_counts(db_path: Path) -> Counter:
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute("SELECT job_type FROM background_jobs").fetchall()
        return Counter(row[0] for row in rows)
    finally:
        conn.close()


def _fetch_job_status_counts(db_path: Path) -> Counter:
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute("SELECT status FROM background_jobs").fetchall()
        return Counter(row[0] for row in rows)
    finally:
        conn.close()


def _fetch_memory_row(db_path: Path, memory_id: str) -> sqlite3.Row | None:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        return conn.execute(
            """SELECT content, kind, task_status, admission_tier, importance, emotion_score, content_version
               FROM memory_items
               WHERE id = ?""",
            (memory_id,),
        ).fetchone()
    finally:
        conn.close()


def _fetch_chat_message_count(db_path: Path, session_id: str) -> int:
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT COUNT(*) FROM chat_messages WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        return int(row[0] or 0)
    finally:
        conn.close()


def _fetch_latest_chat_memory_row(db_path: Path) -> sqlite3.Row | None:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        return conn.execute(
            """SELECT id, content, kind, admission_tier
               FROM memory_items
               WHERE content IS NOT NULL
               ORDER BY created_at DESC, rowid DESC
               LIMIT 1"""
        ).fetchone()
    finally:
        conn.close()


def _drain_registered_jobs(max_iterations: int = 20):
    async def _run():
        db = await database.get_db()
        try:
            for _ in range(max_iterations):
                handled = await background_jobs.run_worker_iteration(
                    db,
                    worker_id="test-drain-worker",
                )
                if not handled:
                    break
        finally:
            await db.close()

    asyncio.run(_run())


def _reset_shared_db():
    asyncio.run(database.close_shared_db())


def _fetch_jobs(db_path: Path, *, status: str | None = None, job_type: str | None = None):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        where_clauses: list[str] = []
        params: list[str] = []
        if status:
            where_clauses.append("status = ?")
            params.append(status)
        if job_type:
            where_clauses.append("job_type = ?")
            params.append(job_type)
        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
        return conn.execute(
            f"""SELECT id, job_type, status, origin, origin_run_id, payload_json, attempt_count, terminal_reason,
                       last_error, created_at, updated_at, available_at, finished_at
                FROM background_jobs
                {where_sql}
                ORDER BY updated_at DESC, created_at DESC""",
            tuple(params),
        ).fetchall()
    finally:
        conn.close()


def _insert_job_row(
    db_path: Path,
    *,
    job_id: str,
    job_type: str,
    status: str,
    origin: str = "pipeline",
    origin_run_id: str | None = None,
    terminal_reason: str | None = None,
    last_error: str | None = None,
):
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """INSERT INTO background_jobs
               (id, job_type, status, origin, origin_run_id, payload_json, dedupe_key, attempt_count, max_attempts,
                available_at, created_at, updated_at, finished_at, terminal_reason, last_error)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'), datetime('now'),
                       CASE WHEN ? IN ('completed', 'failed', 'dead') THEN datetime('now') ELSE NULL END,
                       ?, ?)""",
            (
                job_id,
                job_type,
                status,
                origin,
                origin_run_id,
                "{}",
                f"dedupe:{job_id}",
                1,
                3,
                status,
                terminal_reason,
                last_error,
            ),
        )
        conn.commit()
    finally:
        conn.close()


class TestBackgroundJobRuntime:
    async def test_enqueue_claim_complete_and_dedupe(self):
        db = await _jobs_db()
        try:
            job1, created1 = await background_jobs.enqueue_job(
                db,
                job_type="embedding_index",
                payload={"memory_id": "mem-1", "content_hash": "abc"},
                dedupe_scope="memory:mem-1",
                subject_ref="memory:mem-1",
            )
            job2, created2 = await background_jobs.enqueue_job(
                db,
                job_type="embedding_index",
                payload={"memory_id": "mem-1", "content_hash": "abc"},
                dedupe_scope="memory:mem-1",
                subject_ref="memory:mem-1",
            )
            claimed = await background_jobs.claim_next_job(
                db,
                worker_id="worker-a",
                job_types=("embedding_index",),
                lease_seconds=60,
            )
            completed = await background_jobs.mark_job_completed(
                db,
                job_id=claimed["id"],
                lease_token=claimed["lease_token"],
            )
        finally:
            await db.close()

        assert created1 is True
        assert created2 is False
        assert job1["id"] == job2["id"]
        assert claimed is not None
        assert claimed["status"] == background_jobs.JOB_STATUS_RUNNING
        assert claimed["attempt_count"] == 1
        assert completed is True

    async def test_mark_failed_then_retry(self):
        db = await _jobs_db()
        try:
            job, _ = await background_jobs.enqueue_job(
                db,
                job_type="importance_score",
                payload={"memory_id": "mem-2", "content_hash": "def"},
                dedupe_scope="memory:mem-2",
                subject_ref="memory:mem-2",
            )
            claimed = await background_jobs.claim_next_job(
                db,
                worker_id="worker-a",
                job_types=("importance_score",),
            )
            status = await background_jobs.mark_job_failed(
                db,
                job_id=claimed["id"],
                lease_token=claimed["lease_token"],
                error="temporary failure",
            )
            retried = await background_jobs.retry_job(db, job_id=job["id"])
            cursor = await db.execute(
                "SELECT status, attempt_count, last_error FROM background_jobs WHERE id = ?",
                (job["id"],),
            )
            row = await cursor.fetchone()
        finally:
            await db.close()

        assert status == background_jobs.JOB_STATUS_FAILED
        assert retried is True
        assert row["status"] == background_jobs.JOB_STATUS_PENDING
        assert row["attempt_count"] == 1
        assert row["last_error"] is None

    async def test_stale_running_reclaim_returns_job_to_pending(self):
        db = await _jobs_db()
        try:
            job, _ = await background_jobs.enqueue_job(
                db,
                job_type="graph_extract",
                payload={"memory_id": "mem-3", "content_hash": "ghi"},
                dedupe_scope="memory:mem-3",
                subject_ref="memory:mem-3",
            )
            claimed = await background_jobs.claim_next_job(
                db,
                worker_id="worker-a",
                job_types=("graph_extract",),
                lease_seconds=1,
            )
            await db.execute(
                """UPDATE background_jobs
                   SET lease_expires_at = datetime('now', '-1 minute')
                   WHERE id = ?""",
                (job["id"],),
            )
            await db.commit()

            reclaimed = await background_jobs.reclaim_stale_running_jobs(db)
            reclaimed_job = await background_jobs.claim_next_job(
                db,
                worker_id="worker-b",
                job_types=("graph_extract",),
            )
        finally:
            await db.close()

        assert claimed is not None
        assert reclaimed == 1
        assert reclaimed_job is not None
        assert reclaimed_job["id"] == job["id"]
        assert reclaimed_job["lease_token"] != claimed["lease_token"]

    async def test_stale_lease_guard_rejects_old_completion_token(self):
        db = await _jobs_db()
        try:
            job, _ = await background_jobs.enqueue_job(
                db,
                job_type="admission_score",
                payload={"memory_id": "mem-4", "content_hash": "jkl"},
                dedupe_scope="memory:mem-4",
                subject_ref="memory:mem-4",
            )
            first_claim = await background_jobs.claim_next_job(
                db,
                worker_id="worker-a",
                job_types=("admission_score",),
                lease_seconds=1,
            )
            await db.execute(
                """UPDATE background_jobs
                   SET lease_expires_at = datetime('now', '-1 minute')
                   WHERE id = ?""",
                (job["id"],),
            )
            await db.commit()
            await background_jobs.reclaim_stale_running_jobs(db)
            second_claim = await background_jobs.claim_next_job(
                db,
                worker_id="worker-b",
                job_types=("admission_score",),
            )
            old_completion = await background_jobs.mark_job_completed(
                db,
                job_id=job["id"],
                lease_token=first_claim["lease_token"],
            )
            new_completion = await background_jobs.mark_job_completed(
                db,
                job_id=job["id"],
                lease_token=second_claim["lease_token"],
            )
        finally:
            await db.close()

        assert old_completion is False
        assert new_completion is True

    async def test_scheduler_run_log_helpers(self):
        db = await _jobs_db()
        try:
            run_id = await background_jobs.create_scheduler_run_log(
                db,
                scheduler_name="sleep_agent",
            )
            finished = await background_jobs.finish_scheduler_run_log(
                db,
                run_id=run_id,
                status="completed",
                summary={"reflections_created": 1},
                error_count=0,
            )
            row = await (
                await db.execute(
                    "SELECT scheduler_name, status, summary_json, error_count FROM scheduler_run_log WHERE id = ?",
                    (run_id,),
                )
            ).fetchone()
        finally:
            await db.close()

        assert finished is True
        assert row["scheduler_name"] == "sleep_agent"
        assert row["status"] == "completed"
        assert row["error_count"] == 0
        assert row["summary_json"] == '{"reflections_created":1}'

    async def test_memory_ai_health_summary_reports_stage_evidence(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("GEMINI_API_KEY", "gm-test")
        monkeypatch.setenv("SLEEP_AGENT_ENABLED", "1")

        db = await _jobs_db()
        try:
            await db.executemany(
                """INSERT INTO background_jobs
                   (id, job_type, status, origin, origin_run_id, payload_json, dedupe_key, attempt_count, max_attempts,
                    available_at, created_at, updated_at, finished_at, terminal_reason, last_error)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    (
                        "job-kind-ok",
                        "kind_correction",
                        "completed",
                        "pipeline",
                        None,
                        "{}",
                        "dedupe-kind",
                        1,
                        3,
                        "2026-04-17 10:00:00",
                        "2026-04-17 10:00:00",
                        "2026-04-17 10:01:00",
                        "2026-04-17 10:01:00",
                        "completed",
                        None,
                    ),
                    (
                        "job-admission-fail",
                        "admission_score",
                        "failed",
                        "pipeline",
                        None,
                        "{}",
                        "dedupe-admission",
                        2,
                        3,
                        "2026-04-17 10:00:00",
                        "2026-04-17 10:00:00",
                        "2026-04-17 10:02:00",
                        "2026-04-17 10:02:00",
                        "failed",
                        "provider auth failed",
                    ),
                    (
                        "job-embedding-running",
                        "embedding_index",
                        "running",
                        "pipeline",
                        None,
                        "{}",
                        "dedupe-embedding",
                        1,
                        3,
                        "2026-04-17 10:03:00",
                        "2026-04-17 10:03:00",
                        "2026-04-17 10:04:00",
                        None,
                        None,
                        None,
                    ),
                ],
            )
            await db.execute(
                """INSERT INTO scheduler_run_log
                   (id, scheduler_name, status, started_at, finished_at, summary_json, error_count)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    "sleep-run-1",
                    "sleep_agent",
                    "completed",
                    "2026-04-17 11:00:00",
                    "2026-04-17 11:02:00",
                    json.dumps(
                        {
                            "topic_memory_count": 4,
                            "topic_count": 2,
                            "persona_memory_count": 3,
                            "persona_traits_upserted": 1,
                            "reflection_memory_count": 2,
                            "reflections_created": 1,
                            "errors": ["persona:mem-1:TimeoutError"],
                            "checkpoint_updates": {
                                "topic_regroup": "2026-04-17 10:59:00",
                                "reflection_generation": "2026-04-17 11:01:00",
                            },
                        },
                        ensure_ascii=False,
                    ),
                    1,
                ),
            )
            await db.executemany(
                """INSERT INTO sleep_agent_checkpoint
                   (stage_name, checkpoint_created_at, last_run_id, updated_at)
                   VALUES (?, ?, ?, ?)""",
                [
                    (
                        "topic_regroup",
                        "2026-04-17 10:59:00",
                        "sleep-run-1",
                        "2026-04-17 11:02:00",
                    ),
                    (
                        "reflection_generation",
                        "2026-04-17 11:01:00",
                        "sleep-run-1",
                        "2026-04-17 11:02:00",
                    ),
                ],
            )
            await db.commit()

            summary = await background_jobs.get_memory_ai_health_summary(db)
        finally:
            await db.close()

        assert summary["openai_configured"] is True
        assert summary["embedding_configured"] is True
        assert summary["sleep_agent_enabled"] is True
        assert summary["sleep_agent_last_run_status"] == "completed"
        assert summary["sleep_agent_last_error_count"] == 1

        stages = {stage["stage_id"]: stage for stage in summary["stages"]}

        assert stages["kind_correction"]["health"] == "healthy"
        assert stages["kind_correction"]["counts"]["succeeded"] == 1
        assert stages["kind_correction"]["latest_status"] == "completed"

        assert stages["admission_score"]["health"] == "failing"
        assert stages["admission_score"]["counts"]["failed"] == 1
        assert stages["admission_score"]["last_error"] == "provider auth failed"

        assert stages["embedding_index"]["health"] == "running"
        assert stages["embedding_index"]["counts"]["running"] == 1

        assert stages["topic_regroup"]["health"] == "healthy"
        assert stages["topic_regroup"]["recent_memory_count"] == 4
        assert stages["topic_regroup"]["recent_output_count"] == 2
        assert stages["topic_regroup"]["checkpoint_created_at"] == "2026-04-17 10:59:00"

        assert stages["persona_refresh"]["health"] == "failing"
        assert "persona:mem-1:TimeoutError" in (stages["persona_refresh"]["last_error"] or "")
        assert stages["persona_refresh"]["recent_memory_count"] == 3
        assert stages["persona_refresh"]["recent_output_count"] == 1

        assert stages["reflection_generation"]["health"] == "healthy"
        assert stages["reflection_generation"]["recent_output_count"] == 1

    async def test_obsolete_job_is_completed_without_failure_retry_semantics(self):
        db = await _jobs_db()
        background_jobs.clear_job_handlers()

        async def _obsolete(_job, _db):
            raise background_jobs.ObsoleteJobError("payload no longer applies")

        background_jobs.register_job_handler("kind_correction", _obsolete)
        try:
            job, _ = await background_jobs.enqueue_job(
                db,
                job_type="kind_correction",
                payload={"memory_id": "mem-5", "content_hash": "xyz"},
                dedupe_scope="memory:mem-5",
                subject_ref="memory:mem-5",
            )
            handled = await background_jobs.run_worker_iteration(
                db,
                worker_id="worker-a",
            )
            row = await (
                await db.execute(
                    "SELECT status, terminal_reason, last_error FROM background_jobs WHERE id = ?",
                    (job["id"],),
                )
            ).fetchone()
        finally:
            background_jobs.clear_job_handlers()
            await db.close()

        assert handled is True
        assert row["status"] == background_jobs.JOB_STATUS_COMPLETED
        assert row["terminal_reason"] == background_jobs.TERMINAL_REASON_OBSOLETE
        assert row["last_error"] is None


class TestSystemSummaryIntegration:
    def test_jobs_summary_requires_auth_and_worker_health_is_exposed(self, monkeypatch, tmp_path: Path):
        db_path = tmp_path / "queue-foundation.db"
        monkeypatch.setenv("AUTH_PASSWORD", "secret-pass")
        monkeypatch.setenv("SESSION_SECRET", "x" * 32)
        monkeypatch.setenv("SECURE_COOKIES", "0")
        monkeypatch.setenv("CONSOLIDATION_SCHEDULER_ENABLED", "0")
        monkeypatch.setenv("WEIGHT_DECAY_SCHEDULER_ENABLED", "0")
        monkeypatch.setenv("SLEEP_AGENT_ENABLED", "0")
        monkeypatch.setenv("BACKGROUND_WORKER_POLL_SECONDS", "0.05")
        database.DB_PATH = db_path
        _reset_shared_db()
        background_jobs.reset_worker_health()
        background_jobs.clear_job_handlers()
        limiter._storage.reset()
        enqueue_spy = None

        sys.modules.pop("main", None)
        import main  # noqa: WPS433

        try:
            with TestClient(main.app) as client:
                unauth = client.get("/api/system/jobs/summary")
                assert unauth.status_code == 401

                login = client.post("/api/auth/login", json={"password": "secret-pass"})
                assert login.status_code == 200

                time.sleep(0.15)
                summary = client.get("/api/system/jobs/summary")
                assert summary.status_code == 200
                payload = summary.json()
                assert payload["jobs"]["total"] == 0
                assert payload["jobs"]["by_origin"] == {}
                assert payload["jobs"]["by_origin_status"] == {}
                assert payload["worker"]["running"] is True
                assert payload["worker"]["last_heartbeat_at"]
        finally:
            _reset_shared_db()
            limiter._storage.reset()

    def test_jobs_summary_includes_origin_aggregation(self, monkeypatch, tmp_path: Path):
        db_path = tmp_path / "queue-summary-origin.db"
        monkeypatch.setenv("AUTH_PASSWORD", "secret-pass")
        monkeypatch.setenv("SESSION_SECRET", "x" * 32)
        monkeypatch.setenv("SECURE_COOKIES", "0")
        monkeypatch.setenv("CONSOLIDATION_SCHEDULER_ENABLED", "0")
        monkeypatch.setenv("WEIGHT_DECAY_SCHEDULER_ENABLED", "0")
        monkeypatch.setenv("SLEEP_AGENT_ENABLED", "0")
        monkeypatch.setenv("BACKGROUND_WORKER_POLL_SECONDS", "0.05")
        database.DB_PATH = db_path
        _reset_shared_db()
        background_jobs.reset_worker_health()
        background_jobs.clear_job_handlers()
        limiter._storage.reset()

        sys.modules.pop("main", None)
        import main  # noqa: WPS433

        try:
            with TestClient(main.app) as client:
                login = client.post("/api/auth/login", json={"password": "secret-pass"})
                assert login.status_code == 200

                _insert_job_row(
                    db_path,
                    job_id="job-pipeline-completed",
                    job_type="kind_correction",
                    status="completed",
                    origin="pipeline",
                    terminal_reason="completed",
                )
                _insert_job_row(
                    db_path,
                    job_id="job-pipeline-failed",
                    job_type="embedding_index",
                    status="failed",
                    origin="pipeline",
                    terminal_reason="failed",
                    last_error="boom",
                )
                _insert_job_row(
                    db_path,
                    job_id="job-reprocess-obsolete",
                    job_type="kind_correction",
                    status="completed",
                    origin="manual_reprocess",
                    origin_run_id="run-123",
                    terminal_reason="obsolete",
                )

                summary = client.get("/api/system/jobs/summary")
                assert summary.status_code == 200
                payload = summary.json()["jobs"]
                assert payload["total"] == 3
                assert payload["by_origin"] == {
                    "manual_reprocess": 1,
                    "pipeline": 2,
                }
                assert payload["by_origin_status"] == {
                    "manual_reprocess": {"completed": 1},
                    "pipeline": {"completed": 1, "failed": 1},
                }
        finally:
            _reset_shared_db()
            limiter._storage.reset()

    def test_jobs_list_filters_and_retry_endpoint(self, monkeypatch, tmp_path: Path):
        db_path = tmp_path / "queue-system-jobs.db"
        monkeypatch.setenv("AUTH_PASSWORD", "secret-pass")
        monkeypatch.setenv("SESSION_SECRET", "x" * 32)
        monkeypatch.setenv("SECURE_COOKIES", "0")
        monkeypatch.setenv("CONSOLIDATION_SCHEDULER_ENABLED", "0")
        monkeypatch.setenv("WEIGHT_DECAY_SCHEDULER_ENABLED", "0")
        monkeypatch.setenv("SLEEP_AGENT_ENABLED", "0")
        monkeypatch.setenv("BACKGROUND_WORKER_POLL_SECONDS", "0.05")
        database.DB_PATH = db_path
        _reset_shared_db()
        background_jobs.reset_worker_health()
        background_jobs.clear_job_handlers()
        limiter._storage.reset()

        sys.modules.pop("main", None)
        import main  # noqa: WPS433

        try:
            with TestClient(main.app) as client:
                unauth = client.get("/api/system/jobs")
                assert unauth.status_code == 401

                login = client.post("/api/auth/login", json={"password": "secret-pass"})
                assert login.status_code == 200

                _insert_job_row(
                    db_path,
                    job_id="job-failed",
                    job_type="kind_correction",
                    status="failed",
                    terminal_reason="failed",
                    last_error="boom",
                )
                _insert_job_row(
                    db_path,
                    job_id="job-obsolete",
                    job_type="embedding_index",
                    status="completed",
                    terminal_reason="obsolete",
                )
                _insert_job_row(
                    db_path,
                    job_id="job-dead",
                    job_type="graph_extract",
                    status="dead",
                    terminal_reason="dead",
                    last_error="max attempts",
                )

                failed_only = client.get("/api/system/jobs", params={"status": "failed", "limit": 10})
                assert failed_only.status_code == 200
                failed_payload = failed_only.json()
                assert len(failed_payload["jobs"]) == 1
                assert failed_payload["jobs"][0]["id"] == "job-failed"
                assert failed_payload["jobs"][0]["terminal_reason"] == "failed"

                embedding_only = client.get("/api/system/jobs", params={"job_type": "embedding_index"})
                assert embedding_only.status_code == 200
                embedding_payload = embedding_only.json()
                assert len(embedding_payload["jobs"]) == 1
                assert embedding_payload["jobs"][0]["id"] == "job-obsolete"
                assert embedding_payload["jobs"][0]["terminal_reason"] == "obsolete"

                completed_retry = client.post("/api/system/jobs/job-obsolete/retry")
                assert completed_retry.status_code == 400

                failed_retry = client.post("/api/system/jobs/job-failed/retry")
                assert failed_retry.status_code == 200
                assert failed_retry.json()["ok"] is True
                assert failed_retry.json()["job"]["status"] == "pending"
                assert failed_retry.json()["job"]["terminal_reason"] is None

                dead_retry = client.post("/api/system/jobs/job-dead/retry")
                assert dead_retry.status_code == 200
                assert dead_retry.json()["job"]["status"] == "pending"
        finally:
            _reset_shared_db()
            limiter._storage.reset()


class TestMemoryStoreQueuePipeline:
    def test_memory_store_uses_queue_pipeline_and_respects_stage_dependencies(
        self,
        monkeypatch,
        tmp_path: Path,
    ):
        db_path = tmp_path / "memory-store-queue.db"
        events: list[tuple[str, str]] = []

        monkeypatch.setenv("AUTH_PASSWORD", "secret-pass")
        monkeypatch.setenv("SESSION_SECRET", "x" * 32)
        monkeypatch.setenv("SECURE_COOKIES", "0")
        monkeypatch.setenv("CONSOLIDATION_SCHEDULER_ENABLED", "0")
        monkeypatch.setenv("WEIGHT_DECAY_SCHEDULER_ENABLED", "0")
        monkeypatch.setenv("SLEEP_AGENT_ENABLED", "0")
        monkeypatch.setenv("BACKGROUND_WORKER_POLL_SECONDS", "0.02")
        database.DB_PATH = db_path
        _reset_shared_db()
        background_jobs.reset_worker_health()
        background_jobs.clear_job_handlers()
        limiter._storage.reset()

        sys.modules.pop("main", None)
        import main  # noqa: WPS433

        async def fake_graph(memory_id: str, content: str, kind: str | None = None):
            events.append(("graph", kind or ""))

        async def fake_embedding(memory_id: str, content: str):
            events.append(("embedding", memory_id))

        async def fake_emotion(memory_id: str, content: str):
            events.append(("emotion", memory_id))

        async def fake_importance(memory_id: str, content: str):
            events.append(("importance", memory_id))

        async def fake_persona(memory_id: str, content: str, kind: str):
            events.append(("persona", kind))

        async def fake_admission(content: str, kind: str, db, *, exclude_memory_id: str | None = None):
            events.append(("admission", kind))
            return AdmissionScore(
                utility=0.9,
                confidence=0.8,
                novelty=1.0,
                recency=1.0,
                type_prior=0.95,
                total=0.9,
                tier="standard",
            )

        try:
            enqueue_spy = None
            with patch(
                "memory_core.services.memory_service.classify_memory_kind",
                return_value="other",
            ), patch(
                "memory_core.services.memory_service.classify_memory_kind_with_llm",
                new=AsyncMock(return_value="project_update"),
            ), patch(
                "memory_core.services.memory_service.compute_admission_score",
                new=AsyncMock(side_effect=fake_admission),
            ), patch(
                "memory_core.services.memory_service._extract_graph_in_background",
                new=AsyncMock(side_effect=fake_graph),
            ), patch(
                "memory_core.services.memory_service._index_memory_embedding",
                new=AsyncMock(side_effect=fake_embedding),
            ), patch(
                "memory_core.services.memory_service._update_emotion_score_in_background",
                new=AsyncMock(side_effect=fake_emotion),
            ), patch(
                "memory_core.services.memory_service._update_importance_in_background",
                new=AsyncMock(side_effect=fake_importance),
            ), patch(
                "memory_core.services.memory_service._extract_persona_in_background",
                new=AsyncMock(side_effect=fake_persona),
            ):
                with TestClient(main.app) as client:
                    login = client.post("/api/auth/login", json={"password": "secret-pass"})
                    assert login.status_code == 200

                    response = client.post(
                        "/api/memory/store",
                        json={"content": "请记住：我最近在推进 Pervault 项目"},
                    )
                    assert response.status_code == 200
                    memory_id = response.json()["id"]
                    assert response.json()["kind"] == "other"

                    counts = _wait_for(
                        lambda: _fetch_job_status_counts(db_path),
                        timeout=3.0,
                    )
                    assert counts is not None

                    completed = _wait_for(
                        lambda: _fetch_job_status_counts(db_path).get("completed", 0) == 7,
                        timeout=5.0,
                    )
                    assert completed is True
        finally:
            _reset_shared_db()
            limiter._storage.reset()

        job_types = _fetch_job_type_counts(db_path)
        assert job_types == Counter(
            {
                "kind_correction": 1,
                "embedding_index": 1,
                "emotion_score": 1,
                "importance_score": 1,
                "admission_score": 1,
                "graph_extract": 1,
                "persona_extract": 1,
            }
        )
        row = _fetch_memory_row(db_path, memory_id)
        assert row is not None
        assert row["kind"] == "project_update"
        assert row["admission_tier"] == "standard"
        assert row["content_version"] == 1
        jobs = _fetch_jobs(db_path)
        kind_job = next(job for job in jobs if job["job_type"] == "kind_correction")
        payload = json.loads(kind_job["payload_json"])
        assert payload["subject_version"] == 1
        assert ("admission", "project_update") in events
        assert ("graph", "project_update") in events
        assert ("persona", "project_update") in events
        assert [name for name, _ in events].index("admission") < [name for name, _ in events].index("persona")

    def test_memory_store_does_not_enqueue_persona_for_low_value_admission(
        self,
        monkeypatch,
        tmp_path: Path,
    ):
        db_path = tmp_path / "memory-store-low-value.db"
        persona_calls: list[str] = []

        monkeypatch.setenv("AUTH_PASSWORD", "secret-pass")
        monkeypatch.setenv("SESSION_SECRET", "x" * 32)
        monkeypatch.setenv("SECURE_COOKIES", "0")
        monkeypatch.setenv("CONSOLIDATION_SCHEDULER_ENABLED", "0")
        monkeypatch.setenv("WEIGHT_DECAY_SCHEDULER_ENABLED", "0")
        monkeypatch.setenv("SLEEP_AGENT_ENABLED", "0")
        monkeypatch.setenv("BACKGROUND_WORKER_POLL_SECONDS", "0.02")
        database.DB_PATH = db_path
        _reset_shared_db()
        background_jobs.reset_worker_health()
        background_jobs.clear_job_handlers()
        limiter._storage.reset()

        sys.modules.pop("main", None)
        import main  # noqa: WPS433

        async def fake_persona(memory_id: str, content: str, kind: str):
            persona_calls.append(memory_id)

        async def fake_admission(content: str, kind: str, db, *, exclude_memory_id: str | None = None):
            return AdmissionScore(
                utility=0.1,
                confidence=0.1,
                novelty=1.0,
                recency=1.0,
                type_prior=0.35,
                total=0.32,
                tier="low_value",
            )

        try:
            with patch(
                "memory_core.services.memory_service.classify_memory_kind",
                return_value="other",
            ), patch(
                "memory_core.services.memory_service.classify_memory_kind_with_llm",
                new=AsyncMock(return_value="project_update"),
            ), patch(
                "memory_core.services.memory_service.compute_admission_score",
                new=AsyncMock(side_effect=fake_admission),
            ), patch(
                "memory_core.services.memory_service._extract_graph_in_background",
                new=AsyncMock(return_value=None),
            ), patch(
                "memory_core.services.memory_service._index_memory_embedding",
                new=AsyncMock(return_value=None),
            ), patch(
                "memory_core.services.memory_service._update_emotion_score_in_background",
                new=AsyncMock(return_value=None),
            ), patch(
                "memory_core.services.memory_service._update_importance_in_background",
                new=AsyncMock(return_value=None),
            ), patch(
                "memory_core.services.memory_service._extract_persona_in_background",
                new=AsyncMock(side_effect=fake_persona),
            ):
                with TestClient(main.app) as client:
                    login = client.post("/api/auth/login", json={"password": "secret-pass"})
                    assert login.status_code == 200

                    response = client.post(
                        "/api/memory/store",
                        json={"content": "请记住：我最近在推进 Pervault 项目"},
                    )
                    assert response.status_code == 200
                    memory_id = response.json()["id"]

                    completed = _wait_for(
                        lambda: _fetch_job_status_counts(db_path).get("completed", 0) == 6,
                        timeout=3.0,
                    )
                    assert completed is True
        finally:
            _reset_shared_db()
            limiter._storage.reset()

        job_types = _fetch_job_type_counts(db_path)
        assert job_types["persona_extract"] == 0
        row = _fetch_memory_row(db_path, memory_id)
        assert row is not None
        assert row["kind"] == "project_update"
        assert row["admission_tier"] == "low_value"
        assert persona_calls == []

    def test_subject_version_change_marks_old_job_obsolete(self, monkeypatch, tmp_path: Path):
        db_path = tmp_path / "memory-version-obsolete.db"

        monkeypatch.setenv("AUTH_PASSWORD", "secret-pass")
        monkeypatch.setenv("SESSION_SECRET", "x" * 32)
        monkeypatch.setenv("SECURE_COOKIES", "0")
        monkeypatch.setenv("CONSOLIDATION_SCHEDULER_ENABLED", "0")
        monkeypatch.setenv("WEIGHT_DECAY_SCHEDULER_ENABLED", "0")
        monkeypatch.setenv("SLEEP_AGENT_ENABLED", "0")
        monkeypatch.setenv("BACKGROUND_WORKER_POLL_SECONDS", "0.02")
        database.DB_PATH = db_path
        _reset_shared_db()
        background_jobs.reset_worker_health()
        background_jobs.clear_job_handlers()
        limiter._storage.reset()

        sys.modules.pop("main", None)
        import main  # noqa: WPS433

        try:
            with patch(
                "memory_core.services.memory_service.classify_memory_kind",
                return_value="other",
            ), patch(
                "memory_core.services.memory_service.classify_memory_kind_with_llm",
                new=AsyncMock(return_value="project_update"),
            ), patch(
                "memory_core.services.memory_service.compute_admission_score",
                new=AsyncMock(
                    return_value=AdmissionScore(
                        utility=0.9,
                        confidence=0.8,
                        novelty=1.0,
                        recency=1.0,
                        type_prior=0.95,
                        total=0.9,
                        tier="standard",
                    )
                ),
            ), patch(
                "memory_core.services.memory_service._extract_graph_in_background",
                new=AsyncMock(return_value=None),
            ), patch(
                "memory_core.services.memory_service._index_memory_embedding",
                new=AsyncMock(return_value=None),
            ), patch(
                "memory_core.services.memory_service._update_emotion_score_in_background",
                new=AsyncMock(return_value=None),
            ), patch(
                "memory_core.services.memory_service._update_importance_in_background",
                new=AsyncMock(return_value=None),
            ), patch(
                "memory_core.services.memory_service._extract_persona_in_background",
                new=AsyncMock(return_value=None),
            ):
                with TestClient(main.app) as client:
                    login = client.post("/api/auth/login", json={"password": "secret-pass"})
                    assert login.status_code == 200

                    response = client.post(
                        "/api/memory/store",
                        json={"content": "请记住：我最近在推进 Pervault 项目"},
                    )
                    assert response.status_code == 200
                    memory_id = response.json()["id"]

                    completed = _wait_for(
                        lambda: _fetch_job_status_counts(db_path).get("completed", 0) == 7,
                        timeout=5.0,
                    )
                    assert completed is True

            async def _seed_stale_job():
                db = await database.get_db()
                try:
                    job, _ = await background_jobs.enqueue_job(
                        db,
                        job_type="kind_correction",
                        payload={
                            "memory_id": memory_id,
                            "content_hash": "irrelevant-old-hash",
                            "kind_snapshot": "project_update",
                            "subject_version": 1,
                            "pipeline_version": "memory_pipeline_v1",
                        },
                        dedupe_scope=f"memory:{memory_id}:replay",
                        dedupe_version="memory_pipeline_v1_test",
                        subject_ref=f"memory:{memory_id}",
                        subject_version="1",
                    )
                    return job["id"]
                finally:
                    await db.close()

            async def _run_one_iteration():
                db = await database.get_db()
                try:
                    await background_jobs.run_worker_iteration(
                        db,
                        worker_id="test-worker",
                    )
                finally:
                    await db.close()

            stale_job_id = asyncio.run(_seed_stale_job())
            conn = sqlite3.connect(db_path)
            try:
                conn.execute(
                    """UPDATE memory_items
                       SET content = ?, content_version = 2
                       WHERE id = ?""",
                    ("内容已经更新了", memory_id),
                )
                conn.commit()
            finally:
                conn.close()

            asyncio.run(_run_one_iteration())

            stale_terminal = _wait_for(
                lambda: next(
                    (
                        row["terminal_reason"]
                        for row in _fetch_jobs(db_path)
                        if row["id"] == stale_job_id and row["status"] == "completed"
                    ),
                    None,
                ),
                timeout=3.0,
            )
        finally:
            _reset_shared_db()
            limiter._storage.reset()

        assert stale_terminal == "obsolete"
        stale_job = next(row for row in _fetch_jobs(db_path) if row["id"] == stale_job_id)
        assert stale_job["status"] == "completed"
        assert stale_job["terminal_reason"] == "obsolete"
        assert stale_job["last_error"] is None


class TestMemoryUpdateQueuePipeline:
    def test_memory_update_increments_version_and_reuses_shared_queue_pipeline(
        self,
        monkeypatch,
        tmp_path: Path,
    ):
        db_path = tmp_path / "memory-update-queue.db"
        events: list[tuple[str, str]] = []

        monkeypatch.setenv("AUTH_PASSWORD", "secret-pass")
        monkeypatch.setenv("SESSION_SECRET", "x" * 32)
        monkeypatch.setenv("SECURE_COOKIES", "0")
        monkeypatch.setenv("CONSOLIDATION_SCHEDULER_ENABLED", "0")
        monkeypatch.setenv("WEIGHT_DECAY_SCHEDULER_ENABLED", "0")
        monkeypatch.setenv("SLEEP_AGENT_ENABLED", "0")
        monkeypatch.setenv("BACKGROUND_WORKER_POLL_SECONDS", "0.02")
        database.DB_PATH = db_path
        _reset_shared_db()
        background_jobs.reset_worker_health()
        background_jobs.clear_job_handlers()
        limiter._storage.reset()

        sys.modules.pop("main", None)
        import main  # noqa: WPS433

        def fake_rule_kind(content: str) -> str:
            if "推进" in content or "项目" in content:
                return "project_update"
            if "喜欢" in content:
                return "preference"
            return "other"

        async def fake_llm_kind(content: str) -> str:
            return fake_rule_kind(content)

        async def fake_graph(memory_id: str, content: str, kind: str | None = None):
            events.append(("graph", f"{kind}:{content}"))

        async def fake_embedding(memory_id: str, content: str):
            events.append(("embedding", content))

        async def fake_emotion(memory_id: str, content: str):
            events.append(("emotion", content))

        async def fake_importance(memory_id: str, content: str):
            events.append(("importance", content))

        async def fake_persona(memory_id: str, content: str, kind: str):
            events.append(("persona", f"{kind}:{content}"))

        async def fake_admission(content: str, kind: str, db, *, exclude_memory_id: str | None = None):
            events.append(("admission", f"{kind}:{content}"))
            return AdmissionScore(
                utility=0.9,
                confidence=0.8,
                novelty=1.0,
                recency=1.0,
                type_prior=0.95 if kind == "project_update" else 0.9,
                total=0.9,
                tier="standard",
            )

        try:
            with patch(
                "memory_core.services.memory_service.classify_memory_kind",
                side_effect=fake_rule_kind,
            ), patch(
                "memory_core.services.memory_service.classify_memory_kind_with_llm",
                new=AsyncMock(side_effect=fake_llm_kind),
            ), patch(
                "memory_core.services.memory_service.compute_admission_score",
                new=AsyncMock(side_effect=fake_admission),
            ), patch(
                "memory_core.services.memory_service._extract_graph_in_background",
                new=AsyncMock(side_effect=fake_graph),
            ), patch(
                "memory_core.services.memory_service._index_memory_embedding",
                new=AsyncMock(side_effect=fake_embedding),
            ), patch(
                "memory_core.services.memory_service._update_emotion_score_in_background",
                new=AsyncMock(side_effect=fake_emotion),
            ), patch(
                "memory_core.services.memory_service._update_importance_in_background",
                new=AsyncMock(side_effect=fake_importance),
            ), patch(
                "memory_core.services.memory_service._extract_persona_in_background",
                new=AsyncMock(side_effect=fake_persona),
            ):
                with TestClient(main.app) as client:
                    login = client.post("/api/auth/login", json={"password": "secret-pass"})
                    assert login.status_code == 200

                    created = client.post(
                        "/api/memory/store",
                        json={"content": "请记住：我喜欢吃辣"},
                    )
                    assert created.status_code == 200
                    memory_id = created.json()["id"]

                    first_completed = _wait_for(
                        lambda: _fetch_job_status_counts(db_path).get("completed", 0) == 7,
                        timeout=5.0,
                    )
                    assert first_completed is True

                    async def _seed_old_version_job():
                        db = await database.get_db()
                        try:
                            job, _ = await background_jobs.enqueue_job(
                                db,
                                job_type="kind_correction",
                                payload={
                                    "memory_id": memory_id,
                                    "content_hash": "old-hash",
                                    "kind_snapshot": "preference",
                                    "subject_version": 1,
                                    "pipeline_version": "memory_pipeline_v1",
                                },
                                dedupe_scope=f"memory:{memory_id}:update-replay",
                                dedupe_version="memory_pipeline_v1_update_test",
                                subject_ref=f"memory:{memory_id}",
                                subject_version="1",
                            )
                            return job["id"]
                        finally:
                            await db.close()

                    stale_job_id = asyncio.run(_seed_old_version_job())

                    enqueue_spy = AsyncMock(wraps=main.memory.enqueue_memory_store_jobs)
                    with patch(
                        "routers.memory.enqueue_memory_store_jobs",
                        new=enqueue_spy,
                    ):
                        updated = client.patch(
                            f"/api/memory/{memory_id}",
                            json={"content": "我最近在推进 Pervault 项目"},
                        )

                    assert updated.status_code == 200
                    updated_payload = updated.json()
                    assert updated_payload["id"] == memory_id
                    assert updated_payload["content"] == "我最近在推进 Pervault 项目"
                    assert updated_payload["kind"] == "project_update"
                    assert "content_version" not in updated_payload
                    enqueue_spy.assert_awaited_once()

                    fully_completed = _wait_for(
                        lambda: _fetch_job_status_counts(db_path).get("completed", 0) >= 15,
                        timeout=5.0,
                    )
                    assert fully_completed is True
        finally:
            _reset_shared_db()
            limiter._storage.reset()


class TestMemoryReprocessEndpoint:
    def test_reprocess_endpoint_enqueues_current_version_jobs(
        self,
        monkeypatch,
        tmp_path: Path,
    ):
        db_path = tmp_path / "memory-reprocess.db"
        events: list[tuple[str, str]] = []

        monkeypatch.setenv("AUTH_PASSWORD", "secret-pass")
        monkeypatch.setenv("SESSION_SECRET", "x" * 32)
        monkeypatch.setenv("SECURE_COOKIES", "0")
        monkeypatch.setenv("CONSOLIDATION_SCHEDULER_ENABLED", "0")
        monkeypatch.setenv("WEIGHT_DECAY_SCHEDULER_ENABLED", "0")
        monkeypatch.setenv("SLEEP_AGENT_ENABLED", "0")
        monkeypatch.setenv("BACKGROUND_WORKER_POLL_SECONDS", "0.02")
        database.DB_PATH = db_path
        _reset_shared_db()
        background_jobs.reset_worker_health()
        background_jobs.clear_job_handlers()
        limiter._storage.reset()

        sys.modules.pop("main", None)
        import main  # noqa: WPS433

        async def fake_graph(memory_id: str, content: str, kind: str | None = None):
            events.append(("graph", f"{kind}:{content}"))

        async def fake_embedding(memory_id: str, content: str):
            events.append(("embedding", content))

        async def fake_emotion(memory_id: str, content: str):
            events.append(("emotion", content))

        async def fake_importance(memory_id: str, content: str):
            events.append(("importance", content))

        async def fake_persona(memory_id: str, content: str, kind: str):
            events.append(("persona", f"{kind}:{content}"))

        async def fake_admission(content: str, kind: str, db, *, exclude_memory_id: str | None = None):
            events.append(("admission", f"{kind}:{content}"))
            return AdmissionScore(
                utility=0.9,
                confidence=0.8,
                novelty=1.0,
                recency=1.0,
                type_prior=0.95,
                total=0.9,
                tier="standard",
            )

        try:
            with patch(
                "memory_core.services.memory_service.classify_memory_kind",
                return_value="project_update",
            ), patch(
                "memory_core.services.memory_service.classify_memory_kind_with_llm",
                new=AsyncMock(return_value="project_update"),
            ), patch(
                "memory_core.services.memory_service.compute_admission_score",
                new=AsyncMock(side_effect=fake_admission),
            ), patch(
                "memory_core.services.memory_service._extract_graph_in_background",
                new=AsyncMock(side_effect=fake_graph),
            ), patch(
                "memory_core.services.memory_service._index_memory_embedding",
                new=AsyncMock(side_effect=fake_embedding),
            ), patch(
                "memory_core.services.memory_service._update_emotion_score_in_background",
                new=AsyncMock(side_effect=fake_emotion),
            ), patch(
                "memory_core.services.memory_service._update_importance_in_background",
                new=AsyncMock(side_effect=fake_importance),
            ), patch(
                "memory_core.services.memory_service._extract_persona_in_background",
                new=AsyncMock(side_effect=fake_persona),
            ):
                with TestClient(main.app) as client:
                    login = client.post("/api/auth/login", json={"password": "secret-pass"})
                    assert login.status_code == 200

                    created = client.post(
                        "/api/memory/store",
                        json={"content": "请记住：我最近在推进 Pervault 项目"},
                    )
                    assert created.status_code == 200
                    memory_id = created.json()["id"]

                    first_completed = _wait_for(
                        lambda: _fetch_job_status_counts(db_path).get("completed", 0) == 7,
                        timeout=5.0,
                    )
                    assert first_completed is True

                    reprocess = client.post(f"/api/memory/{memory_id}/reprocess")
                    assert reprocess.status_code == 200
                    payload = reprocess.json()
                    assert payload["memory_id"] == memory_id
                    assert payload["content_version"] == 1
                    assert payload["origin"] == "manual_reprocess"
                    assert payload["origin_run_id"]
                    origin_run_id = payload["origin_run_id"]
                    assert [job["job_type"] for job in payload["jobs"]] == [
                        "kind_correction",
                        "embedding_index",
                        "emotion_score",
                        "importance_score",
                    ]
                    assert all(job["reused_existing"] is False for job in payload["jobs"])

                    second_completed = _wait_for(
                        lambda: _fetch_job_status_counts(db_path).get("completed", 0) >= 14,
                        timeout=5.0,
                    )
                    assert second_completed is True

                    jobs_list = client.get("/api/system/jobs", params={"job_type": "kind_correction", "limit": 20})
                    assert jobs_list.status_code == 200
                    jobs_payload = jobs_list.json()["jobs"]
                    assert any(
                        job["origin"] == "manual_reprocess"
                        and job["origin_run_id"] == origin_run_id
                        for job in jobs_payload
                    )
        finally:
            _reset_shared_db()
            limiter._storage.reset()

        jobs = _fetch_jobs(db_path)
        reprocess_kind_jobs = [
            json.loads(job["payload_json"])
            for job in jobs
            if job["job_type"] == "kind_correction"
        ]
        assert any(payload.get("subject_version") == 1 for payload in reprocess_kind_jobs)
        assert any(str(payload.get("run_token", "")).startswith("manual_reprocess:") for payload in reprocess_kind_jobs)
        assert any(
            job["origin"] == "manual_reprocess" and job["origin_run_id"] == origin_run_id
            for job in jobs
            if job["job_type"] == "kind_correction"
        )
        assert any(
            str(json.loads(job["payload_json"]).get("run_token", "")).startswith("manual_reprocess:")
            for job in jobs
            if job["job_type"] == "persona_extract"
        )
        assert ("admission", "project_update:请记住：我最近在推进 Pervault 项目") in events

    def test_reprocess_endpoint_returns_404_for_missing_memory(
        self,
        monkeypatch,
        tmp_path: Path,
    ):
        db_path = tmp_path / "memory-reprocess-missing.db"

        monkeypatch.setenv("AUTH_PASSWORD", "secret-pass")
        monkeypatch.setenv("SESSION_SECRET", "x" * 32)
        monkeypatch.setenv("SECURE_COOKIES", "0")
        monkeypatch.setenv("CONSOLIDATION_SCHEDULER_ENABLED", "0")
        monkeypatch.setenv("WEIGHT_DECAY_SCHEDULER_ENABLED", "0")
        monkeypatch.setenv("SLEEP_AGENT_ENABLED", "0")
        monkeypatch.setenv("BACKGROUND_WORKER_POLL_SECONDS", "0.02")
        database.DB_PATH = db_path
        _reset_shared_db()
        background_jobs.reset_worker_health()
        background_jobs.clear_job_handlers()
        limiter._storage.reset()

        sys.modules.pop("main", None)
        import main  # noqa: WPS433

        try:
            with TestClient(main.app) as client:
                login = client.post("/api/auth/login", json={"password": "secret-pass"})
                assert login.status_code == 200

                response = client.post("/api/memory/non-existent-memory/reprocess")
                assert response.status_code == 404
                assert response.json()["detail"] == "记忆不存在"
        finally:
            _reset_shared_db()
            limiter._storage.reset()

    def test_reprocess_jobs_become_obsolete_after_later_update(
        self,
        monkeypatch,
        tmp_path: Path,
    ):
        db_path = tmp_path / "memory-reprocess-obsolete.db"

        monkeypatch.setenv("AUTH_PASSWORD", "secret-pass")
        monkeypatch.setenv("SESSION_SECRET", "x" * 32)
        monkeypatch.setenv("SECURE_COOKIES", "0")
        monkeypatch.setenv("CONSOLIDATION_SCHEDULER_ENABLED", "0")
        monkeypatch.setenv("WEIGHT_DECAY_SCHEDULER_ENABLED", "0")
        monkeypatch.setenv("SLEEP_AGENT_ENABLED", "0")
        monkeypatch.setenv("BACKGROUND_WORKER_POLL_SECONDS", "0.02")
        database.DB_PATH = db_path
        _reset_shared_db()
        background_jobs.reset_worker_health()
        background_jobs.clear_job_handlers()
        limiter._storage.reset()

        async def _disabled_worker():
            return None

        sys.modules.pop("main", None)
        with patch("memory_core.services.background_jobs.run_background_jobs_worker", new=_disabled_worker):
            import main  # noqa: WPS433

        try:
            with patch(
                "memory_core.services.memory_service.classify_memory_kind",
                side_effect=lambda content: "project_update" if "推进" in content or "项目" in content else "preference",
            ), patch(
                "memory_core.services.memory_service.classify_memory_kind_with_llm",
                new=AsyncMock(side_effect=lambda content: "project_update" if "推进" in content or "项目" in content else "preference"),
            ), patch(
                "memory_core.services.memory_service.compute_admission_score",
                new=AsyncMock(
                    return_value=AdmissionScore(
                        utility=0.9,
                        confidence=0.8,
                        novelty=1.0,
                        recency=1.0,
                        type_prior=0.95,
                        total=0.9,
                        tier="standard",
                    )
                ),
            ), patch(
                "memory_core.services.memory_service._extract_graph_in_background",
                new=AsyncMock(return_value=None),
            ), patch(
                "memory_core.services.memory_service._index_memory_embedding",
                new=AsyncMock(return_value=None),
            ), patch(
                "memory_core.services.memory_service._update_emotion_score_in_background",
                new=AsyncMock(return_value=None),
            ), patch(
                "memory_core.services.memory_service._update_importance_in_background",
                new=AsyncMock(return_value=None),
            ), patch(
                "memory_core.services.memory_service._extract_persona_in_background",
                new=AsyncMock(return_value=None),
            ):
                with TestClient(main.app) as client:
                    login = client.post("/api/auth/login", json={"password": "secret-pass"})
                    assert login.status_code == 200

                    created = client.post(
                        "/api/memory/store",
                        json={"content": "请记住：我喜欢吃辣"},
                    )
                    assert created.status_code == 200
                    memory_id = created.json()["id"]

                    _drain_registered_jobs()

                    reprocess = client.post(f"/api/memory/{memory_id}/reprocess")
                    assert reprocess.status_code == 200
                    reprocess_payload = reprocess.json()
                    assert reprocess_payload["content_version"] == 1
                    reprocess_origin_run_id = reprocess_payload["origin_run_id"]

                    updated = client.patch(
                        f"/api/memory/{memory_id}",
                        json={"content": "我最近在推进 Pervault 项目"},
                    )
                    assert updated.status_code == 200

                    _drain_registered_jobs(max_iterations=40)
        finally:
            _reset_shared_db()
            limiter._storage.reset()

        row = _fetch_memory_row(db_path, memory_id)
        assert row is not None
        assert row["content"] == "我最近在推进 Pervault 项目"
        assert row["kind"] == "project_update"
        assert row["content_version"] == 2

        jobs = _fetch_jobs(db_path)
        obsolete_reprocess_jobs = [
            job
            for job in jobs
            if job["status"] == "completed"
            and job["terminal_reason"] == "obsolete"
            and job["origin"] == "manual_reprocess"
            and job["origin_run_id"] == reprocess_origin_run_id
        ]
        assert obsolete_reprocess_jobs
        assert all(json.loads(job["payload_json"]).get("subject_version") == 1 for job in obsolete_reprocess_jobs)
        assert any(
            job["status"] == "completed"
            and job["terminal_reason"] == "completed"
            and json.loads(job["payload_json"]).get("subject_version") == 2
            for job in jobs
        )

    def test_memory_update_returns_404_for_missing_memory(
        self,
        monkeypatch,
        tmp_path: Path,
    ):
        db_path = tmp_path / "memory-update-missing.db"

        monkeypatch.setenv("AUTH_PASSWORD", "secret-pass")
        monkeypatch.setenv("SESSION_SECRET", "x" * 32)
        monkeypatch.setenv("SECURE_COOKIES", "0")
        monkeypatch.setenv("CONSOLIDATION_SCHEDULER_ENABLED", "0")
        monkeypatch.setenv("WEIGHT_DECAY_SCHEDULER_ENABLED", "0")
        monkeypatch.setenv("SLEEP_AGENT_ENABLED", "0")
        monkeypatch.setenv("BACKGROUND_WORKER_POLL_SECONDS", "0.02")
        database.DB_PATH = db_path
        _reset_shared_db()
        background_jobs.reset_worker_health()
        background_jobs.clear_job_handlers()
        limiter._storage.reset()

        sys.modules.pop("main", None)
        import main  # noqa: WPS433

        try:
            with TestClient(main.app) as client:
                login = client.post("/api/auth/login", json={"password": "secret-pass"})
                assert login.status_code == 200

                response = client.patch(
                    "/api/memory/non-existent-memory",
                    json={"content": "新的内容"},
                )
                assert response.status_code == 404
                assert response.json()["detail"] == "记忆不存在"
        finally:
            _reset_shared_db()
            limiter._storage.reset()
