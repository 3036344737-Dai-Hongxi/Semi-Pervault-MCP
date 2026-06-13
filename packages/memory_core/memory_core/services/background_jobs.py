"""Persistent background job runtime.

M1 scope:
  - persistent queue schema helpers
  - enqueue / claim / complete / fail / retry primitives
  - lease + heartbeat + stale-reclaim foundation
  - minimal worker loop with dedicated DB connection
  - scheduler run ledger helpers

This module intentionally does NOT migrate existing memory/chat write paths yet.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
import hashlib
import json
import logging
import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from memory_core.database import get_db

logger = logging.getLogger("uvicorn.error")


class ObsoleteJobError(Exception):
    """Raised when a job is still structurally valid but no longer applies."""

JOB_STATUS_PENDING = "pending"
JOB_STATUS_RUNNING = "running"
JOB_STATUS_COMPLETED = "completed"
JOB_STATUS_FAILED = "failed"
JOB_STATUS_DEAD = "dead"

TERMINAL_REASON_COMPLETED = "completed"
TERMINAL_REASON_OBSOLETE = "obsolete"
TERMINAL_REASON_FAILED = "failed"
TERMINAL_REASON_DEAD = "dead"

ACTIVE_JOB_STATUSES = frozenset(
    {
        JOB_STATUS_PENDING,
        JOB_STATUS_RUNNING,
        JOB_STATUS_FAILED,
        JOB_STATUS_DEAD,
    }
)

DEFAULT_JOB_MAX_ATTEMPTS = 3
DEFAULT_JOB_LEASE_SECONDS = int(os.getenv("BACKGROUND_JOB_LEASE_SECONDS", "60"))
DEFAULT_WORKER_POLL_SECONDS = float(os.getenv("BACKGROUND_WORKER_POLL_SECONDS", "1.0"))

JobHandler = Callable[[dict[str, Any], Any], Awaitable[None]]

_JOB_HANDLERS: dict[str, JobHandler] = {}
_WORKER_HEALTH: dict[str, Any] = {
    "worker_id": None,
    "started_at": None,
    "last_heartbeat_at": None,
    "last_error": None,
    "registered_job_types": [],
}

MEMORY_AI_PIPELINE_STAGES: tuple[dict[str, str], ...] = (
    {
        "stage_id": "kind_correction",
        "label": "类型校正",
        "provider": "llm",
        "config_key": "OPENAI_API_KEY",
    },
    {
        "stage_id": "embedding_index",
        "label": "向量索引",
        "provider": "embedding",
        "config_key": "GEMINI_API_KEY",
    },
    {
        "stage_id": "emotion_score",
        "label": "情绪评分",
        "provider": "llm",
        "config_key": "OPENAI_API_KEY",
    },
    {
        "stage_id": "importance_score",
        "label": "重要度评分",
        "provider": "llm",
        "config_key": "OPENAI_API_KEY",
    },
    {
        "stage_id": "admission_score",
        "label": "准入评分",
        "provider": "llm",
        "config_key": "OPENAI_API_KEY",
    },
    {
        "stage_id": "graph_extract",
        "label": "图谱抽取",
        "provider": "llm",
        "config_key": "OPENAI_API_KEY",
    },
    {
        "stage_id": "persona_extract",
        "label": "画像提取",
        "provider": "llm",
        "config_key": "OPENAI_API_KEY",
    },
)

SLEEP_AGENT_AI_STAGES: tuple[dict[str, str], ...] = (
    {
        "stage_id": "topic_regroup",
        "label": "主题整理",
        "provider": "llm",
        "config_key": "OPENAI_API_KEY",
    },
    {
        "stage_id": "persona_refresh",
        "label": "画像刷新",
        "provider": "llm",
        "config_key": "OPENAI_API_KEY",
    },
    {
        "stage_id": "reflection_generation",
        "label": "洞察生成",
        "provider": "llm",
        "config_key": "OPENAI_API_KEY",
    },
)

_SLEEP_AGENT_STAGE_ERROR_PREFIXES: dict[str, tuple[str, ...]] = {
    "topic_regroup": ("topic_regroup",),
    "persona_refresh": ("persona_refresh", "persona:", "persona_db:"),
    "reflection_generation": ("reflection_generation", "reflection_db"),
}

_SLEEP_AGENT_STAGE_MEMORY_COUNT_KEYS: dict[str, str] = {
    "topic_regroup": "topic_memory_count",
    "persona_refresh": "persona_memory_count",
    "reflection_generation": "reflection_memory_count",
}

_SLEEP_AGENT_STAGE_OUTPUT_COUNT_KEYS: dict[str, str] = {
    "topic_regroup": "topic_count",
    "persona_refresh": "persona_traits_upserted",
    "reflection_generation": "reflections_created",
}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _sqlite_datetime(value: datetime | None = None) -> str:
    normalized = (value or _utc_now()).astimezone(timezone.utc).replace(microsecond=0)
    return normalized.strftime("%Y-%m-%d %H:%M:%S")


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _payload_fingerprint(payload: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def build_job_dedupe_key(
    job_type: str,
    payload: dict[str, Any],
    *,
    dedupe_scope: str | None = None,
    dedupe_version: str = "v1",
    subject_ref: str | None = None,
    subject_version: str | None = None,
) -> str:
    """Build a version-aware dedupe key.

    We do not have a first-class memory version column yet, so the M1 scheme uses:
      job_type + scope + subject_ref + subject_version + dedupe_version + payload fingerprint.

    Future M2/M3 can swap in a real subject version without changing queue semantics.
    """
    scope = dedupe_scope or "global"
    payload_hash = _payload_fingerprint(payload)
    subject_ref_part = subject_ref or "none"
    subject_version_part = subject_version or "none"
    raw = "|".join(
        [
            job_type,
            scope,
            subject_ref_part,
            subject_version_part,
            dedupe_version,
            payload_hash,
        ]
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def register_job_handler(job_type: str, handler: JobHandler) -> None:
    _JOB_HANDLERS[job_type] = handler
    _WORKER_HEALTH["registered_job_types"] = sorted(_JOB_HANDLERS.keys())


def clear_job_handlers() -> None:
    _JOB_HANDLERS.clear()
    _WORKER_HEALTH["registered_job_types"] = []


def reset_worker_health() -> None:
    _WORKER_HEALTH.update(
        {
            "worker_id": None,
            "started_at": None,
            "last_heartbeat_at": None,
            "last_error": None,
            "registered_job_types": sorted(_JOB_HANDLERS.keys()),
        }
    )


def _set_worker_heartbeat(worker_id: str) -> None:
    _WORKER_HEALTH["worker_id"] = worker_id
    _WORKER_HEALTH["started_at"] = _WORKER_HEALTH.get("started_at") or _sqlite_datetime()
    _WORKER_HEALTH["last_heartbeat_at"] = _sqlite_datetime()
    _WORKER_HEALTH["registered_job_types"] = sorted(_JOB_HANDLERS.keys())


def _set_worker_error(error: str) -> None:
    _WORKER_HEALTH["last_error"] = error
    _WORKER_HEALTH["last_heartbeat_at"] = _sqlite_datetime()


def get_worker_health_snapshot() -> dict[str, Any]:
    last_heartbeat_at = _WORKER_HEALTH.get("last_heartbeat_at")
    return {
        "worker_id": _WORKER_HEALTH.get("worker_id"),
        "started_at": _WORKER_HEALTH.get("started_at"),
        "last_heartbeat_at": last_heartbeat_at,
        "last_error": _WORKER_HEALTH.get("last_error"),
        "registered_job_types": list(_WORKER_HEALTH.get("registered_job_types") or []),
        "running": bool(last_heartbeat_at),
    }


def _env_configured(name: str) -> bool:
    return bool(os.getenv(name, "").strip())


def _stage_count_summary(
    status_counts: dict[str, int] | None = None,
    terminal_reason_counts: dict[str, int] | None = None,
) -> dict[str, int]:
    status_counts = status_counts or {}
    terminal_reason_counts = terminal_reason_counts or {}
    return {
        "total": sum(int(value or 0) for value in status_counts.values()),
        "pending": int(status_counts.get(JOB_STATUS_PENDING, 0) or 0),
        "running": int(status_counts.get(JOB_STATUS_RUNNING, 0) or 0),
        "completed": int(status_counts.get(JOB_STATUS_COMPLETED, 0) or 0),
        "failed": int(status_counts.get(JOB_STATUS_FAILED, 0) or 0),
        "dead": int(status_counts.get(JOB_STATUS_DEAD, 0) or 0),
        "obsolete": int(terminal_reason_counts.get(TERMINAL_REASON_OBSOLETE, 0) or 0),
        "succeeded": int(terminal_reason_counts.get(TERMINAL_REASON_COMPLETED, 0) or 0),
    }


def _pipeline_stage_health(*, configured: bool, counts: dict[str, int]) -> str:
    if not configured:
        return "blocked"
    if counts["running"] > 0 or counts["pending"] > 0:
        return "degraded" if counts["failed"] > 0 or counts["dead"] > 0 else "running"
    if counts["succeeded"] > 0:
        return "degraded" if counts["failed"] > 0 or counts["dead"] > 0 else "healthy"
    if counts["failed"] > 0 or counts["dead"] > 0:
        return "failing"
    return "idle"


def _sleep_stage_health(
    *,
    enabled: bool,
    configured: bool,
    latest_run_status: str | None,
    checkpoint_created_at: str | None,
    stage_errors: list[str],
    recent_memory_count: int | None,
) -> str:
    if not enabled:
        return "disabled"
    if not configured:
        return "blocked"
    if latest_run_status == "running":
        return "running"
    if latest_run_status and latest_run_status != "completed":
        return "failing"
    if stage_errors:
        return "degraded" if checkpoint_created_at else "failing"
    if checkpoint_created_at or (recent_memory_count is not None and recent_memory_count > 0):
        return "healthy"
    return "idle"


def _row_to_job(row) -> dict[str, Any]:
    payload_raw = row["payload_json"] or "{}"
    try:
        payload = json.loads(payload_raw)
    except (json.JSONDecodeError, TypeError):
        payload = {}
    return {
        "id": row["id"],
        "job_type": row["job_type"],
        "status": row["status"],
        "origin": row["origin"],
        "origin_run_id": row["origin_run_id"],
        "payload": payload,
        "payload_json": payload_raw,
        "dedupe_key": row["dedupe_key"],
        "attempt_count": int(row["attempt_count"] or 0),
        "max_attempts": int(row["max_attempts"] or DEFAULT_JOB_MAX_ATTEMPTS),
        "available_at": row["available_at"],
        "started_at": row["started_at"],
        "finished_at": row["finished_at"],
        "last_error": row["last_error"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "lease_expires_at": row["lease_expires_at"],
        "heartbeat_at": row["heartbeat_at"],
        "lease_token": row["lease_token"],
        "terminal_reason": row["terminal_reason"],
    }


async def enqueue_job(
    db,
    *,
    job_type: str,
    payload: dict[str, Any],
    origin: str = "pipeline",
    origin_run_id: str | None = None,
    dedupe_scope: str | None = None,
    dedupe_version: str = "v1",
    subject_ref: str | None = None,
    subject_version: str | None = None,
    max_attempts: int = DEFAULT_JOB_MAX_ATTEMPTS,
    available_at: datetime | None = None,
) -> tuple[dict[str, Any], bool]:
    dedupe_key = build_job_dedupe_key(
        job_type,
        payload,
        dedupe_scope=dedupe_scope,
        dedupe_version=dedupe_version,
        subject_ref=subject_ref,
        subject_version=subject_version,
    )

    cursor = await db.execute(
        """SELECT *
           FROM background_jobs
           WHERE dedupe_key = ?
           ORDER BY created_at DESC
           LIMIT 1""",
        (dedupe_key,),
    )
    existing = await cursor.fetchone()
    if existing is not None:
        return _row_to_job(existing), False

    now = _sqlite_datetime()
    available_at_value = _sqlite_datetime(available_at) if available_at else now
    job_id = str(uuid.uuid4())
    payload_json = _canonical_json(payload)
    await db.execute(
        """INSERT INTO background_jobs
           (id, job_type, status, origin, origin_run_id, payload_json, dedupe_key, attempt_count, max_attempts,
            available_at, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            job_id,
            job_type,
            JOB_STATUS_PENDING,
            origin,
            origin_run_id,
            payload_json,
            dedupe_key,
            0,
            max(1, int(max_attempts)),
            available_at_value,
            now,
            now,
        ),
    )
    await db.commit()

    cursor = await db.execute(
        "SELECT * FROM background_jobs WHERE id = ?",
        (job_id,),
    )
    row = await cursor.fetchone()
    if row is None:
        raise RuntimeError(f"enqueued job disappeared job_id={job_id}")
    return _row_to_job(row), True


async def claim_next_job(
    db,
    *,
    worker_id: str,
    job_types: tuple[str, ...] | None = None,
    lease_seconds: int = DEFAULT_JOB_LEASE_SECONDS,
) -> dict[str, Any] | None:
    now = _sqlite_datetime()
    lease_expires_at = _sqlite_datetime(
        _utc_now() + timedelta(seconds=max(lease_seconds, 1))
    )
    lease_token = str(uuid.uuid4())
    params: list[Any] = [JOB_STATUS_PENDING, now]
    job_type_clause = ""
    if job_types:
        placeholders = ",".join("?" for _ in job_types)
        job_type_clause = f" AND job_type IN ({placeholders})"
        params.extend(job_types)

    try:
        await db.execute("BEGIN IMMEDIATE")
        cursor = await db.execute(
            f"""SELECT *
                FROM background_jobs
                WHERE status = ?
                  AND available_at <= ?
                  {job_type_clause}
                ORDER BY available_at ASC, created_at ASC
                LIMIT 1""",
            tuple(params),
        )
        row = await cursor.fetchone()
        if row is None:
            await db.commit()
            return None

        await db.execute(
            """UPDATE background_jobs
               SET status = ?,
                   attempt_count = attempt_count + 1,
                   started_at = ?,
                   finished_at = NULL,
                   updated_at = ?,
                   lease_expires_at = ?,
                   heartbeat_at = ?,
                   lease_token = ?,
                   last_error = NULL,
                   terminal_reason = NULL
               WHERE id = ?
                 AND status = ?""",
            (
                JOB_STATUS_RUNNING,
                now,
                now,
                lease_expires_at,
                now,
                lease_token,
                row["id"],
                JOB_STATUS_PENDING,
            ),
        )
        await db.commit()
    except Exception:
        await db.rollback()
        raise

    cursor = await db.execute(
        "SELECT * FROM background_jobs WHERE id = ?",
        (row["id"],),
    )
    claimed = await cursor.fetchone()
    if claimed is None:
        return None
    logger.info(
        "background job claimed job_id=%s job_type=%s worker_id=%s",
        row["id"],
        row["job_type"],
        worker_id,
    )
    return _row_to_job(claimed)


async def mark_job_heartbeat(
    db,
    *,
    job_id: str,
    lease_token: str,
    lease_seconds: int = DEFAULT_JOB_LEASE_SECONDS,
) -> bool:
    now = _sqlite_datetime()
    lease_expires_at = _sqlite_datetime(
        _utc_now() + timedelta(seconds=max(lease_seconds, 1))
    )
    cursor = await db.execute(
        """UPDATE background_jobs
           SET heartbeat_at = ?,
               lease_expires_at = ?,
               updated_at = ?
           WHERE id = ?
             AND status = ?
             AND lease_token = ?""",
        (
            now,
            lease_expires_at,
            now,
            job_id,
            JOB_STATUS_RUNNING,
            lease_token,
        ),
    )
    await db.commit()
    return bool(cursor.rowcount and cursor.rowcount > 0)


async def mark_job_completed(
    db,
    *,
    job_id: str,
    lease_token: str,
    terminal_reason: str = TERMINAL_REASON_COMPLETED,
) -> bool:
    now = _sqlite_datetime()
    cursor = await db.execute(
        """UPDATE background_jobs
           SET status = ?,
               finished_at = ?,
               updated_at = ?,
               lease_expires_at = NULL,
               heartbeat_at = ?,
               lease_token = NULL,
               terminal_reason = ?
           WHERE id = ?
             AND status = ?
             AND lease_token = ?""",
        (
            JOB_STATUS_COMPLETED,
            now,
            now,
            now,
            terminal_reason,
            job_id,
            JOB_STATUS_RUNNING,
            lease_token,
        ),
    )
    await db.commit()
    return bool(cursor.rowcount and cursor.rowcount > 0)


async def mark_job_failed(
    db,
    *,
    job_id: str,
    lease_token: str,
    error: str,
    force_dead: bool = False,
) -> str | None:
    truncated_error = error.strip()[:2000] if error else "unknown error"
    now = _sqlite_datetime()
    try:
        await db.execute("BEGIN IMMEDIATE")
        cursor = await db.execute(
            """SELECT attempt_count, max_attempts
               FROM background_jobs
               WHERE id = ?
                 AND status = ?
                 AND lease_token = ?""",
            (job_id, JOB_STATUS_RUNNING, lease_token),
        )
        row = await cursor.fetchone()
        if row is None:
            await db.commit()
            return None

        next_status = (
            JOB_STATUS_DEAD
            if force_dead or int(row["attempt_count"] or 0) >= int(row["max_attempts"] or 1)
            else JOB_STATUS_FAILED
        )
        await db.execute(
            """UPDATE background_jobs
               SET status = ?,
                   finished_at = ?,
                   updated_at = ?,
                   last_error = ?,
                   lease_expires_at = NULL,
                   heartbeat_at = ?,
                   lease_token = NULL,
                   terminal_reason = ?
               WHERE id = ?
                 AND status = ?
                 AND lease_token = ?""",
            (
                next_status,
                now,
                now,
                truncated_error,
                now,
                TERMINAL_REASON_DEAD if next_status == JOB_STATUS_DEAD else TERMINAL_REASON_FAILED,
                job_id,
                JOB_STATUS_RUNNING,
                lease_token,
            ),
        )
        await db.commit()
        return next_status
    except Exception:
        await db.rollback()
        raise


async def retry_job(
    db,
    *,
    job_id: str,
    reset_attempt_count: bool = False,
    available_at: datetime | None = None,
) -> bool:
    now = _sqlite_datetime()
    available_at_value = _sqlite_datetime(available_at) if available_at else now
    attempt_count_sql = "0" if reset_attempt_count else "attempt_count"
    try:
        await db.execute("BEGIN IMMEDIATE")
        cursor = await db.execute(
            """UPDATE background_jobs
               SET status = ?,
                   available_at = ?,
                   started_at = NULL,
                   finished_at = NULL,
                   updated_at = ?,
                   last_error = NULL,
                   lease_expires_at = NULL,
                   heartbeat_at = NULL,
                   lease_token = NULL,
                   terminal_reason = NULL,
                   attempt_count = """
            + attempt_count_sql
            + """
               WHERE id = ?
                 AND status IN (?, ?)""",
            (
                JOB_STATUS_PENDING,
                available_at_value,
                now,
                job_id,
                JOB_STATUS_FAILED,
                JOB_STATUS_DEAD,
            ),
        )
        await db.commit()
        return bool(cursor.rowcount and cursor.rowcount > 0)
    except Exception:
        await db.rollback()
        raise


async def reclaim_stale_running_jobs(db) -> int:
    now = _sqlite_datetime()
    reclaimed = 0
    try:
        await db.execute("BEGIN IMMEDIATE")
        cursor = await db.execute(
            """SELECT id, attempt_count, max_attempts
               FROM background_jobs
               WHERE status = ?
                 AND lease_expires_at IS NOT NULL
                 AND lease_expires_at < ?""",
            (JOB_STATUS_RUNNING, now),
        )
        rows = await cursor.fetchall()
        for row in rows:
            attempt_count = int(row["attempt_count"] or 0)
            max_attempts = int(row["max_attempts"] or 1)
            next_status = (
                JOB_STATUS_DEAD if attempt_count >= max_attempts else JOB_STATUS_PENDING
            )
            finished_at = now if next_status == JOB_STATUS_DEAD else None
            await db.execute(
                """UPDATE background_jobs
                   SET status = ?,
                       available_at = ?,
                       finished_at = ?,
                       updated_at = ?,
                       last_error = ?,
                       lease_expires_at = NULL,
                       heartbeat_at = NULL,
                       lease_token = NULL,
                       terminal_reason = ?
                   WHERE id = ?
                     AND status = ?""",
                (
                    next_status,
                    now,
                    finished_at,
                    now,
                    "stale running job reclaimed after lease expiry",
                    TERMINAL_REASON_DEAD if next_status == JOB_STATUS_DEAD else None,
                    row["id"],
                    JOB_STATUS_RUNNING,
                ),
            )
            reclaimed += 1
        await db.commit()
        return reclaimed
    except Exception:
        await db.rollback()
        raise


async def get_jobs_summary(db) -> dict[str, Any]:
    cursor = await db.execute(
        """SELECT status, COUNT(*) AS cnt
           FROM background_jobs
           GROUP BY status"""
    )
    status_rows = await cursor.fetchall()
    by_status = {row["status"]: int(row["cnt"] or 0) for row in status_rows}

    cursor = await db.execute(
        """SELECT job_type, status, COUNT(*) AS cnt
           FROM background_jobs
           GROUP BY job_type, status
           ORDER BY job_type ASC, status ASC"""
    )
    rows = await cursor.fetchall()
    by_job_type: dict[str, dict[str, int]] = {}
    for row in rows:
        by_job_type.setdefault(row["job_type"], {})[row["status"]] = int(row["cnt"] or 0)

    cursor = await db.execute(
        """SELECT origin, COUNT(*) AS cnt
           FROM background_jobs
           GROUP BY origin
           ORDER BY origin ASC"""
    )
    origin_rows = await cursor.fetchall()
    by_origin = {row["origin"]: int(row["cnt"] or 0) for row in origin_rows}

    cursor = await db.execute(
        """SELECT origin, status, COUNT(*) AS cnt
           FROM background_jobs
           GROUP BY origin, status
           ORDER BY origin ASC, status ASC"""
    )
    origin_status_rows = await cursor.fetchall()
    by_origin_status: dict[str, dict[str, int]] = {}
    for row in origin_status_rows:
        by_origin_status.setdefault(row["origin"], {})[row["status"]] = int(row["cnt"] or 0)

    cursor = await db.execute("SELECT COUNT(*) AS cnt FROM background_jobs")
    total_row = await cursor.fetchone()
    return {
        "total": int(total_row["cnt"] or 0),
        "by_status": by_status,
        "by_job_type": by_job_type,
        "by_origin": by_origin,
        "by_origin_status": by_origin_status,
    }


async def get_memory_ai_health_summary(db) -> dict[str, Any]:
    jobs_summary = await get_jobs_summary(db)

    cursor = await db.execute(
        """SELECT job_type, terminal_reason, COUNT(*) AS cnt
           FROM background_jobs
           WHERE terminal_reason IS NOT NULL
           GROUP BY job_type, terminal_reason
           ORDER BY job_type ASC, terminal_reason ASC"""
    )
    terminal_rows = await cursor.fetchall()
    terminal_reason_counts: dict[str, dict[str, int]] = {}
    for row in terminal_rows:
        terminal_reason_counts.setdefault(row["job_type"], {})[
            row["terminal_reason"]
        ] = int(row["cnt"] or 0)

    latest_jobs: dict[str, dict[str, Any] | None] = {}
    for stage in MEMORY_AI_PIPELINE_STAGES:
        stage_id = stage["stage_id"]
        cursor = await db.execute(
            """SELECT status, terminal_reason, updated_at, finished_at, last_error
               FROM background_jobs
               WHERE job_type = ?
               ORDER BY updated_at DESC, created_at DESC
               LIMIT 1""",
            (stage_id,),
        )
        row = await cursor.fetchone()
        latest_jobs[stage_id] = (
            {
                "status": row["status"],
                "terminal_reason": row["terminal_reason"],
                "updated_at": row["updated_at"],
                "finished_at": row["finished_at"],
                "last_error": row["last_error"],
            }
            if row is not None
            else None
        )

    pipeline_stages: list[dict[str, Any]] = []
    for stage in MEMORY_AI_PIPELINE_STAGES:
        stage_id = stage["stage_id"]
        configured = _env_configured(stage["config_key"])
        counts = _stage_count_summary(
            jobs_summary["by_job_type"].get(stage_id),
            terminal_reason_counts.get(stage_id),
        )
        latest = latest_jobs.get(stage_id)
        pipeline_stages.append(
            {
                "stage_id": stage_id,
                "label": stage["label"],
                "category": "memory_pipeline",
                "provider": stage["provider"],
                "configured": configured,
                "health": _pipeline_stage_health(
                    configured=configured,
                    counts=counts,
                ),
                "counts": counts,
                "latest_status": latest["status"] if latest else None,
                "latest_terminal_reason": latest["terminal_reason"] if latest else None,
                "last_started_at": None,
                "last_updated_at": latest["updated_at"] if latest else None,
                "last_finished_at": latest["finished_at"] if latest else None,
                "last_error": latest["last_error"] if latest else None,
                "checkpoint_created_at": None,
                "recent_memory_count": None,
                "recent_output_count": None,
            }
        )

    cursor = await db.execute(
        """SELECT status, started_at, finished_at, summary_json, error_count
           FROM scheduler_run_log
           WHERE scheduler_name = ?
           ORDER BY started_at DESC, rowid DESC
           LIMIT 1""",
        ("sleep_agent",),
    )
    latest_sleep_run = await cursor.fetchone()

    sleep_summary_raw = (
        str(latest_sleep_run["summary_json"] or "{}")
        if latest_sleep_run is not None
        else "{}"
    )
    try:
        sleep_summary = json.loads(sleep_summary_raw)
    except json.JSONDecodeError:
        sleep_summary = {}
    if not isinstance(sleep_summary, dict):
        sleep_summary = {}

    sleep_errors = sleep_summary.get("errors", [])
    if not isinstance(sleep_errors, list):
        sleep_errors = []
    normalized_sleep_errors = [str(item).strip() for item in sleep_errors if str(item).strip()]

    cursor = await db.execute(
        """SELECT stage_name, checkpoint_created_at
           FROM sleep_agent_checkpoint"""
    )
    checkpoint_rows = await cursor.fetchall()
    sleep_checkpoints = {
        str(row["stage_name"]): (
            str(row["checkpoint_created_at"])
            if row["checkpoint_created_at"]
            else None
        )
        for row in checkpoint_rows
    }

    sleep_agent_enabled = os.getenv("SLEEP_AGENT_ENABLED", "1") != "0"
    llm_configured = _env_configured("OPENAI_API_KEY")
    sleep_stages: list[dict[str, Any]] = []
    latest_sleep_run_status = (
        str(latest_sleep_run["status"]) if latest_sleep_run is not None else None
    )
    latest_sleep_started_at = (
        str(latest_sleep_run["started_at"])
        if latest_sleep_run is not None and latest_sleep_run["started_at"]
        else None
    )
    latest_sleep_finished_at = (
        str(latest_sleep_run["finished_at"])
        if latest_sleep_run is not None and latest_sleep_run["finished_at"]
        else None
    )

    for stage in SLEEP_AGENT_AI_STAGES:
        stage_id = stage["stage_id"]
        prefixes = _SLEEP_AGENT_STAGE_ERROR_PREFIXES.get(stage_id, ())
        stage_errors = [
            error
            for error in normalized_sleep_errors
            if any(error == prefix or error.startswith(prefix) for prefix in prefixes)
        ]
        memory_count_key = _SLEEP_AGENT_STAGE_MEMORY_COUNT_KEYS[stage_id]
        output_count_key = _SLEEP_AGENT_STAGE_OUTPUT_COUNT_KEYS[stage_id]
        recent_memory_count = sleep_summary.get(memory_count_key)
        recent_output_count = sleep_summary.get(output_count_key)
        sleep_stages.append(
            {
                "stage_id": stage_id,
                "label": stage["label"],
                "category": "sleep_agent",
                "provider": stage["provider"],
                "configured": llm_configured,
                "health": _sleep_stage_health(
                    enabled=sleep_agent_enabled,
                    configured=llm_configured,
                    latest_run_status=latest_sleep_run_status,
                    checkpoint_created_at=sleep_checkpoints.get(stage_id),
                    stage_errors=stage_errors,
                    recent_memory_count=(
                        int(recent_memory_count)
                        if isinstance(recent_memory_count, (int, float))
                        else None
                    ),
                ),
                "counts": _stage_count_summary(),
                "latest_status": latest_sleep_run_status,
                "latest_terminal_reason": None,
                "last_started_at": latest_sleep_started_at,
                "last_updated_at": latest_sleep_finished_at or latest_sleep_started_at,
                "last_finished_at": latest_sleep_finished_at,
                "last_error": "；".join(stage_errors[:3]) if stage_errors else None,
                "checkpoint_created_at": sleep_checkpoints.get(stage_id),
                "recent_memory_count": (
                    int(recent_memory_count)
                    if isinstance(recent_memory_count, (int, float))
                    else None
                ),
                "recent_output_count": (
                    int(recent_output_count)
                    if isinstance(recent_output_count, (int, float))
                    else None
                ),
            }
        )

    return {
        "openai_configured": llm_configured,
        "embedding_configured": _env_configured("GEMINI_API_KEY"),
        "sleep_agent_enabled": sleep_agent_enabled,
        "worker_running": get_worker_health_snapshot()["running"],
        "sleep_agent_last_run_status": latest_sleep_run_status,
        "sleep_agent_last_started_at": latest_sleep_started_at,
        "sleep_agent_last_finished_at": latest_sleep_finished_at,
        "sleep_agent_last_error_count": (
            int(latest_sleep_run["error_count"] or 0)
            if latest_sleep_run is not None
            else 0
        ),
        "stages": pipeline_stages + sleep_stages,
    }


async def list_jobs(
    db,
    *,
    status: str | None = None,
    job_type: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    where_clauses: list[str] = []
    params: list[Any] = []
    if status:
        where_clauses.append("status = ?")
        params.append(status)
    if job_type:
        where_clauses.append("job_type = ?")
        params.append(job_type)

    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
    cursor = await db.execute(
        f"""SELECT id, job_type, status, origin, origin_run_id, attempt_count, created_at, updated_at,
                   available_at, finished_at, terminal_reason, last_error
            FROM background_jobs
            {where_sql}
            ORDER BY updated_at DESC, created_at DESC
            LIMIT ?""",
        tuple(params + [max(1, min(int(limit), 200))]),
    )
    rows = await cursor.fetchall()
    return [
        {
            "id": row["id"],
            "job_type": row["job_type"],
            "status": row["status"],
            "origin": row["origin"],
            "origin_run_id": row["origin_run_id"],
            "attempt_count": int(row["attempt_count"] or 0),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "available_at": row["available_at"],
            "finished_at": row["finished_at"],
            "terminal_reason": row["terminal_reason"],
            "last_error": row["last_error"],
        }
        for row in rows
    ]


async def list_memory_jobs(
    db,
    *,
    memory_id: str,
    subject_version: int | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    where_clauses = ["json_extract(payload_json, '$.memory_id') = ?"]
    params: list[Any] = [memory_id]
    if subject_version is not None:
        where_clauses.append(
            "CAST(json_extract(payload_json, '$.subject_version') AS INTEGER) = ?"
        )
        params.append(int(subject_version))

    where_sql = f"WHERE {' AND '.join(where_clauses)}"
    cursor = await db.execute(
        f"""SELECT job_type, status, origin, origin_run_id, attempt_count,
                   created_at, updated_at, finished_at, terminal_reason, last_error,
                   CAST(json_extract(payload_json, '$.subject_version') AS INTEGER) AS subject_version
            FROM background_jobs
            {where_sql}
            ORDER BY created_at ASC, updated_at ASC
            LIMIT ?""",
        tuple(params + [max(1, min(int(limit), 100))]),
    )
    rows = await cursor.fetchall()
    return [
        {
            "job_type": row["job_type"],
            "status": row["status"],
            "origin": row["origin"],
            "origin_run_id": row["origin_run_id"],
            "attempt_count": int(row["attempt_count"] or 0),
            "subject_version": (
                int(row["subject_version"])
                if row["subject_version"] is not None
                else None
            ),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "finished_at": row["finished_at"],
            "terminal_reason": row["terminal_reason"],
            "last_error": row["last_error"],
        }
        for row in rows
    ]


def summarize_memory_job_runs(
    jobs: list[dict[str, Any]],
    *,
    current_subject_version: int | None = None,
) -> list[dict[str, Any]]:
    grouped_runs: dict[str, dict[str, Any]] = {}
    run_order: list[str] = []

    for job in jobs:
        origin = str(job.get("origin") or "pipeline")
        subject_version = (
            int(job["subject_version"])
            if job.get("subject_version") is not None
            else None
        )
        origin_run_id = (
            str(job["origin_run_id"]).strip()
            if job.get("origin_run_id")
            else None
        )
        group_key = origin_run_id or f"{origin}:v{subject_version or 'none'}:implicit"

        if group_key not in grouped_runs:
            grouped_runs[group_key] = {
                "origin_run_id": origin_run_id,
                "origin": origin,
                "subject_version": subject_version,
                "job_count": 0,
                "status_counts": {},
                "started_at": None,
                "updated_at": None,
                "finished_at": None,
                "is_current_version": (
                    current_subject_version is None
                    or subject_version == current_subject_version
                ),
                "jobs": [],
            }
            run_order.append(group_key)

        run = grouped_runs[group_key]
        run["job_count"] += 1
        status = str(job.get("status") or "pending")
        status_counts = run["status_counts"]
        status_counts[status] = int(status_counts.get(status, 0)) + 1
        run["jobs"].append(job)

        created_at = job.get("created_at")
        if created_at and (
            run["started_at"] is None or str(created_at) < str(run["started_at"])
        ):
            run["started_at"] = created_at

        updated_at = job.get("updated_at") or job.get("created_at")
        if updated_at and (
            run["updated_at"] is None or str(updated_at) > str(run["updated_at"])
        ):
            run["updated_at"] = updated_at

        finished_at = job.get("finished_at")
        if finished_at:
            if run["finished_at"] is None or str(finished_at) > str(run["finished_at"]):
                run["finished_at"] = finished_at
        else:
            run["finished_at"] = None

    runs = [grouped_runs[key] for key in run_order]
    runs.sort(
        key=lambda run: (
            str(run.get("updated_at") or run.get("started_at") or ""),
            str(run.get("started_at") or ""),
        ),
        reverse=True,
    )
    return runs


async def get_job(db, *, job_id: str) -> dict[str, Any] | None:
    cursor = await db.execute(
        """SELECT id, job_type, status, origin, origin_run_id, attempt_count, created_at, updated_at,
                  available_at, finished_at, terminal_reason, last_error
           FROM background_jobs
           WHERE id = ?""",
        (job_id,),
    )
    row = await cursor.fetchone()
    if row is None:
        return None
    return {
        "id": row["id"],
        "job_type": row["job_type"],
        "status": row["status"],
        "origin": row["origin"],
        "origin_run_id": row["origin_run_id"],
        "attempt_count": int(row["attempt_count"] or 0),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "available_at": row["available_at"],
        "finished_at": row["finished_at"],
        "terminal_reason": row["terminal_reason"],
        "last_error": row["last_error"],
    }


async def create_scheduler_run_log(db, *, scheduler_name: str, status: str = "running") -> str:
    run_id = str(uuid.uuid4())
    await db.execute(
        """INSERT INTO scheduler_run_log
           (id, scheduler_name, status, started_at, error_count)
           VALUES (?, ?, ?, ?, ?)""",
        (
            run_id,
            scheduler_name,
            status,
            _sqlite_datetime(),
            0,
        ),
    )
    await db.commit()
    return run_id


async def finish_scheduler_run_log(
    db,
    *,
    run_id: str,
    status: str,
    summary: dict[str, Any] | None = None,
    error_count: int = 0,
) -> bool:
    cursor = await db.execute(
        """UPDATE scheduler_run_log
           SET status = ?,
               finished_at = ?,
               summary_json = ?,
               error_count = ?
           WHERE id = ?""",
        (
            status,
            _sqlite_datetime(),
            _canonical_json(summary or {}),
            max(0, int(error_count)),
            run_id,
        ),
    )
    await db.commit()
    return bool(cursor.rowcount and cursor.rowcount > 0)


async def run_worker_iteration(
    db,
    *,
    worker_id: str,
    lease_seconds: int = DEFAULT_JOB_LEASE_SECONDS,
) -> bool:
    if not _JOB_HANDLERS:
        return False

    job = await claim_next_job(
        db,
        worker_id=worker_id,
        job_types=tuple(sorted(_JOB_HANDLERS.keys())),
        lease_seconds=lease_seconds,
    )
    if job is None:
        return False

    handler = _JOB_HANDLERS.get(job["job_type"])
    if handler is None:
        await mark_job_failed(
            db,
            job_id=job["id"],
            lease_token=job["lease_token"],
            error=f"no handler registered for job_type={job['job_type']}",
        )
        return True

    try:
        await handler(job, db)
    except ObsoleteJobError as exc:
        completed = await mark_job_completed(
            db,
            job_id=job["id"],
            lease_token=job["lease_token"],
            terminal_reason=TERMINAL_REASON_OBSOLETE,
        )
        logger.info(
            "background job skipped as obsolete job_id=%s job_type=%s reason=%s completed=%s",
            job["id"],
            job["job_type"],
            str(exc),
            completed,
        )
    except Exception as exc:
        status = await mark_job_failed(
            db,
            job_id=job["id"],
            lease_token=job["lease_token"],
            error=f"{type(exc).__name__}: {exc}",
        )
        logger.exception(
            "background job handler failed job_id=%s job_type=%s resulting_status=%s",
            job["id"],
            job["job_type"],
            status,
        )
    else:
        completed = await mark_job_completed(
            db,
            job_id=job["id"],
            lease_token=job["lease_token"],
        )
        if not completed:
            logger.warning(
                "background job completion skipped by stale lease guard job_id=%s job_type=%s",
                job["id"],
                job["job_type"],
            )
    return True


async def run_background_jobs_worker(
    *,
    poll_interval_seconds: float = DEFAULT_WORKER_POLL_SECONDS,
    lease_seconds: int = DEFAULT_JOB_LEASE_SECONDS,
) -> None:
    worker_id = str(uuid.uuid4())
    db = await get_db()
    reset_worker_health()
    _set_worker_heartbeat(worker_id)
    logger.info("background jobs worker started worker_id=%s", worker_id)
    try:
        while True:
            _set_worker_heartbeat(worker_id)
            try:
                reclaimed = await reclaim_stale_running_jobs(db)
                if reclaimed:
                    logger.info(
                        "background jobs worker reclaimed stale jobs worker_id=%s count=%s",
                        worker_id,
                        reclaimed,
                    )
                handled = await run_worker_iteration(
                    db,
                    worker_id=worker_id,
                    lease_seconds=lease_seconds,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                _set_worker_error(f"{type(exc).__name__}: {exc}")
                logger.exception("background jobs worker iteration failed worker_id=%s", worker_id)
                handled = False

            await asyncio.sleep(0 if handled else max(poll_interval_seconds, 0.1))
    except asyncio.CancelledError:
        logger.info("background jobs worker cancelled worker_id=%s", worker_id)
        raise
    finally:
        await db.close()
