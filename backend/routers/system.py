import logging

from fastapi import APIRouter, HTTPException, Query

from memory_core.models import MemoryAIHealthResponse
from memory_core.database import get_db
from memory_core.services.background_jobs import (
    get_job,
    get_jobs_summary,
    get_memory_ai_health_summary,
    get_worker_health_snapshot,
    list_jobs,
    retry_job,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/system", tags=["system"])


@router.get("/jobs/summary")
async def jobs_summary():
    db = await get_db(read_only=True)
    try:
        summary = await get_jobs_summary(db)
        return {
            "jobs": summary,
            "worker": get_worker_health_snapshot(),
        }
    finally:
        await db.close()


@router.get("/memory-ai-health", response_model=MemoryAIHealthResponse)
async def memory_ai_health():
    db = await get_db(read_only=True)
    try:
        return MemoryAIHealthResponse(**(await get_memory_ai_health_summary(db)))
    finally:
        await db.close()


@router.get("/jobs")
async def jobs_list(
    status: str | None = Query(default=None),
    job_type: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
):
    db = await get_db(read_only=True)
    try:
        jobs = await list_jobs(
            db,
            status=status.strip() if status else None,
            job_type=job_type.strip() if job_type else None,
            limit=limit,
        )
        return {"jobs": jobs}
    finally:
        await db.close()


@router.post("/jobs/{job_id}/retry")
async def retry_system_job(job_id: str):
    db = await get_db()
    try:
        job = await get_job(db, job_id=job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job 不存在")
        if job["status"] not in {"failed", "dead"}:
            raise HTTPException(status_code=400, detail="只有 failed 或 dead job 可以重试")

        retried = await retry_job(db, job_id=job_id)
        if not retried:
            raise HTTPException(status_code=409, detail="job 当前不可重试")
        refreshed = await get_job(db, job_id=job_id)
        return {
            "ok": True,
            "job": refreshed,
        }
    finally:
        await db.close()
