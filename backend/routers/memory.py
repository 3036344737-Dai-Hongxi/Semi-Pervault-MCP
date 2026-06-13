import json as _json
import logging
import sqlite3
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Query, HTTPException, Request
from fastapi.responses import Response

from memory_core.database import get_db, get_shared_db
from memory_core.models import (
    LongTermOverviewResponse,
    LongTermLayersResponse,
    MemoryAdmissionExplanation,
    MemoryAdmissionExplanationResponse,
    MemoryPipelineTraceJob,
    MemoryPipelineTraceRun,
    MemoryPipelineTraceResponse,
    MemoryExportRequest,
    MemoryItem,
    PersonaItem,
    ReflectionListItem,
    MemoryReprocessResponse,
    MemorySearchResult,
    MemoryStoreRequest,
    MemoryUpdateRequest,
)
from memory_core.services.background_jobs import list_memory_jobs, summarize_memory_job_runs
from memory_core.services.memory_policy import contains_cjk
from memory_core.services.memory_service import (
    create_memory_item,
    enqueue_memory_store_jobs,
    reprocess_memory_item,
    update_memory_item_content,
    row_to_item,
)
from services.rate_limit import limiter

logger = logging.getLogger("uvicorn.error")

router = APIRouter(prefix="/api/memory", tags=["memory"])


def _request_ip(request: Request) -> str | None:
    return request.client.host if request.client else None


def _request_user_agent(request: Request) -> str | None:
    return request.headers.get("user-agent")


async def _write_export_audit_log(
    db,
    *,
    auth_session_id: str | None,
    status: str,
    request: Request,
    counts: dict[str, int] | None = None,
) -> None:
    counts = counts or {}
    await db.execute(
        """INSERT INTO data_export_log
           (id, auth_session_id, status, client_ip, user_agent,
            memory_count, fact_count, persona_count, reflection_count,
            revision_count, graph_node_count, graph_edge_count)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            str(uuid.uuid4()),
            auth_session_id,
            status,
            _request_ip(request),
            _request_user_agent(request),
            counts.get("memory_count"),
            counts.get("fact_count"),
            counts.get("persona_count"),
            counts.get("reflection_count"),
            counts.get("revision_count"),
            counts.get("graph_node_count"),
            counts.get("graph_edge_count"),
        ),
    )


def _row_to_admission_explanation(row) -> MemoryAdmissionExplanation:
    return MemoryAdmissionExplanation(
        memory_id=str(row["memory_id"]),
        utility=float(row["score_utility"] or 0.0),
        confidence=float(row["score_confidence"] or 0.0),
        novelty=float(row["score_novelty"] or 0.0),
        recency=float(row["score_recency"] or 0.0),
        type_prior=float(row["score_type_prior"] or 0.0),
        total_score=float(row["total_score"] or 0.0),
        tier=str(row["tier"] or "standard"),
        created_at=str(row["created_at"]) if row["created_at"] else None,
    )


def _parse_json_string_list(value: str | None) -> list[str]:
    if not value:
        return []
    try:
        raw = _json.loads(value)
    except (_json.JSONDecodeError, TypeError):
        return []
    if not isinstance(raw, list):
        return []
    return [str(item) for item in raw if isinstance(item, str) and item.strip()]


@router.post("/store", response_model=MemoryItem)
@limiter.limit("60/minute")
async def store_memory(request: Request, req: MemoryStoreRequest):
    db = await get_db()
    try:
        item = await create_memory_item(
            content=req.content,
            voice_record_id=req.voice_record_id,
            tags=req.tags,
            extract_structured_facts_enabled=True,
            db=db,
        )

        await enqueue_memory_store_jobs(
            memory_id=item.id,
            content=req.content,
            kind=item.kind,
            db=db,
        )

        return item
    finally:
        await db.close()


@router.patch("/{memory_id}", response_model=MemoryItem)
@limiter.limit("60/minute")
async def update_memory(request: Request, memory_id: str, req: MemoryUpdateRequest):
    db = await get_db()
    try:
        item = await update_memory_item_content(
            memory_item_id=memory_id,
            content=req.content,
            extract_structured_facts_enabled=True,
            db=db,
        )

        await enqueue_memory_store_jobs(
            memory_id=item.id,
            content=req.content,
            kind=item.kind,
            db=db,
        )

        return item
    finally:
        await db.close()


@router.post("/{memory_id}/reprocess", response_model=MemoryReprocessResponse)
@limiter.limit("60/minute")
async def reprocess_memory(request: Request, memory_id: str):
    db = await get_db()
    try:
        return await reprocess_memory_item(memory_id, db=db)
    finally:
        await db.close()


@router.get(
    "/{memory_id}/admission-explanation",
    response_model=MemoryAdmissionExplanationResponse,
)
async def get_memory_admission_explanation(memory_id: str):
    db = await get_db(read_only=True)
    try:
        cursor = await db.execute(
            """SELECT memory_id, score_utility, score_confidence, score_novelty,
                      score_recency, score_type_prior, total_score, tier, created_at
               FROM memory_admission_log
               WHERE memory_id = ?
               ORDER BY created_at DESC, rowid DESC
               LIMIT 1""",
            (memory_id,),
        )
        row = await cursor.fetchone()
        return MemoryAdmissionExplanationResponse(
            memory_id=memory_id,
            explanation=_row_to_admission_explanation(row) if row else None,
        )
    finally:
        await db.close()


@router.get(
    "/{memory_id}/pipeline-trace",
    response_model=MemoryPipelineTraceResponse,
)
async def get_memory_pipeline_trace(memory_id: str):
    db = await get_db(read_only=True)
    try:
        memory_row = await (
            await db.execute(
                """SELECT id, content_version
                   FROM memory_items
                   WHERE id = ?""",
                (memory_id,),
            )
        ).fetchone()
        if memory_row is None:
            raise HTTPException(status_code=404, detail="记忆不存在")

        content_version = int(memory_row["content_version"] or 1)
        jobs = await list_memory_jobs(
            db,
            memory_id=memory_id,
            subject_version=content_version,
            limit=50,
        )
        hidden_row = await (
            await db.execute(
                """SELECT COUNT(*) AS cnt
                   FROM background_jobs
                   WHERE json_extract(payload_json, '$.memory_id') = ?
                     AND (
                       json_extract(payload_json, '$.subject_version') IS NULL
                       OR CAST(json_extract(payload_json, '$.subject_version') AS INTEGER) != ?
                     )""",
                (memory_id, content_version),
            )
        ).fetchone()

        return MemoryPipelineTraceResponse(
            memory_id=memory_id,
            content_version=content_version,
            hidden_job_count=int(hidden_row["cnt"] or 0),
            runs=[
                MemoryPipelineTraceRun(**run)
                for run in summarize_memory_job_runs(
                    jobs,
                    current_subject_version=content_version,
                )
            ],
            jobs=[MemoryPipelineTraceJob(**job) for job in jobs],
        )
    finally:
        await db.close()


@router.get("/layer-overview", response_model=LongTermOverviewResponse)
async def get_long_term_layer_overview():
    db = await get_db(read_only=True)
    try:
        persona_row = await (
            await db.execute("SELECT COUNT(*) AS cnt FROM user_persona")
        ).fetchone()
        reflection_row = await (
            await db.execute("SELECT COUNT(*) AS cnt FROM memory_reflection")
        ).fetchone()
        pending_graph_row = await (
            await db.execute(
                "SELECT COUNT(*) AS cnt FROM graph_nodes WHERE status = 'pending'"
            )
        ).fetchone()
        low_value_row = await (
            await db.execute(
                """SELECT COUNT(*) AS cnt
                   FROM memory_items
                   WHERE COALESCE(admission_tier, 'standard') = 'low_value'"""
            )
        ).fetchone()

        return LongTermOverviewResponse(
            persona_count=int(persona_row["cnt"] or 0),
            reflection_count=int(reflection_row["cnt"] or 0),
            pending_graph_node_count=int(pending_graph_row["cnt"] or 0),
            low_value_memory_count=int(low_value_row["cnt"] or 0),
        )
    finally:
        await db.close()


@router.get("/long-term-layers", response_model=LongTermLayersResponse)
async def get_long_term_layers():
    db = await get_db(read_only=True)
    try:
        persona_rows = await (
            await db.execute(
                """SELECT id, trait_key, trait_value, confidence, evidence_count,
                          source_memory_ids, last_updated
                   FROM user_persona
                   ORDER BY confidence DESC, evidence_count DESC, last_updated DESC"""
            )
        ).fetchall()
        reflection_rows = await (
            await db.execute(
                """SELECT id, insight, source_memory_ids, importance, created_at
                   FROM memory_reflection
                   ORDER BY importance DESC, created_at DESC"""
            )
        ).fetchall()

        persona_items = [
            PersonaItem(
                id=str(row["id"]),
                trait_key=str(row["trait_key"] or ""),
                trait_value=str(row["trait_value"] or ""),
                confidence=float(row["confidence"] or 0.0),
                evidence_count=int(row["evidence_count"] or 0),
                source_memory_ids=_parse_json_string_list(row["source_memory_ids"]),
                last_updated=str(row["last_updated"]) if row["last_updated"] else None,
            )
            for row in persona_rows
        ]
        reflection_items: list[ReflectionListItem] = []
        for row in reflection_rows:
            source_memory_ids = _parse_json_string_list(row["source_memory_ids"])
            reflection_items.append(
                ReflectionListItem(
                    id=str(row["id"]),
                    insight=str(row["insight"] or ""),
                    source_memory_ids=source_memory_ids,
                    source_memory_count=len(source_memory_ids),
                    importance=float(row["importance"] or 0.0),
                    created_at=str(row["created_at"]) if row["created_at"] else None,
                )
            )

        return LongTermLayersResponse(
            persona_items=persona_items,
            reflection_items=reflection_items,
        )
    finally:
        await db.close()


@router.get("/search", response_model=MemorySearchResult)
async def search_memories(
    q: str = Query(default="", description="搜索关键词"),
    kind: str = Query(default="", description="按 kind 类型筛选"),
    admission_tier: str = Query(default="", description="按 admission tier 筛选"),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
):
    kind_filter = kind.strip()
    admission_tier_filter = admission_tier.strip()
    # kind_clause / kind_params are injected into queries with existing WHERE clauses
    kind_clause = " AND m.kind = ?" if kind_filter else ""
    kind_params: tuple = (kind_filter,) if kind_filter else ()
    admission_tier_clause = (
        " AND COALESCE(m.admission_tier, 'standard') = ?"
        if admission_tier_filter
        else ""
    )
    admission_tier_params: tuple = (
        (admission_tier_filter,) if admission_tier_filter else ()
    )

    db = await get_db(read_only=True)
    try:
        if q.strip():
            search_term = q.strip()
            if contains_cjk(search_term):
                like_term = f"%{search_term}%"
                cursor = await db.execute(
                    f"""SELECT m.* FROM memory_items m
                       WHERE (m.content LIKE ? OR m.tags LIKE ?){kind_clause}{admission_tier_clause}
                       ORDER BY m.weight DESC, m.created_at DESC
                       LIMIT ? OFFSET ?""",
                    (like_term, like_term)
                    + kind_params
                    + admission_tier_params
                    + (limit, offset),
                )
                rows = await cursor.fetchall()

                count_cursor = await db.execute(
                    f"""SELECT COUNT(*) as cnt FROM memory_items m
                       WHERE (m.content LIKE ? OR m.tags LIKE ?){kind_clause}{admission_tier_clause}""",
                    (like_term, like_term) + kind_params + admission_tier_params,
                )
                count_row = await count_cursor.fetchone()
                total = count_row["cnt"]
            else:
                try:
                    cursor = await db.execute(
                        f"""SELECT m.* FROM memory_items m
                           JOIN memory_fts f ON m.rowid = f.rowid
                           WHERE memory_fts MATCH ?{kind_clause}{admission_tier_clause}
                           ORDER BY m.weight DESC, m.created_at DESC
                           LIMIT ? OFFSET ?""",
                        (search_term,)
                        + kind_params
                        + admission_tier_params
                        + (limit, offset),
                    )
                    rows = await cursor.fetchall()

                    count_cursor = await db.execute(
                        f"""SELECT COUNT(*) as cnt FROM memory_items m
                           JOIN memory_fts f ON m.rowid = f.rowid
                           WHERE memory_fts MATCH ?{kind_clause}{admission_tier_clause}""",
                        (search_term,) + kind_params + admission_tier_params,
                    )
                    count_row = await count_cursor.fetchone()
                    total = count_row["cnt"]
                except sqlite3.OperationalError as e:
                    raise HTTPException(
                        400, f"搜索语法错误，请简化关键词：{e}"
                    ) from e
        else:
            # No keyword query — WHERE clause depends solely on kind_filter
            where_conditions: list[str] = []
            where_params_list: list[object] = []
            if kind_filter:
                where_conditions.append("kind = ?")
                where_params_list.append(kind_filter)
            if admission_tier_filter:
                where_conditions.append("COALESCE(admission_tier, 'standard') = ?")
                where_params_list.append(admission_tier_filter)
            where_clause = f"WHERE {' AND '.join(where_conditions)}" if where_conditions else ""
            where_params = tuple(where_params_list)
            cursor = await db.execute(
                f"""SELECT * FROM memory_items
                   {where_clause}
                   ORDER BY created_at DESC
                   LIMIT ? OFFSET ?""",
                where_params + (limit, offset),
            )
            rows = await cursor.fetchall()

            count_cursor = await db.execute(
                f"SELECT COUNT(*) as cnt FROM memory_items {where_clause}",
                where_params,
            )
            count_row = await count_cursor.fetchone()
            total = count_row["cnt"]

        return MemorySearchResult(
            items=[row_to_item(r) for r in rows],
            total=total,
        )
    finally:
        await db.close()


@router.post("/export")
async def export_memories(request: Request, req: MemoryExportRequest):
    if req.confirm_export is not True:
        raise HTTPException(status_code=400, detail="导出前需要显式确认")

    db = await get_shared_db()
    auth_session_id = getattr(request.state, "auth_session_id", None)
    try:
        cur = await db.execute("SELECT * FROM memory_items ORDER BY created_at ASC")
        memories = [dict(r) for r in await cur.fetchall()]

        cur = await db.execute("SELECT * FROM structured_facts ORDER BY created_at ASC")
        facts = [dict(r) for r in await cur.fetchall()]

        cur = await db.execute("SELECT * FROM graph_nodes ORDER BY created_at ASC")
        nodes = [dict(r) for r in await cur.fetchall()]

        cur = await db.execute("SELECT * FROM graph_edges ORDER BY created_at ASC")
        edges = [dict(r) for r in await cur.fetchall()]

        cur = await db.execute("SELECT * FROM user_persona ORDER BY last_updated DESC")
        persona = [dict(r) for r in await cur.fetchall()]

        cur = await db.execute("SELECT * FROM memory_reflection ORDER BY importance DESC, created_at DESC")
        reflections = [dict(r) for r in await cur.fetchall()]

        cur = await db.execute("SELECT * FROM preference_revision_log ORDER BY created_at DESC")
        revision_log = [dict(r) for r in await cur.fetchall()]

        payload = {
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "version": "1.0",
            "memories": memories,
            "structured_facts": facts,
            "user_persona": persona,
            "memory_reflection": reflections,
            "preference_revision_log": revision_log,
            "graph": {"nodes": nodes, "edges": edges},
        }
        counts = {
            "memory_count": len(memories),
            "fact_count": len(facts),
            "persona_count": len(persona),
            "reflection_count": len(reflections),
            "revision_count": len(revision_log),
            "graph_node_count": len(nodes),
            "graph_edge_count": len(edges),
        }
        await _write_export_audit_log(
            db,
            auth_session_id=auth_session_id,
            status="completed",
            request=request,
            counts=counts,
        )
        await db.commit()
        logger.info(
            "memory export completed session_id=%s memory_count=%s fact_count=%s persona_count=%s reflection_count=%s revision_count=%s graph_nodes=%s graph_edges=%s",
            auth_session_id,
            counts["memory_count"],
            counts["fact_count"],
            counts["persona_count"],
            counts["reflection_count"],
            counts["revision_count"],
            counts["graph_node_count"],
            counts["graph_edge_count"],
        )
        return Response(
            content=_json.dumps(payload, ensure_ascii=False, indent=2),
            media_type="application/json",
            headers={"Content-Disposition": 'attachment; filename="voicevault_export.json"'},
        )
    except Exception:
        try:
            await _write_export_audit_log(
                db,
                auth_session_id=auth_session_id,
                status="failed",
                request=request,
            )
            await db.commit()
        except Exception:
            await db.rollback()
            logger.exception("memory export audit write failed session_id=%s", auth_session_id)
        logger.exception("memory export failed session_id=%s", auth_session_id)
        raise


@router.get("/stats")
async def memory_stats():
    db = await get_db(read_only=True)
    try:
        cursor = await db.execute("SELECT COUNT(*) as cnt FROM memory_items")
        row = await cursor.fetchone()
        total = row["cnt"]

        cursor2 = await db.execute(
            """SELECT COUNT(*) as cnt FROM memory_items
               WHERE date(created_at) = date('now')"""
        )
        row2 = await cursor2.fetchone()
        today = row2["cnt"]

        return {"total_memories": total, "today_count": today}
    finally:
        await db.close()
