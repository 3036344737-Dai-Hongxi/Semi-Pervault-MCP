"""Loopback Core API（/core/*）：本地宿主（MCP 桥接、浏览器扩展）专用入口。

信任模型：仅 127.0.0.1 + X-Pervault-Token 头（token 见 memory_core.local_auth）。
/core 前缀不走浏览器 cookie 会话中间件（main.py 只拦 /api/*），由本模块强制 token。
"""

import hashlib as _hashlib
import logging
import secrets as _secrets
from typing import Any, cast, get_args

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from pydantic import BaseModel, Field

from memory_core.database import get_db, get_shared_db
from memory_core.local_auth import read_or_create_core_token
from memory_core.models import MemoryItem
from memory_core.services.graph_retrieval import retrieve_graph_context
from memory_core.services.memory_service import (
    create_memory_item,
    enqueue_memory_store_jobs,
    update_memory_item_content,
)
from memory_core.services.provenance import explain_belief
from memory_core.services.retrieval_constants import QueryIntent
from memory_core.services.retrieval_context import retrieve_context

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/core", tags=["core"])

# 从内核 QueryIntent 单一真源派生，避免手抄副本与内核漂移（新增意图自动放行，
# 删除的意图自动拒绝；不再需要 # type: ignore 屏蔽类型不一致）。
_VALID_INTENTS = frozenset(get_args(QueryIntent))


async def require_core_token(
    x_pervault_token: str = Header(default=""),
) -> None:
    expected = read_or_create_core_token()
    if not x_pervault_token or not _secrets.compare_digest(
        x_pervault_token, expected
    ):
        raise HTTPException(status_code=401, detail="invalid core token")


class CoreMemoryStoreRequest(BaseModel):
    content: str = Field(min_length=1, max_length=10000)
    tags: list[str] = Field(default_factory=list)
    source: str = Field(default="mcp", max_length=32)


class CoreMemoryUpdateRequest(BaseModel):
    content: str = Field(min_length=1, max_length=10000)


@router.get("/health", dependencies=[Depends(require_core_token)])
async def core_health():
    from memory_core.database import DB_PATH

    # 不回显绝对路径（含用户名）。返回路径的稳定哈希 db_id，既不泄漏文件系统信息，
    # 又能让本地编排（如 MemArena）校验「连上的是我这次启动的 daemon」（sec-4）。
    db_id = _hashlib.sha256(str(DB_PATH).encode("utf-8")).hexdigest()[:16]
    return {"status": "ok", "db_id": db_id}


@router.post(
    "/memory", response_model=MemoryItem, dependencies=[Depends(require_core_token)]
)
async def core_store_memory(req: CoreMemoryStoreRequest):
    db = await get_db()
    try:
        tags = list(dict.fromkeys([*req.tags, f"source:{req.source}"]))
        item = await create_memory_item(
            content=req.content,
            tags=tags,
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


@router.patch(
    "/memory/{memory_id}",
    response_model=MemoryItem,
    dependencies=[Depends(require_core_token)],
)
async def core_update_memory(memory_id: str, req: CoreMemoryUpdateRequest):
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


@router.get("/recall", dependencies=[Depends(require_core_token)])
async def core_recall(
    q: str = Query(min_length=1, max_length=500),
    intent: str | None = Query(default=None),
    limit: int = Query(default=10, ge=1, le=50),
):
    if intent is not None and intent not in _VALID_INTENTS:
        raise HTTPException(
            status_code=400,
            detail=f"invalid intent, must be one of: {sorted(_VALID_INTENTS)}",
        )
    # 已校验为合法 QueryIntent 字面量；cast 让类型检查通过而无需 type: ignore
    validated_intent = cast("QueryIntent | None", intent)
    db = await get_shared_db()
    results = await retrieve_context(q, db, intent=validated_intent)
    trimmed: list[dict[str, Any]] = []
    for item in results[:limit]:
        trimmed.append(
            {
                "content": item.get("content"),
                "source": item.get("_source", "unknown"),
                "memory_id": item.get("id"),
                "kind": item.get("kind"),
                "created_at": item.get("created_at"),
            }
        )
    return {"query": q, "intent": intent, "results": trimmed}


@router.get("/why", dependencies=[Depends(require_core_token)])
async def core_why(q: str = Query(min_length=1, max_length=500)):
    """证据链：返回「为什么这么认为」——信念 + 来源记忆 + 准入打分 + 修正日志。"""
    db = await get_shared_db()
    return await explain_belief(q, db)


@router.get("/graph", dependencies=[Depends(require_core_token)])
async def core_graph(q: str = Query(min_length=1, max_length=500)):
    db = await get_shared_db()
    context = await retrieve_graph_context(q, db)
    return {"query": q, "graph_context": context}


@router.get("/persona", dependencies=[Depends(require_core_token)])
async def core_persona(limit: int = Query(default=10, ge=1, le=50)):
    db = await get_db(read_only=True)
    try:
        cursor = await db.execute(
            """SELECT trait_key, trait_value, confidence, evidence_count, last_updated
               FROM user_persona
               ORDER BY confidence DESC, evidence_count DESC, last_updated DESC
               LIMIT ?""",
            (limit,),
        )
        rows = await cursor.fetchall()
        return {"traits": [dict(row) for row in rows]}
    finally:
        await db.close()


@router.get("/reflections", dependencies=[Depends(require_core_token)])
async def core_reflections(limit: int = Query(default=10, ge=1, le=50)):
    db = await get_db(read_only=True)
    try:
        cursor = await db.execute(
            """SELECT id, insight, importance, created_at
               FROM memory_reflection
               ORDER BY importance DESC, created_at DESC
               LIMIT ?""",
            (limit,),
        )
        rows = await cursor.fetchall()
        return {"reflections": [dict(row) for row in rows]}
    finally:
        await db.close()


@router.get("/stats", dependencies=[Depends(require_core_token)])
async def core_stats():
    db = await get_db(read_only=True)
    try:
        cursor = await db.execute("SELECT COUNT(*) AS cnt FROM memory_items")
        total = (await cursor.fetchone())["cnt"]
        cursor = await db.execute(
            "SELECT COUNT(*) AS cnt FROM memory_items WHERE date(created_at) = date('now')"
        )
        today = (await cursor.fetchone())["cnt"]
        cursor = await db.execute(
            """SELECT admission_tier AS tier, COUNT(*) AS cnt
               FROM memory_items GROUP BY admission_tier"""
        )
        tiers = {row["tier"]: row["cnt"] async for row in cursor}
        return {"total_memories": total, "today_count": today, "by_tier": tiers}
    finally:
        await db.close()
