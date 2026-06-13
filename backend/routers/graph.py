import json
import logging
from fastapi import APIRouter, HTTPException, Query
from memory_core.database import get_db
from memory_core.models import (
    GraphNode,
    GraphEdge,
    GraphEdgeWithLabels,
    GraphExtractRequest,
    GraphExtractResponse,
    GraphSubgraphResponse,
    GraphNodeDetailResponse,
    GraphPendingResponse,
    MemoryBrief,
)
from memory_core.services.graph_extract import GraphExtractionError
from memory_core.services.graph_pipeline import extract_and_store_graph
from memory_core.services.llm import AIServiceUnavailableError
from memory_core.services.memory_policy import ALL_QUERY_NODE_TYPES, GRAPH_ELIGIBLE_MEMORY_KINDS, is_graph_eligible_kind

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/graph", tags=["graph"])

VALID_NODE_TYPES = ALL_QUERY_NODE_TYPES
VALID_STATUSES = {"confirmed", "pending", "all"}


# ── helpers ─────────────────────────────────────────────


def _row_to_node(row) -> GraphNode:
    props_raw = row["properties"] or "{}"
    try:
        props = json.loads(props_raw)
    except (json.JSONDecodeError, TypeError):
        props = {}
    row_keys = set(row.keys())
    return GraphNode(
        id=row["id"],
        type=row["type"],
        label=row["label"],
        properties=props,
        weight=row["weight"],
        source_memory_count=row["source_memory_count"],
        created_at=row["created_at"],
        last_seen_at=row["last_seen_at"],
        status=row["status"] if "status" in row_keys else "confirmed",
        possible_duplicate_of=(
            row["possible_duplicate_of"] if "possible_duplicate_of" in row_keys else None
        ),
    )


def _row_to_edge(row) -> GraphEdge:
    return GraphEdge(
        id=row["id"],
        source_id=row["source_id"],
        target_id=row["target_id"],
        relation=row["relation"],
        weight=row["weight"],
        source_memory_id=row["source_memory_id"],
        created_at=row["created_at"],
    )


# ── endpoints ───────────────────────────────────────────


@router.post("/extract", response_model=GraphExtractResponse)
async def extract_and_store(req: GraphExtractRequest):
    read_db = await get_db(read_only=True)
    try:
        cur = await read_db.execute(
            "SELECT id, kind FROM memory_items WHERE id = ?", (req.memory_item_id,)
        )
        row = await cur.fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="记忆条目不存在")
        memory_kind: str = row["kind"] or "other"
    finally:
        await read_db.close()

    if not is_graph_eligible_kind(memory_kind):
        raise HTTPException(
            status_code=422,
            detail=(
                f"该记忆 kind='{memory_kind}' 不符合图谱提取条件，"
                f"允许的 kind：{sorted(GRAPH_ELIGIBLE_MEMORY_KINDS)}"
            ),
        )

    try:
        nodes, edges = await extract_and_store_graph(req.memory_item_id, req.content)
    except AIServiceUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except GraphExtractionError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return GraphExtractResponse(nodes=nodes, edges=edges)


@router.get("/subgraph", response_model=GraphSubgraphResponse)
async def get_subgraph(
    keyword: str = Query(default="", description="按 label 模糊搜索"),
    node_type: str = Query(default="", description="按节点类型筛选"),
    status: str = Query(
        default="confirmed",
        description="节点状态筛选: confirmed（默认）/ pending / all",
    ),
    limit: int = Query(default=50, ge=1, le=200),
):
    if node_type and node_type not in VALID_NODE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"无效的节点类型，允许值: {', '.join(sorted(VALID_NODE_TYPES))}",
        )
    if status not in VALID_STATUSES:
        raise HTTPException(
            status_code=400,
            detail=f"无效的状态，允许值: {', '.join(sorted(VALID_STATUSES))}",
        )

    db = await get_db(read_only=True)
    try:
        conditions: list[str] = []
        params: list[object] = []

        if keyword.strip():
            conditions.append("label LIKE ?")
            params.append(f"%{keyword.strip()}%")
        if node_type:
            conditions.append("type = ?")
            params.append(node_type)
        if status != "all":
            conditions.append("status = ?")
            params.append(status)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        order = "ORDER BY last_seen_at DESC"

        sql = f"SELECT * FROM graph_nodes {where} {order} LIMIT ?"
        params.append(limit)

        cur = await db.execute(sql, params)
        node_rows = await cur.fetchall()
        nodes = [_row_to_node(r) for r in node_rows]

        if not nodes:
            return GraphSubgraphResponse(nodes=[], edges=[])

        node_ids = {n.id for n in nodes}
        placeholders = ",".join("?" for _ in node_ids)
        id_list = list(node_ids)

        cur = await db.execute(
            f"""SELECT * FROM graph_edges
                WHERE source_id IN ({placeholders})
                   OR target_id IN ({placeholders})""",
            id_list + id_list,
        )
        edge_rows = await cur.fetchall()
        edges = [
            _row_to_edge(r)
            for r in edge_rows
            if r["source_id"] in node_ids and r["target_id"] in node_ids
        ]

        return GraphSubgraphResponse(nodes=nodes, edges=edges)
    finally:
        await db.close()


@router.get("/pending", response_model=GraphPendingResponse)
async def get_pending_nodes():
    """Return all pending nodes plus the confirmed nodes they may duplicate."""
    db = await get_db(read_only=True)
    try:
        cur = await db.execute(
            "SELECT * FROM graph_nodes WHERE status = 'pending' ORDER BY created_at DESC"
        )
        node_rows = await cur.fetchall()
        nodes = [_row_to_node(r) for r in node_rows]

        candidate_ids = list({
            r["possible_duplicate_of"]
            for r in node_rows
            if r["possible_duplicate_of"]
        })
        candidates: list[GraphNode] = []
        if candidate_ids:
            placeholders = ",".join("?" for _ in candidate_ids)
            cur = await db.execute(
                f"SELECT * FROM graph_nodes WHERE id IN ({placeholders})",
                candidate_ids,
            )
            candidates = [_row_to_node(r) for r in await cur.fetchall()]

        return GraphPendingResponse(nodes=nodes, candidates=candidates)
    finally:
        await db.close()


@router.patch("/node/{node_id}/confirm", response_model=GraphNode)
async def confirm_pending_node(node_id: str):
    """Confirm a pending node: set status='confirmed' and clear possible_duplicate_of."""
    db = await get_db()
    try:
        cur = await db.execute(
            "SELECT id, status FROM graph_nodes WHERE id = ?", (node_id,)
        )
        row = await cur.fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="节点不存在")
        if row["status"] != "pending":
            raise HTTPException(
                status_code=422,
                detail=f"节点当前 status='{row['status']}'，只能确认 pending 节点",
            )

        cur = await db.execute(
            """UPDATE graph_nodes
               SET status = 'confirmed',
                   possible_duplicate_of = NULL
               WHERE id = ?
               RETURNING *""",
            (node_id,),
        )
        updated = await cur.fetchone()
        await db.commit()
        return _row_to_node(updated)
    except HTTPException:
        raise
    except Exception:
        await db.rollback()
        logger.exception("confirm_pending_node failed id=%s", node_id)
        raise HTTPException(status_code=500, detail="确认节点失败")
    finally:
        await db.close()


@router.patch("/node/{node_id}/reject")
async def reject_pending_node(node_id: str):
    """Delete a pending node and its edges (treat it as a duplicate to be discarded)."""
    db = await get_db()
    try:
        cur = await db.execute(
            "SELECT id, status FROM graph_nodes WHERE id = ?", (node_id,)
        )
        row = await cur.fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="节点不存在")
        if row["status"] != "pending":
            raise HTTPException(
                status_code=422,
                detail=f"节点当前 status='{row['status']}'，只能拒绝 pending 节点",
            )

        # Delete edges first to satisfy FK constraints.
        await db.execute(
            "DELETE FROM graph_edges WHERE source_id = ? OR target_id = ?",
            (node_id, node_id),
        )
        await db.execute("DELETE FROM graph_nodes WHERE id = ?", (node_id,))
        await db.commit()
        return {"ok": True, "deleted_id": node_id}
    except HTTPException:
        raise
    except Exception:
        await db.rollback()
        logger.exception("reject_pending_node failed id=%s", node_id)
        raise HTTPException(status_code=500, detail="拒绝节点失败")
    finally:
        await db.close()


@router.get("/node/{node_id}", response_model=GraphNodeDetailResponse)
async def get_node_detail(node_id: str):
    db = await get_db(read_only=True)
    try:
        cur = await db.execute(
            "SELECT * FROM graph_nodes WHERE id = ?", (node_id,)
        )
        row = await cur.fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="节点不存在")
        node = _row_to_node(row)

        cur = await db.execute(
            """SELECT * FROM graph_edges
               WHERE source_id = ? OR target_id = ?""",
            (node_id, node_id),
        )
        edge_rows = await cur.fetchall()

        peer_ids: set[str] = set()
        for r in edge_rows:
            peer_ids.add(r["source_id"])
            peer_ids.add(r["target_id"])
        peer_ids.discard(node_id)

        label_map: dict[str, str] = {node_id: node.label}
        if peer_ids:
            ph = ",".join("?" for _ in peer_ids)
            cur = await db.execute(
                f"SELECT id, label FROM graph_nodes WHERE id IN ({ph})",
                list(peer_ids),
            )
            for pr in await cur.fetchall():
                label_map[pr["id"]] = pr["label"]

        edges: list[GraphEdgeWithLabels] = []
        for r in edge_rows:
            base = _row_to_edge(r)
            e = GraphEdgeWithLabels(
                **base.model_dump(),
                source_label=label_map.get(r["source_id"]),
                target_label=label_map.get(r["target_id"]),
            )
            edges.append(e)

        memory_ids: set[str] = set()
        for e in edge_rows:
            mid = e["source_memory_id"]
            if mid:
                memory_ids.add(mid)

        memories: list[MemoryBrief] = []
        if memory_ids:
            placeholders = ",".join("?" for _ in memory_ids)
            cur = await db.execute(
                f"SELECT id, content, created_at FROM memory_items WHERE id IN ({placeholders})",
                list(memory_ids),
            )
            for mr in await cur.fetchall():
                memories.append(
                    MemoryBrief(id=mr["id"], content=mr["content"], created_at=mr["created_at"])
                )

        return GraphNodeDetailResponse(node=node, edges=edges, source_memories=memories)
    finally:
        await db.close()
