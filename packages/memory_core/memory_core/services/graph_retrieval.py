"""Graph context retrieval — query → graph triples → natural-language string.

This module owns the public ``retrieve_graph_context`` function called by
``routers/chat.py`` to inject knowledge-graph context into the LLM prompt.

It is intentionally separate from the memory-recall retrieval layer
(retrieval_primitives.py / retrieval_context.py) because:
  - Graph retrieval is a parallel concept to memory recall, not subordinate.
  - Future extensions (multi-hop reasoning, graph summaries) should live here
    rather than inflating the already-large retrieval_primitives module.

Public API:
  retrieve_graph_context(query, db) → str   semicolon-joined graph triples
"""

import logging
import re

from memory_core.services.retrieval_constants import GRAPH_STOPWORDS, MAX_GRAPH_CONTEXT_EDGES, MAX_LAYER_RESULTS

logger = logging.getLogger("uvicorn.error")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_graph_terms(query: str) -> tuple[list[str], list[str]]:
    """Extract entity terms and node-type hints from a user query.

    Returns:
        terms      — candidate graph-node label fragments (de-duplicated)
        type_hints — node type strings inferred from keyword signals
    """
    raw_terms = re.findall(r"[A-Za-z][A-Za-z0-9._-]{1,31}|[\u4e00-\u9fff]{2,6}", query)
    terms: list[str] = []
    seen_terms: set[str] = set()

    for raw_term in raw_terms:
        term = raw_term.strip()
        if not term or term in GRAPH_STOPWORDS:
            continue
        if re.search(r"[什么谁吗呢呀吧啊哦哈咯]", term) and len(term) <= 3:
            continue
        if term not in seen_terms:
            seen_terms.add(term)
            terms.append(term)

    type_hints: list[str] = []
    if "项目" in query:
        type_hints.append("project")
    if "任务" in query or "待办" in query or "TODO" in query.upper():
        type_hints.append("task")
    if "想法" in query or "主意" in query or "思路" in query:
        type_hints.append("idea")
    if "谁" in query or "人物" in query or "我是谁" in query:
        type_hints.append("person")
    if "事件" in query or "里程碑" in query or "活动" in query or "会议" in query:
        type_hints.append("event")

    return terms[:MAX_LAYER_RESULTS], list(dict.fromkeys(type_hints))


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


async def retrieve_graph_context(query: str, db) -> str:
    """Return graph triples relevant to *query* as a semicolon-joined string.

    Steps:
      1. Extract entity terms and type hints from the query.
      2. Match confirmed graph nodes by label similarity.
      3. Fall back to type-hint node lookup when no label matches are found.
      4. Fetch edges incident to matched nodes; format as "A -> rel -> B" triples.
    """
    search_terms, type_hints = _extract_graph_terms(query)
    node_ids: list[str] = []

    for term in search_terms:
        cursor = await db.execute(
            """SELECT id
               FROM graph_nodes
               WHERE label LIKE ?
                 AND status = 'confirmed'
               ORDER BY last_seen_at DESC
               LIMIT ?""",
            (f"%{term}%", 3),
        )
        rows = await cursor.fetchall()
        for row in rows:
            node_id = row["id"]
            if node_id not in node_ids:
                node_ids.append(node_id)
        if len(node_ids) >= MAX_LAYER_RESULTS:
            break

    if not node_ids and type_hints:
        for node_type in type_hints:
            cursor = await db.execute(
                """SELECT id
                   FROM graph_nodes
                   WHERE type = ?
                     AND status = 'confirmed'
                   ORDER BY last_seen_at DESC
                   LIMIT ?""",
                (node_type, 3),
            )
            rows = await cursor.fetchall()
            for row in rows:
                node_id = row["id"]
                if node_id not in node_ids:
                    node_ids.append(node_id)
            if len(node_ids) >= MAX_LAYER_RESULTS:
                break

    if not node_ids:
        return ""

    placeholders = ",".join("?" for _ in node_ids)
    cursor = await db.execute(
        f"""SELECT s.label AS source_label,
                   e.relation AS relation,
                   t.label AS target_label
            FROM graph_edges e
            JOIN graph_nodes s ON s.id = e.source_id AND s.status = 'confirmed'
            JOIN graph_nodes t ON t.id = e.target_id AND t.status = 'confirmed'
            LEFT JOIN memory_items m ON m.id = e.source_memory_id
            WHERE (e.source_id IN ({placeholders}) OR e.target_id IN ({placeholders}))
              AND (
                  e.source_memory_id IS NULL
                  OR COALESCE(m.admission_tier, 'standard') = 'standard'
              )
            ORDER BY e.created_at DESC
            LIMIT ?""",
        node_ids + node_ids + [MAX_GRAPH_CONTEXT_EDGES],
    )
    rows = await cursor.fetchall()

    triples: list[str] = []
    seen_triples: set[str] = set()
    for row in rows:
        triple = f'{row["source_label"]} -> {row["relation"]} -> {row["target_label"]}'
        if triple in seen_triples:
            continue
        seen_triples.add(triple)
        triples.append(triple)

    return "; ".join(triples)
