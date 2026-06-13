"""证据链服务：回答「你为什么这么认为？」。

把库里已有的溯源资产翻到台面上：
- user_persona.source_memory_ids   人设特征 ← 来源记忆
- structured_facts.memory_id       结构化事实 ← 来源记忆（规则法，无 LLM 也有）
- memory_reflection.source_memory_ids 洞察 ← 来源记忆
- memory_admission_log             每条记忆的五维收录打分
- preference_revision_log          人设修正审计日志

全部确定性 SQL，不调用 LLM。
"""

from __future__ import annotations

import json
import logging
from typing import Any

from memory_core.services.memory_policy import normalize_query_key

logger = logging.getLogger(__name__)

_MAX_BELIEFS_PER_TYPE = 5
_MAX_EVIDENCE_PER_BELIEF = 5
_MAX_SUPPORTING_MEMORIES = 5


def _like_terms(query: str) -> list[str]:
    """生成 LIKE 匹配词：原查询 + 规范化形态（与检索层 fallback 同源策略）。"""
    terms = []
    raw = query.strip()
    if raw:
        terms.append(raw)
    normalized = normalize_query_key(raw)
    if normalized and normalized not in terms:
        terms.append(normalized)
    return terms


def _load_id_list(value: str | None) -> list[str]:
    if not value:
        return []
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return []
    return [str(x) for x in parsed if x] if isinstance(parsed, list) else []


async def _fetch_memory_evidence(db, memory_ids: list[str]) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for memory_id in memory_ids[:_MAX_EVIDENCE_PER_BELIEF]:
        cursor = await db.execute(
            """SELECT id, content, kind, weight, importance, created_at
               FROM memory_items WHERE id = ?""",
            (memory_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            evidence.append({"memory_id": memory_id, "missing": True})
            continue
        admission_cursor = await db.execute(
            """SELECT tier, total_score, score_utility, score_confidence,
                      score_novelty, score_recency, created_at
               FROM memory_admission_log
               WHERE memory_id = ?
               ORDER BY created_at DESC LIMIT 1""",
            (memory_id,),
        )
        admission_row = await admission_cursor.fetchone()
        evidence.append(
            {
                "memory_id": row["id"],
                "content": row["content"],
                "kind": row["kind"],
                "weight": row["weight"],
                "importance": row["importance"],
                "created_at": row["created_at"],
                "admission": dict(admission_row) if admission_row else None,
            }
        )
    return evidence


async def _persona_beliefs(db, terms: list[str]) -> list[dict[str, Any]]:
    clause = " OR ".join(["trait_value LIKE ? OR trait_key LIKE ?"] * len(terms))
    params: list[str] = []
    for term in terms:
        params.extend([f"%{term}%", f"%{term}%"])
    cursor = await db.execute(
        f"""SELECT id, trait_key, trait_value, confidence, evidence_count,
                   source_memory_ids, last_updated
            FROM user_persona WHERE {clause}
            ORDER BY confidence DESC LIMIT {_MAX_BELIEFS_PER_TYPE}""",
        params,
    )
    beliefs = []
    for row in await cursor.fetchall():
        revisions_cursor = await db.execute(
            """SELECT old_value, new_value, trigger, created_at
               FROM preference_revision_log
               WHERE persona_id = ? ORDER BY created_at DESC LIMIT 5""",
            (row["id"],),
        )
        revisions = [dict(r) for r in await revisions_cursor.fetchall()]
        beliefs.append(
            {
                "type": "persona",
                "statement": f"{row['trait_key']}: {row['trait_value']}",
                "confidence": row["confidence"],
                "evidence_count": row["evidence_count"],
                "last_updated": row["last_updated"],
                "evidence": await _fetch_memory_evidence(
                    db, _load_id_list(row["source_memory_ids"])
                ),
                "revisions": revisions,
            }
        )
    return beliefs


async def _fact_beliefs(db, terms: list[str]) -> list[dict[str, Any]]:
    clause = " OR ".join(
        ["subject LIKE ? OR predicate LIKE ? OR object LIKE ?"] * len(terms)
    )
    params: list[str] = []
    for term in terms:
        params.extend([f"%{term}%"] * 3)
    cursor = await db.execute(
        f"""SELECT id, memory_id, kind, subject, predicate, object, created_at
            FROM structured_facts
            WHERE status = 'accepted' AND ({clause})
            ORDER BY created_at DESC LIMIT {_MAX_BELIEFS_PER_TYPE}""",
        params,
    )
    beliefs = []
    for row in await cursor.fetchall():
        beliefs.append(
            {
                "type": "fact",
                "statement": f"{row['subject']} {row['predicate']} {row['object']}".strip(),
                "kind": row["kind"],
                "created_at": row["created_at"],
                "evidence": await _fetch_memory_evidence(db, [row["memory_id"]]),
            }
        )
    return beliefs


async def _reflection_beliefs(db, terms: list[str]) -> list[dict[str, Any]]:
    clause = " OR ".join(["insight LIKE ?"] * len(terms))
    params = [f"%{term}%" for term in terms]
    cursor = await db.execute(
        f"""SELECT id, insight, importance, source_memory_ids, created_at
            FROM memory_reflection WHERE {clause}
            ORDER BY importance DESC LIMIT {_MAX_BELIEFS_PER_TYPE}""",
        params,
    )
    beliefs = []
    for row in await cursor.fetchall():
        beliefs.append(
            {
                "type": "reflection",
                "statement": row["insight"],
                "importance": row["importance"],
                "created_at": row["created_at"],
                "evidence": await _fetch_memory_evidence(
                    db, _load_id_list(row["source_memory_ids"])
                ),
            }
        )
    return beliefs


async def _supporting_from_retrieval(query: str, db, retriever) -> tuple[
    list[dict[str, Any]], dict[str, int]
]:
    """用**真实混合检索**（FTS/向量/图谱/意图路由）找支撑记忆，并标注召回通道。

    这是「可对账」招牌的关键：证据来源必须与 retrieve_context 真正召回的一致，
    否则会出现「向量/图谱召回到了、`/why` 却用 content LIKE 解释不出」的洞——
    同义改写、跨语言、图谱关联的记忆此前会被旧的子串匹配漏掉。
    """
    results = await retriever(query, db)
    supporting: list[dict[str, Any]] = []
    channels: dict[str, int] = {}
    seen: set[str] = set()
    for item in results[: _MAX_SUPPORTING_MEMORIES]:
        channel = item.get("_source", "unknown")
        channels[channel] = channels.get(channel, 0) + 1
        memory_id = item.get("id")
        if memory_id and memory_id not in seen:
            seen.add(memory_id)
            evidence = await _fetch_memory_evidence(db, [memory_id])
            entry = evidence[0] if evidence else {"memory_id": memory_id}
        else:
            # 非 memory_items 来源（如结构化事实/图谱）或无 id：直接用检索内容
            entry = {"memory_id": memory_id, "content": item.get("content")}
        entry["retrieval_channel"] = channel
        supporting.append(entry)
    return supporting, channels


async def explain_belief(query: str, db, *, retriever=None) -> dict[str, Any]:
    """组装「为什么相信 X」的完整证据链。

    retriever 可注入（测试用）；默认使用内核真实混合检索 retrieve_context，
    保证证据来源与系统实际召回一致。
    """
    terms = _like_terms(query)
    if not terms:
        return {
            "query": query,
            "beliefs": [],
            "supporting_memories": [],
            "retrieval_channels": {},
        }

    if retriever is None:
        # 延迟导入避免 service 间环依赖
        from memory_core.services.retrieval_context import retrieve_context

        retriever = retrieve_context

    beliefs = (
        await _persona_beliefs(db, terms)
        + await _fact_beliefs(db, terms)
        + await _reflection_beliefs(db, terms)
    )
    supporting, channels = await _supporting_from_retrieval(query, db, retriever)

    logger.info(
        "provenance query_len=%d beliefs=%d supporting=%d channels=%s",
        len(query),
        len(beliefs),
        len(supporting),
        channels,
    )
    return {
        "query": query,
        "beliefs": beliefs,
        "supporting_memories": supporting,
        "retrieval_channels": channels,
    }
