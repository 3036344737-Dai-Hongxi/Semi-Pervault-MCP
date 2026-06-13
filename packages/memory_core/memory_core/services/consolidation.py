"""Offline consolidation service for Layer 1 -> Layer 2/3 promotion.

This service scans unconsolidated episodic memories, asks the LLM to
route each memory into one of three outcomes:

- fact: promote to structured_facts with ADD / UPDATE / NOOP semantics
- graph: promote only high-value person / project / event graph items
- noop: keep the memory in Layer 1 and only mark it as consolidated

The whole flow is designed to run asynchronously and periodically,
never blocking the main chat path.
"""

import asyncio
import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from typing import Literal

from memory_core.database import get_db
from memory_core.services.graph_extract import extract_consolidation_graph
from memory_core.services.graph_pipeline import persist_graph
from memory_core.services.llm import get_client, raise_ai_service_error
from memory_core.services.logging_policy import maybe_sensitive_preview
from memory_core.services.memory_policy import (
    normalize_text,
    fact_supported_kinds,
    should_update_memory_kind,
    task_fact_is_stable_long_term,
    CONSOLIDATION_NODE_TYPES,
    ALLOWED_RELATIONS,
)

logger = logging.getLogger(__name__)

DEFAULT_BATCH_LIMIT = 50
DEFAULT_LLM_CONCURRENCY = int(os.getenv("CONSOLIDATION_LLM_CONCURRENCY", "5"))
DEFAULT_CONSOLIDATION_INTERVAL_SECONDS = int(
    os.getenv("CONSOLIDATION_INTERVAL_SECONDS", "14400")
)
DEFAULT_CONSOLIDATION_STARTUP_DELAY_SECONDS = int(
    os.getenv("CONSOLIDATION_STARTUP_DELAY_SECONDS", "120")
)


ConsolidationRoute = Literal["fact", "graph", "noop"]
FactAction = Literal["add", "update", "noop"]
PlannedAction = Literal[
    "fact_add",
    "fact_update",
    "fact_noop",
    "graph_add",
    "graph_noop",
    "noop_mark",
]

CONSOLIDATION_DECISION_PROMPT = """\
你是 Pervault 的离线记忆整合器。你的任务是把一条情节记忆判断为以下三种路由之一：

1. fact
- 这条记忆包含适合进入 structured_facts 的稳定信息
- 包括：偏好、习惯、长期事实、可复用的人际事实、项目状态

2. graph
- 这条记忆更适合进入关系图谱
- 只在存在高价值 person / project / event 关系时选择 graph
- 不要把 task、idea、情绪、碎片信息送进 graph

3. noop
- 这条记忆不值得升级到长期层
- 例如日常碎片、瞬时感受、闲聊、噪音、只有当下价值的片段

如果选择 fact，必须返回一个规范化 fact：
- kind 只允许：project_update / preference / relationship_event / fact / task
- subject 要简洁稳定，例如 user、Pervault、小王
- predicate 要简洁稳定，例如 likes、works_on、status_update、met_with
- object 是该事实当前值

严格返回 JSON：
{
  "route": "fact|graph|noop",
  "reason": "简短原因",
  "fact": {
    "kind": "project_update|preference|relationship_event|fact|task",
    "subject": "string",
    "predicate": "string",
    "object": "string"
  }
}

规则：
- route=noop 时 fact 必须为 null
- route=graph 时 fact 必须为 null
- 不要返回多条 fact，v1 只保留一条最核心 fact
- 如果 memory_kind_hint=task，默认选择 noop，不要选择 graph
- 只有当 task 已经明确表达为长期习惯、长期偏好、长期身份/状态事实时，才允许选择 fact
- 例如“我一直在控制饮食”“我长期坚持早起跑步”“我一贯偏好高蛋白饮食”“我现在在长期减脂阶段”才可以进入 fact
- 例如“开始减肥”“下周开始运动”“我要少吃点”“计划控制饮食”一律不要进入 fact
- 只基于输入文本，不要编造
- 只返回 JSON，不要有其它文字"""


@dataclass
class ConsolidationDecision:
    route: ConsolidationRoute
    reason: str = ""
    fact: dict[str, str] | None = None


@dataclass
class FactPlan:
    action: FactAction
    match_key: str
    target_id: str | None = None
    target_status: str | None = None
    next_object: str | None = None
    existing_object: str | None = None


@dataclass
class GraphPlan:
    action: Literal["add", "noop"]
    nodes: list[dict[str, str]] = field(default_factory=list)
    edges: list[dict[str, str]] = field(default_factory=list)
    candidate_types: list[str] = field(default_factory=list)


@dataclass
class MemoryReview:
    memory_id: str
    original_kind: str
    route_decision: ConsolidationRoute
    planned_action: PlannedAction
    structured_fact_match_key: str | None = None
    graph_candidate_types: list[str] = field(default_factory=list)
    short_reason: str = ""
    error: str | None = None


@dataclass
class ConsolidationResult:
    scanned_count: int = 0
    processed: int = 0
    fact_count: int = 0
    graph_count: int = 0
    noop_count: int = 0
    fact_added: int = 0
    fact_updated: int = 0
    fact_noop: int = 0
    graph_added: int = 0
    graph_noop: int = 0
    noop_marked_candidate: int = 0
    errors: list[str] = field(default_factory=list)
    kind_distribution: dict[str, int] = field(default_factory=dict)
    route_distribution: dict[str, int] = field(default_factory=dict)
    processed_ids: list[str] = field(default_factory=list)
    reviews: list[MemoryReview] = field(default_factory=list)

    @property
    def scanned(self) -> int:
        return self.scanned_count

    @property
    def error_count(self) -> int:
        return len(self.errors)


def _normalize_fact_payload(raw_fact: dict | None) -> dict[str, str] | None:
    if not isinstance(raw_fact, dict):
        return None

    fact_kind = normalize_text(str(raw_fact.get("kind", "")).lower())
    subject = normalize_text(str(raw_fact.get("subject", "")))
    predicate = normalize_text(str(raw_fact.get("predicate", "")).lower())
    object_value = normalize_text(str(raw_fact.get("object", "")))

    if fact_kind not in fact_supported_kinds(include_task=True):
        return None
    if not (subject and predicate and object_value):
        return None

    return {
        "kind": fact_kind,
        "subject": subject,
        "predicate": predicate,
        "object": object_value,
    }


def _apply_task_route_guard(
    *,
    original_kind: str,
    content: str,
    decision: ConsolidationDecision,
) -> ConsolidationDecision:
    if original_kind != "task":
        return decision

    if decision.route != "fact":
        return ConsolidationDecision(
            route="noop",
            reason=decision.reason or "task_default_layer1",
        )

    if decision.fact is None or not task_fact_is_stable_long_term(content, decision.fact["kind"]):
        return ConsolidationDecision(
            route="noop",
            reason=decision.reason or "task_not_long_term_enough",
        )

    return decision


async def _decide_consolidation(content: str, kind: str) -> ConsolidationDecision:
    client = get_client()
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")

    try:
        resp = await client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {"role": "system", "content": CONSOLIDATION_DECISION_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "memory_kind_hint": kind or "other",
                            "content": content,
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            response_format={"type": "json_object"},
        )
    except Exception as exc:
        raise_ai_service_error(exc, "离线整合决策服务")

    raw = resp.choices[0].message.content or "{}"
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning(
            "consolidation decision returned invalid json preview=%s",
            maybe_sensitive_preview(raw, limit=200),
        )
        return ConsolidationDecision(route="noop", reason="invalid_json")

    route = str(parsed.get("route", "")).strip().lower()
    reason = normalize_text(str(parsed.get("reason", "")))
    if route not in {"fact", "graph", "noop"}:
        return ConsolidationDecision(route="noop", reason=reason or "invalid_route")

    fact = _normalize_fact_payload(parsed.get("fact"))
    if route == "fact" and fact is None:
        return ConsolidationDecision(route="noop", reason=reason or "invalid_fact_payload")

    return _apply_task_route_guard(
        original_kind=kind,
        content=content,
        decision=ConsolidationDecision(route=route, reason=reason, fact=fact),
    )


async def _load_unconsolidated_memories(limit: int) -> list:
    db = await get_db(read_only=True)
    try:
        cursor = await db.execute(
            """SELECT id, content, kind
               FROM memory_items
               WHERE consolidated = 0
                 AND content IS NOT NULL
                 AND TRIM(content) != ''
                 AND COALESCE(admission_tier, 'standard') = 'standard'
               ORDER BY created_at ASC
               LIMIT ?""",
            (limit,),
        )
        return await cursor.fetchall()
    finally:
        await db.close()


async def _mark_memory_consolidated(db, memory_id: str) -> None:
    await db.execute(
        "UPDATE memory_items SET consolidated = 1 WHERE id = ?",
        (memory_id,),
    )


async def _collapse_duplicate_facts(
    db,
    *,
    keep_id: str,
    kind: str,
    subject: str,
    predicate: str,
) -> None:
    await db.execute(
        """UPDATE structured_facts
           SET status = 'superseded'
           WHERE kind = ?
             AND subject = ?
             AND predicate = ?
             AND status = 'accepted'
             AND id != ?""",
        (kind, subject, predicate, keep_id),
    )


async def _plan_structured_fact(
    db,
    *,
    fact: dict[str, str],
) -> FactPlan:
    match_key = f'{fact["kind"]}|{fact["subject"]}|{fact["predicate"]}'
    cursor = await db.execute(
        """SELECT id, object, status
           FROM structured_facts
           WHERE kind = ?
             AND subject = ?
             AND predicate = ?
           ORDER BY CASE WHEN status = 'accepted' THEN 0 ELSE 1 END,
                    created_at DESC,
                    id DESC""",
        (fact["kind"], fact["subject"], fact["predicate"]),
    )
    rows = await cursor.fetchall()

    if not rows:
        return FactPlan(
            action="add",
            match_key=match_key,
            next_object=fact["object"],
        )

    next_object = normalize_text(fact["object"])
    accepted_rows = [row for row in rows if row["status"] == "accepted"]
    exact_row = next(
        (
            row
            for row in rows
            if normalize_text(str(row["object"] or "")) == next_object
        ),
        None,
    )

    if exact_row is not None:
        return FactPlan(
            action="noop" if exact_row["status"] == "accepted" else "update",
            match_key=match_key,
            target_id=exact_row["id"],
            target_status=exact_row["status"],
            next_object=next_object,
            existing_object=normalize_text(str(exact_row["object"] or "")),
        )

    target_row = accepted_rows[0] if accepted_rows else rows[0]
    return FactPlan(
        action="update",
        match_key=match_key,
        target_id=target_row["id"],
        target_status=target_row["status"],
        next_object=next_object,
        existing_object=normalize_text(str(target_row["object"] or "")),
    )


async def _apply_structured_fact_plan(
    db,
    *,
    memory_id: str,
    fact: dict[str, str],
    plan: FactPlan,
) -> FactAction:
    if plan.action == "add":
        fact_id = str(uuid.uuid4())
        await db.execute(
            """INSERT INTO structured_facts
               (id, memory_id, kind, subject, predicate, object, status)
               VALUES (?, ?, ?, ?, ?, ?, 'accepted')""",
            (
                fact_id,
                memory_id,
                fact["kind"],
                fact["subject"],
                fact["predicate"],
                fact["object"],
            ),
        )
        return "add"

    if plan.target_id is None:
        return "noop"

    # Even for a noop (same object value, no update needed), refresh created_at so
    # the fact is treated as recently confirmed, and collapse any duplicate rows for
    # the same (kind, subject, predicate) key that may have accumulated over time.
    await db.execute(
        """UPDATE structured_facts
           SET status = 'accepted',
               created_at = datetime('now')
           WHERE id = ?""",
        (plan.target_id,),
    )
    await _collapse_duplicate_facts(
        db,
        keep_id=plan.target_id,
        kind=fact["kind"],
        subject=fact["subject"],
        predicate=fact["predicate"],
    )

    if plan.action == "noop":
        return "noop"

    await db.execute(
        """UPDATE structured_facts
           SET object = ?,
               status = 'accepted',
               created_at = datetime('now')
           WHERE id = ?""",
        (plan.next_object, plan.target_id),
    )
    return "update"


async def _plan_graph_promotion(content: str) -> GraphPlan:
    extracted = await extract_consolidation_graph(content)
    nodes = [
        node for node in extracted.get("nodes", []) if node.get("type") in CONSOLIDATION_NODE_TYPES
    ]
    edges = [
        edge
        for edge in extracted.get("edges", [])
        if edge.get("relation") in ALLOWED_RELATIONS
    ]

    candidate_types = sorted({node.get("type", "") for node in nodes if node.get("type")})
    if not nodes:
        return GraphPlan(action="noop", candidate_types=candidate_types)

    return GraphPlan(
        action="add",
        nodes=nodes,
        edges=edges,
        candidate_types=candidate_types,
    )


async def _apply_graph_plan(
    db,
    *,
    memory_id: str,
    plan: GraphPlan,
) -> Literal["add", "noop"]:
    if plan.action == "noop" or not plan.nodes:
        return "noop"

    persisted_nodes, persisted_edges = await persist_graph(
        db,
        plan.nodes,
        plan.edges,
        memory_id,
    )
    if persisted_nodes or persisted_edges:
        return "add"
    return "noop"


def _build_memory_review(
    *,
    memory_id: str,
    kind: str,
    decision: ConsolidationDecision,
    planned_action: PlannedAction,
    fact_plan: FactPlan | None = None,
    graph_plan: GraphPlan | None = None,
    error: str | None = None,
) -> MemoryReview:
    return MemoryReview(
        memory_id=memory_id,
        original_kind=kind,
        route_decision=decision.route,
        planned_action=planned_action,
        structured_fact_match_key=fact_plan.match_key if fact_plan else None,
        graph_candidate_types=graph_plan.candidate_types if graph_plan else [],
        short_reason=decision.reason,
        error=error,
    )


def _record_route_count(result: ConsolidationResult, route: ConsolidationRoute) -> None:
    result.route_distribution[route] = result.route_distribution.get(route, 0) + 1
    if route == "fact":
        result.fact_count += 1
    elif route == "graph":
        result.graph_count += 1
    else:
        result.noop_count += 1


async def _decide_for_row(row) -> tuple[ConsolidationDecision, GraphPlan | None]:
    """LLM-only phase: no DB access. Raises on any LLM failure."""
    content = row["content"] or ""
    kind = row["kind"] or "other"
    decision = await asyncio.wait_for(
        _decide_consolidation(content, kind),
        timeout=45.0,
    )
    graph_plan: GraphPlan | None = None
    if decision.route == "graph":
        graph_plan = await asyncio.wait_for(
            _plan_graph_promotion(content),
            timeout=45.0,
        )
    return decision, graph_plan


async def _apply_memory_decision(
    row,
    decision: ConsolidationDecision,
    graph_plan: GraphPlan | None,
    result: ConsolidationResult,
    db,
    *,
    dry_run: bool,
) -> None:
    """DB-only phase: apply a pre-computed decision using a shared DB connection.

    Does not open or close the connection. Commits per-memory when not dry_run.
    Rolls back only the uncommitted writes of the current memory on failure.
    """
    memory_id = row["id"]
    kind = row["kind"] or "other"

    try:
        review: MemoryReview
        if decision.route == "fact" and decision.fact is not None:
            fact_plan = await _plan_structured_fact(db, fact=decision.fact)
            action = fact_plan.action
            if not dry_run:
                action = await _apply_structured_fact_plan(
                    db,
                    memory_id=memory_id,
                    fact=decision.fact,
                    plan=fact_plan,
                )

            if action == "add":
                result.fact_added += 1
                planned_action: PlannedAction = "fact_add"
            elif action == "update":
                result.fact_updated += 1
                planned_action = "fact_update"
            else:
                result.fact_noop += 1
                planned_action = "fact_noop"

            # Backwrite memory_items.kind when the LLM determined a more
            # accurate kind than the keyword classifier.  Only runs when
            # should_update_memory_kind() returns True (see memory_policy.py for
            # the exact rules: new_kind must differ, must not be "task", and
            # must be a valid _BASE_FACT_KINDS value).
            new_kind = decision.fact["kind"]
            if not dry_run and should_update_memory_kind(kind, new_kind):
                await db.execute(
                    "UPDATE memory_items SET kind = ? WHERE id = ?",
                    (new_kind, memory_id),
                )
                logger.info(
                    "consolidation kind corrected memory_id=%s original_kind=%s new_kind=%s",
                    memory_id,
                    kind,
                    new_kind,
                )

            review = _build_memory_review(
                memory_id=memory_id,
                kind=kind,
                decision=decision,
                planned_action=planned_action,
                fact_plan=fact_plan,
            )
        elif decision.route == "graph":
            effective_graph_plan = graph_plan if graph_plan is not None else GraphPlan(action="noop")
            graph_action = effective_graph_plan.action
            if not dry_run:
                graph_action = await _apply_graph_plan(
                    db,
                    memory_id=memory_id,
                    plan=effective_graph_plan,
                )

            if graph_action == "add":
                result.graph_added += 1
                planned_action = "graph_add"
            else:
                result.graph_noop += 1
                planned_action = "graph_noop"
            review = _build_memory_review(
                memory_id=memory_id,
                kind=kind,
                decision=decision,
                planned_action=planned_action,
                graph_plan=effective_graph_plan,
            )
        else:
            result.noop_marked_candidate += 1
            review = _build_memory_review(
                memory_id=memory_id,
                kind=kind,
                decision=decision,
                planned_action="noop_mark",
            )

        if not dry_run:
            await _mark_memory_consolidated(db, memory_id)
            await db.commit()

        result.processed += 1
        result.processed_ids.append(memory_id)
        result.reviews.append(review)
        logger.info(
            "consolidation processed memory_id=%s kind=%s route=%s action=%s dry_run=%s reason=%s",
            memory_id,
            kind,
            decision.route,
            review.planned_action,
            dry_run,
            decision.reason,
        )
    except Exception:
        if not dry_run:
            await db.rollback()
        raise


async def run_once(
    *,
    limit: int = DEFAULT_BATCH_LIMIT,
    dry_run: bool = False,
) -> ConsolidationResult:
    """Run a single consolidation pass over unconsolidated memories.

    Phase 1 – concurrent LLM decisions: up to CONSOLIDATION_LLM_CONCURRENCY
    parallel requests; each failure is captured per-row and does not abort
    the batch.

    Phase 2 – sequential DB writes: a single shared connection is opened for
    the whole batch; each memory is committed (or rolled back) individually so
    one failure cannot corrupt a previously committed row.
    """
    result = ConsolidationResult()
    rows = await _load_unconsolidated_memories(limit)

    result.scanned_count = len(rows)
    if not rows:
        logger.info("consolidation: nothing to process dry_run=%s", dry_run)
        return result

    # ── Phase 1: concurrent LLM decisions (no DB) ─────────────────────────
    sem = asyncio.Semaphore(DEFAULT_LLM_CONCURRENCY)

    async def _decide_with_sem(row):
        async with sem:
            return await _decide_for_row(row)

    decision_results = await asyncio.gather(
        *[_decide_with_sem(row) for row in rows],
        return_exceptions=True,
    )

    # ── Phase 2: sequential DB writes with a single shared connection ──────
    db = await get_db(read_only=dry_run)
    try:
        for row, decision_result in zip(rows, decision_results):
            memory_id = row["id"]
            kind = row["kind"] or "other"
            result.kind_distribution[kind] = result.kind_distribution.get(kind, 0) + 1

            if isinstance(decision_result, BaseException):
                logger.error(
                    "consolidation LLM decision failed memory_id=%s kind=%s dry_run=%s",
                    memory_id,
                    kind,
                    dry_run,
                    exc_info=decision_result,
                )
                result.errors.append(memory_id)
                result.reviews.append(
                    MemoryReview(
                        memory_id=memory_id,
                        original_kind=kind,
                        route_decision="noop",
                        planned_action="noop_mark",
                        short_reason="llm_decision_error",
                        error=str(decision_result),
                    )
                )
                continue

            decision, graph_plan = decision_result
            _record_route_count(result, decision.route)

            try:
                await _apply_memory_decision(
                    row, decision, graph_plan, result, db, dry_run=dry_run
                )
            except Exception as exc:
                logger.exception(
                    "consolidation DB apply failed memory_id=%s kind=%s dry_run=%s",
                    memory_id,
                    kind,
                    dry_run,
                )
                result.errors.append(memory_id)
                result.reviews.append(
                    MemoryReview(
                        memory_id=memory_id,
                        original_kind=kind,
                        route_decision=decision.route,
                        planned_action="noop_mark",
                        short_reason="db_apply_error",
                        error=str(exc),
                    )
                )
    finally:
        await db.close()

    logger.info(
        "consolidation summary: scanned=%s processed=%s fact_count=%s graph_count=%s noop_count=%s fact_added=%s fact_updated=%s fact_noop=%s graph_added=%s graph_noop=%s noop_marked_candidate=%s errors=%s dry_run=%s kind_distribution=%s route_distribution=%s",
        result.scanned_count,
        result.processed,
        result.fact_count,
        result.graph_count,
        result.noop_count,
        result.fact_added,
        result.fact_updated,
        result.fact_noop,
        result.graph_added,
        result.graph_noop,
        result.noop_marked_candidate,
        len(result.errors),
        dry_run,
        result.kind_distribution,
        result.route_distribution,
    )
    return result


async def run_periodically(
    *,
    interval_seconds: int = DEFAULT_CONSOLIDATION_INTERVAL_SECONDS,
    startup_delay_seconds: int = DEFAULT_CONSOLIDATION_STARTUP_DELAY_SECONDS,
    batch_limit: int = DEFAULT_BATCH_LIMIT,
    dry_run: bool = False,
) -> None:
    """Background scheduler loop for offline consolidation."""
    if startup_delay_seconds > 0:
        await asyncio.sleep(startup_delay_seconds)

    while True:
        try:
            await run_once(limit=batch_limit, dry_run=dry_run)
        except asyncio.CancelledError:
            logger.info("consolidation scheduler cancelled")
            raise
        except Exception:
            logger.exception("consolidation scheduler iteration failed")

        await asyncio.sleep(max(interval_seconds, 1))
