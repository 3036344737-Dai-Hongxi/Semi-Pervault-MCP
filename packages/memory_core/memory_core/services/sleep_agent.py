"""Daily long-memory maintenance for Persona and Reflection layers.

Sleep Agent is intentionally separate from consolidation:
consolidation promotes episodic memories into structured facts / graph, while
this service refreshes persona evidence and produces higher-level reflections.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any

from memory_core.database import get_db
from memory_core.services.background_jobs import create_scheduler_run_log, finish_scheduler_run_log
from memory_core.services.llm import get_client
from memory_core.services.logging_policy import maybe_sensitive_preview
from memory_core.services.memory_policy import normalize_query_key
from memory_core.services.persona_service import (
    PERSONA_ELIGIBLE_KINDS,
    PersonaTraitCandidate,
    extract_persona_traits_with_llm,
    upsert_persona_traits,
)

logger = logging.getLogger("uvicorn.error")

DEFAULT_SLEEP_AGENT_INTERVAL_SECONDS = int(
    os.getenv("SLEEP_AGENT_INTERVAL_SECONDS", "86400")
)
DEFAULT_SLEEP_AGENT_STARTUP_DELAY_SECONDS = int(
    os.getenv("SLEEP_AGENT_STARTUP_DELAY_SECONDS", "300")
)
DEFAULT_SLEEP_AGENT_LLM_TIMEOUT_SECONDS = float(
    os.getenv("SLEEP_AGENT_LLM_TIMEOUT_SECONDS", "60")
)
DEFAULT_SLEEP_AGENT_LLM_CONCURRENCY = int(
    os.getenv("SLEEP_AGENT_LLM_CONCURRENCY", "3")
)

TOPIC_MEMORY_LIMIT = 60
PERSONA_REFRESH_MEMORY_LIMIT = 80
REFLECTION_MEMORY_LIMIT = 50
REFLECTION_MIN_TOTAL_IMPORTANCE = 50.0
REFLECTION_MAX_CANDIDATES = 3

SLEEP_AGENT_TOPIC_STAGE = "topic_regroup"
SLEEP_AGENT_PERSONA_STAGE = "persona_refresh"
SLEEP_AGENT_REFLECTION_STAGE = "reflection_generation"


TOPIC_CLUSTER_PROMPT = """你是 Pervault 的 Sleep Agent 主题整理器。

请把输入的近期高价值记忆聚类成 0 到 5 个主题。主题只作为后续长期洞察的中间材料，不要编造输入之外的信息。

严格输出 JSON：
{
  "topics": [
    {
      "title": "简短主题名",
      "summary": "一句话主题摘要",
      "source_memory_ids": ["输入里存在的 memory id"]
    }
  ]
}

规则：
- 只引用输入中存在的 memory id
- 不确定时输出 {"topics": []}
- 不要输出 JSON 之外的任何文字"""


REFLECTION_GENERATION_PROMPT = """你是 Pervault 的长期记忆洞察生成器。

请根据近期高重要性记忆和可选主题，生成 1 到 3 条长期 reflection。Reflection 应该描述跨记忆的稳定趋势、持续目标、反复出现的偏好或值得长期保留的洞察。

严格输出 JSON：
{
  "reflections": [
    {
      "insight": "长期洞察，使用简洁中文",
      "importance": 8.5,
      "source_memory_ids": ["输入里存在的 memory id"]
    }
  ]
}

规则：
- importance 是 1.0 到 10.0
- 只引用输入中存在的 memory id
- 不要总结单条琐碎事件
- 不要编造输入外事实
- 没有足够洞察时输出 {"reflections": []}
- 不要输出 JSON 之外的任何文字"""


@dataclass(frozen=True)
class SleepMemory:
    id: str
    content: str
    kind: str
    importance: float
    created_at: str


@dataclass(frozen=True)
class SleepTopic:
    title: str
    summary: str
    source_memory_ids: list[str]


@dataclass
class SleepAgentResult:
    topic_memory_count: int = 0
    topic_count: int = 0
    persona_memory_count: int = 0
    persona_traits_upserted: int = 0
    reflection_memory_count: int = 0
    reflections_created: int = 0
    skipped_reason: str | None = None
    errors: list[str] = field(default_factory=list)

    @property
    def error_count(self) -> int:
        return len(self.errors)


@dataclass(frozen=True)
class _ReflectionCandidate:
    insight: str
    importance: float
    source_memory_ids: list[str]


def _clamp_float(value: object, *, default: float, lower: float, upper: float) -> float:
    if not isinstance(value, (int, float)):
        return default
    return max(lower, min(upper, float(value)))


def _row_to_sleep_memory(row) -> SleepMemory:
    return SleepMemory(
        id=str(row["id"]),
        content=str(row["content"] or ""),
        kind=str(row["kind"] or "other"),
        importance=_clamp_float(row["importance"], default=5.0, lower=1.0, upper=10.0),
        created_at=str(row["created_at"] or ""),
    )


def _memory_prompt_payload(memories: list[SleepMemory]) -> list[dict[str, Any]]:
    return [
        {
            "id": memory.id,
            "kind": memory.kind,
            "importance": memory.importance,
            "created_at": memory.created_at,
            "content": memory.content[:700],
        }
        for memory in memories
    ]


async def _load_sleep_agent_checkpoint(db, *, stage_name: str) -> str | None:
    cursor = await db.execute(
        """SELECT checkpoint_created_at
           FROM sleep_agent_checkpoint
           WHERE stage_name = ?""",
        (stage_name,),
    )
    row = await cursor.fetchone()
    return str(row["checkpoint_created_at"]) if row and row["checkpoint_created_at"] else None


async def _advance_sleep_agent_checkpoint(
    db,
    *,
    stage_name: str,
    checkpoint_created_at: str,
    run_id: str | None,
) -> None:
    await db.execute(
        """INSERT INTO sleep_agent_checkpoint
           (stage_name, checkpoint_created_at, last_run_id, updated_at)
           VALUES (?, ?, ?, datetime('now'))
           ON CONFLICT(stage_name) DO UPDATE SET
               checkpoint_created_at = excluded.checkpoint_created_at,
               last_run_id = excluded.last_run_id,
               updated_at = datetime('now')""",
        (stage_name, checkpoint_created_at, run_id),
    )
    await db.commit()


def _stage_checkpoint_candidate(memories: list[SleepMemory]) -> str | None:
    created_at_values = [memory.created_at for memory in memories if memory.created_at]
    return max(created_at_values) if created_at_values else None


def _sleep_agent_created_after_clause(
    created_after: str | None,
    *,
    fallback_window_modifier: str,
) -> tuple[str, tuple[object, ...]]:
    if created_after:
        return "AND created_at > ?", (created_after,)
    return f"AND created_at > datetime('now', '{fallback_window_modifier}')", ()


def _normalize_topic_payload(payload: object, allowed_ids: set[str]) -> list[SleepTopic]:
    if not isinstance(payload, dict):
        return []
    raw_topics = payload.get("topics", [])
    if not isinstance(raw_topics, list):
        return []

    topics: list[SleepTopic] = []
    for raw_topic in raw_topics[:5]:
        if not isinstance(raw_topic, dict):
            continue
        title = str(raw_topic.get("title", "")).strip()
        summary = str(raw_topic.get("summary", "")).strip()
        raw_source_ids = raw_topic.get("source_memory_ids", [])
        if not isinstance(raw_source_ids, list):
            continue
        source_ids = [
            str(source_id)
            for source_id in raw_source_ids
            if str(source_id) in allowed_ids
        ]
        if not title or not summary or not source_ids:
            continue
        topics.append(
            SleepTopic(
                title=title[:120],
                summary=summary[:500],
                source_memory_ids=list(dict.fromkeys(source_ids)),
            )
        )
    return topics


def _normalize_reflection_payload(
    payload: object,
    allowed_ids: set[str],
) -> list[_ReflectionCandidate]:
    if not isinstance(payload, dict):
        return []
    raw_reflections = payload.get("reflections", [])
    if not isinstance(raw_reflections, list):
        return []

    candidates: list[_ReflectionCandidate] = []
    for raw_reflection in raw_reflections[:REFLECTION_MAX_CANDIDATES]:
        if not isinstance(raw_reflection, dict):
            continue
        insight = str(raw_reflection.get("insight", "")).strip()
        if len(normalize_query_key(insight)) < 6:
            continue
        raw_source_ids = raw_reflection.get("source_memory_ids", [])
        if not isinstance(raw_source_ids, list):
            continue
        source_ids = [
            str(source_id)
            for source_id in raw_source_ids
            if str(source_id) in allowed_ids
        ]
        source_ids = list(dict.fromkeys(source_ids))
        if not source_ids:
            continue
        importance = _clamp_float(
            raw_reflection.get("importance"),
            default=8.0,
            lower=1.0,
            upper=10.0,
        )
        candidates.append(
            _ReflectionCandidate(
                insight=insight[:700],
                importance=importance,
                source_memory_ids=source_ids,
            )
        )
    return candidates


def _reflection_insight_dedupe_key(insight: str) -> str:
    return normalize_query_key(insight)


def _reflection_source_fingerprint(source_memory_ids: list[str]) -> str:
    normalized_ids = sorted(
        {
            source_id.strip()
            for source_id in source_memory_ids
            if isinstance(source_id, str) and source_id.strip()
        }
    )
    return json.dumps(normalized_ids, ensure_ascii=False, separators=(",", ":"))


async def _load_topic_memories(db, *, created_after: str | None = None) -> list[SleepMemory]:
    created_after_clause, created_after_params = _sleep_agent_created_after_clause(
        created_after,
        fallback_window_modifier="-1 day",
    )
    cursor = await db.execute(
        """SELECT id, content, kind, importance, created_at
           FROM memory_items
           WHERE 1 = 1
             {created_after_clause}
             AND COALESCE(admission_tier, 'standard') = 'standard'
             AND importance >= 6.0
             AND content IS NOT NULL
             AND TRIM(content) != ''
           ORDER BY importance DESC, created_at DESC
           LIMIT ?""".format(created_after_clause=created_after_clause),
        created_after_params + (TOPIC_MEMORY_LIMIT,),
    )
    rows = await cursor.fetchall()
    return [_row_to_sleep_memory(row) for row in rows]


async def _load_persona_refresh_memories(
    db,
    *,
    created_after: str | None = None,
) -> list[SleepMemory]:
    placeholders = ",".join("?" for _ in PERSONA_ELIGIBLE_KINDS)
    created_after_clause, created_after_params = _sleep_agent_created_after_clause(
        created_after,
        fallback_window_modifier="-7 days",
    )
    cursor = await db.execute(
        f"""SELECT id, content, kind, importance, created_at
            FROM memory_items
            WHERE 1 = 1
              {created_after_clause}
              AND COALESCE(admission_tier, 'standard') = 'standard'
              AND importance >= 7.0
              AND kind IN ({placeholders})
              AND content IS NOT NULL
              AND TRIM(content) != ''
            ORDER BY importance DESC, created_at DESC
            LIMIT ?""",
        created_after_params + tuple(PERSONA_ELIGIBLE_KINDS) + (PERSONA_REFRESH_MEMORY_LIMIT,),
    )
    rows = await cursor.fetchall()
    return [_row_to_sleep_memory(row) for row in rows]


async def _load_reflection_memories(
    db,
    *,
    created_after: str | None = None,
) -> list[SleepMemory]:
    created_after_clause, created_after_params = _sleep_agent_created_after_clause(
        created_after,
        fallback_window_modifier="-1 day",
    )
    cursor = await db.execute(
        """SELECT id, content, kind, importance, created_at
           FROM memory_items
           WHERE 1 = 1
             {created_after_clause}
             AND COALESCE(admission_tier, 'standard') = 'standard'
             AND importance >= 7.0
             AND content IS NOT NULL
             AND TRIM(content) != ''
           ORDER BY importance DESC, created_at DESC
           LIMIT ?""".format(created_after_clause=created_after_clause),
        created_after_params + (REFLECTION_MEMORY_LIMIT,),
    )
    rows = await cursor.fetchall()
    return [_row_to_sleep_memory(row) for row in rows]


async def _cluster_topics_with_llm(memories: list[SleepMemory]) -> list[SleepTopic]:
    if not memories:
        return []

    allowed_ids = {memory.id for memory in memories}
    try:
        client = get_client()
        model = os.getenv("LLM_MODEL", "gpt-4o-mini")
        resp = await asyncio.wait_for(
            client.chat.completions.create(
                model=model,
                temperature=0,
                messages=[
                    {"role": "system", "content": TOPIC_CLUSTER_PROMPT},
                    {
                        "role": "user",
                        "content": json.dumps(
                            {"memories": _memory_prompt_payload(memories)},
                            ensure_ascii=False,
                        ),
                    },
                ],
                response_format={"type": "json_object"},
            ),
            timeout=DEFAULT_SLEEP_AGENT_LLM_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        logger.warning("sleep agent topic clustering timed out")
        return []
    except Exception:
        logger.exception("sleep agent topic clustering failed")
        return []

    raw = resp.choices[0].message.content or "{}"
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning(
            "sleep agent topic clustering returned invalid JSON preview=%s",
            maybe_sensitive_preview(raw, limit=200),
        )
        return []
    return _normalize_topic_payload(payload, allowed_ids)


async def _generate_reflections_with_llm(
    memories: list[SleepMemory],
    topics: list[SleepTopic],
) -> list[_ReflectionCandidate]:
    if not memories:
        return []

    allowed_ids = {memory.id for memory in memories}
    topic_payload = [
        {
            "title": topic.title,
            "summary": topic.summary,
            "source_memory_ids": [
                source_id for source_id in topic.source_memory_ids if source_id in allowed_ids
            ],
        }
        for topic in topics
    ]
    try:
        client = get_client()
        model = os.getenv("LLM_MODEL", "gpt-4o-mini")
        resp = await asyncio.wait_for(
            client.chat.completions.create(
                model=model,
                temperature=0,
                messages=[
                    {"role": "system", "content": REFLECTION_GENERATION_PROMPT},
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "memories": _memory_prompt_payload(memories),
                                "topics": topic_payload,
                            },
                            ensure_ascii=False,
                        ),
                    },
                ],
                response_format={"type": "json_object"},
            ),
            timeout=DEFAULT_SLEEP_AGENT_LLM_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        logger.warning("sleep agent reflection generation timed out")
        return []
    except Exception:
        logger.exception("sleep agent reflection generation failed")
        return []

    raw = resp.choices[0].message.content or "{}"
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning(
            "sleep agent reflection generation returned invalid JSON preview=%s",
            maybe_sensitive_preview(raw, limit=200),
        )
        return []
    return _normalize_reflection_payload(payload, allowed_ids)


async def _topic_regroup(db, result: SleepAgentResult) -> tuple[list[SleepTopic], str | None]:
    checkpoint = await _load_sleep_agent_checkpoint(db, stage_name=SLEEP_AGENT_TOPIC_STAGE)
    memories = await _load_topic_memories(db, created_after=checkpoint)
    result.topic_memory_count = len(memories)
    topics = await _cluster_topics_with_llm(memories)
    result.topic_count = len(topics)
    logger.info(
        "sleep agent topic regroup memory_count=%s topic_count=%s",
        result.topic_memory_count,
        result.topic_count,
    )
    return topics, _stage_checkpoint_candidate(memories)


async def _persona_refresh(db, result: SleepAgentResult) -> str | None:
    checkpoint = await _load_sleep_agent_checkpoint(db, stage_name=SLEEP_AGENT_PERSONA_STAGE)
    memories = await _load_persona_refresh_memories(db, created_after=checkpoint)
    result.persona_memory_count = len(memories)
    if not memories:
        logger.info("sleep agent persona refresh found no eligible memories")
        return None

    semaphore = asyncio.Semaphore(max(DEFAULT_SLEEP_AGENT_LLM_CONCURRENCY, 1))

    async def _extract(memory: SleepMemory) -> tuple[SleepMemory, list[PersonaTraitCandidate], BaseException | None]:
        try:
            async with semaphore:
                traits = await asyncio.wait_for(
                    extract_persona_traits_with_llm(memory.content, memory.kind),
                    timeout=DEFAULT_SLEEP_AGENT_LLM_TIMEOUT_SECONDS,
                )
            return memory, traits, None
        except Exception as exc:
            return memory, [], exc

    extracted = await asyncio.gather(*[_extract(memory) for memory in memories])

    for memory, traits, error in extracted:
        if error is not None:
            logger.warning(
                "sleep agent persona extraction failed memory_id=%s",
                memory.id,
                exc_info=error,
            )
            result.errors.append(f"persona:{memory.id}:{type(error).__name__}")
            continue
        if not traits:
            continue

        try:
            stored_count = await upsert_persona_traits(
                memory.id,
                traits,
                db,
                conflict_strategy="lower_confidence",
            )
            await db.commit()
            result.persona_traits_upserted += stored_count
        except Exception:
            await db.rollback()
            logger.exception("sleep agent persona DB write failed memory_id=%s", memory.id)
            result.errors.append(f"persona_db:{memory.id}")

    logger.info(
        "sleep agent persona refresh memory_count=%s traits_upserted=%s",
        result.persona_memory_count,
        result.persona_traits_upserted,
    )
    return _stage_checkpoint_candidate(memories)


async def _reflection_exists(db, candidate: _ReflectionCandidate) -> bool:
    candidate_key = _reflection_insight_dedupe_key(candidate.insight)
    if not candidate_key:
        return False

    cursor = await db.execute(
        """SELECT 1
           FROM memory_reflection
           WHERE insight_dedupe_key = ?
           LIMIT 1""",
        (candidate_key,),
    )
    if await cursor.fetchone():
        return True

    candidate_source_fingerprint = _reflection_source_fingerprint(
        candidate.source_memory_ids
    )
    if not candidate_source_fingerprint or candidate_source_fingerprint == "[]":
        return False

    cursor = await db.execute(
        """SELECT insight_dedupe_key
           FROM memory_reflection
           WHERE source_memory_fingerprint = ?""",
        (candidate_source_fingerprint,),
    )
    rows = await cursor.fetchall()
    for row in rows:
        existing_key = str(row["insight_dedupe_key"] or "")
        if not existing_key:
            continue
        similarity = SequenceMatcher(None, candidate_key, existing_key).ratio()
        if similarity >= 0.85:
            return True
    return False


async def _generate_reflections(
    db,
    result: SleepAgentResult,
    *,
    topics: list[SleepTopic] | None = None,
) -> str | None:
    checkpoint = await _load_sleep_agent_checkpoint(db, stage_name=SLEEP_AGENT_REFLECTION_STAGE)
    memories = await _load_reflection_memories(db, created_after=checkpoint)
    result.reflection_memory_count = len(memories)
    total_importance = sum(memory.importance for memory in memories)
    if total_importance < REFLECTION_MIN_TOTAL_IMPORTANCE:
        result.skipped_reason = "insufficient_total_importance"
        logger.info(
            "sleep agent reflection skipped reason=%s memory_count=%s total_importance=%.2f",
            result.skipped_reason,
            result.reflection_memory_count,
            total_importance,
        )
        return _stage_checkpoint_candidate(memories)

    candidates = await _generate_reflections_with_llm(memories, topics or [])
    if not candidates:
        logger.info("sleep agent reflection generation produced no candidates")
        return _stage_checkpoint_candidate(memories)

    created = 0
    try:
        for candidate in candidates:
            if await _reflection_exists(db, candidate):
                continue
            insight_dedupe_key = _reflection_insight_dedupe_key(candidate.insight)
            source_memory_fingerprint = _reflection_source_fingerprint(
                candidate.source_memory_ids
            )
            await db.execute(
                """INSERT INTO memory_reflection
                   (
                       id,
                       insight,
                       source_memory_ids,
                       insight_dedupe_key,
                       source_memory_fingerprint,
                       importance
                   )
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    str(uuid.uuid4()),
                    candidate.insight,
                    json.dumps(candidate.source_memory_ids, ensure_ascii=False),
                    insight_dedupe_key,
                    source_memory_fingerprint,
                    candidate.importance,
                ),
            )
            created += 1

        if created:
            await db.commit()
            result.reflections_created += created
        logger.info(
            "sleep agent reflections memory_count=%s total_importance=%.2f created=%s",
            result.reflection_memory_count,
            total_importance,
            created,
        )
    except Exception:
        await db.rollback()
        logger.exception("sleep agent reflection DB write failed")
        result.errors.append("reflection_db")
    return _stage_checkpoint_candidate(memories)


async def run_sleep_agent_once() -> SleepAgentResult:
    db = await get_db()
    result = SleepAgentResult()
    topics: list[SleepTopic] = []
    run_log_id: str | None = None
    checkpoint_updates: dict[str, str] = {}

    try:
        try:
            run_log_id = await create_scheduler_run_log(
                db,
                scheduler_name="sleep_agent",
            )
        except Exception:
            logger.exception("sleep agent run log start failed")

        try:
            topics, checkpoint_candidate = await _topic_regroup(db, result)
            if checkpoint_candidate:
                await _advance_sleep_agent_checkpoint(
                    db,
                    stage_name=SLEEP_AGENT_TOPIC_STAGE,
                    checkpoint_created_at=checkpoint_candidate,
                    run_id=run_log_id,
                )
                checkpoint_updates[SLEEP_AGENT_TOPIC_STAGE] = checkpoint_candidate
        except Exception:
            logger.exception("sleep agent topic regroup stage failed")
            result.errors.append("topic_regroup")

        try:
            checkpoint_candidate = await _persona_refresh(db, result)
            if checkpoint_candidate:
                await _advance_sleep_agent_checkpoint(
                    db,
                    stage_name=SLEEP_AGENT_PERSONA_STAGE,
                    checkpoint_created_at=checkpoint_candidate,
                    run_id=run_log_id,
                )
                checkpoint_updates[SLEEP_AGENT_PERSONA_STAGE] = checkpoint_candidate
        except Exception:
            logger.exception("sleep agent persona refresh stage failed")
            result.errors.append("persona_refresh")

        try:
            checkpoint_candidate = await _generate_reflections(db, result, topics=topics)
            if checkpoint_candidate:
                await _advance_sleep_agent_checkpoint(
                    db,
                    stage_name=SLEEP_AGENT_REFLECTION_STAGE,
                    checkpoint_created_at=checkpoint_candidate,
                    run_id=run_log_id,
                )
                checkpoint_updates[SLEEP_AGENT_REFLECTION_STAGE] = checkpoint_candidate
        except Exception:
            logger.exception("sleep agent reflection stage failed")
            result.errors.append("reflection_generation")

        logger.info(
            "sleep agent summary topic_memories=%s topics=%s persona_memories=%s persona_traits=%s reflection_memories=%s reflections=%s errors=%s skipped_reason=%s",
            result.topic_memory_count,
            result.topic_count,
            result.persona_memory_count,
            result.persona_traits_upserted,
            result.reflection_memory_count,
            result.reflections_created,
            len(result.errors),
            result.skipped_reason,
        )
        if run_log_id:
            try:
                await finish_scheduler_run_log(
                    db,
                    run_id=run_log_id,
                    status="completed",
                    summary={
                        "topic_memory_count": result.topic_memory_count,
                        "topic_count": result.topic_count,
                        "persona_memory_count": result.persona_memory_count,
                        "persona_traits_upserted": result.persona_traits_upserted,
                        "reflection_memory_count": result.reflection_memory_count,
                        "reflections_created": result.reflections_created,
                        "skipped_reason": result.skipped_reason,
                        "errors": list(result.errors),
                        "checkpoint_updates": checkpoint_updates,
                    },
                    error_count=result.error_count,
                )
            except Exception:
                logger.exception("sleep agent run log finish failed")
        return result
    finally:
        await db.close()


async def run_sleep_agent_periodically(
    *,
    interval_seconds: int = DEFAULT_SLEEP_AGENT_INTERVAL_SECONDS,
    startup_delay_seconds: int = DEFAULT_SLEEP_AGENT_STARTUP_DELAY_SECONDS,
) -> None:
    if startup_delay_seconds > 0:
        await asyncio.sleep(startup_delay_seconds)

    while True:
        try:
            await run_sleep_agent_once()
        except asyncio.CancelledError:
            logger.info("sleep agent scheduler cancelled")
            raise
        except Exception:
            logger.exception("sleep agent scheduler iteration failed")

        await asyncio.sleep(max(interval_seconds, 1))
