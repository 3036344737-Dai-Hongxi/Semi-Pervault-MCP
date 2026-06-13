"""Memory service layer.

Owns all core memory business logic:
  - kind classification and emotion scoring
  - structured-facts extraction and storage
  - memory item creation
  - background side-effect coordinators (graph extraction, embedding indexing)

Routers should only import from this module, never define their own memory
business logic.  This module must NOT import from any routers/* module.
"""

import asyncio
import hashlib
import json
import logging
import re
import uuid

from memory_core.exceptions import MemoryCoreError, MemoryNotFoundError, StorageError
from sqlite_vec import serialize_float32

from memory_core.database import get_db
from memory_core.models import MemoryItem
from memory_core.services.background_jobs import ObsoleteJobError, enqueue_job, register_job_handler
from memory_core.services.graph_pipeline import extract_and_store_graph
from memory_core.services.llm import (
    classify_memory_kind_with_llm,
    embed_text,
    score_emotion_with_llm,
    score_importance_with_llm,
)
from memory_core.services.memory_admission import compute_admission_score
from memory_core.services.memory_policy import (
    contains_any,
    fact_supported_kinds,
    is_graph_eligible_kind,
    normalize_fact_text,
    should_update_memory_kind,
)
from memory_core.services.persona_service import (
    PERSONA_ELIGIBLE_KINDS,
    extract_persona_traits_with_llm,
    upsert_persona_traits,
)

logger = logging.getLogger("uvicorn.error")

JOB_TYPE_KIND_CORRECTION = "kind_correction"
JOB_TYPE_EMBEDDING_INDEX = "embedding_index"
JOB_TYPE_EMOTION_SCORE = "emotion_score"
JOB_TYPE_IMPORTANCE_SCORE = "importance_score"
JOB_TYPE_ADMISSION_SCORE = "admission_score"
JOB_TYPE_GRAPH_EXTRACT = "graph_extract"
JOB_TYPE_PERSONA_EXTRACT = "persona_extract"

MEMORY_PIPELINE_JOB_TYPES = (
    JOB_TYPE_KIND_CORRECTION,
    JOB_TYPE_EMBEDDING_INDEX,
    JOB_TYPE_EMOTION_SCORE,
    JOB_TYPE_IMPORTANCE_SCORE,
    JOB_TYPE_ADMISSION_SCORE,
    JOB_TYPE_GRAPH_EXTRACT,
    JOB_TYPE_PERSONA_EXTRACT,
)
MEMORY_PIPELINE_DEDUPE_VERSION = "memory_pipeline_v1"

# ---------------------------------------------------------------------------
# Pattern constants
# ---------------------------------------------------------------------------

PROJECT_UPDATE_PATTERNS = (
    "我现在在做",
    "我在做",
    "正在做",
    "今天推进了",
    "推进了",
    "项目",
    "版本",
    "stage",
)
PREFERENCE_PATTERNS = (
    "我更喜欢",
    "我想吃",
    "我更偏向",
    "我偏向",
    "爱吃",
    "口味",
    "偏好",
    "想买",
    "想喝",
    "不喜欢",
)
RELATIONSHIP_ACTION_PATTERNS = (
    "讨论了",
    "见了",
    "联系了",
    "一起做了",
    "一起做",
    "聊了",
    "沟通了",
    "开了会",
)
TASK_PATTERNS = (
    "我要做",
    "我需要做",
    "打算",
    "下一步",
    "明天要",
    "计划",
    "准备",
    "待办",
    "todo",
)
FACT_PATTERNS = (
    "我是",
    "我叫",
    "我在",
    "我住在",
    "我的工作是",
    "长期",
    "一直",
    "通常",
    "经常",
    "习惯",
)
RELATIONSHIP_ACTION_PREDICATES = {
    "讨论了": "discussed_with",
    "见了": "met_with",
    "联系了": "contacted",
    "一起做了": "worked_with",
    "一起做": "worked_with",
    "聊了": "chatted_with",
    "沟通了": "communicated_with",
    "开了会": "met_with",
}
POSITIVE_EMOTION_KEYWORDS = (
    "开心",
    "高兴",
    "顺利",
    "成功",
    "满意",
    "轻松",
    "期待",
    "兴奋",
    "很好",
    "不错",
)
NEGATIVE_EMOTION_KEYWORDS = (
    "焦虑",
    "烦",
    "压力",
    "难受",
    "难过",
    "崩溃",
    "沮丧",
    "糟糕",
    "担心",
    "痛苦",
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _count_keyword_hits(text: str, keywords: tuple[str, ...]) -> int:
    return sum(1 for keyword in keywords if keyword in text)


def _is_relationship_event(text: str) -> bool:
    normalized = text.strip()
    has_social_action = contains_any(normalized, RELATIONSHIP_ACTION_PATTERNS)
    if not has_social_action:
        return False
    return any(marker in normalized for marker in ("和", "跟", "与", "一起", "联系了", "见了"))


# ---------------------------------------------------------------------------
# Business rules: classification and scoring
# ---------------------------------------------------------------------------


def classify_memory_kind(content: str) -> str:
    normalized = content.strip()
    if not normalized:
        return "other"

    if _is_relationship_event(normalized):
        return "relationship_event"
    if contains_any(normalized, PREFERENCE_PATTERNS):
        return "preference"
    if contains_any(normalized, TASK_PATTERNS):
        return "task"
    if contains_any(normalized, PROJECT_UPDATE_PATTERNS):
        return "project_update"
    if contains_any(normalized, FACT_PATTERNS):
        return "fact"
    return "other"


def default_task_status(kind: str) -> str | None:
    return "open" if kind == "task" else None


def estimate_emotion_score(content: str) -> float:
    normalized = content.strip()
    if not normalized:
        return 0.0

    positive_hits = _count_keyword_hits(normalized, POSITIVE_EMOTION_KEYWORDS)
    negative_hits = _count_keyword_hits(normalized, NEGATIVE_EMOTION_KEYWORDS)

    if positive_hits == negative_hits:
        return 0.0
    if positive_hits > negative_hits:
        return min(0.6 + 0.1 * (positive_hits - 1), 0.8)
    return -min(0.6 + 0.1 * (negative_hits - 1), 0.8)


# ---------------------------------------------------------------------------
# Structured facts extraction
# ---------------------------------------------------------------------------


def _build_structured_fact(
    kind: str, subject: str, predicate: str, object_value: str
) -> dict[str, str] | None:
    subject = normalize_fact_text(subject)
    predicate = normalize_fact_text(predicate)
    object_value = normalize_fact_text(object_value)
    if not (subject and predicate and object_value):
        return None
    return {
        "kind": kind,
        "subject": subject,
        "predicate": predicate,
        "object": object_value,
    }


def _extract_project_name(content: str) -> str:
    match = re.search(
        r"([A-Za-z][A-Za-z0-9._-]{1,31}|[\u4e00-\u9fffA-Za-z0-9._-]{2,32})\s*项目",
        content,
    )
    if match:
        return normalize_fact_text(match.group(1))

    english_match = re.search(r"\b[A-Z][A-Za-z0-9._-]{2,31}\b", content)
    if english_match:
        return normalize_fact_text(english_match.group(0))

    if "项目" in content:
        return "当前项目"
    return ""


def _extract_project_update_facts(content: str) -> list[dict[str, str]]:
    subject = _extract_project_name(content)
    if not subject:
        return []

    detail = content
    for marker in ("当前进展是", "进展是", "今天推进了", "推进了"):
        if marker in content:
            candidate = normalize_fact_text(content.split(marker, 1)[1])
            if candidate:
                detail = candidate
                break

    fact = _build_structured_fact("project_update", subject, "status_update", detail)
    return [fact] if fact else []


def _extract_preference_facts(content: str) -> list[dict[str, str]]:
    preference_patterns = (
        ("我更喜欢", "prefers"),
        ("我更偏向", "prefers"),
        ("我偏向", "prefers"),
        ("我喜欢", "likes"),
        ("我想吃", "wants_to_eat"),
        ("我想喝", "wants_to_drink"),
        ("我想买", "wants_to_buy"),
        ("爱吃", "likes_to_eat"),
        ("不喜欢", "dislikes"),
    )
    for marker, predicate in preference_patterns:
        if marker not in content:
            continue
        object_value = normalize_fact_text(content.split(marker, 1)[1])
        fact = _build_structured_fact("preference", "user", predicate, object_value)
        return [fact] if fact else []
    return []


def _extract_people_segment(content: str, action: str) -> str:
    patterns = (
        rf"(?:和|跟|与)(.+?){re.escape(action)}",
        rf"(.+?){re.escape(action)}",
    )
    for pattern in patterns:
        match = re.search(pattern, content)
        if match:
            return normalize_fact_text(match.group(1))
    return ""


def _split_people_names(segment: str) -> list[str]:
    normalized = segment
    for separator in ("、", "，", ",", "和", "跟", "与", "及"):
        normalized = normalized.replace(separator, "|")

    names: list[str] = []
    for raw_name in normalized.split("|"):
        name = normalize_fact_text(raw_name)
        if not name or len(name) > 16:
            continue
        names.append(name)
    return list(dict.fromkeys(names))


def _extract_relationship_event_facts(content: str) -> list[dict[str, str]]:
    for action, predicate in RELATIONSHIP_ACTION_PREDICATES.items():
        if action not in content:
            continue
        people_segment = _extract_people_segment(content, action)
        people = _split_people_names(people_segment)
        facts: list[dict[str, str]] = []
        for person in people:
            fact = _build_structured_fact("relationship_event", "user", predicate, person)
            if fact:
                facts.append(fact)
        if facts:
            return facts
    return []


def _extract_fact_facts(content: str) -> list[dict[str, str]]:
    fact_patterns = (
        ("我是", "is"),
        ("我叫", "name"),
        ("我住在", "lives_in"),
        ("我的工作是", "works_as"),
        ("我在", "associated_with"),
    )
    for marker, predicate in fact_patterns:
        if marker not in content:
            continue
        object_value = normalize_fact_text(content.split(marker, 1)[1])
        fact = _build_structured_fact("fact", "user", predicate, object_value)
        return [fact] if fact else []
    return []


def extract_structured_facts(content: str, kind: str) -> list[dict[str, str]]:
    normalized = content.strip()
    if not normalized or kind not in fact_supported_kinds():
        return []

    if kind == "project_update":
        candidates = _extract_project_update_facts(normalized)
    elif kind == "preference":
        candidates = _extract_preference_facts(normalized)
    elif kind == "relationship_event":
        candidates = _extract_relationship_event_facts(normalized)
    elif kind == "fact":
        candidates = _extract_fact_facts(normalized)
    else:
        candidates = []

    deduped: list[dict[str, str]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for fact in candidates:
        fact_key = (
            fact["kind"],
            fact["subject"],
            fact["predicate"],
            fact["object"],
        )
        if fact_key in seen:
            continue
        seen.add(fact_key)
        deduped.append(fact)
    return deduped


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------


async def _store_structured_facts(
    memory_item_id: str,
    kind: str,
    facts: list[dict[str, str]],
    *,
    db=None,
) -> int:
    if not facts:
        logger.info("structured facts extracted memory_id=%s kind=%s count=0", memory_item_id, kind)
        return 0

    owns_db = db is None
    if db is None:
        db = await get_db()
    stored_count = 0
    try:
        for fact in facts:
            cursor = await db.execute(
                """INSERT OR IGNORE INTO structured_facts
                   (id, memory_id, kind, subject, predicate, object, status)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    str(uuid.uuid4()),
                    memory_item_id,
                    fact["kind"],
                    fact["subject"],
                    fact["predicate"],
                    fact["object"],
                    "accepted",
                ),
            )
            if cursor.rowcount and cursor.rowcount > 0:
                stored_count += 1
                logger.info(
                    "structured fact stored memory_id=%s kind=%s subject=%r predicate=%r object=%r",
                    memory_item_id,
                    fact["kind"],
                    fact["subject"],
                    fact["predicate"],
                    fact["object"],
                )
        await db.commit()
    except Exception:
        await db.rollback()
        logger.exception("structured fact storage failed memory_id=%s kind=%s", memory_item_id, kind)
        return 0
    finally:
        if owns_db:
            await db.close()

    logger.info(
        "structured facts extracted memory_id=%s kind=%s count=%s",
        memory_item_id,
        kind,
        stored_count,
    )
    return stored_count


async def _replace_structured_facts(
    memory_item_id: str,
    kind: str,
    facts: list[dict[str, str]],
    *,
    db=None,
) -> int:
    owns_db = db is None
    if db is None:
        db = await get_db()
    stored_count = 0
    try:
        await db.execute(
            "DELETE FROM structured_facts WHERE memory_id = ?",
            (memory_item_id,),
        )
        for fact in facts:
            cursor = await db.execute(
                """INSERT INTO structured_facts
                   (id, memory_id, kind, subject, predicate, object, status)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    str(uuid.uuid4()),
                    memory_item_id,
                    fact["kind"],
                    fact["subject"],
                    fact["predicate"],
                    fact["object"],
                    "accepted",
                ),
            )
            if cursor.rowcount and cursor.rowcount > 0:
                stored_count += 1
        await db.commit()
    except Exception:
        await db.rollback()
        logger.exception("structured fact replace failed memory_id=%s kind=%s", memory_item_id, kind)
        return 0
    finally:
        if owns_db:
            await db.close()

    logger.info(
        "structured facts replaced memory_id=%s kind=%s count=%s",
        memory_item_id,
        kind,
        stored_count,
    )
    return stored_count


async def _load_memory_kind(memory_item_id: str, *, db=None) -> str:
    owns_db = db is None
    if db is None:
        db = await get_db(read_only=True)
    cursor = await db.execute(
        "SELECT kind FROM memory_items WHERE id = ?",
        (memory_item_id,),
    )
    row = await cursor.fetchone()
    try:
        if row is None:
            return "other"
        return row["kind"] or "other"
    finally:
        if owns_db:
            await db.close()


async def _load_memory_snapshot(memory_item_id: str, *, db=None):
    owns_db = db is None
    if db is None:
        db = await get_db(read_only=True)
    cursor = await db.execute(
        """SELECT id, content, kind, admission_tier, content_version
           FROM memory_items
           WHERE id = ?""",
        (memory_item_id,),
    )
    try:
        return await cursor.fetchone()
    finally:
        if owns_db:
            await db.close()


def _memory_content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _build_memory_job_payload(
    *,
    memory_id: str,
    content: str,
    kind: str,
    subject_version: int | None = None,
    run_token: str | None = None,
) -> dict[str, str | int | None]:
    payload: dict[str, str | int | None] = {
        "memory_id": memory_id,
        "content_hash": _memory_content_hash(content),
        "kind_snapshot": kind,
        "subject_version": subject_version,
        "pipeline_version": MEMORY_PIPELINE_DEDUPE_VERSION,
    }
    if run_token:
        payload["run_token"] = run_token
    return payload


async def _enqueue_memory_job(
    db,
    *,
    job_type: str,
    memory_id: str,
    content: str,
    kind: str,
    subject_version: int | None = None,
    run_token: str | None = None,
    origin: str = "pipeline",
    origin_run_id: str | None = None,
) -> dict[str, str | bool]:
    payload = _build_memory_job_payload(
        memory_id=memory_id,
        content=content,
        kind=kind,
        subject_version=subject_version,
        run_token=run_token,
    )
    job, created = await enqueue_job(
        db,
        job_type=job_type,
        payload=payload,
        origin=origin,
        origin_run_id=origin_run_id,
        dedupe_scope=f"memory:{memory_id}",
        dedupe_version=MEMORY_PIPELINE_DEDUPE_VERSION,
        subject_ref=f"memory:{memory_id}",
        subject_version=str(subject_version) if subject_version is not None else None,
    )
    logger.info(
        "memory pipeline enqueue job_id=%s job_type=%s memory_id=%s created=%s",
        job["id"],
        job_type,
        memory_id,
        created,
    )
    return {
        "job_id": str(job["id"]),
        "job_type": job_type,
        "reused_existing": not created,
    }


async def enqueue_memory_store_jobs(
    *,
    memory_id: str,
    content: str,
    kind: str,
    subject_version: int | None = None,
    run_token: str | None = None,
    origin: str = "pipeline",
    origin_run_id: str | None = None,
    db=None,
) -> list[dict[str, str | bool]]:
    owns_db = db is None
    if db is None:
        db = await get_db()
    try:
        if subject_version is None:
            snapshot = await _load_memory_snapshot(memory_id, db=db)
            if snapshot is None:
                raise RuntimeError(f"memory not found before enqueue memory_id={memory_id}")
            subject_version = int(snapshot["content_version"] or 1)
        queued_jobs: list[dict[str, str | bool]] = []
        for job_type in (
            JOB_TYPE_KIND_CORRECTION,
            JOB_TYPE_EMBEDDING_INDEX,
            JOB_TYPE_EMOTION_SCORE,
            JOB_TYPE_IMPORTANCE_SCORE,
        ):
            job_info = await _enqueue_memory_job(
                db,
                job_type=job_type,
                memory_id=memory_id,
                content=content,
                kind=kind,
                subject_version=subject_version,
                run_token=run_token,
                origin=origin,
                origin_run_id=origin_run_id,
            )
            queued_jobs.append(job_info)
        return queued_jobs
    finally:
        if owns_db:
            await db.close()


async def reprocess_memory_item(memory_item_id: str, *, db=None) -> dict[str, object]:
    owns_db = db is None
    if db is None:
        db = await get_db()
    try:
        snapshot = await _load_memory_snapshot(memory_item_id, db=db)
        if snapshot is None:
            raise MemoryNotFoundError("记忆不存在")

        content = snapshot["content"] or ""
        kind = snapshot["kind"] or "other"
        content_version = int(snapshot["content_version"] or 1)
        origin_run_id = str(uuid.uuid4())
        run_token = f"manual_reprocess:{origin_run_id}"
        queued_jobs = await enqueue_memory_store_jobs(
            memory_id=memory_item_id,
            content=content,
            kind=kind,
            subject_version=content_version,
            run_token=run_token,
            origin="manual_reprocess",
            origin_run_id=origin_run_id,
            db=db,
        )
        logger.info(
            "memory reprocess queued memory_id=%s content_version=%s queued_jobs=%s",
            memory_item_id,
            content_version,
            len(queued_jobs),
        )
        return {
            "memory_id": memory_item_id,
            "content_version": content_version,
            "origin": "manual_reprocess",
            "origin_run_id": origin_run_id,
            "jobs": queued_jobs,
        }
    finally:
        if owns_db:
            await db.close()


async def _resolve_memory_job_snapshot(job: dict[str, object]):
    payload = job.get("payload", {})
    if not isinstance(payload, dict):
        raise RuntimeError(f"invalid job payload job_id={job.get('id')}")
    memory_id = str(payload.get("memory_id", "")).strip()
    if not memory_id:
        raise RuntimeError(f"missing memory_id job_id={job.get('id')}")

    db = await get_db(read_only=True)
    try:
        row = await _load_memory_snapshot(memory_id, db=db)
        if row is None:
            raise ObsoleteJobError(f"memory missing memory_id={memory_id}")

        payload_subject_version = payload.get("subject_version")
        current_subject_version = int(row["content_version"] or 1)
        if payload_subject_version is not None:
            try:
                payload_version = int(payload_subject_version)
            except (TypeError, ValueError):
                raise RuntimeError(f"invalid subject_version job_id={job.get('id')}")
            if payload_version != current_subject_version:
                raise ObsoleteJobError(
                    f"stale memory job version mismatch memory_id={memory_id} payload_version={payload_version} current_version={current_subject_version}"
                )

        content = row["content"] or ""
        current_hash = _memory_content_hash(content)
        payload_hash = str(payload.get("content_hash", "")).strip()
        if payload_hash and payload_hash != current_hash:
            raise ObsoleteJobError(
                f"stale memory job payload mismatch memory_id={memory_id}"
            )
        return row
    finally:
        await db.close()


# ---------------------------------------------------------------------------
# Row adapter (shared by router search handler and create_memory_item)
# ---------------------------------------------------------------------------


def row_to_item(row) -> MemoryItem:
    row_keys = set(row.keys())
    tags_raw = row["tags"] if row["tags"] else "[]"
    try:
        tags = json.loads(tags_raw)
    except (json.JSONDecodeError, TypeError):
        tags = []
    return MemoryItem(
        id=row["id"],
        voice_record_id=row["voice_record_id"],
        content=row["content"],
        tags=tags,
        kind=row["kind"] if "kind" in row_keys and row["kind"] else "other",
        task_status=row["task_status"] if "task_status" in row_keys else None,
        emotion_score=float(row["emotion_score"] or 0.0)
        if "emotion_score" in row_keys
        else 0.0,
        consolidated=bool(row["consolidated"])
        if "consolidated" in row_keys
        else False,
        importance=float(row["importance"] or 5.0)
        if "importance" in row_keys
        else 5.0,
        admission_score=(
            float(row["admission_score"])
            if "admission_score" in row_keys and row["admission_score"] is not None
            else None
        ),
        admission_tier=(
            row["admission_tier"]
            if "admission_tier" in row_keys and row["admission_tier"]
            else "standard"
        ),
        weight=row["weight"],
        last_referenced_at=row["last_referenced_at"],
        created_at=row["created_at"],
    )


# ---------------------------------------------------------------------------
# Core memory creation
# ---------------------------------------------------------------------------


async def create_memory_item(
    content: str,
    voice_record_id: str | None = None,
    tags: list[str] | None = None,
    *,
    extract_structured_facts_enabled: bool = False,
    db=None,
) -> MemoryItem:
    item_id = str(uuid.uuid4())
    tags_json = json.dumps(tags or [], ensure_ascii=False)
    kind = classify_memory_kind(content)
    task_status = default_task_status(kind)
    emotion_score = estimate_emotion_score(content)  # keyword estimate; LLM update runs in background
    consolidated = 0

    row = None
    owns_db = db is None
    if db is None:
        db = await get_db()
    try:
        try:
            await db.execute(
                """INSERT INTO memory_items
                   (id, voice_record_id, content, tags, kind, task_status, emotion_score, consolidated)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    item_id,
                    voice_record_id,
                    content,
                    tags_json,
                    kind,
                    task_status,
                    emotion_score,
                    consolidated,
                ),
            )
            if voice_record_id:
                await db.execute(
                    """UPDATE voice_records
                       SET status = 'stored', updated_at = datetime('now')
                       WHERE id = ?""",
                    (voice_record_id,),
                )
            await db.commit()

            cursor = await db.execute(
                "SELECT * FROM memory_items WHERE id = ?", (item_id,)
            )
            row = await cursor.fetchone()
        except Exception:
            await db.rollback()
            raise

        if row is None:
            raise StorageError("记录写入后查询失败")

        item = row_to_item(row)
        logger.info(
            "memory stored: %s kind=%s task_status=%s emotion_score=%.2f consolidated=%s",
            item.id,
            kind,
            task_status,
            emotion_score,
            False,
        )
        if extract_structured_facts_enabled:
            try:
                facts = extract_structured_facts(content, kind)
                await _store_structured_facts(item.id, kind, facts, db=db)
            except Exception:
                logger.exception(
                    "structured facts pipeline failed memory_id=%s kind=%s",
                    item.id,
                    kind,
                )
        return item
    finally:
        if owns_db:
            await db.close()


async def update_memory_item_content(
    memory_item_id: str,
    content: str,
    *,
    extract_structured_facts_enabled: bool = False,
    db=None,
) -> MemoryItem:
    kind = classify_memory_kind(content)
    task_status = default_task_status(kind)
    emotion_score = estimate_emotion_score(content)

    owns_db = db is None
    if db is None:
        db = await get_db()
    try:
        cursor = await db.execute(
            """UPDATE memory_items
               SET content = ?,
                   content_version = COALESCE(content_version, 1) + 1,
                   kind = ?,
                   task_status = ?,
                   emotion_score = ?
               WHERE id = ?""",
            (
                content,
                kind,
                task_status,
                emotion_score,
                memory_item_id,
            ),
        )
        if not cursor.rowcount:
            await db.rollback()
            raise MemoryNotFoundError("记忆不存在")
        await db.commit()

        refreshed_cursor = await db.execute(
            "SELECT * FROM memory_items WHERE id = ?",
            (memory_item_id,),
        )
        row = await refreshed_cursor.fetchone()
        if row is None:
            raise StorageError("记录更新后查询失败")

        item = row_to_item(row)
        logger.info(
            "memory updated: %s kind=%s task_status=%s emotion_score=%.2f content_version=%s",
            item.id,
            kind,
            task_status,
            emotion_score,
            row["content_version"],
        )
        if extract_structured_facts_enabled:
            try:
                facts = extract_structured_facts(content, kind)
                await _replace_structured_facts(item.id, kind, facts, db=db)
            except Exception:
                logger.exception(
                    "structured facts replace pipeline failed memory_id=%s kind=%s",
                    item.id,
                    kind,
                )
        return item
    except MemoryCoreError:
        raise
    except Exception:
        await db.rollback()
        raise
    finally:
        if owns_db:
            await db.close()


# ---------------------------------------------------------------------------
# Background side-effect coordinators
# ---------------------------------------------------------------------------


async def _extract_graph_in_background(
    memory_item_id: str, content: str, kind: str | None = None
) -> None:
    db = await get_db()
    try:
        resolved_kind = kind or await _load_memory_kind(memory_item_id, db=db)
        if not is_graph_eligible_kind(resolved_kind):
            logger.info(
                "graph extraction skipped: %s kind=%s",
                memory_item_id,
                resolved_kind,
            )
            return

        logger.info("graph extraction started: %s", memory_item_id)
        try:
            nodes, edges = await asyncio.wait_for(
                extract_and_store_graph(memory_item_id, content, db=db),
                timeout=45.0,
            )
            logger.info(
                "graph extraction finished: %s nodes=%s edges=%s",
                memory_item_id,
                len(nodes),
                len(edges),
            )
        except asyncio.TimeoutError:
            logger.warning(
                "graph extraction timed out after 45s, skipping: %s", memory_item_id
            )
        except Exception as exc:
            logger.exception("graph extraction failed: %s, %s", memory_item_id, exc)
    finally:
        await db.close()


async def _correct_memory_kind_in_background(
    memory_item_id: str, content: str, initial_kind: str
) -> None:
    """LLM-based kind correction: backwrites memory_items.kind when LLM disagrees with keyword classifier."""
    try:
        llm_kind = await asyncio.wait_for(
            classify_memory_kind_with_llm(content),
            timeout=30.0,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "kind correction timed out after 30s, skipping: %s", memory_item_id
        )
        return
    except Exception:
        logger.exception("kind correction LLM call failed: %s", memory_item_id)
        return

    if not should_update_memory_kind(initial_kind, llm_kind):
        logger.info(
            "kind correction: no update needed memory_id=%s initial=%s llm=%s",
            memory_item_id,
            initial_kind,
            llm_kind,
        )
        return

    db = await get_db()
    try:
        await db.execute(
            "UPDATE memory_items SET kind = ? WHERE id = ?",
            (llm_kind, memory_item_id),
        )
        await db.commit()
        logger.info(
            "kind corrected memory_id=%s %s → %s",
            memory_item_id,
            initial_kind,
            llm_kind,
        )
    except Exception:
        await db.rollback()
        logger.exception("kind correction DB write failed: %s", memory_item_id)
    finally:
        await db.close()


async def _update_emotion_score_in_background(
    memory_item_id: str, content: str
) -> None:
    """LLM emotion scoring: backwrites memory_items.emotion_score after store returns."""
    try:
        score = await asyncio.wait_for(
            score_emotion_with_llm(content),
            timeout=30.0,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "emotion score timed out after 30s, skipping: %s", memory_item_id
        )
        return
    except Exception:
        logger.exception("emotion score LLM failed: %s", memory_item_id)
        return

    db = await get_db()
    try:
        await db.execute(
            "UPDATE memory_items SET emotion_score = ? WHERE id = ?",
            (score, memory_item_id),
        )
        await db.commit()
        logger.info(
            "emotion score updated memory_id=%s score=%.2f", memory_item_id, score
        )
    except Exception:
        await db.rollback()
        logger.exception("emotion score DB write failed: %s", memory_item_id)
    finally:
        await db.close()


async def _update_importance_in_background(
    memory_item_id: str, content: str
) -> None:
    """LLM importance scoring: backwrites memory_items.importance after store returns."""
    try:
        score = await asyncio.wait_for(
            score_importance_with_llm(content),
            timeout=30.0,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "importance score timed out after 30s, skipping: %s", memory_item_id
        )
        return
    except Exception:
        logger.exception("importance score LLM failed: %s", memory_item_id)
        return

    db = await get_db()
    try:
        await db.execute(
            "UPDATE memory_items SET importance = ? WHERE id = ?",
            (score, memory_item_id),
        )
        await db.commit()
        logger.info(
            "importance score updated memory_id=%s score=%.2f", memory_item_id, score
        )
    except Exception:
        await db.rollback()
        logger.exception("importance score DB write failed: %s", memory_item_id)
    finally:
        await db.close()


async def _persist_memory_admission_score(
    memory_item_id: str,
    content: str,
    kind: str,
):
    db = await get_db()
    try:
        score = await asyncio.wait_for(
            compute_admission_score(
                content,
                kind,
                db,
                exclude_memory_id=memory_item_id,
            ),
            timeout=30.0,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "admission score timed out after 30s, skipping: %s", memory_item_id
        )
        return None
    except Exception:
        logger.exception("admission score failed: %s", memory_item_id)
        return None

    try:
        await db.execute(
            """UPDATE memory_items
               SET admission_score = ?, admission_tier = ?
               WHERE id = ?""",
            (score.total, score.tier, memory_item_id),
        )
        await db.execute(
            """INSERT INTO memory_admission_log
               (id, memory_id, raw_content, score_utility, score_confidence,
                score_novelty, score_recency, score_type_prior, total_score,
                admitted, tier)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                str(uuid.uuid4()),
                memory_item_id,
                content,
                score.utility,
                score.confidence,
                score.novelty,
                score.recency,
                score.type_prior,
                score.total,
                1 if score.tier == "standard" else 0,
                score.tier,
            ),
        )
        await db.commit()
        logger.info(
            "admission score updated memory_id=%s score=%.4f tier=%s kind=%s",
            memory_item_id,
            score.total,
            score.tier,
            kind,
        )
        return score
    except Exception:
        await db.rollback()
        logger.exception("admission score DB write failed: %s", memory_item_id)
        return None
    finally:
        await db.close()


async def _score_memory_admission_in_background(
    memory_item_id: str, content: str, kind: str
) -> None:
    """Score retrieval admission and backwrite memory_items admission fields."""
    score = await _persist_memory_admission_score(memory_item_id, content, kind)
    if score is None:
        return

    if score.tier == "standard":
        await _extract_persona_in_background(memory_item_id, content, kind)


async def _extract_persona_in_background(
    memory_item_id: str, content: str, kind: str
) -> None:
    """Extract stable persona traits from an admitted memory."""
    if kind not in PERSONA_ELIGIBLE_KINDS:
        logger.info(
            "persona extraction skipped memory_id=%s kind=%s reason=ineligible_kind",
            memory_item_id,
            kind,
        )
        return

    try:
        traits = await asyncio.wait_for(
            extract_persona_traits_with_llm(content, kind),
            timeout=30.0,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "persona extraction timed out after 30s, skipping: %s", memory_item_id
        )
        return
    except Exception:
        logger.exception("persona extraction failed: %s", memory_item_id)
        return

    if not traits:
        logger.info("persona extraction produced no traits memory_id=%s", memory_item_id)
        return

    db = await get_db()
    try:
        stored_count = await upsert_persona_traits(memory_item_id, traits, db)
        await db.commit()
        logger.info(
            "persona traits upserted memory_id=%s count=%s",
            memory_item_id,
            stored_count,
        )
    except Exception:
        await db.rollback()
        logger.exception("persona upsert DB write failed: %s", memory_item_id)
    finally:
        await db.close()


async def _index_memory_embedding(memory_item_id: str, content: str) -> None:
    db = await get_db()
    try:
        if not getattr(db, "sqlite_vec_loaded", False):
            logger.warning("sqlite-vec unavailable, skip embedding for memory %s", memory_item_id)
            return

        try:
            embedding = await asyncio.wait_for(embed_text(content), timeout=45.0)
        except asyncio.TimeoutError:
            logger.warning(
                "embedding timed out after 45s, skipping: %s", memory_item_id
            )
            return
        await db.execute("DELETE FROM vec_items WHERE ref_id = ? AND ref_type = ?", (memory_item_id, "memory"))
        await db.execute(
            """INSERT INTO vec_items (ref_id, ref_type, embedding)
               VALUES (?, ?, ?)""",
            (memory_item_id, "memory", serialize_float32(embedding)),
        )
        await db.commit()
        logger.info("Embedding indexed for memory %s", memory_item_id)
    except Exception:
        await db.rollback()
        logger.exception("Embedding index failed for memory %s", memory_item_id)
    finally:
        await db.close()


async def _handle_kind_correction_job(job: dict[str, object], queue_db) -> None:
    row = await _resolve_memory_job_snapshot(job)
    memory_id = row["id"]
    content = row["content"] or ""
    initial_kind = row["kind"] or "other"
    payload = job["payload"] if isinstance(job.get("payload"), dict) else {}
    await _correct_memory_kind_in_background(memory_id, content, initial_kind)

    snapshot_db = await get_db(read_only=True)
    try:
        refreshed = await _load_memory_snapshot(memory_id, db=snapshot_db)
    finally:
        await snapshot_db.close()
    if refreshed is None:
        raise RuntimeError(
            f"memory disappeared after kind correction memory_id={memory_id}"
        )
    final_kind = refreshed["kind"] or "other"
    for downstream_job_type in (JOB_TYPE_ADMISSION_SCORE, JOB_TYPE_GRAPH_EXTRACT):
        await _enqueue_memory_job(
            queue_db,
            job_type=downstream_job_type,
            memory_id=memory_id,
            content=content,
            kind=final_kind,
            subject_version=payload.get("subject_version"),
            run_token=str(payload.get("run_token")) if payload.get("run_token") else None,
            origin=str(job.get("origin") or "pipeline"),
            origin_run_id=str(job.get("origin_run_id")) if job.get("origin_run_id") else None,
        )


async def _handle_embedding_index_job(job: dict[str, object], _queue_db) -> None:
    row = await _resolve_memory_job_snapshot(job)
    await _index_memory_embedding(row["id"], row["content"] or "")


async def _handle_emotion_score_job(job: dict[str, object], _queue_db) -> None:
    row = await _resolve_memory_job_snapshot(job)
    await _update_emotion_score_in_background(row["id"], row["content"] or "")


async def _handle_importance_score_job(job: dict[str, object], _queue_db) -> None:
    row = await _resolve_memory_job_snapshot(job)
    await _update_importance_in_background(row["id"], row["content"] or "")


async def _handle_admission_score_job(job: dict[str, object], queue_db) -> None:
    row = await _resolve_memory_job_snapshot(job)
    memory_id = row["id"]
    content = row["content"] or ""
    final_kind = row["kind"] or "other"
    payload = job["payload"] if isinstance(job.get("payload"), dict) else {}
    score = await _persist_memory_admission_score(memory_id, content, final_kind)
    if score is None or score.tier != "standard":
        return
    await _enqueue_memory_job(
        queue_db,
        job_type=JOB_TYPE_PERSONA_EXTRACT,
        memory_id=memory_id,
        content=content,
        kind=final_kind,
        subject_version=payload.get("subject_version"),
        run_token=str(payload.get("run_token")) if payload.get("run_token") else None,
        origin=str(job.get("origin") or "pipeline"),
        origin_run_id=str(job.get("origin_run_id")) if job.get("origin_run_id") else None,
    )


async def _handle_graph_extract_job(job: dict[str, object], _queue_db) -> None:
    row = await _resolve_memory_job_snapshot(job)
    await _extract_graph_in_background(row["id"], row["content"] or "", row["kind"] or "other")


async def _handle_persona_extract_job(job: dict[str, object], _queue_db) -> None:
    row = await _resolve_memory_job_snapshot(job)
    if (row["admission_tier"] or "standard") != "standard":
        logger.info(
            "persona extraction skipped by admission tier memory_id=%s tier=%s",
            row["id"],
            row["admission_tier"],
        )
        return
    await _extract_persona_in_background(row["id"], row["content"] or "", row["kind"] or "other")


def register_memory_pipeline_job_handlers() -> None:
    register_job_handler(JOB_TYPE_KIND_CORRECTION, _handle_kind_correction_job)
    register_job_handler(JOB_TYPE_EMBEDDING_INDEX, _handle_embedding_index_job)
    register_job_handler(JOB_TYPE_EMOTION_SCORE, _handle_emotion_score_job)
    register_job_handler(JOB_TYPE_IMPORTANCE_SCORE, _handle_importance_score_job)
    register_job_handler(JOB_TYPE_ADMISSION_SCORE, _handle_admission_score_job)
    register_job_handler(JOB_TYPE_GRAPH_EXTRACT, _handle_graph_extract_job)
    register_job_handler(JOB_TYPE_PERSONA_EXTRACT, _handle_persona_extract_job)
