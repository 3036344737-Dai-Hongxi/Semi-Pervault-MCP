"""Persona revision and PAHF clarification helpers.

This service owns the user-correction path for Persona.  It intentionally does
not rewrite structured_facts, graph data, or original memory_items.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import os
import re
import uuid
from typing import Any

from memory_core.services.llm import get_client
from memory_core.services.logging_policy import maybe_sensitive_preview
from memory_core.services.memory_policy import normalize_query_key
from memory_core.services.retrieval_constants import CORRECTION_QUERY_PATTERNS

logger = logging.getLogger("uvicorn.error")

TRAIT_KEY_PATTERN = re.compile(r"^[a-z0-9_.-]{3,64}$")
REVISION_CONFIDENCE_FLOOR = 0.85
LOW_CONFIDENCE_DEFAULT_THRESHOLD = 0.6

REVISION_EXTRACTION_PROMPT = """你是 Pervault 的 Persona 纠偏解析器。用户正在纠正系统对自己的长期画像。

你会收到：
1. 用户原话
2. 当前已有的 Persona 候选

请判断用户是否在修正 Persona，并输出 JSON：
{
  "is_revision": true,
  "persona_id": "候选 persona id，若没有合适候选则为 null",
  "trait_key": "communication_style.direct",
  "new_value": "用户不喜欢被催促",
  "confidence": 0.9,
  "trigger": "用户明确纠正：你记错了，我不喜欢被催"
}

规则：
- 只处理长期画像、稳定偏好、沟通风格、工作方式、习惯、长期限制条件
- 如果用户只是普通聊天或信息不足，输出 {"is_revision": false}
- trait_key 只能用小写英文、数字、点、下划线、短横线
- new_value 使用简洁中文
- persona_id 只能来自候选列表；没有合适候选则用 null
- 不要输出 JSON 之外的任何文字"""

CLARIFICATION_PROMPT = """你是 Pervault 的用户画像确认助手。

用户的问题：{query}

以下 Persona 置信度较低，请生成一个简短、自然的确认问题。不要解释系统，不要列清单。

低置信 Persona：
{personas}

只输出 JSON：
{{"question": "一句确认问题"}}
"""

TRAIT_KEYWORD_HINTS: dict[str, tuple[str, ...]] = {
    "communication": ("沟通", "说话", "表达", "直接", "委婉", "风格"),
    "style": ("风格", "方式"),
    "work": ("工作", "项目", "做事", "协作"),
    "food": ("吃", "饮食", "口味", "辣", "甜", "喝"),
    "preference": ("偏好", "喜欢", "不喜欢", "更喜欢"),
    "habit": ("习惯", "通常", "经常", "长期"),
    "goal": ("目标", "计划", "长期"),
    "health": ("健康", "跑步", "运动", "睡眠"),
}

VALUE_KEYWORDS = (
    "沟通",
    "风格",
    "工作",
    "项目",
    "饮食",
    "口味",
    "喜欢",
    "不喜欢",
    "更喜欢",
    "直接",
    "委婉",
    "催",
    "提醒",
    "跑步",
    "运动",
    "健康",
    "习惯",
    "目标",
)


@dataclass(frozen=True)
class PersonaRevisionDraft:
    is_revision: bool
    persona_id: str | None
    trait_key: str
    old_value: str | None
    new_value: str
    confidence: float
    trigger: str


@dataclass(frozen=True)
class PersonaRevisionResult:
    applied: bool
    needs_clarification: bool = False
    clarification_question: str | None = None
    persona_id: str | None = None
    trait_key: str | None = None
    old_value: str | None = None
    new_value: str | None = None


def _clamp_unit(value: object, *, default: float = REVISION_CONFIDENCE_FLOOR) -> float:
    if not isinstance(value, (int, float)):
        return default
    return max(0.0, min(1.0, float(value)))


def is_persona_correction_message(text: str) -> bool:
    normalized = text.strip()
    if not normalized:
        return False
    return any(pattern in normalized for pattern in CORRECTION_QUERY_PATTERNS)


async def load_revision_persona_candidates(db, *, limit: int = 30) -> list[dict[str, Any]]:
    cursor = await db.execute(
        """SELECT id, trait_key, trait_value, confidence, evidence_count, last_updated
           FROM user_persona
           ORDER BY last_updated DESC, confidence ASC, evidence_count DESC
           LIMIT ?""",
        (limit,),
    )
    rows = await cursor.fetchall()
    return [dict(row) for row in rows]


def _normalize_revision_payload(
    payload: object,
    candidate_ids: set[str],
) -> PersonaRevisionDraft | None:
    if not isinstance(payload, dict):
        return None
    if payload.get("is_revision") is not True:
        return None

    persona_id_raw = payload.get("persona_id")
    persona_id = str(persona_id_raw).strip() if persona_id_raw is not None else ""
    if persona_id in {"", "null", "None"} or persona_id not in candidate_ids:
        persona_id = None

    trait_key = str(payload.get("trait_key", "")).strip().lower()
    if not TRAIT_KEY_PATTERN.fullmatch(trait_key):
        return None

    new_value = str(payload.get("new_value", "")).strip()
    if not new_value:
        return None

    old_value = payload.get("old_value")
    if old_value is not None:
        old_value = str(old_value).strip() or None

    trigger = str(payload.get("trigger", "")).strip()
    return PersonaRevisionDraft(
        is_revision=True,
        persona_id=persona_id,
        trait_key=trait_key,
        old_value=old_value,
        new_value=new_value,
        confidence=_clamp_unit(payload.get("confidence")),
        trigger=trigger or "用户纠正了系统对自己的理解",
    )


async def extract_persona_revision_with_llm(
    message: str,
    candidates: list[dict],
) -> PersonaRevisionDraft | None:
    normalized = message.strip()
    if not normalized:
        return None
    if not is_persona_correction_message(normalized):
        return None

    candidate_ids = {str(candidate["id"]) for candidate in candidates if candidate.get("id")}
    candidate_payload = [
        {
            "id": candidate.get("id"),
            "trait_key": candidate.get("trait_key"),
            "trait_value": candidate.get("trait_value"),
            "confidence": candidate.get("confidence"),
        }
        for candidate in candidates
    ]

    client = get_client()
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    try:
        resp = await client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {"role": "system", "content": REVISION_EXTRACTION_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "message": normalized,
                            "persona_candidates": candidate_payload,
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            response_format={"type": "json_object"},
        )
    except Exception:
        logger.exception("persona revision extraction failed")
        return None

    raw = resp.choices[0].message.content or "{}"
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning(
            "persona revision returned invalid JSON preview=%s",
            maybe_sensitive_preview(raw, limit=200),
        )
        return None
    return _normalize_revision_payload(payload, candidate_ids)


async def _find_persona_by_trait_key(db, trait_key: str) -> dict[str, Any] | None:
    cursor = await db.execute(
        """SELECT id, trait_key, trait_value, confidence, evidence_count
           FROM user_persona
           WHERE trait_key = ?""",
        (trait_key,),
    )
    row = await cursor.fetchone()
    return dict(row) if row is not None else None


async def _find_persona_by_id(db, persona_id: str) -> dict[str, Any] | None:
    cursor = await db.execute(
        """SELECT id, trait_key, trait_value, confidence, evidence_count
           FROM user_persona
           WHERE id = ?""",
        (persona_id,),
    )
    row = await cursor.fetchone()
    return dict(row) if row is not None else None


async def revise_persona(
    draft: PersonaRevisionDraft,
    db,
) -> PersonaRevisionResult:
    if not draft.is_revision:
        return PersonaRevisionResult(applied=False)

    try:
        target = None
        if draft.persona_id:
            target = await _find_persona_by_id(db, draft.persona_id)
        trait_target = await _find_persona_by_trait_key(db, draft.trait_key)
        if trait_target is not None and (
            target is None or trait_target["id"] != target["id"]
        ):
            target = trait_target
        elif target is None:
            target = trait_target

        if target is None:
            persona_id = str(uuid.uuid4())
            confidence = max(draft.confidence, REVISION_CONFIDENCE_FLOOR)
            await db.execute(
                """INSERT INTO user_persona
                   (id, trait_key, trait_value, confidence, evidence_count, source_memory_ids)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    persona_id,
                    draft.trait_key,
                    draft.new_value,
                    confidence,
                    1,
                    "[]",
                ),
            )
            old_value = draft.old_value
        else:
            persona_id = target["id"]
            old_value = str(target["trait_value"] or "")
            confidence = max(
                float(target["confidence"] or 0.0),
                draft.confidence,
                REVISION_CONFIDENCE_FLOOR,
            )
            confidence = max(0.0, min(1.0, confidence))
            evidence_count = int(target["evidence_count"] or 0) + 1
            await db.execute(
                """UPDATE user_persona
                   SET trait_key = ?,
                       trait_value = ?,
                       confidence = ?,
                       evidence_count = ?,
                       last_updated = datetime('now')
                   WHERE id = ?""",
                (
                    draft.trait_key,
                    draft.new_value,
                    confidence,
                    evidence_count,
                    persona_id,
                ),
            )

        await db.execute(
            """INSERT INTO preference_revision_log
               (id, persona_id, old_value, new_value, trigger)
               VALUES (?, ?, ?, ?, ?)""",
            (
                str(uuid.uuid4()),
                persona_id,
                old_value,
                draft.new_value,
                draft.trigger,
            ),
        )
        await db.commit()
        return PersonaRevisionResult(
            applied=True,
            persona_id=persona_id,
            trait_key=draft.trait_key,
            old_value=old_value,
            new_value=draft.new_value,
        )
    except Exception:
        await db.rollback()
        logger.exception("persona revision DB write failed trait_key=%s", draft.trait_key)
        return PersonaRevisionResult(applied=False)


async def handle_persona_revision_message(
    message: str,
    db,
) -> PersonaRevisionResult:
    if not is_persona_correction_message(message):
        return PersonaRevisionResult(applied=False)

    try:
        candidates = await load_revision_persona_candidates(db)
        draft = await extract_persona_revision_with_llm(message, candidates)
    except Exception:
        logger.exception("persona revision handling failed")
        return PersonaRevisionResult(
            applied=False,
            needs_clarification=True,
            clarification_question="我想确认一下：你想纠正我关于你的哪一条理解？",
        )

    if draft is None:
        return PersonaRevisionResult(
            applied=False,
            needs_clarification=True,
            clarification_question="我想确认一下：你是想纠正我关于沟通风格、偏好，还是工作方式的哪一条理解？",
        )
    return await revise_persona(draft, db)


def _trait_key_terms(trait_key: str) -> set[str]:
    terms: set[str] = set()
    for raw_part in re.split(r"[._-]+", trait_key.lower()):
        if len(raw_part) >= 2:
            terms.add(raw_part)
        for hint in TRAIT_KEYWORD_HINTS.get(raw_part, ()):
            terms.add(hint)
    return terms


def _value_terms(value: str) -> set[str]:
    return {keyword for keyword in VALUE_KEYWORDS if keyword in value}


def _persona_matches_query(query: str, persona: dict[str, Any]) -> bool:
    query_key = normalize_query_key(query)
    if not query_key:
        return False

    trait_key = str(persona.get("trait_key") or "")
    trait_value = str(persona.get("trait_value") or "")
    trait_value_key = normalize_query_key(trait_value)
    if trait_value_key and (trait_value_key in query_key or query_key in trait_value_key):
        return True

    terms = _trait_key_terms(trait_key) | _value_terms(trait_value)
    return any(normalize_query_key(term) in query_key for term in terms if term)


async def get_low_confidence_personas(
    query: str,
    db,
    *,
    threshold: float = LOW_CONFIDENCE_DEFAULT_THRESHOLD,
    limit: int = 3,
) -> list[dict[str, Any]]:
    cursor = await db.execute(
        """SELECT id, trait_key, trait_value, confidence, evidence_count, last_updated
           FROM user_persona
           WHERE confidence < ?
           ORDER BY confidence ASC, last_updated DESC
           LIMIT ?""",
        (threshold, max(limit * 5, limit)),
    )
    rows = await cursor.fetchall()
    matches: list[dict[str, Any]] = []
    for row in rows:
        persona = dict(row)
        if _persona_matches_query(query, persona):
            matches.append(persona)
        if len(matches) >= limit:
            break
    return matches


def _fallback_clarification(personas: list[dict[str, Any]]) -> str:
    if not personas:
        return "我对这条关于你的理解还不太确定。这是准确的吗？"
    value = str(personas[0].get("trait_value") or "").strip()
    return f"我对这条关于你的理解还不太确定：“{value}”。这是准确的吗？"


async def generate_persona_clarification(
    query: str,
    personas: list[dict],
) -> str:
    if not personas:
        return ""

    persona_payload = [
        {
            "trait_key": persona.get("trait_key"),
            "trait_value": persona.get("trait_value"),
            "confidence": persona.get("confidence"),
        }
        for persona in personas
    ]
    client = get_client()
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    try:
        resp = await client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {"role": "system", "content": "你是一个只输出 JSON 的用户画像确认助手。"},
                {
                    "role": "user",
                    "content": CLARIFICATION_PROMPT.format(
                        query=query,
                        personas=json.dumps(persona_payload, ensure_ascii=False),
                    ),
                },
            ],
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content or "{}"
        payload = json.loads(raw)
        question = str(payload.get("question", "")).strip()
        return question or _fallback_clarification(personas)
    except Exception:
        logger.exception("persona clarification generation failed")
        return _fallback_clarification(personas)
