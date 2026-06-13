"""Persona extraction and persistence.

Persona is the stable user-profile layer. It is derived from admitted memories
after storage, then retrieved separately from episodic memory.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import os
import re
import uuid
from typing import Literal

from memory_core.services.llm import get_client, raise_ai_service_error
from memory_core.services.logging_policy import maybe_sensitive_preview

logger = logging.getLogger("uvicorn.error")

PERSONA_ELIGIBLE_KINDS = frozenset(
    {"preference", "fact", "project_update", "relationship_event", "task"}
)
PERSONA_CONFIDENCE_THRESHOLD = 0.7
PERSONA_SOURCE_MEMORY_LIMIT = 20
TRAIT_KEY_PATTERN = re.compile(r"^[a-z0-9_.-]{3,64}$")

PERSONA_EXTRACTION_PROMPT = """你是一个长期用户画像提取器。请从文本中提取稳定、可长期使用的用户画像。

记忆类别：{kind}

只接受这些稳定画像：
- 身份事实、长期习惯、沟通风格、工作方式、长期目标、稳定偏好、长期限制条件

必须跳过这些内容：
- 当天计划、一次性任务、临时情绪、寒暄、低价值噪音、没有清晰含义的片段

输出 JSON，格式为：
{{
  "traits": [
    {{
      "trait_key": "communication_style.direct",
      "trait_value": "用户偏好直接、清晰的沟通",
      "confidence": 0.85
    }}
  ]
}}

要求：
- trait_key 只能使用小写英文、数字、下划线、点和短横线
- trait_value 使用简洁中文
- confidence 是 0.0 到 1.0
- 如果没有稳定画像，输出 {{"traits": []}}
- 不要输出任何其他内容

文本：{content}"""


@dataclass(frozen=True)
class PersonaTraitCandidate:
    trait_key: str
    trait_value: str
    confidence: float


def _clamp_unit(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _normalize_trait_payload(payload: object) -> list[PersonaTraitCandidate]:
    if not isinstance(payload, dict):
        return []
    raw_traits = payload.get("traits", [])
    if not isinstance(raw_traits, list):
        return []

    traits: list[PersonaTraitCandidate] = []
    for raw_trait in raw_traits:
        if not isinstance(raw_trait, dict):
            continue
        trait_key = str(raw_trait.get("trait_key", "")).strip().lower()
        trait_value = str(raw_trait.get("trait_value", "")).strip()
        confidence_raw = raw_trait.get("confidence")
        if not TRAIT_KEY_PATTERN.fullmatch(trait_key):
            continue
        if not trait_value:
            continue
        if not isinstance(confidence_raw, (int, float)):
            continue
        confidence = _clamp_unit(confidence_raw)
        if confidence < PERSONA_CONFIDENCE_THRESHOLD:
            continue
        traits.append(
            PersonaTraitCandidate(
                trait_key=trait_key,
                trait_value=trait_value,
                confidence=confidence,
            )
        )
    return traits


async def extract_persona_traits_with_llm(
    content: str,
    kind: str,
) -> list[PersonaTraitCandidate]:
    normalized = content.strip()
    if not normalized:
        return []
    if kind not in PERSONA_ELIGIBLE_KINDS:
        return []

    c = get_client()
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    try:
        resp = await c.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {"role": "system", "content": "你是一个只输出 JSON 的长期用户画像提取器。"},
                {
                    "role": "user",
                    "content": PERSONA_EXTRACTION_PROMPT.format(
                        content=normalized,
                        kind=kind,
                    ),
                },
            ],
            response_format={"type": "json_object"},
        )
    except Exception as exc:
        raise_ai_service_error(exc, "Persona 提取服务")

    raw = resp.choices[0].message.content or "{}"
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning(
            "persona extraction returned invalid JSON preview=%s",
            maybe_sensitive_preview(raw, limit=200),
        )
        return []
    return _normalize_trait_payload(payload)


def _load_source_memory_ids(value: str | None) -> list[str]:
    if not value:
        return []
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item) for item in parsed if isinstance(item, str)]


def _append_source_memory_id(existing: list[str], memory_id: str) -> list[str]:
    deduped = [item for item in existing if item != memory_id]
    deduped.append(memory_id)
    return deduped[-PERSONA_SOURCE_MEMORY_LIMIT:]


async def upsert_persona_traits(
    memory_id: str,
    traits: list[PersonaTraitCandidate],
    db,
    *,
    conflict_strategy: Literal["skip", "lower_confidence"] = "skip",
) -> int:
    stored_count = 0
    for trait in traits:
        cursor = await db.execute(
            """SELECT id, trait_value, confidence, evidence_count, source_memory_ids
               FROM user_persona
               WHERE trait_key = ?""",
            (trait.trait_key,),
        )
        row = await cursor.fetchone()
        if row is None:
            await db.execute(
                """INSERT INTO user_persona
                   (id, trait_key, trait_value, confidence, evidence_count, source_memory_ids)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    str(uuid.uuid4()),
                    trait.trait_key,
                    trait.trait_value,
                    trait.confidence,
                    1,
                    json.dumps([memory_id], ensure_ascii=False),
                ),
            )
            stored_count += 1
            continue

        if row["trait_value"] != trait.trait_value:
            if conflict_strategy == "lower_confidence":
                confidence = max(0.0, float(row["confidence"] or 0.0) - 0.05)
                await db.execute(
                    """UPDATE user_persona
                       SET confidence = ?,
                           last_updated = datetime('now')
                       WHERE id = ?""",
                    (confidence, row["id"]),
                )
                logger.info(
                    "persona conflict lowered confidence trait_key=%s old_value=%r new_value=%r confidence=%.2f",
                    trait.trait_key,
                    row["trait_value"],
                    trait.trait_value,
                    confidence,
                )
                continue
            logger.info(
                "persona conflict skipped trait_key=%s old_value=%r new_value=%r",
                trait.trait_key,
                row["trait_value"],
                trait.trait_value,
            )
            continue

        existing_source_ids = _load_source_memory_ids(row["source_memory_ids"])
        if memory_id in existing_source_ids:
            logger.info(
                "persona upsert skipped duplicate source trait_key=%s memory_id=%s",
                trait.trait_key,
                memory_id,
            )
            continue

        source_ids = _append_source_memory_id(existing_source_ids, memory_id)
        confidence = min(1.0, max(float(row["confidence"] or 0.0), trait.confidence) + 0.05)
        evidence_count = int(row["evidence_count"] or 0) + 1
        await db.execute(
            """UPDATE user_persona
               SET confidence = ?,
                   evidence_count = ?,
                   source_memory_ids = ?,
                   last_updated = datetime('now')
               WHERE id = ?""",
            (
                confidence,
                evidence_count,
                json.dumps(source_ids, ensure_ascii=False),
                row["id"],
            ),
        )
        stored_count += 1
    return stored_count
