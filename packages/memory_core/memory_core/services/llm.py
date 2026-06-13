import os
import json
import logging
import math
from openai import (
    AsyncOpenAI,
    AuthenticationError,
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
    APIStatusError,
)

from memory_core.services.logging_policy import sensitive_debug_logs_enabled

client: AsyncOpenAI | None = None
embedding_client: AsyncOpenAI | None = None
logger = logging.getLogger(__name__)
prompt_logger = logging.getLogger("uvicorn.error")

GEMINI_EMBEDDING_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GEMINI_EMBEDDING_MODEL = "gemini-embedding-2-preview"
DEFAULT_EMBEDDING_DIM = 768
PROMPT_LOG_PREVIEW_CHARS = 400


def get_client() -> AsyncOpenAI:
    global client
    if client is None:
        client = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        )
    return client


def get_embedding_client() -> AsyncOpenAI:
    global embedding_client
    if embedding_client is None:
        embedding_client = AsyncOpenAI(
            api_key=os.getenv("GEMINI_API_KEY"),
            base_url=os.getenv("EMBEDDING_BASE_URL", GEMINI_EMBEDDING_BASE_URL),
        )
        logger.info("Embedding client configured with Gemini endpoint")
    return embedding_client


def get_embedding_model() -> str:
    return os.getenv("EMBEDDING_MODEL", GEMINI_EMBEDDING_MODEL)


def get_embedding_dim() -> int:
    return int(os.getenv("EMBEDDING_DIM", str(DEFAULT_EMBEDDING_DIM)))


def _normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0:
        raise AIServiceUnavailableError("Embedding 向量范数为 0")
    return [x / norm for x in vec]


class AIServiceUnavailableError(Exception):
    pass


def raise_ai_service_error(exc: Exception, service_name: str) -> None:
    if isinstance(exc, AuthenticationError):
        if service_name.startswith("Embedding"):
            raise AIServiceUnavailableError(
                "Embedding 服务不可用，请检查 GEMINI_API_KEY 或 Gemini embedding 配置"
            ) from exc
        raise AIServiceUnavailableError(
            f"{service_name}不可用，请检查 OPENAI_API_KEY 或上游配置"
        ) from exc
    if isinstance(exc, (APIConnectionError, APITimeoutError, RateLimitError, APIStatusError)):
        raise AIServiceUnavailableError(f"{service_name}暂时不可用，请稍后重试") from exc
    raise exc


CLARIFY_SYSTEM_PROMPT = """你是一个语音记忆助手。用户通过语音输入了一段话，经过 ASR 转写后可能有误。

你的任务：
1. 判断这段转写文本是否语义完整、表意清晰
2. 如果清晰：提取核心信息，输出规范化文本
3. 如果不清晰（有明显 ASR 错误、断句不完整、指代不明）：提出一个具体的澄清问题

以 JSON 格式回复，二选一：
清晰时：{"status": "clear", "normalized_text": "规范化后的文本"}
不清晰时：{"status": "unclear", "question": "你想问的具体澄清问题"}

注意：
- normalized_text 应该是完整、通顺的中文句子
- 保留原始语义，不要添加原文没有的信息
- 澄清问题要具体，不要问"你能重说一遍吗"这种笼统问题"""

CHAT_SYSTEM_PROMPT_TEMPLATE = """你是 Pervault 的聊天助手。

以下是用户的历史记忆：
{context}

回答要求：
1. 优先基于给定记忆回答，不要编造不存在的历史事实
2. 如果引用了记忆，请自然地写出“你在 X 提到过……”之类的来源标注
3. 如果记忆不足以支持结论，要明确说明“我目前只能根据现有记忆判断”
4. 回答保持简洁、自然，默认用中文
"""

MEMORY_DECISION_PROMPT = """判断以下用户消息是否包含值得长期记忆的信息（人物、事件、任务、偏好、重要事实等）。
如果是闲聊、感叹词、无意义内容，返回 {{"should_store": false}}。
如果包含有价值的信息，返回 {{"should_store": true, "reason": "原因"}}。

用户消息：{content}"""

INTENT_CLASSIFICATION_PROMPT = """你是一个查询意图分类器。

将用户查询分类为以下类别之一：
- correction：用户在纠正系统对自己的记忆或画像（你记错了、不对、不是这样、我改变主意了等）
- project：询问项目相关信息（项目进展、在做什么项目、项目推进等）
- persona：询问长期用户画像、自我特征、习惯、沟通风格、工作方式、长期偏好
- preference：询问个人偏好（喜欢什么、爱吃什么、口味偏好等）
- people：询问人际关系（和谁见面、联系过谁、与谁有交集等）
- task：询问任务或待办（有什么任务、待办是什么、还没完成什么等）
- summary：询问近期总结（我最近都干什么了、你记得我说过什么等）
- generic：以上均不符合，或问题太模糊

只输出 JSON，必须包含以下三个字段：
{{
  "intent": "<correction|project|persona|preference|people|task|summary|generic>",
  "confidence": <0.0-1.0>,
  "reason": "<一句话说明分类依据>"
}}

用户查询：{query}"""

VALID_INTENT_LABELS = frozenset(
    {"correction", "project", "persona", "preference", "people", "task", "summary", "generic"}
)


async def clarify_transcript(raw_text: str) -> dict:
    """Ask LLM to judge clarity and normalize if clear."""
    c = get_client()
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    try:
        resp = await c.chat.completions.create(
            model=model,
            temperature=0.3,
            messages=[
                {"role": "system", "content": CLARIFY_SYSTEM_PROMPT},
                {"role": "user", "content": f"ASR 转写结果：\n{raw_text}"},
            ],
            response_format={"type": "json_object"},
        )
    except Exception as exc:
        raise_ai_service_error(exc, "LLM 澄清服务")
    content = resp.choices[0].message.content or "{}"
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return {"status": "unclear", "question": "AI 响应格式异常，请重新尝试"}

    if not isinstance(parsed, dict):
        return {"status": "unclear", "question": "AI 响应结构异常，请重新尝试"}

    status = parsed.get("status")
    if status == "clear" and parsed.get("normalized_text", "").strip():
        return parsed
    if status == "unclear" and parsed.get("question", "").strip():
        return parsed

    return {"status": "unclear", "question": "AI 未能给出明确判断，请补充更多信息后重试"}


async def answer_with_context(
    message: str,
    context: str,
    *,
    history: list[dict] | None = None,
) -> str:
    c = get_client()
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    prompt_context = context.strip() or "暂无相关历史记忆。"
    system_prompt = CHAT_SYSTEM_PROMPT_TEMPLATE.format(context=prompt_context)

    prompt_logger.info(
        "llm chat prompt history_turns=%d context_chars=%d sensitive_debug=%s",
        len(history) if history else 0,
        len(prompt_context),
        sensitive_debug_logs_enabled(),
    )

    messages: list[dict] = [{"role": "system", "content": system_prompt}]
    if history:
        messages.extend({"role": h["role"], "content": h["content"]} for h in history)
    messages.append({"role": "user", "content": message})

    try:
        resp = await c.chat.completions.create(
            model=model,
            temperature=0.4,
            messages=messages,
        )
    except Exception as exc:
        raise_ai_service_error(exc, "LLM 对话服务")

    content = resp.choices[0].message.content or ""
    return content.strip() or "我暂时没能生成有效回答，请稍后再试。"


async def classify_query_intent(query: str) -> dict:
    """Call LLM to classify query intent.

    Returns a dict with keys: intent, confidence, reason.
    Raises on any LLM/network/parse failure so callers can fallback.
    """
    c = get_client()
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")

    resp = await c.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": "你是一个只输出 JSON 的查询意图分类器。"},
            {
                "role": "user",
                "content": INTENT_CLASSIFICATION_PROMPT.format(query=query),
            },
        ],
        response_format={"type": "json_object"},
    )

    raw = resp.choices[0].message.content or "{}"
    parsed = json.loads(raw)

    intent = str(parsed.get("intent", "")).strip()
    if intent not in VALID_INTENT_LABELS:
        raise ValueError(f"LLM returned invalid intent label: {intent!r}")

    return {
        "intent": intent,
        "confidence": float(parsed.get("confidence", 0.0)),
        "reason": str(parsed.get("reason", "")).strip(),
    }


async def should_store_memory(content: str) -> dict:
    c = get_client()
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")

    try:
        resp = await c.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {"role": "system", "content": "你是一个只输出 JSON 的记忆过滤器。"},
                {"role": "user", "content": MEMORY_DECISION_PROMPT.format(content=content)},
            ],
            response_format={"type": "json_object"},
        )
    except Exception as exc:
        raise_ai_service_error(exc, "记忆过滤服务")

    raw = resp.choices[0].message.content or "{}"
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {"should_store": False, "reason": "记忆过滤返回格式异常"}

    should_store = parsed.get("should_store") is True
    reason = str(parsed.get("reason", "")).strip()
    return {"should_store": should_store, "reason": reason}


async def embed_text(text: str) -> list[float]:
    c = get_embedding_client()
    model = get_embedding_model()
    embedding_dim = get_embedding_dim()

    try:
        resp = await c.embeddings.create(
            model=model,
            input=text,
            dimensions=embedding_dim,
        )
    except Exception as exc:
        raise_ai_service_error(exc, "Embedding 服务")

    if not resp.data or not resp.data[0].embedding:
        raise AIServiceUnavailableError("Embedding 服务返回了空向量")

    vector = resp.data[0].embedding
    if len(vector) != embedding_dim:
        raise AIServiceUnavailableError(
            f"Embedding 维度不匹配：期望 {embedding_dim}，实际 {len(vector)}"
        )

    return _normalize(vector)


async def embedding_smoke_test(text: str = "你好") -> dict:
    vector = await embed_text(text)
    return {
        "model": get_embedding_model(),
        "expected_dimension": get_embedding_dim(),
        "dimension": len(vector),
        "preview": vector[:3],
    }


KIND_CORRECTION_PROMPT = """判断以下文本属于哪个记忆类别，只输出 JSON。

类别说明：
- project_update：关于项目、工作进展、版本迭代等
- preference：个人偏好、口味、喜好、习惯
- relationship_event：与特定人物的互动、见面、沟通
- task：待办事项、计划做的事、明确的行动意图
- fact：关于自身的客观事实（名字、职业、居住地等）
- other：以上均不符合

只输出 JSON，格式：{{"kind": "<类别>"}}

文本：{content}"""

VALID_MEMORY_KINDS: frozenset[str] = frozenset({
    "project_update", "preference", "relationship_event", "task", "fact", "other"
})


async def classify_memory_kind_with_llm(content: str) -> str:
    """Ask LLM to classify memory kind. Returns 'other' on any failure."""
    normalized = content.strip()
    if not normalized:
        return "other"
    try:
        c = get_client()
        model = os.getenv("LLM_MODEL", "gpt-4o-mini")
        resp = await c.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {"role": "system", "content": "你是一个只输出 JSON 的记忆分类器。"},
                {"role": "user", "content": KIND_CORRECTION_PROMPT.format(content=normalized)},
            ],
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content or "{}"
        parsed = json.loads(raw)
        kind = str(parsed.get("kind", "")).strip()
        if kind not in VALID_MEMORY_KINDS:
            raise ValueError(f"LLM returned invalid kind: {kind!r}")
        return kind
    except Exception as exc:
        logger.warning("classify_memory_kind_with_llm failed, returning 'other': %r", exc)
        return "other"


EMOTION_SCORE_PROMPT = """你是一个情绪强度分析器。对以下文本进行情绪评分。

评分规则：
- 输出一个 -1.0 到 1.0 之间的浮点数
- 1.0 = 非常正面（极度开心、兴奋、成功）
- 0.0 = 中性或无明显情绪
- -1.0 = 非常负面（极度痛苦、崩溃、绝望）
- 只输出 JSON，格式为：{{"emotion_score": <float>}}
- 不要输出任何其他内容

文本：{content}"""

IMPORTANCE_SCORE_PROMPT = """你是一个长期记忆重要性评分器。对以下文本判断它对个人长期记忆系统的重要程度。

评分规则：
- 输出一个 1.0 到 10.0 之间的浮点数
- 1.0 = 几乎没有长期价值，例如寒暄、口头禅、临时感叹
- 5.0 = 普通记忆，有一定上下文价值但不关键
- 10.0 = 非常重要，例如长期偏好、身份事实、关键项目、人际关系、明确任务或重大事件
- 只输出 JSON，格式为：{{"importance": <float>}}
- 不要输出任何其他内容

文本：{content}"""

ADMISSION_SCORE_PROMPT = """你是一个长期记忆准入评分器。判断以下文本是否值得进入聊天检索上下文。

记忆类别：{kind}

评分规则：
- utility：0.0 到 1.0，表示内容对未来回答用户问题的长期实用价值
- confidence：0.0 到 1.0，表示内容是否清晰、具体、可被可靠使用
- 高分内容：稳定偏好、身份事实、项目进展、人际事件、明确任务、长期习惯
- 低分内容：寒暄、口头禅、无具体指代的感叹、纯噪音、无法确定含义的片段
- 只输出 JSON，格式为：{{"utility": <float>, "confidence": <float>}}
- 不要输出任何其他内容

文本：{content}"""


async def score_emotion_with_llm(content: str) -> float:
    """Score the emotional intensity of *content* using the LLM.

    Returns a float in [-1.0, 1.0].
    Falls back to ``estimate_emotion_score`` on any failure.
    Empty / whitespace-only input short-circuits to 0.0 without calling LLM.
    """
    from memory_core.services.memory_service import estimate_emotion_score  # local import to avoid circular

    normalized = content.strip()
    if not normalized:
        return 0.0

    try:
        c = get_client()
        model = os.getenv("LLM_MODEL", "gpt-4o-mini")
        resp = await c.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {"role": "system", "content": "你是一个只输出 JSON 的情绪分析器。"},
                {"role": "user", "content": EMOTION_SCORE_PROMPT.format(content=normalized)},
            ],
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content or "{}"
        parsed = json.loads(raw)
        value = parsed["emotion_score"]
        if not isinstance(value, (int, float)):
            raise ValueError(f"emotion_score 字段类型错误: {type(value)}")
        return max(-1.0, min(1.0, float(value)))
    except Exception as exc:
        fallback = estimate_emotion_score(content)
        logger.warning(
            "score_emotion_with_llm failed, fallback to keyword score=%.2f error=%r",
            fallback,
            exc,
        )
        return fallback


async def score_importance_with_llm(content: str) -> float:
    """Score long-term memory importance using the LLM.

    Returns a float in [1.0, 10.0].
    Falls back to 5.0 on any failure. Empty input short-circuits to 5.0.
    """
    normalized = content.strip()
    if not normalized:
        return 5.0

    try:
        c = get_client()
        model = os.getenv("LLM_MODEL", "gpt-4o-mini")
        resp = await c.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {"role": "system", "content": "你是一个只输出 JSON 的重要性评分器。"},
                {"role": "user", "content": IMPORTANCE_SCORE_PROMPT.format(content=normalized)},
            ],
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content or "{}"
        parsed = json.loads(raw)
        value = parsed["importance"]
        if not isinstance(value, (int, float)):
            raise ValueError(f"importance 字段类型错误: {type(value)}")
        return max(1.0, min(10.0, float(value)))
    except Exception as exc:
        logger.warning(
            "score_importance_with_llm failed, fallback to default importance=5.0 error=%r",
            exc,
        )
        return 5.0


async def score_admission_with_llm(content: str, kind: str) -> dict[str, float]:
    """Score whether a memory should participate in generative retrieval.

    Returns ``{"utility": float, "confidence": float}`` with both values in
    [0.0, 1.0]. Unlike emotion/importance scoring, failures are raised so the
    caller can skip DB writes and keep the memory at its safe default tier.
    """
    normalized = content.strip()
    if not normalized:
        raise ValueError("admission scoring requires non-empty content")

    c = get_client()
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    try:
        resp = await c.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {"role": "system", "content": "你是一个只输出 JSON 的记忆准入评分器。"},
                {
                    "role": "user",
                    "content": ADMISSION_SCORE_PROMPT.format(
                        content=normalized,
                        kind=kind or "other",
                    ),
                },
            ],
            response_format={"type": "json_object"},
        )
    except Exception as exc:
        raise_ai_service_error(exc, "记忆准入评分服务")

    raw = resp.choices[0].message.content or "{}"
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("admission scoring returned invalid JSON") from exc

    utility = parsed.get("utility")
    confidence = parsed.get("confidence")
    if not isinstance(utility, (int, float)):
        raise ValueError(f"utility 字段类型错误: {type(utility)}")
    if not isinstance(confidence, (int, float)):
        raise ValueError(f"confidence 字段类型错误: {type(confidence)}")

    return {
        "utility": max(0.0, min(1.0, float(utility))),
        "confidence": max(0.0, min(1.0, float(confidence))),
    }
