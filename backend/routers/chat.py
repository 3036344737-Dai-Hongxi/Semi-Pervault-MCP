import logging
import re
import uuid

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request

from memory_core.database import get_db
from memory_core.models import ChatMessage, ChatRequest, ChatResponse, ChatSession, ChatSessionsResponse, ChatSource
from memory_core.services.memory_service import (
    create_memory_item,
    enqueue_memory_store_jobs,
)
from memory_core.services.llm import AIServiceUnavailableError, answer_with_context, should_store_memory
from memory_core.services.memory_revision import (
    PersonaRevisionResult,
    generate_persona_clarification,
    get_low_confidence_personas,
    handle_persona_revision_message,
)
from memory_core.services.logging_policy import sensitive_debug_logs_enabled
from memory_core.services.memory_policy import normalize_query_key
from memory_core.services.graph_retrieval import retrieve_graph_context
from memory_core.services.retrieval_boot import get_boot_context
from memory_core.services.retrieval_context import retrieve_context
from memory_core.services.retrieval_intent import detect_query_intent
from memory_core.services.retrieval_intent import is_low_value_content
from memory_core.services.retrieval_primitives import _source_composition
from services.rate_limit import limiter
from memory_core.services.weight_decay import reset_referenced_weights

logger = logging.getLogger(__name__)
prompt_logger = logging.getLogger("uvicorn.error")

router = APIRouter(prefix="/api/chat", tags=["chat"])

MAX_CONTEXT_CHARS = 2000
MAX_BOOT_CONTEXT_ITEM_CHARS = 160
PROMPT_CONTEXT_PREVIEW_CHARS = 400
CHAT_HISTORY_LIMIT = 10
RECORD_INTENT_PATTERNS = (
    r"帮我记一下[:：,，\s]*(.*)",
    r"帮我记录一下[:：,，\s]*(.*)",
    r"记一下[:：,，\s]*(.*)",
    r"记录一下[:：,，\s]*(.*)",
    r"记住这个[:：,，\s]*(.*)",
)
GENERIC_RECORD_PLACEHOLDERS = {"这个", "这件事", "这条", "一下", "一下子"}


def _summarize_content(content: str, limit: int) -> str:
    return content if len(content) <= limit else f"{content[:limit]}..."


def _get_session_title(messages: list[ChatMessage]) -> str:
    first_user_message = next((message for message in messages if message.role == "user"), None)
    base = first_user_message.content if first_user_message else messages[0].content
    trimmed = base.strip()
    if not trimmed:
        return "新对话"
    return _summarize_content(trimmed, 18)


def _build_context(sources: list[dict], graph_context: str = "") -> str:
    parts: list[str] = []
    total_chars = 0

    for source in sources:
        chunk = f'[{source["created_at"]}] {source["content"]}'
        if total_chars + len(chunk) > MAX_CONTEXT_CHARS:
            remaining = MAX_CONTEXT_CHARS - total_chars
            if remaining > 0:
                parts.append(chunk[:remaining])
            break
        parts.append(chunk)
        total_chars += len(chunk)

    return "\n".join(parts)


def _normalize_memory_key(text: str) -> str:
    return normalize_query_key(text)


def _build_boot_context(boot_items: list[dict]) -> str:
    if not boot_items:
        return ""

    kind_labels = {
        "persona": "用户画像",
        "reflection": "长期洞察",
        "project_update": "项目动态",
        "preference": "用户偏好",
        "relationship_event": "人际事件",
        "task": "近期任务",
    }
    grouped: dict[str, list[dict]] = {
        "persona": [],
        "reflection": [],
        "project_update": [],
        "preference": [],
        "relationship_event": [],
        "task": [],
    }
    for item in boot_items:
        grouped.setdefault(item["kind"], []).append(item)

    lines = [
        "【Boot Context】",
        "以下是用户最近的重要背景，回答前请优先参考；若和即时检索重复，以更具体、更新的记录为准。",
    ]
    for kind in ("persona", "reflection", "project_update", "preference", "relationship_event", "task"):
        entries = grouped.get(kind, [])
        if not entries:
            continue
        lines.append(f"{kind_labels[kind]}:")
        for entry in entries:
            lines.append(
                f'- [{entry["created_at"]}] '
                f'{_summarize_content(entry["content"], MAX_BOOT_CONTEXT_ITEM_CHARS)}'
            )
    return "\n".join(lines)


def _build_prompt_context(
    sources: list[dict], boot_context: str = "", graph_context: str = ""
) -> str:
    memory_context = _build_context(sources, graph_context)
    sections: list[str] = []

    if boot_context.strip():
        sections.append(boot_context.strip())
    if memory_context:
        if boot_context.strip():
            sections.append(f"【即时检索记忆】\n{memory_context}")
        else:
            sections.append(memory_context)
    if graph_context.strip():
        sections.append(f"【图谱上下文】{graph_context.strip()}")

    return "\n\n".join(sections)


def _build_revision_success_reply(result: PersonaRevisionResult) -> str:
    if result.old_value:
        return f"我改过来了：之前我理解为“{result.old_value}”，现在更新为“{result.new_value}”。"
    return f"我记下这个修正了：{result.new_value}。"


def _normalize_record_content(content: str) -> str:
    cleaned = content.strip()
    cleaned = re.sub(r"^(是|就是|关于)\s*", "", cleaned)
    return cleaned.strip("，,。.!！?？ ")


def _extract_explicit_record_content(message: str) -> str | None:
    for pattern in RECORD_INTENT_PATTERNS:
        match = re.fullmatch(pattern, message.strip())
        if not match:
            continue
        content = _normalize_record_content(match.group(1))
        if content and content not in GENERIC_RECORD_PLACEHOLDERS:
            return content
        return ""
    return None


async def _should_store_chat_memory(message: str) -> tuple[bool, str]:
    explicit_record_content = _extract_explicit_record_content(message)
    if explicit_record_content is not None:
        if explicit_record_content:
            return True, "explicit_record_intent"
        return False, "empty_explicit_record_content"

    if await detect_query_intent(message) == "summary_query":
        return False, "summary_query"

    if is_low_value_content(message):
        return False, "low_value_blacklist"

    decision = await should_store_memory(message)
    if decision.get("should_store"):
        return True, decision.get("reason", "")
    return False, decision.get("reason", "")


async def _persist_chat_messages(
    session_id: str,
    message: str,
    reply: str,
    *,
    assistant_needs_clarification: bool = False,
    assistant_clarification_question: str | None = None,
) -> None:
    db = await get_db()
    try:
        await db.execute(
            """INSERT INTO chat_messages
               (id, session_id, role, content, needs_clarification, clarification_question)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (str(uuid.uuid4()), session_id, "user", message, 0, None),
        )
        await db.execute(
            """INSERT INTO chat_messages
               (id, session_id, role, content, needs_clarification, clarification_question)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                str(uuid.uuid4()),
                session_id,
                "assistant",
                reply,
                1 if assistant_needs_clarification else 0,
                assistant_clarification_question,
            ),
        )
        await db.commit()
    except Exception:
        await db.rollback()
        raise
    finally:
        await db.close()


async def _persist_chat_side_effects(
    session_id: str,
    message: str,
    reply: str,
    *,
    assistant_needs_clarification: bool = False,
    assistant_clarification_question: str | None = None,
) -> None:
    try:
        await _persist_chat_messages(
            session_id,
            message,
            reply,
            assistant_needs_clarification=assistant_needs_clarification,
            assistant_clarification_question=assistant_clarification_question,
        )
    except Exception:
        logger.exception("Failed to persist chat messages for session_id=%s", session_id)

    explicit_record_content = _extract_explicit_record_content(message)
    if explicit_record_content is not None:
        logger.info(
            "Skipped async chat memory storage for session_id=%s reason=explicit_record_handled_sync",
            session_id,
        )
        return

    try:
        should_store, reason = await _should_store_chat_memory(message)
    except Exception:
        logger.exception("Failed memory filter for session_id=%s", session_id)
        return

    if not should_store:
        logger.info(
            "Skipped chat memory storage for session_id=%s reason=%s",
            session_id,
            reason,
        )
        return

    try:
        memory_item = await create_memory_item(content=message, tags=["chat"])
    except Exception:
        logger.exception("Failed to persist chat message into memory for session_id=%s", session_id)
        return

    try:
        await enqueue_memory_store_jobs(
            memory_id=memory_item.id,
            content=message,
            kind=memory_item.kind,
        )
    except Exception:
        logger.exception(
            "Failed to enqueue chat memory pipeline for session_id=%s memory_id=%s",
            session_id,
            memory_item.id,
        )


async def _load_chat_history(
    session_id: str,
    *,
    limit: int = CHAT_HISTORY_LIMIT,
    db=None,
) -> list[dict[str, str]]:
    owns_db = db is None
    history_db = db if db is not None else await get_db(read_only=True)
    try:
        history_cursor = await history_db.execute(
            """SELECT role, content FROM chat_messages
               WHERE session_id = ?
               ORDER BY created_at DESC, rowid DESC
               LIMIT ?""",
            (session_id, limit),
        )
        history_rows = await history_cursor.fetchall()
        await history_cursor.close()
        return [
            {"role": row["role"], "content": row["content"]}
            for row in reversed(history_rows)
        ]
    finally:
        if owns_db:
            await history_db.close()


@router.get("/sessions", response_model=ChatSessionsResponse)
async def list_chat_sessions():
    db = await get_db(read_only=True)
    try:
        cursor = await db.execute(
            """SELECT id, session_id, role, content, needs_clarification, clarification_question, created_at
               FROM chat_messages
               ORDER BY created_at ASC, rowid ASC"""
        )
        rows = await cursor.fetchall()
        await cursor.close()

        session_map: dict[str, list[ChatMessage]] = {}
        last_timestamp_map: dict[str, str] = {}

        for row in rows:
            message = ChatMessage(
                id=row["id"],
                role=row["role"],
                content=row["content"],
                timestamp=row["created_at"],
                needs_clarification=bool(row["needs_clarification"] or 0),
                clarification_question=row["clarification_question"],
            )
            session_id = row["session_id"]
            session_map.setdefault(session_id, []).append(message)
            last_timestamp_map[session_id] = message.timestamp

        sessions: list[ChatSession] = []
        for session_id, messages in session_map.items():
            sessions.append(
                ChatSession(
                    id=session_id,
                    title=_get_session_title(messages),
                    last_message=messages[-1].content,
                    timestamp=last_timestamp_map[session_id],
                    message_count=len(messages),
                    messages=messages,
                )
            )

        sessions.sort(key=lambda session: session.timestamp, reverse=True)
        return ChatSessionsResponse(sessions=sessions)
    finally:
        await db.close()


@router.post("", response_model=ChatResponse)
@limiter.limit("30/minute")
async def chat(request: Request, req: ChatRequest, background_tasks: BackgroundTasks):
    message = req.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="消息不能为空")

    session_id = req.session_id or str(uuid.uuid4())
    explicit_record_content = _extract_explicit_record_content(message)
    if explicit_record_content is not None:
        if not explicit_record_content:
            reply = "你想让我记什么？请把要记录的内容再说具体一点。"
            background_tasks.add_task(_persist_chat_messages, session_id, message, reply)
            return ChatResponse(reply=reply, sources=[])

        try:
            memory_item = await create_memory_item(
                content=explicit_record_content,
                tags=["chat"],
                extract_structured_facts_enabled=True,
            )
        except Exception:
            logger.exception(
                "Synchronous memory storage failed for session_id=%s", session_id
            )
            reply = "我尝试记录但失败了，请稍后再试。"
            background_tasks.add_task(_persist_chat_messages, session_id, message, reply)
            return ChatResponse(reply=reply, sources=[])

        try:
            await enqueue_memory_store_jobs(
                memory_id=memory_item.id,
                content=explicit_record_content,
                kind=memory_item.kind,
            )
        except Exception:
            logger.exception(
                "Failed to enqueue explicit chat memory pipeline for session_id=%s memory_id=%s",
                session_id,
                memory_item.id,
            )
        reply = f"已帮你记录：{explicit_record_content}"
        background_tasks.add_task(_persist_chat_messages, session_id, message, reply)
        return ChatResponse(
            reply=reply,
            sources=[
                ChatSource(
                    id=memory_item.id,
                    content=memory_item.content,
                    created_at=memory_item.created_at,
                )
            ],
        )

    query_intent = await detect_query_intent(message)

    if query_intent == "correction_intent":
        correction_db = await get_db()
        try:
            revision_result = await handle_persona_revision_message(message, correction_db)
        finally:
            await correction_db.close()

        if revision_result.applied:
            reply = _build_revision_success_reply(revision_result)
            background_tasks.add_task(_persist_chat_messages, session_id, message, reply)
            return ChatResponse(
                reply=reply,
                sources=[],
                needs_clarification=False,
            )

        if revision_result.needs_clarification:
            reply = revision_result.clarification_question or "你想纠正哪一条关于你的记忆？"
            background_tasks.add_task(
                _persist_chat_messages,
                session_id,
                message,
                reply,
                assistant_needs_clarification=True,
                assistant_clarification_question=reply,
            )
            return ChatResponse(
                reply=reply,
                sources=[],
                needs_clarification=True,
                clarification_question=reply,
            )

    read_db = await get_db(read_only=True)
    try:
        sources = await retrieve_context(message, read_db, intent=query_intent)
        try:
            graph_context = await retrieve_graph_context(message, read_db)
        except Exception:
            logger.exception("Graph context retrieval failed for session_id=%s", req.session_id)
            graph_context = ""
        try:
            source_ids = {source["id"] for source in sources}
            source_content_keys = {
                _normalize_memory_key(source["content"])
                for source in sources
                if source.get("content")
            }
            boot_items = await get_boot_context(
                read_db,
                exclude_ids=source_ids,
                exclude_content_keys=source_content_keys,
            )
        except Exception:
            logger.exception("Boot context retrieval failed for session_id=%s", req.session_id)
            boot_items = []
        try:
            chat_history = await _load_chat_history(
                session_id,
                limit=CHAT_HISTORY_LIMIT,
                db=read_db,
            )
        except Exception:
            logger.exception("Chat history retrieval failed for session_id=%s", session_id)
            chat_history = []

        boot_context = _build_boot_context(boot_items)
        context = _build_prompt_context(sources, boot_context, graph_context)
        clarification_question: str | None = None
        try:
            low_confidence_personas = await get_low_confidence_personas(
                message,
                read_db,
                threshold=0.6,
                limit=3,
            )
            if low_confidence_personas:
                clarification_question = await generate_persona_clarification(
                    message,
                    low_confidence_personas,
                )
                if clarification_question:
                    context = (
                        f"{context}\n\n"
                        f"【需要确认的用户画像】\n"
                        f"{clarification_question}"
                    )
        except Exception:
            logger.exception(
                "Low-confidence persona clarification failed for session_id=%s",
                session_id,
            )
    finally:
        await read_db.close()

    if sources:
        referenced_ids = list(dict.fromkeys(s["id"] for s in sources if s.get("id")))
        background_tasks.add_task(reset_referenced_weights, referenced_ids)
    prompt_logger.info(
        "chat prompt context session_id=%s intent=%s source_count=%s boot_count=%s source_composition=%s boot_composition=%s context_chars=%s graph_context_chars=%s sensitive_debug=%s",
        session_id,
        query_intent,
        len(sources),
        len(boot_items),
        _source_composition(sources),
        _source_composition(boot_items),
        len(context),
        len(graph_context),
        sensitive_debug_logs_enabled(),
    )
    if query_intent == "summary_query":
        if not sources:
            reply = "我还没有足够记录，暂时无法总结你最近都在做什么。"
            background_tasks.add_task(_persist_chat_side_effects, session_id, message, reply)
            return ChatResponse(reply=reply, sources=[])
        context = (
            "以下是用户最近的高价值记忆与启动背景，请优先基于这些记录直接总结回答；"
            "除非没有任何记录，否则不要说没有相关记录。\n"
            f"{context}"
        )

    try:
        reply = await answer_with_context(message, context, history=chat_history)
    except AIServiceUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception:
        logger.exception("Unexpected chat error for session_id=%s", req.session_id)
        raise HTTPException(status_code=500, detail="聊天服务暂时不可用，请稍后重试")

    if clarification_question and clarification_question not in reply:
        reply = f"{reply}\n\n顺便确认一下：{clarification_question}"

    background_tasks.add_task(
        _persist_chat_side_effects,
        session_id,
        message,
        reply,
        assistant_needs_clarification=bool(clarification_question),
        assistant_clarification_question=clarification_question,
    )

    return ChatResponse(
        reply=reply,
        sources=[ChatSource(**source) for source in sources],
        needs_clarification=bool(clarification_question),
        clarification_question=clarification_question,
    )
