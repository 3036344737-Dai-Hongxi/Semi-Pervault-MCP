import sys
import types
from unittest.mock import AsyncMock, MagicMock, patch

import aiosqlite
from fastapi import BackgroundTasks
from starlette.requests import Request

from memory_core.database import ensure_chat_messages_schema
limiter = types.SimpleNamespace(limit=lambda _rule: (lambda func: func))
sys.modules.setdefault("main", types.SimpleNamespace(limiter=limiter))

from memory_core.models import ChatRequest
from routers.chat import _persist_chat_messages, chat, list_chat_sessions
from memory_core.services.memory_revision import PersonaRevisionResult


class _Cursor:
    def __init__(self, rows=None):
        self._rows = rows or []

    async def fetchall(self):
        return self._rows

    async def close(self):
        return None


class _Db:
    def __init__(self):
        self.execute = AsyncMock(return_value=_Cursor([]))
        self.close = AsyncMock()


def _request() -> Request:
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/api/chat",
            "headers": [],
            "client": ("testclient", 50000),
            "server": ("testserver", 80),
            "scheme": "http",
        }
    )


class TestChatPahf:
    async def test_explicit_record_uses_shared_enqueue_helper_and_reply_shape_is_compatible(self):
        memory_item = MagicMock(
            id="mem-1",
            content="我喜欢清晰直接的沟通",
            kind="preference",
            created_at="2026-04-17 00:00:00",
        )
        with patch(
            "routers.chat.create_memory_item",
            new=AsyncMock(return_value=memory_item),
        ), patch(
            "routers.chat.enqueue_memory_store_jobs",
            new=AsyncMock(),
        ) as mock_enqueue:
            response = await chat(
                _request(),
                ChatRequest(message="帮我记一下：我喜欢清晰直接的沟通"),
                BackgroundTasks(),
            )

        mock_enqueue.assert_awaited_once_with(
            memory_id="mem-1",
            content="我喜欢清晰直接的沟通",
            kind="preference",
        )
        assert response.reply == "已帮你记录：我喜欢清晰直接的沟通"
        assert response.sources[0].id == "mem-1"
        assert response.needs_clarification is False

    async def test_persist_chat_side_effects_uses_enqueue_helper_for_auto_memory(self):
        from routers.chat import _persist_chat_side_effects

        memory_item = MagicMock(id="mem-auto", kind="other")
        with patch(
            "routers.chat._persist_chat_messages",
            new=AsyncMock(),
        ), patch(
            "routers.chat._extract_explicit_record_content",
            return_value=None,
        ), patch(
            "routers.chat._should_store_chat_memory",
            new=AsyncMock(return_value=(True, "llm_yes")),
        ), patch(
            "routers.chat.create_memory_item",
            new=AsyncMock(return_value=memory_item),
        ), patch(
            "routers.chat.enqueue_memory_store_jobs",
            new=AsyncMock(),
        ) as mock_enqueue:
            await _persist_chat_side_effects(
                "session-1",
                "我最近在推进 Pervault 项目",
                "收到",
            )

        mock_enqueue.assert_awaited_once_with(
            memory_id="mem-auto",
            content="我最近在推进 Pervault 项目",
            kind="other",
        )

    async def test_correction_success_returns_revision_reply_without_normal_answer(self):
        result = PersonaRevisionResult(
            applied=True,
            persona_id="persona-1",
            trait_key="communication_style.direct",
            old_value="用户喜欢被提醒",
            new_value="用户不喜欢被催促",
        )
        with patch("routers.chat.detect_query_intent", new=AsyncMock(return_value="correction_intent")), patch(
            "routers.chat.get_db", new=AsyncMock(return_value=_Db())
        ), patch(
            "routers.chat.handle_persona_revision_message", new=AsyncMock(return_value=result)
        ) as mock_revision, patch(
            "routers.chat.retrieve_context", new=AsyncMock(return_value=[])
        ) as mock_retrieve, patch(
            "routers.chat.answer_with_context", new=AsyncMock(return_value="普通回答")
        ) as mock_answer:
            response = await chat(
                _request(),
                ChatRequest(message="你记错了，我不喜欢被催"),
                BackgroundTasks(),
            )

        mock_revision.assert_awaited_once()
        mock_retrieve.assert_not_awaited()
        mock_answer.assert_not_awaited()
        assert response.reply == "我改过来了：之前我理解为“用户喜欢被提醒”，现在更新为“用户不喜欢被催促”。"
        assert response.sources == []
        assert response.needs_clarification is False

    async def test_correction_uncertain_returns_clarification_without_db_write(self):
        result = PersonaRevisionResult(
            applied=False,
            needs_clarification=True,
            clarification_question="你想纠正我关于你的哪一条理解？",
        )
        with patch("routers.chat.detect_query_intent", new=AsyncMock(return_value="correction_intent")), patch(
            "routers.chat.get_db", new=AsyncMock(return_value=_Db())
        ), patch(
            "routers.chat.handle_persona_revision_message", new=AsyncMock(return_value=result)
        ), patch(
            "routers.chat.retrieve_context", new=AsyncMock(return_value=[])
        ) as mock_retrieve:
            response = await chat(
                _request(),
                ChatRequest(message="你记错了"),
                BackgroundTasks(),
            )

        mock_retrieve.assert_not_awaited()
        assert response.reply == "你想纠正我关于你的哪一条理解？"
        assert response.needs_clarification is True
        assert response.clarification_question == response.reply

    async def test_low_confidence_persona_appends_clarification_to_reply(self):
        with patch("routers.chat.detect_query_intent", new=AsyncMock(return_value="generic")), patch(
            "routers.chat.get_db", new=AsyncMock(return_value=_Db())
        ), patch(
            "routers.chat.retrieve_context", new=AsyncMock(return_value=[])
        ), patch(
            "routers.chat.retrieve_graph_context", new=AsyncMock(return_value="")
        ), patch(
            "routers.chat.get_boot_context", new=AsyncMock(return_value=[])
        ), patch(
            "routers.chat.get_low_confidence_personas",
            new=AsyncMock(return_value=[{"trait_value": "用户偏好直接沟通"}]),
        ), patch(
            "routers.chat.generate_persona_clarification",
            new=AsyncMock(return_value="你确实偏好直接沟通吗？"),
        ), patch(
            "routers.chat.answer_with_context", new=AsyncMock(return_value="这是回答")
        ):
            response = await chat(
                _request(),
                ChatRequest(message="我的沟通风格是什么"),
                BackgroundTasks(),
            )

        assert response.reply == "这是回答\n\n顺便确认一下：你确实偏好直接沟通吗？"
        assert response.needs_clarification is True
        assert response.clarification_question == "你确实偏好直接沟通吗？"

    async def test_regular_chat_without_low_confidence_persona_is_unchanged(self):
        with patch("routers.chat.detect_query_intent", new=AsyncMock(return_value="generic")), patch(
            "routers.chat.get_db", new=AsyncMock(return_value=_Db())
        ), patch(
            "routers.chat.retrieve_context", new=AsyncMock(return_value=[])
        ), patch(
            "routers.chat.retrieve_graph_context", new=AsyncMock(return_value="")
        ), patch(
            "routers.chat.get_boot_context", new=AsyncMock(return_value=[])
        ), patch(
            "routers.chat.get_low_confidence_personas", new=AsyncMock(return_value=[])
        ), patch(
            "routers.chat.answer_with_context", new=AsyncMock(return_value="这是回答")
        ):
            response = await chat(
                _request(),
                ChatRequest(message="随便聊聊"),
                BackgroundTasks(),
            )

        assert response.reply == "这是回答"
        assert response.needs_clarification is False
        assert response.clarification_question is None

    async def test_list_chat_sessions_preserves_persisted_clarification_metadata(self):
        db = await aiosqlite.connect(":memory:")
        db.row_factory = aiosqlite.Row
        original_close = db.close
        try:
            await ensure_chat_messages_schema(db)
            await db.commit()
            db.close = AsyncMock()
            with patch("routers.chat.get_db", new=AsyncMock(return_value=db)):
                await _persist_chat_messages(
                    "session-clarify",
                    "你记错了",
                    "你想纠正我关于你的哪一条理解？",
                    assistant_needs_clarification=True,
                    assistant_clarification_question="你想纠正我关于你的哪一条理解？",
                )
                response = await list_chat_sessions()
        finally:
            await original_close()

        assert len(response.sessions) == 1
        assistant_message = response.sessions[0].messages[-1]
        assert assistant_message.role == "assistant"
        assert assistant_message.needs_clarification is True
        assert assistant_message.clarification_question == "你想纠正我关于你的哪一条理解？"
