import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from starlette.requests import Request

import memory_core.database as database
from memory_core.models import MemoryItem


def _make_db_mock(*, rowcount: int = 0, fetchone=None):
    cursor = AsyncMock()
    cursor.rowcount = rowcount
    cursor.fetchone = AsyncMock(return_value=fetchone)

    db = AsyncMock()
    db.execute = AsyncMock(return_value=cursor)
    db.commit = AsyncMock()
    db.rollback = AsyncMock()
    db.close = AsyncMock()
    return db


def _make_cursor_mock(*, rowcount: int = 0, fetchone=None, fetchall=None):
    cursor = AsyncMock()
    cursor.rowcount = rowcount
    cursor.fetchone = AsyncMock(return_value=fetchone)
    cursor.fetchall = AsyncMock(return_value=fetchall if fetchall is not None else [])
    return cursor


def _request(*, path: str, method: str = "GET", headers: list[tuple[bytes, bytes]] | None = None):
    return Request(
        {
            "type": "http",
            "method": method,
            "path": path,
            "headers": headers or [],
            "client": ("testclient", 50000),
            "server": ("testserver", 80),
            "scheme": "http",
        }
    )


class TestWeightDecayConnectionStrategy:
    async def test_decay_weights_once_uses_dedicated_connection(self):
        db = _make_db_mock(rowcount=3)

        with patch("memory_core.services.weight_decay.get_db", new=AsyncMock(return_value=db)):
            from memory_core.services.weight_decay import decay_weights_once

            updated = await decay_weights_once()

        assert updated == 3
        db.commit.assert_awaited_once()
        db.close.assert_awaited_once()

    async def test_reset_referenced_weights_uses_dedicated_connection(self):
        db = _make_db_mock()

        with patch("memory_core.services.weight_decay.get_db", new=AsyncMock(return_value=db)):
            from memory_core.services.weight_decay import reset_referenced_weights

            await reset_referenced_weights(["m-1", "m-2"])

        db.execute.assert_awaited_once()
        db.commit.assert_awaited_once()
        db.close.assert_awaited_once()


class TestMainLifespanConnectionStrategy:
    async def test_lifespan_does_not_eagerly_open_shared_db(self, monkeypatch):
        monkeypatch.setenv("AUTH_PASSWORD", "secret-pass")
        monkeypatch.setenv("SESSION_SECRET", "x" * 32)
        monkeypatch.setenv("CONSOLIDATION_SCHEDULER_ENABLED", "0")
        monkeypatch.setenv("WEIGHT_DECAY_SCHEDULER_ENABLED", "0")
        monkeypatch.setenv("SLEEP_AGENT_ENABLED", "0")

        database._shared_db = None
        sys.modules.pop("main", None)
        import main

        with patch(
            "memory_core.runtime.init_db", new=AsyncMock()
        ), patch(
            "memory_core.runtime.register_memory_pipeline_job_handlers"
        ), patch(
            "memory_core.runtime.run_background_jobs_worker",
            new=AsyncMock(return_value=None),
        ), patch(
            "memory_core.runtime.close_shared_db",
            new=AsyncMock(),
        ):
            async with main.lifespan(FastAPI()):
                assert database._shared_db is None


class TestAuthConnectionStrategy:
    async def test_get_active_session_touch_uses_read_only_lookup_and_dedicated_write_connection(self, monkeypatch):
        monkeypatch.setenv("SESSION_SECRET", "x" * 32)
        read_db = _make_db_mock(
            fetchone={
                "id": "session-1",
                "created_at": "2026-04-17 10:00:00",
                "expires_at": "2026-05-17 10:00:00",
                "last_seen_at": "2026-04-17 10:00:00",
                "revoked_at": None,
                "ip_address": "127.0.0.1",
                "user_agent": "pytest",
            }
        )
        write_db = _make_db_mock()

        get_db_mock = AsyncMock(side_effect=[read_db, write_db])

        with patch(
            "routers.auth.get_db",
            new=get_db_mock,
        ):
            from routers.auth import get_active_session

            session = await get_active_session("token-1", touch=True)

        assert session is not None
        get_db_mock.assert_any_await(read_only=True)
        read_db.execute.assert_awaited_once()
        read_db.close.assert_awaited_once()
        write_db.execute.assert_awaited_once()
        write_db.commit.assert_awaited_once()
        write_db.close.assert_awaited_once()

    async def test_get_active_session_without_touch_uses_only_read_only_connection(self, monkeypatch):
        monkeypatch.setenv("SESSION_SECRET", "x" * 32)
        read_db = _make_db_mock(
            fetchone={
                "id": "session-1",
                "created_at": "2026-04-17 10:00:00",
                "expires_at": "2026-05-17 10:00:00",
                "last_seen_at": "2026-04-17 10:00:00",
                "revoked_at": None,
                "ip_address": "127.0.0.1",
                "user_agent": "pytest",
            }
        )

        get_db_mock = AsyncMock(return_value=read_db)

        with patch("routers.auth.get_db", new=get_db_mock):
            from routers.auth import get_active_session

            session = await get_active_session("token-1", touch=False)

        assert session is not None
        get_db_mock.assert_awaited_once_with(read_only=True)
        read_db.execute.assert_awaited_once()
        read_db.close.assert_awaited_once()


class TestChatConnectionStrategy:
    async def test_persist_chat_messages_uses_dedicated_connection(self):
        db = _make_db_mock()

        with patch("routers.chat.get_db", new=AsyncMock(return_value=db)):
            from routers.chat import _persist_chat_messages

            await _persist_chat_messages("session-1", "你好", "收到")

        assert db.execute.await_count == 2
        db.commit.assert_awaited_once()
        db.close.assert_awaited_once()

    async def test_list_chat_sessions_uses_read_only_connection(self):
        db = _make_db_mock()
        get_db_mock = AsyncMock(return_value=db)
        db.execute = AsyncMock(
            return_value=_make_cursor_mock(
                fetchall=[
                    {
                        "id": "msg-1",
                        "session_id": "session-1",
                        "role": "user",
                        "content": "你好",
                        "needs_clarification": 0,
                        "clarification_question": None,
                        "created_at": "2026-04-18 10:00:00",
                    },
                    {
                        "id": "msg-2",
                        "session_id": "session-1",
                        "role": "assistant",
                        "content": "收到",
                        "needs_clarification": 0,
                        "clarification_question": None,
                        "created_at": "2026-04-18 10:00:01",
                    },
                ]
            )
        )

        with patch("routers.chat.get_db", new=get_db_mock):
            from routers.chat import list_chat_sessions

            response = await list_chat_sessions()

        assert len(response.sessions) == 1
        assert response.sessions[0].id == "session-1"
        get_db_mock.assert_awaited_once_with(read_only=True)
        db.close.assert_awaited_once()

    async def test_load_chat_history_uses_read_only_connection(self):
        db = _make_db_mock()
        get_db_mock = AsyncMock(return_value=db)
        db.execute = AsyncMock(
            return_value=_make_cursor_mock(
                fetchall=[
                    {"role": "assistant", "content": "后来的回复"},
                    {"role": "user", "content": "更早的问题"},
                ]
            )
        )

        with patch("routers.chat.get_db", new=get_db_mock):
            from routers.chat import _load_chat_history

            history = await _load_chat_history("session-1", limit=5)

        assert history == [
            {"role": "user", "content": "更早的问题"},
            {"role": "assistant", "content": "后来的回复"},
        ]
        get_db_mock.assert_awaited_once_with(read_only=True)
        db.close.assert_awaited_once()

    async def test_chat_main_read_pipeline_uses_one_read_only_connection(self):
        read_db = _make_db_mock()
        get_db_mock = AsyncMock(return_value=read_db)
        retrieve_context_mock = AsyncMock(
            return_value=[
                {
                    "id": "memory-1",
                    "content": "用户最近在推进项目",
                    "created_at": "2026-04-18 10:00:00",
                }
            ]
        )
        retrieve_graph_context_mock = AsyncMock(return_value="项目A -> depends_on -> 项目B")
        get_boot_context_mock = AsyncMock(return_value=[])
        load_chat_history_mock = AsyncMock(
            return_value=[{"role": "user", "content": "之前的问题"}]
        )
        get_low_confidence_personas_mock = AsyncMock(return_value=[])
        answer_mock = AsyncMock(return_value="这是回答")

        with patch("routers.chat.detect_query_intent", new=AsyncMock(return_value="generic")), patch(
            "routers.chat.get_db",
            new=get_db_mock,
        ), patch(
            "routers.chat.retrieve_context",
            new=retrieve_context_mock,
        ), patch(
            "routers.chat.retrieve_graph_context",
            new=retrieve_graph_context_mock,
        ), patch(
            "routers.chat.get_boot_context",
            new=get_boot_context_mock,
        ), patch(
            "routers.chat._load_chat_history",
            new=load_chat_history_mock,
        ), patch(
            "routers.chat.get_low_confidence_personas",
            new=get_low_confidence_personas_mock,
        ), patch(
            "routers.chat.answer_with_context",
            new=answer_mock,
        ):
            from routers.chat import chat

            response = await chat(
                _request(path="/api/chat", method="POST"),
                SimpleNamespace(message="聊聊我的项目", session_id="session-1"),
                SimpleNamespace(add_task=lambda *args, **kwargs: None),
            )

        assert response.reply == "这是回答"
        get_db_mock.assert_awaited_once_with(read_only=True)
        retrieve_context_mock.assert_awaited_once_with("聊聊我的项目", read_db, intent="generic")
        retrieve_graph_context_mock.assert_awaited_once_with("聊聊我的项目", read_db)
        get_boot_context_mock.assert_awaited_once()
        load_chat_history_mock.assert_awaited_once_with(
            "session-1",
            limit=10,
            db=read_db,
        )
        get_low_confidence_personas_mock.assert_awaited_once_with(
            "聊聊我的项目",
            read_db,
            threshold=0.6,
            limit=3,
        )
        read_db.close.assert_awaited_once()


class TestVoiceConnectionStrategy:
    async def test_upload_voice_uses_dedicated_connection(self):
        db = _make_db_mock()
        upload = SimpleNamespace(
            content_type="audio/webm",
            filename="sample.webm",
            read=AsyncMock(return_value=b"voice-bytes"),
        )
        request = _request(
            path="/api/voice/upload",
            method="POST",
            headers=[(b"content-length", b"11")],
        )

        with patch("routers.voice.transcribe", new=AsyncMock(return_value=("你好", 0.91))), patch(
            "routers.voice.get_db",
            new=AsyncMock(return_value=db),
        ):
            from routers.voice import upload_voice

            response = await upload_voice(request, upload)

        assert response.transcript == "你好"
        db.execute.assert_awaited_once()
        db.commit.assert_awaited_once()
        db.close.assert_awaited_once()


class TestGraphConnectionStrategy:
    async def test_extract_and_store_graph_uses_dedicated_connection_by_default(self):
        db = _make_db_mock()
        inserted_node = {
            "id": "node-1",
            "type": "person",
            "label": "小王",
            "properties": "{}",
            "weight": 1.0,
            "source_memory_count": 1,
            "created_at": "2026-04-17 10:00:00",
            "last_seen_at": "2026-04-17 10:00:00",
            "status": "confirmed",
            "possible_duplicate_of": None,
        }
        db.execute = AsyncMock(
            side_effect=[
                _make_cursor_mock(fetchone=None),
                _make_cursor_mock(fetchall=[]),
                _make_cursor_mock(fetchone=inserted_node),
            ]
        )

        with patch("memory_core.services.graph_pipeline.get_db", new=AsyncMock(return_value=db)), patch(
            "memory_core.services.graph_pipeline.extract_graph",
            new=AsyncMock(return_value={"nodes": [{"type": "person", "label": "小王"}], "edges": []}),
        ):
            from memory_core.services.graph_pipeline import extract_and_store_graph

            nodes, edges = await extract_and_store_graph("memory-1", "小王帮我处理项目")

        assert len(nodes) == 1
        assert edges == []
        db.commit.assert_awaited_once()
        db.close.assert_awaited_once()

    async def test_get_subgraph_uses_read_only_connection(self):
        db = _make_db_mock()
        get_db_mock = AsyncMock(return_value=db)
        db.execute = AsyncMock(
            side_effect=[
                _make_cursor_mock(
                    fetchall=[
                        {
                            "id": "node-1",
                            "type": "person",
                            "label": "小王",
                            "properties": "{}",
                            "weight": 1.0,
                            "source_memory_count": 2,
                            "created_at": "2026-04-17 10:00:00",
                            "last_seen_at": "2026-04-17 10:05:00",
                            "status": "confirmed",
                            "possible_duplicate_of": None,
                        }
                    ]
                ),
                _make_cursor_mock(
                    fetchall=[
                        {
                            "id": "edge-1",
                            "source_id": "node-1",
                            "target_id": "node-1",
                            "relation": "works_with",
                            "weight": 1.0,
                            "source_memory_id": "memory-1",
                            "created_at": "2026-04-17 10:06:00",
                        }
                    ]
                ),
            ]
        )

        with patch("routers.graph.get_db", new=get_db_mock):
            from routers.graph import get_subgraph

            response = await get_subgraph(
                keyword="小王",
                node_type="",
                status="confirmed",
                limit=10,
            )

        assert len(response.nodes) == 1
        assert len(response.edges) == 1
        get_db_mock.assert_awaited_once_with(read_only=True)
        db.close.assert_awaited_once()

    async def test_get_pending_nodes_uses_read_only_connection(self):
        db = _make_db_mock()
        get_db_mock = AsyncMock(return_value=db)
        db.execute = AsyncMock(
            side_effect=[
                _make_cursor_mock(
                    fetchall=[
                        {
                            "id": "node-pending",
                            "type": "person",
                            "label": "王老师",
                            "properties": "{}",
                            "weight": 1.0,
                            "source_memory_count": 1,
                            "created_at": "2026-04-17 10:00:00",
                            "last_seen_at": "2026-04-17 10:00:00",
                            "status": "pending",
                            "possible_duplicate_of": "node-confirmed",
                        }
                    ]
                ),
                _make_cursor_mock(
                    fetchall=[
                        {
                            "id": "node-confirmed",
                            "type": "person",
                            "label": "小王",
                            "properties": "{}",
                            "weight": 3.0,
                            "source_memory_count": 4,
                            "created_at": "2026-04-16 10:00:00",
                            "last_seen_at": "2026-04-17 09:00:00",
                            "status": "confirmed",
                            "possible_duplicate_of": None,
                        }
                    ]
                ),
            ]
        )

        with patch("routers.graph.get_db", new=get_db_mock):
            from routers.graph import get_pending_nodes

            response = await get_pending_nodes()

        assert len(response.nodes) == 1
        assert len(response.candidates) == 1
        get_db_mock.assert_awaited_once_with(read_only=True)
        db.close.assert_awaited_once()

    async def test_get_node_detail_uses_read_only_connection(self):
        db = _make_db_mock()
        get_db_mock = AsyncMock(return_value=db)
        db.execute = AsyncMock(
            side_effect=[
                _make_cursor_mock(
                    fetchone={
                        "id": "node-1",
                        "type": "person",
                        "label": "小王",
                        "properties": "{}",
                        "weight": 1.0,
                        "source_memory_count": 2,
                        "created_at": "2026-04-17 10:00:00",
                        "last_seen_at": "2026-04-17 10:05:00",
                        "status": "confirmed",
                        "possible_duplicate_of": None,
                    }
                ),
                _make_cursor_mock(
                    fetchall=[
                        {
                            "id": "edge-1",
                            "source_id": "node-1",
                            "target_id": "node-2",
                            "relation": "works_with",
                            "weight": 1.0,
                            "source_memory_id": "memory-1",
                            "created_at": "2026-04-17 10:06:00",
                        }
                    ]
                ),
                _make_cursor_mock(fetchall=[{"id": "node-2", "label": "项目A"}]),
                _make_cursor_mock(
                    fetchall=[
                        {
                            "id": "memory-1",
                            "content": "小王和项目A对接",
                            "created_at": "2026-04-17 10:07:00",
                        }
                    ]
                ),
            ]
        )

        with patch("routers.graph.get_db", new=get_db_mock):
            from routers.graph import get_node_detail

            response = await get_node_detail("node-1")

        assert response.node.id == "node-1"
        assert len(response.edges) == 1
        assert len(response.source_memories) == 1
        get_db_mock.assert_awaited_once_with(read_only=True)
        db.close.assert_awaited_once()

    async def test_confirm_pending_node_uses_dedicated_connection(self):
        db = _make_db_mock()
        updated_row = {
            "id": "node-1",
            "type": "person",
            "label": "小王",
            "properties": "{}",
            "weight": 1.0,
            "source_memory_count": 1,
            "created_at": "2026-04-17 10:00:00",
            "last_seen_at": "2026-04-17 10:05:00",
            "status": "confirmed",
            "possible_duplicate_of": None,
        }
        db.execute = AsyncMock(
            side_effect=[
                _make_cursor_mock(fetchone={"id": "node-1", "status": "pending"}),
                _make_cursor_mock(fetchone=updated_row),
            ]
        )

        with patch("routers.graph.get_db", new=AsyncMock(return_value=db)):
            from routers.graph import confirm_pending_node

            node = await confirm_pending_node("node-1")

        assert node.id == "node-1"
        assert node.status == "confirmed"
        db.commit.assert_awaited_once()
        db.close.assert_awaited_once()

    async def test_reject_pending_node_uses_dedicated_connection(self):
        db = _make_db_mock()
        db.execute = AsyncMock(
            side_effect=[
                _make_cursor_mock(fetchone={"id": "node-1", "status": "pending"}),
                _make_cursor_mock(),
                _make_cursor_mock(),
            ]
        )

        with patch("routers.graph.get_db", new=AsyncMock(return_value=db)):
            from routers.graph import reject_pending_node

            response = await reject_pending_node("node-1")

        assert response == {"ok": True, "deleted_id": "node-1"}
        db.commit.assert_awaited_once()
        db.close.assert_awaited_once()


class TestSystemConnectionStrategy:
    async def test_jobs_summary_uses_read_only_connection(self):
        db = _make_db_mock()
        get_db_mock = AsyncMock(return_value=db)
        get_jobs_summary_mock = AsyncMock(return_value={"total": 3, "by_status": {"pending": 3}})

        with patch("routers.system.get_db", new=get_db_mock), patch(
            "routers.system.get_jobs_summary",
            new=get_jobs_summary_mock,
        ):
            from routers.system import jobs_summary

            response = await jobs_summary()

        assert response["jobs"]["total"] == 3
        get_db_mock.assert_awaited_once_with(read_only=True)
        get_jobs_summary_mock.assert_awaited_once_with(db)
        db.close.assert_awaited_once()

    async def test_memory_ai_health_uses_read_only_connection(self):
        db = _make_db_mock()
        get_db_mock = AsyncMock(return_value=db)
        get_memory_ai_health_summary_mock = AsyncMock(
            return_value={
                "openai_configured": True,
                "embedding_configured": False,
                "sleep_agent_enabled": True,
                "worker_running": True,
                "sleep_agent_last_run_status": "completed",
                "sleep_agent_last_started_at": "2026-04-17 10:00:00",
                "sleep_agent_last_finished_at": "2026-04-17 10:05:00",
                "sleep_agent_last_error_count": 0,
                "stages": [],
            }
        )

        with patch("routers.system.get_db", new=get_db_mock), patch(
            "routers.system.get_memory_ai_health_summary",
            new=get_memory_ai_health_summary_mock,
        ):
            from routers.system import memory_ai_health

            response = await memory_ai_health()

        assert response.openai_configured is True
        get_db_mock.assert_awaited_once_with(read_only=True)
        get_memory_ai_health_summary_mock.assert_awaited_once_with(db)
        db.close.assert_awaited_once()

    async def test_jobs_list_uses_read_only_connection(self):
        db = _make_db_mock()
        get_db_mock = AsyncMock(return_value=db)
        list_jobs_mock = AsyncMock(
            return_value=[
                {
                    "id": "job-1",
                    "job_type": "graph_extract",
                    "status": "pending",
                    "origin": "pipeline",
                    "origin_run_id": "run-1",
                    "attempt_count": 0,
                    "created_at": "2026-04-17 10:00:00",
                    "updated_at": "2026-04-17 10:00:00",
                    "available_at": "2026-04-17 10:00:00",
                    "finished_at": None,
                    "terminal_reason": None,
                    "last_error": None,
                }
            ]
        )

        with patch("routers.system.get_db", new=get_db_mock), patch(
            "routers.system.list_jobs",
            new=list_jobs_mock,
        ):
            from routers.system import jobs_list

            response = await jobs_list(status="pending", job_type="graph_extract", limit=10)

        assert len(response["jobs"]) == 1
        get_db_mock.assert_awaited_once_with(read_only=True)
        list_jobs_mock.assert_awaited_once_with(
            db,
            status="pending",
            job_type="graph_extract",
            limit=10,
        )
        db.close.assert_awaited_once()

    async def test_retry_system_job_uses_dedicated_connection(self):
        db = _make_db_mock()
        get_job_mock = AsyncMock(
            side_effect=[
                {"id": "job-1", "status": "failed"},
                {"id": "job-1", "status": "pending"},
            ]
        )
        retry_job_mock = AsyncMock(return_value=True)

        with patch("routers.system.get_db", new=AsyncMock(return_value=db)), patch(
            "routers.system.get_job",
            new=get_job_mock,
        ), patch(
            "routers.system.retry_job",
            new=retry_job_mock,
        ):
            from routers.system import retry_system_job

            response = await retry_system_job("job-1")

        assert response["ok"] is True
        assert response["job"]["status"] == "pending"
        get_job_mock.assert_any_await(db, job_id="job-1")
        retry_job_mock.assert_awaited_once_with(db, job_id="job-1")
        db.close.assert_awaited_once()


class TestMemoryConnectionStrategy:
    async def test_get_memory_admission_explanation_uses_read_only_connection(self):
        db = _make_db_mock()
        get_db_mock = AsyncMock(return_value=db)
        db.execute = AsyncMock(
            return_value=_make_cursor_mock(
                fetchone={
                    "memory_id": "memory-1",
                    "score_utility": 0.9,
                    "score_confidence": 0.8,
                    "score_novelty": 0.7,
                    "score_recency": 0.6,
                    "score_type_prior": 0.5,
                    "total_score": 0.75,
                    "tier": "standard",
                    "created_at": "2026-04-18 10:00:00",
                }
            )
        )

        with patch("routers.memory.get_db", new=get_db_mock):
            from routers.memory import get_memory_admission_explanation

            response = await get_memory_admission_explanation("memory-1")

        assert response.memory_id == "memory-1"
        assert response.explanation is not None
        assert response.explanation.total_score == 0.75
        get_db_mock.assert_awaited_once_with(read_only=True)
        db.close.assert_awaited_once()

    async def test_get_memory_pipeline_trace_uses_read_only_connection(self):
        db = _make_db_mock()
        get_db_mock = AsyncMock(return_value=db)
        jobs = [
            {
                "job_type": "graph_extract",
                "status": "completed",
                "origin": "pipeline",
                "origin_run_id": "run-1",
                "attempt_count": 1,
                "subject_version": 3,
                "created_at": "2026-04-18 10:00:00",
                "updated_at": "2026-04-18 10:01:00",
                "finished_at": "2026-04-18 10:01:00",
                "terminal_reason": None,
                "last_error": None,
            }
        ]
        db.execute = AsyncMock(
            side_effect=[
                _make_cursor_mock(fetchone={"id": "memory-1", "content_version": 3}),
                _make_cursor_mock(fetchone={"cnt": 2}),
            ]
        )
        list_memory_jobs_mock = AsyncMock(return_value=jobs)
        def summarize_runs_mock(_jobs, *, current_subject_version):
            assert current_subject_version == 3
            return [
                {
                    "origin_run_id": "run-1",
                    "origin": "pipeline",
                    "subject_version": 3,
                    "job_count": 1,
                    "status_counts": {"completed": 1},
                    "started_at": "2026-04-18 10:00:00",
                    "updated_at": "2026-04-18 10:01:00",
                    "finished_at": "2026-04-18 10:01:00",
                    "is_current_version": True,
                    "jobs": jobs,
                }
            ]

        with patch("routers.memory.get_db", new=get_db_mock), patch(
            "routers.memory.list_memory_jobs",
            new=list_memory_jobs_mock,
        ), patch(
            "routers.memory.summarize_memory_job_runs",
            new=summarize_runs_mock,
        ):
            from routers.memory import get_memory_pipeline_trace

            response = await get_memory_pipeline_trace("memory-1")

        assert response.memory_id == "memory-1"
        assert response.content_version == 3
        assert response.hidden_job_count == 2
        assert len(response.jobs) == 1
        assert len(response.runs) == 1
        get_db_mock.assert_awaited_once_with(read_only=True)
        list_memory_jobs_mock.assert_awaited_once_with(
            db,
            memory_id="memory-1",
            subject_version=3,
            limit=50,
        )
        db.close.assert_awaited_once()

    async def test_get_long_term_layer_overview_uses_read_only_connection(self):
        db = _make_db_mock()
        get_db_mock = AsyncMock(return_value=db)
        db.execute = AsyncMock(
            side_effect=[
                _make_cursor_mock(fetchone={"cnt": 2}),
                _make_cursor_mock(fetchone={"cnt": 3}),
                _make_cursor_mock(fetchone={"cnt": 1}),
                _make_cursor_mock(fetchone={"cnt": 4}),
            ]
        )

        with patch("routers.memory.get_db", new=get_db_mock):
            from routers.memory import get_long_term_layer_overview

            response = await get_long_term_layer_overview()

        assert response.persona_count == 2
        assert response.reflection_count == 3
        assert response.pending_graph_node_count == 1
        assert response.low_value_memory_count == 4
        get_db_mock.assert_awaited_once_with(read_only=True)
        db.close.assert_awaited_once()

    async def test_get_long_term_layers_uses_read_only_connection(self):
        db = _make_db_mock()
        get_db_mock = AsyncMock(return_value=db)
        db.execute = AsyncMock(
            side_effect=[
                _make_cursor_mock(
                    fetchall=[
                        {
                            "id": "persona-1",
                            "trait_key": "style",
                            "trait_value": "直接沟通",
                            "confidence": 0.9,
                            "evidence_count": 2,
                            "source_memory_ids": '["memory-1","memory-2"]',
                            "last_updated": "2026-04-18 10:00:00",
                        }
                    ]
                ),
                _make_cursor_mock(
                    fetchall=[
                        {
                            "id": "reflection-1",
                            "insight": "用户偏好直接沟通风格",
                            "source_memory_ids": '["memory-1"]',
                            "importance": 8.5,
                            "created_at": "2026-04-18 09:00:00",
                        }
                    ]
                ),
            ]
        )

        with patch("routers.memory.get_db", new=get_db_mock):
            from routers.memory import get_long_term_layers

            response = await get_long_term_layers()

        assert len(response.persona_items) == 1
        assert len(response.reflection_items) == 1
        assert response.reflection_items[0].source_memory_count == 1
        get_db_mock.assert_awaited_once_with(read_only=True)
        db.close.assert_awaited_once()

    async def test_search_memories_uses_read_only_connection(self):
        db = _make_db_mock()
        get_db_mock = AsyncMock(return_value=db)
        db.execute = AsyncMock(
            side_effect=[
                _make_cursor_mock(fetchall=[{"id": "memory-1"}]),
                _make_cursor_mock(fetchone={"cnt": 1}),
            ]
        )
        row_to_item_mock = lambda _row: MemoryItem(
            id="memory-1",
            voice_record_id=None,
            content="请记住这个项目进展",
            tags=[],
            kind="project_update",
            task_status="active",
            emotion_score=0.0,
            consolidated=False,
            importance=5.0,
            admission_score=None,
            admission_tier="standard",
            weight=1.0,
            last_referenced_at=None,
            created_at="2026-04-18 10:00:00",
        )

        with patch("routers.memory.get_db", new=get_db_mock), patch(
            "routers.memory.row_to_item",
            new=row_to_item_mock,
        ):
            from routers.memory import search_memories

            response = await search_memories(q="", kind="", admission_tier="", limit=10, offset=0)

        assert response.total == 1
        assert len(response.items) == 1
        get_db_mock.assert_awaited_once_with(read_only=True)
        db.close.assert_awaited_once()

    async def test_memory_stats_uses_read_only_connection(self):
        db = _make_db_mock()
        get_db_mock = AsyncMock(return_value=db)
        db.execute = AsyncMock(
            side_effect=[
                _make_cursor_mock(fetchone={"cnt": 12}),
                _make_cursor_mock(fetchone={"cnt": 3}),
            ]
        )

        with patch("routers.memory.get_db", new=get_db_mock):
            from routers.memory import memory_stats

            response = await memory_stats()

        assert response["total_memories"] == 12
        assert response["today_count"] == 3
        get_db_mock.assert_awaited_once_with(read_only=True)
        db.close.assert_awaited_once()
