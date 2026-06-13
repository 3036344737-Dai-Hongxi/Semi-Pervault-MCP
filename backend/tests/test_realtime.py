import asyncio
import json
from pathlib import Path
import sys
from unittest.mock import patch

import memory_core.database as database
import pytest
from fastapi.testclient import TestClient
from fastapi.websockets import WebSocketState
from services.rate_limit import limiter
from starlette.websockets import WebSocketDisconnect

from memory_core.services.llm import AIServiceUnavailableError
from routers.realtime import _close_browser_socket
from routers.realtime import (
    PROVIDER_CLOSE_TIMEOUT_SECONDS,
    PROVIDER_PING_INTERVAL_SECONDS,
    PROVIDER_PING_TIMEOUT_SECONDS,
)
from services.realtime_session import (
    build_glm_session_update_event,
    build_glm_transcription_session_update_event,
    build_provider_headers,
    build_provider_websocket_url,
    build_realtime_session_bootstrap,
)


def _reset_shared_db():
    asyncio.run(database.close_shared_db())


class FakeProviderWebSocket:
    def __init__(self, incoming_messages: list[str] | None = None):
        self._queue: asyncio.Queue[str | None] = asyncio.Queue()
        for message in incoming_messages or []:
            self._queue.put_nowait(message)
        self.sent_messages: list[str] = []
        self.closed = False

    async def send(self, message: str) -> None:
        self.sent_messages.append(message)

    async def close(self) -> None:
        self.closed = True
        self._queue.put_nowait(None)

    def __aiter__(self):
        return self

    async def __anext__(self):
        message = await self._queue.get()
        if message is None:
            raise StopAsyncIteration
        return message


class FakeProviderContextManager:
    def __init__(self, provider_ws: FakeProviderWebSocket):
        self.provider_ws = provider_ws

    async def __aenter__(self) -> FakeProviderWebSocket:
        return self.provider_ws

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.provider_ws.close()


class FakeBrowserWebSocket:
    def __init__(self):
        self.application_state = WebSocketState.CONNECTED

    async def close(self) -> None:
        raise RuntimeError("Unexpected ASGI message 'websocket.close'")


class TestRealtimeSessionService:
    def test_close_browser_socket_ignores_double_close_runtime_error(self):
        websocket = FakeBrowserWebSocket()

        asyncio.run(_close_browser_socket(websocket))

    def test_build_realtime_session_bootstrap_requires_glm_api_key(self, monkeypatch):
        monkeypatch.setenv("VOICECALL_GLM_API_KEY", " ")

        with pytest.raises(AIServiceUnavailableError) as exc_info:
            build_realtime_session_bootstrap(websocket_url="ws://localhost:8000/api/realtime/ws")

        assert "VOICECALL_GLM_API_KEY" in str(exc_info.value)

    def test_build_realtime_session_bootstrap_returns_glm_config(self, monkeypatch):
        monkeypatch.setenv("VOICECALL_GLM_API_KEY", "glm-test-key")
        monkeypatch.setenv("VOICECALL_GLM_REALTIME_MODEL", "glm-realtime")
        monkeypatch.setenv("VOICECALL_GLM_VOICE", "tongtong")
        monkeypatch.setenv("VOICECALL_GLM_INPUT_AUDIO_FORMAT", "pcm16")
        monkeypatch.setenv("VOICECALL_GLM_OUTPUT_AUDIO_FORMAT", "pcm")

        payload = build_realtime_session_bootstrap(
            websocket_url="ws://localhost:8000/api/realtime/ws"
        )

        assert payload == {
            "provider": "glm",
            "websocket_url": "ws://localhost:8000/api/realtime/ws",
            "recommended_interaction_mode": "push_to_talk",
            "session": {
                "model": "glm-realtime",
                "voice": "tongtong",
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm",
                "turn_detection": {
                    "type": "client_vad",
                    "create_response": False,
                    "interrupt_response": False,
                },
            },
        }

    def test_build_provider_websocket_url_and_headers(self, monkeypatch):
        monkeypatch.setenv("VOICECALL_GLM_API_KEY", "glm-test-key")
        monkeypatch.setenv("VOICECALL_GLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")

        assert build_provider_websocket_url() == "wss://open.bigmodel.cn/api/paas/v4/realtime"
        assert build_provider_headers() == {"Authorization": "Bearer glm-test-key"}

    def test_build_session_events_default_to_audio_client_vad(self, monkeypatch):
        monkeypatch.setenv("VOICECALL_GLM_REALTIME_MODEL", "glm-realtime")
        monkeypatch.setenv("VOICECALL_GLM_VOICE", "tongtong")
        monkeypatch.setenv("VOICECALL_GLM_INPUT_AUDIO_FORMAT", "pcm16")
        monkeypatch.setenv("VOICECALL_GLM_OUTPUT_AUDIO_FORMAT", "pcm")

        session_update = build_glm_session_update_event()
        transcription_update = build_glm_transcription_session_update_event()

        assert session_update["type"] == "session.update"
        assert session_update["session"]["beta_fields"]["chat_mode"] == "audio"
        assert session_update["session"]["turn_detection"] == {
            "type": "client_vad",
            "create_response": False,
            "interrupt_response": False,
        }
        assert transcription_update["type"] == "transcription_session.update"
        assert transcription_update["session"]["input_audio_format"] == "pcm"
        assert transcription_update["session"]["turn_detection"] == {
            "type": "client_vad",
            "create_response": False,
            "interrupt_response": False,
        }


class TestRealtimeRouter:
    def _build_client(self, tmp_path: Path, monkeypatch, db_name: str) -> TestClient:
        monkeypatch.setenv("AUTH_PASSWORD", "secret-pass")
        monkeypatch.setenv("SESSION_SECRET", "x" * 32)
        monkeypatch.setenv("SECURE_COOKIES", "0")
        monkeypatch.setenv("CONSOLIDATION_SCHEDULER_ENABLED", "0")
        monkeypatch.setenv("WEIGHT_DECAY_SCHEDULER_ENABLED", "0")
        monkeypatch.setenv("SLEEP_AGENT_ENABLED", "0")
        monkeypatch.setattr(database, "DB_PATH", tmp_path / db_name)
        _reset_shared_db()
        limiter._storage.reset()

        sys.modules.pop("main", None)
        import main

        return TestClient(main.app)

    def test_session_route_requires_auth(self, tmp_path, monkeypatch):
        monkeypatch.setenv("VOICECALL_GLM_API_KEY", "glm-test-key")
        with self._build_client(tmp_path, monkeypatch, "realtime-auth.db") as client:
            response = client.get("/api/realtime/session")

        _reset_shared_db()
        limiter._storage.reset()
        assert response.status_code == 401

    def test_session_route_returns_bootstrap_for_authenticated_session(self, tmp_path, monkeypatch):
        monkeypatch.setenv("VOICECALL_GLM_API_KEY", "glm-test-key")
        with self._build_client(tmp_path, monkeypatch, "realtime-success.db") as client:
            login = client.post("/api/auth/login", json={"password": "secret-pass"})
            assert login.status_code == 200

            response = client.get("/api/realtime/session")

        _reset_shared_db()
        limiter._storage.reset()
        assert response.status_code == 200
        assert response.json() == {
            "provider": "glm",
            "websocket_url": "ws://testserver/api/realtime/ws",
            "recommended_interaction_mode": "push_to_talk",
            "session": {
                "model": "glm-realtime",
                "voice": "tongtong",
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm",
                "turn_detection": {
                    "type": "client_vad",
                    "create_response": False,
                    "interrupt_response": False,
                },
            },
        }

    def test_session_route_returns_503_when_service_unavailable(self, tmp_path, monkeypatch):
        monkeypatch.setenv("VOICECALL_GLM_API_KEY", " ")

        with self._build_client(tmp_path, monkeypatch, "realtime-unavailable.db") as client:
            login = client.post("/api/auth/login", json={"password": "secret-pass"})
            assert login.status_code == 200

            response = client.get("/api/realtime/session")

        _reset_shared_db()
        limiter._storage.reset()
        assert response.status_code == 503
        assert "VOICECALL_GLM_API_KEY" in response.json()["detail"]

    def test_realtime_ws_rejects_unauthenticated_browser(self, tmp_path, monkeypatch):
        monkeypatch.setenv("VOICECALL_GLM_API_KEY", "glm-test-key")
        with self._build_client(tmp_path, monkeypatch, "realtime-ws-auth.db") as client:
            with pytest.raises(WebSocketDisconnect):
                with client.websocket_connect("/api/realtime/ws"):
                    pass

        _reset_shared_db()
        limiter._storage.reset()

    def test_realtime_ws_relays_messages_after_initial_session_updates(self, tmp_path, monkeypatch):
        monkeypatch.setenv("VOICECALL_GLM_API_KEY", "glm-test-key")
        provider_ws = FakeProviderWebSocket(
            incoming_messages=[
                json.dumps({"type": "session.updated", "session": {"model": "glm-realtime"}}),
                json.dumps({"type": "response.done", "response": {"status": "completed"}}),
            ]
        )

        with self._build_client(tmp_path, monkeypatch, "realtime-ws-relay.db") as client:
            with patch(
                "routers.realtime.websockets.connect",
                return_value=FakeProviderContextManager(provider_ws),
            ) as mocked_connect:
                login = client.post("/api/auth/login", json={"password": "secret-pass"})
                assert login.status_code == 200

                with client.websocket_connect("/api/realtime/ws") as websocket:
                    first_message = websocket.receive_json()
                    assert first_message["type"] == "session.updated"

                    websocket.send_json(
                        {
                            "type": "input_audio_buffer.append",
                            "audio": "ZmFrZS1hdWRpby1jaHVuaw==",
                        }
                    )
                    second_message = websocket.receive_json()
                    assert second_message["type"] == "response.done"
                    assert second_message["response"]["status"] == "completed"

        _reset_shared_db()
        limiter._storage.reset()
        assert provider_ws.closed is True
        mocked_connect.assert_called_once()
        assert mocked_connect.call_args.kwargs["ping_interval"] == PROVIDER_PING_INTERVAL_SECONDS
        assert mocked_connect.call_args.kwargs["ping_timeout"] == PROVIDER_PING_TIMEOUT_SECONDS
        assert mocked_connect.call_args.kwargs["close_timeout"] == PROVIDER_CLOSE_TIMEOUT_SECONDS
        assert [json.loads(message)["type"] for message in provider_ws.sent_messages[:2]] == [
            "session.update",
            "transcription_session.update",
        ]
        assert json.loads(provider_ws.sent_messages[2]) == {
            "type": "input_audio_buffer.append",
            "audio": "ZmFrZS1hdWRpby1jaHVuaw==",
        }

    def test_realtime_ws_reports_provider_connection_failures_to_browser(self, tmp_path, monkeypatch):
        monkeypatch.setenv("VOICECALL_GLM_API_KEY", "glm-test-key")
        with self._build_client(tmp_path, monkeypatch, "realtime-ws-error.db") as client:
            with patch(
                "routers.realtime.websockets.connect",
                side_effect=RuntimeError("provider down"),
            ):
                login = client.post("/api/auth/login", json={"password": "secret-pass"})
                assert login.status_code == 200

                with client.websocket_connect("/api/realtime/ws") as websocket:
                    error_event = websocket.receive_json()

        _reset_shared_db()
        limiter._storage.reset()
        assert error_event == {
            "type": "error",
            "error": {
                "type": "relay_error",
                "code": "voicecall_glm_relay_failed",
                "message": "Realtime relay 暂时不可用，请稍后重试",
            },
        }
