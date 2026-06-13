import asyncio
from datetime import UTC, datetime
import os
from pathlib import Path
import re
import sys

import memory_core.database as database
from fastapi.testclient import TestClient
import pytest
from services.rate_limit import limiter

from services.voice_v2_session import (
    VoiceV2ConfigurationError,
    VoiceV2ProviderError,
    build_voice_v2_session_bootstrap,
    prepare_voice_v2_provider_session,
    stop_voice_v2_provider_session,
    trigger_voice_v2_finish_speech_recognition,
)


def _reset_shared_db():
    asyncio.run(database.close_shared_db())


class TestVoiceV2SessionService:
    def test_build_voice_v2_session_bootstrap_defaults_to_disabled_feature_flag(self, monkeypatch):
        monkeypatch.delenv("VOICE_V2_ENABLED", raising=False)
        monkeypatch.delenv("VOICE_V2_VOLC_AK", raising=False)
        monkeypatch.delenv("VOICE_V2_VOLC_SK", raising=False)
        monkeypatch.delenv("VOICE_V2_VOLC_RTC_APP_ID", raising=False)
        monkeypatch.delenv("VOICE_V2_VOLC_RTC_APP_KEY", raising=False)
        monkeypatch.delenv("VOICE_V2_VOLC_VOICE_CHAT_CONFIG_JSON", raising=False)

        payload = build_voice_v2_session_bootstrap()

        assert payload == {
            "phase": "phase_1_bootstrap",
            "feature_flag": False,
            "provider": "doubao",
            "session_mode": "rtc_voice_chat",
            "browser_policy": {
                "system_browser_required": True,
                "embedded_browser_support": "degraded",
            },
            "memory_policy": {
                "ownership": "pervault",
                "bridge_mode": "not_connected",
                "write_enabled": False,
            },
            "connection_policy": {
                "lifecycle": "server_bootstrap",
                "status": "blocked",
                "blocked_reasons": ["VOICE_V2 feature flag 关闭"],
                "missing_config": [
                    "VOICE_V2_VOLC_AK",
                    "VOICE_V2_VOLC_SK",
                    "VOICE_V2_VOLC_RTC_APP_ID",
                    "VOICE_V2_VOLC_RTC_APP_KEY",
                    "VOICE_V2_VOLC_VOICE_CHAT_CONFIG_JSON",
                ],
                "required_config": [
                    "VOICE_V2_VOLC_AK",
                    "VOICE_V2_VOLC_SK",
                    "VOICE_V2_VOLC_RTC_APP_ID",
                    "VOICE_V2_VOLC_RTC_APP_KEY",
                    "VOICE_V2_VOLC_VOICE_CHAT_CONFIG_JSON",
                ],
            },
            "provider_config": {
                "rtc_app_id_configured": False,
                "rtc_app_key_configured": False,
                "account_credentials_configured": False,
                "voice_chat_configured": False,
                "voice_chat_config_keys": [],
            },
            "legacy_fallback_url": "/voice",
            "next_step": "下一步是接 RTC token / StartVoiceChat，接通真实媒体链路后再开始 read-only memory bridge。",
        }

    def test_build_voice_v2_session_bootstrap_reports_ready_when_provider_config_exists(self, monkeypatch):
        monkeypatch.setenv("VOICE_V2_ENABLED", "1")
        monkeypatch.setenv("VOICE_V2_VOLC_AK", "ak")
        monkeypatch.setenv("VOICE_V2_VOLC_SK", "sk")
        monkeypatch.setenv("VOICE_V2_VOLC_RTC_APP_ID", "rtc-app")
        monkeypatch.setenv("VOICE_V2_VOLC_RTC_APP_KEY", "rtc-key")
        monkeypatch.setenv(
            "VOICE_V2_VOLC_VOICE_CHAT_CONFIG_JSON",
            '{"AgentConfig": {"WelcomeMessage": "hi"}, "ASRConfig": {"Provider": "volc"}}',
        )

        payload = build_voice_v2_session_bootstrap()

        assert payload["connection_policy"] == {
            "lifecycle": "server_bootstrap",
            "status": "ready",
            "blocked_reasons": [],
            "missing_config": [],
            "required_config": [
                "VOICE_V2_VOLC_AK",
                "VOICE_V2_VOLC_SK",
                "VOICE_V2_VOLC_RTC_APP_ID",
                "VOICE_V2_VOLC_RTC_APP_KEY",
                "VOICE_V2_VOLC_VOICE_CHAT_CONFIG_JSON",
            ],
        }
        assert payload["provider_config"] == {
            "rtc_app_id_configured": True,
            "rtc_app_key_configured": True,
            "account_credentials_configured": True,
            "voice_chat_configured": True,
            "voice_chat_config_keys": ["AgentConfig", "Config"],
        }

    def test_build_voice_v2_session_bootstrap_blocks_invalid_voice_chat_config_shape(self, monkeypatch):
        monkeypatch.setenv("VOICE_V2_ENABLED", "1")
        monkeypatch.setenv("VOICE_V2_VOLC_AK", "ak")
        monkeypatch.setenv("VOICE_V2_VOLC_SK", "sk")
        monkeypatch.setenv("VOICE_V2_VOLC_RTC_APP_ID", "rtc-app")
        monkeypatch.setenv("VOICE_V2_VOLC_RTC_APP_KEY", "rtc-key")
        monkeypatch.setenv(
            "VOICE_V2_VOLC_VOICE_CHAT_CONFIG_JSON",
            '{"AgentConfig": {"WelcomeMessage": "hi"}, "ASRConfig": "volc"}',
        )

        payload = build_voice_v2_session_bootstrap()

        assert payload["connection_policy"] == {
            "lifecycle": "server_bootstrap",
            "status": "blocked",
            "blocked_reasons": [
                "VOICE_V2_VOLC_VOICE_CHAT_CONFIG_JSON 中以下段必须是对象：ASRConfig",
            ],
            "missing_config": [],
            "required_config": [
                "VOICE_V2_VOLC_AK",
                "VOICE_V2_VOLC_SK",
                "VOICE_V2_VOLC_RTC_APP_ID",
                "VOICE_V2_VOLC_RTC_APP_KEY",
                "VOICE_V2_VOLC_VOICE_CHAT_CONFIG_JSON",
            ],
        }
        assert payload["provider_config"] == {
            "rtc_app_id_configured": True,
            "rtc_app_key_configured": True,
            "account_credentials_configured": True,
            "voice_chat_configured": False,
            "voice_chat_config_keys": [],
        }

    def test_prepare_voice_v2_provider_session_requires_ready_bootstrap(self, monkeypatch):
        monkeypatch.setenv("VOICE_V2_ENABLED", "0")

        with pytest.raises(VoiceV2ConfigurationError, match="未就绪"):
            prepare_voice_v2_provider_session(auth_session_id="session-1")

    def test_prepare_voice_v2_provider_session_rejects_invalid_voice_chat_config_shape(self, monkeypatch):
        monkeypatch.setenv("VOICE_V2_ENABLED", "1")
        monkeypatch.setenv("VOICE_V2_VOLC_AK", "ak")
        monkeypatch.setenv("VOICE_V2_VOLC_SK", "sk")
        monkeypatch.setenv("VOICE_V2_VOLC_RTC_APP_ID", "rtc-app")
        monkeypatch.setenv("VOICE_V2_VOLC_RTC_APP_KEY", "rtc-key")
        monkeypatch.setenv(
            "VOICE_V2_VOLC_VOICE_CHAT_CONFIG_JSON",
            '{"AgentConfig": {"WelcomeMessage": "hi"}}',
        )

        with pytest.raises(
            VoiceV2ConfigurationError,
            match="Voice V2 provider bootstrap 未就绪：VOICE_V2_VOLC_VOICE_CHAT_CONFIG_JSON 缺少最小必需段：ASRConfig",
        ):
            prepare_voice_v2_provider_session(auth_session_id="session-1")

    def test_build_voice_v2_session_bootstrap_blocks_when_rtc_app_key_missing(self, monkeypatch):
        monkeypatch.setenv("VOICE_V2_ENABLED", "1")
        monkeypatch.setenv("VOICE_V2_VOLC_AK", "ak")
        monkeypatch.setenv("VOICE_V2_VOLC_SK", "sk")
        monkeypatch.setenv("VOICE_V2_VOLC_RTC_APP_ID", "rtc-app")
        monkeypatch.delenv("VOICE_V2_VOLC_RTC_APP_KEY", raising=False)
        monkeypatch.setenv(
            "VOICE_V2_VOLC_VOICE_CHAT_CONFIG_JSON",
            '{"AgentConfig": {"WelcomeMessage": "hi"}, "ASRConfig": {"Provider": "volc"}}',
        )

        payload = build_voice_v2_session_bootstrap()

        assert payload["connection_policy"] == {
            "lifecycle": "server_bootstrap",
            "status": "blocked",
            "blocked_reasons": ["provider bootstrap 配置未齐"],
            "missing_config": ["VOICE_V2_VOLC_RTC_APP_KEY"],
            "required_config": [
                "VOICE_V2_VOLC_AK",
                "VOICE_V2_VOLC_SK",
                "VOICE_V2_VOLC_RTC_APP_ID",
                "VOICE_V2_VOLC_RTC_APP_KEY",
                "VOICE_V2_VOLC_VOICE_CHAT_CONFIG_JSON",
            ],
        }
        assert payload["provider_config"] == {
            "rtc_app_id_configured": True,
            "rtc_app_key_configured": False,
            "account_credentials_configured": True,
            "voice_chat_configured": True,
            "voice_chat_config_keys": ["AgentConfig", "Config"],
        }

    def test_prepare_voice_v2_provider_session_requests_real_provider_bootstrap(self, monkeypatch):
        monkeypatch.setenv("VOICE_V2_ENABLED", "1")
        monkeypatch.setenv("VOICE_V2_VOLC_AK", "ak")
        monkeypatch.setenv("VOICE_V2_VOLC_SK", "sk")
        monkeypatch.setenv("VOICE_V2_VOLC_RTC_APP_ID", "rtc-app")
        monkeypatch.setenv("VOICE_V2_VOLC_RTC_APP_KEY", "rtc-key")
        monkeypatch.setenv(
            "VOICE_V2_VOLC_VOICE_CHAT_CONFIG_JSON",
            '{"AgentConfig": {"WelcomeMessage": "hi"}, "ASRConfig": {"Provider": "volc"}}',
        )
        captured_calls: list[tuple[str, str, dict[str, object]]] = []

        def fake_post_signed_openapi_json(*, action: str, version: str, payload: dict[str, object], now: datetime):
            captured_calls.append((action, version, payload))
            assert now.tzinfo is UTC
            if action == "StartVoiceChat":
                return {
                    "ResponseMetadata": {"RequestId": "start-voice-chat-request"},
                    "Result": "ok",
                }
            raise AssertionError(f"unexpected action {action}")

        monkeypatch.setattr(
            "services.voice_v2_session._post_signed_openapi_json",
            fake_post_signed_openapi_json,
        )

        payload = prepare_voice_v2_provider_session(auth_session_id="auth-session-123")

        assert payload["status"] == "prepared"
        assert payload["app_id"] == "rtc-app"
        assert payload["room_id"].startswith("pv-room-")
        assert payload["user_id"].isdigit()
        assert payload["agent_user_id"] == f"pv-agent-{payload['bootstrap_id'].split('-')[0]}"
        assert payload["task_id"].startswith("pv-task-")
        assert re.match(r"^[0-9a-f-]{36}$", payload["bootstrap_id"])
        assert payload["ttl_seconds"] == 300
        assert payload["rtc_app_token"].startswith("001rtc-app")
        assert payload["rtc_token_request_id"] is None
        assert payload["start_voice_chat_request_id"] == "start-voice-chat-request"
        assert payload["provider_session_status"] == "started"
        assert payload["voice_chat_config_keys"] == ["AgentConfig", "Config"]
        assert "StartVoiceChat 已调用" in payload["next_action"]
        assert captured_calls == [
            (
                "StartVoiceChat",
                "2024-12-01",
                {
                    "AppId": "rtc-app",
                    "RoomId": payload["room_id"],
                    "TaskId": payload["task_id"],
                    "AgentConfig": {
                        "WelcomeMessage": "hi",
                        "TargetUserId": [payload["user_id"]],
                        "UserId": f"pv-agent-{payload['bootstrap_id'].split('-')[0]}",
                    },
                    "Config": {
                        "ASRConfig": {"Provider": "volc"},
                    },
                },
            ),
        ]

    def test_prepare_voice_v2_provider_session_accepts_official_demo_voice_chat_shape(self, monkeypatch):
        monkeypatch.setenv("VOICE_V2_ENABLED", "1")
        monkeypatch.setenv("VOICE_V2_VOLC_AK", "ak")
        monkeypatch.setenv("VOICE_V2_VOLC_SK", "sk")
        monkeypatch.setenv("VOICE_V2_VOLC_RTC_APP_ID", "rtc-app")
        monkeypatch.setenv("VOICE_V2_VOLC_RTC_APP_KEY", "rtc-key")
        monkeypatch.setenv(
            "VOICE_V2_VOLC_VOICE_CHAT_CONFIG_JSON",
            """
            {
              "VoiceChat": {
                "AppId": "demo-app",
                "RoomId": "demo-room",
                "TaskId": "demo-task",
                "AgentConfig": {"WelcomeMessage": "hi"},
                "Config": {
                  "ASRConfig": {"Provider": "volcano"},
                  "TTSConfig": {"Provider": "volcano"}
                }
              }
            }
            """,
        )
        captured_calls: list[tuple[str, str, dict[str, object]]] = []

        def fake_post_signed_openapi_json(*, action: str, version: str, payload: dict[str, object], now: datetime):
            captured_calls.append((action, version, payload))
            if action == "StartVoiceChat":
                return {
                    "ResponseMetadata": {"RequestId": "start-voice-chat-request"},
                    "Result": "ok",
                }
            raise AssertionError(f"unexpected action {action}")

        monkeypatch.setattr(
            "services.voice_v2_session._post_signed_openapi_json",
            fake_post_signed_openapi_json,
        )

        payload = prepare_voice_v2_provider_session(auth_session_id="auth-session-123")

        assert payload["voice_chat_config_keys"] == ["AgentConfig", "Config"]
        assert captured_calls == [(
            "StartVoiceChat",
            "2024-12-01",
            {
                "AppId": "rtc-app",
                "RoomId": payload["room_id"],
                "TaskId": payload["task_id"],
                "AgentConfig": {
                    "WelcomeMessage": "hi",
                    "TargetUserId": [payload["user_id"]],
                    "UserId": f"pv-agent-{payload['bootstrap_id'].split('-')[0]}",
                },
                "Config": {
                    "ASRConfig": {"Provider": "volcano"},
                    "TTSConfig": {"Provider": "volcano"},
                },
            },
        )]

    def test_prepare_voice_v2_provider_session_stops_provider_when_start_voice_chat_fails(self, monkeypatch):
        monkeypatch.setenv("VOICE_V2_ENABLED", "1")
        monkeypatch.setenv("VOICE_V2_VOLC_AK", "ak")
        monkeypatch.setenv("VOICE_V2_VOLC_SK", "sk")
        monkeypatch.setenv("VOICE_V2_VOLC_RTC_APP_ID", "rtc-app")
        monkeypatch.setenv("VOICE_V2_VOLC_RTC_APP_KEY", "rtc-key")
        monkeypatch.setenv(
            "VOICE_V2_VOLC_VOICE_CHAT_CONFIG_JSON",
            '{"AgentConfig": {"WelcomeMessage": "hi"}, "ASRConfig": {"Provider": "volc"}}',
        )
        captured_calls: list[tuple[str, str, dict[str, object]]] = []

        def fake_post_signed_openapi_json(*, action: str, version: str, payload: dict[str, object], now: datetime):
            captured_calls.append((action, version, payload))
            assert now.tzinfo is UTC
            if action == "StartVoiceChat":
                raise VoiceV2ProviderError("start failed")
            if action == "StopVoiceChat":
                return {
                    "ResponseMetadata": {"RequestId": "stop-after-start-fail-request"},
                    "Result": "ok",
                }
            raise AssertionError(f"unexpected action {action}")

        monkeypatch.setattr(
            "services.voice_v2_session._post_signed_openapi_json",
            fake_post_signed_openapi_json,
        )

        with pytest.raises(VoiceV2ProviderError, match="start failed；prepare 失败后已补偿 StopVoiceChat"):
            prepare_voice_v2_provider_session(auth_session_id="auth-session-123")

        assert [call[0] for call in captured_calls] == [
            "StartVoiceChat",
            "StopVoiceChat",
        ]
        assert captured_calls[0][2]["RoomId"] == captured_calls[1][2]["RoomId"]
        assert captured_calls[0][2]["TaskId"] == captured_calls[1][2]["TaskId"]

    def test_prepare_voice_v2_provider_session_reports_cleanup_failure_when_start_voice_chat_fails(self, monkeypatch):
        monkeypatch.setenv("VOICE_V2_ENABLED", "1")
        monkeypatch.setenv("VOICE_V2_VOLC_AK", "ak")
        monkeypatch.setenv("VOICE_V2_VOLC_SK", "sk")
        monkeypatch.setenv("VOICE_V2_VOLC_RTC_APP_ID", "rtc-app")
        monkeypatch.setenv("VOICE_V2_VOLC_RTC_APP_KEY", "rtc-key")
        monkeypatch.setenv(
            "VOICE_V2_VOLC_VOICE_CHAT_CONFIG_JSON",
            '{"AgentConfig": {"WelcomeMessage": "hi"}, "ASRConfig": {"Provider": "volc"}}',
        )
        call_actions: list[str] = []

        def fake_post_signed_openapi_json(*, action: str, version: str, payload: dict[str, object], now: datetime):
            call_actions.append(action)
            if action == "StartVoiceChat":
                raise VoiceV2ProviderError("start failed")
            if action == "StopVoiceChat":
                raise VoiceV2ProviderError("stop cleanup failed")
            raise AssertionError(f"unexpected action {action}")

        monkeypatch.setattr(
            "services.voice_v2_session._post_signed_openapi_json",
            fake_post_signed_openapi_json,
        )

        with pytest.raises(
            VoiceV2ProviderError,
            match="start failed；prepare 失败后补偿 StopVoiceChat 也失败：stop cleanup failed",
        ):
            prepare_voice_v2_provider_session(auth_session_id="auth-session-123")

        assert call_actions == ["StartVoiceChat", "StopVoiceChat"]

    def test_stop_voice_v2_provider_session_calls_real_provider_stop(self, monkeypatch):
        monkeypatch.setenv("VOICE_V2_VOLC_AK", "ak")
        monkeypatch.setenv("VOICE_V2_VOLC_SK", "sk")
        captured_calls: list[tuple[str, str, dict[str, object]]] = []

        def fake_post_signed_openapi_json(*, action: str, version: str, payload: dict[str, object], now: datetime):
            captured_calls.append((action, version, payload))
            assert now.tzinfo is UTC
            return {
                "ResponseMetadata": {"RequestId": "stop-voice-chat-request"},
                "Result": "ok",
            }

        monkeypatch.setattr(
            "services.voice_v2_session._post_signed_openapi_json",
            fake_post_signed_openapi_json,
        )

        payload = stop_voice_v2_provider_session(
            bootstrap_id="bootstrap-123",
            app_id="rtc-app",
            room_id="pv-room-123",
            task_id="pv-task-123",
            reason="manual_disconnect",
        )

        assert payload["status"] == "stopped"
        assert payload["provider_session_status"] == "stopped"
        assert payload["stop_reason"] == "manual_disconnect"
        assert payload["stop_voice_chat_request_id"] == "stop-voice-chat-request"
        assert "用户手动断开" in payload["next_action"]
        assert captured_calls == [
            (
                "StopVoiceChat",
                "2024-12-01",
                {
                    "AppId": "rtc-app",
                    "RoomId": "pv-room-123",
                    "TaskId": "pv-task-123",
                },
            ),
        ]

    def test_stop_voice_v2_provider_session_supports_page_hide_reason(self, monkeypatch):
        monkeypatch.setenv("VOICE_V2_VOLC_AK", "ak")
        monkeypatch.setenv("VOICE_V2_VOLC_SK", "sk")
        monkeypatch.setattr(
            "services.voice_v2_session._post_signed_openapi_json",
            lambda **_: {
                "ResponseMetadata": {"RequestId": "stop-page-hide-request"},
                "Result": "ok",
            },
        )

        payload = stop_voice_v2_provider_session(
            bootstrap_id="bootstrap-123",
            app_id="rtc-app",
            room_id="pv-room-123",
            task_id="pv-task-123",
            reason="page_hide",
        )

        assert payload["stop_reason"] == "page_hide"
        assert "页面离开 / 刷新" in payload["next_action"]

    def test_stop_voice_v2_provider_session_supports_provider_session_expired_reason(self, monkeypatch):
        monkeypatch.setenv("VOICE_V2_VOLC_AK", "ak")
        monkeypatch.setenv("VOICE_V2_VOLC_SK", "sk")
        monkeypatch.setattr(
            "services.voice_v2_session._post_signed_openapi_json",
            lambda **_: {
                "ResponseMetadata": {"RequestId": "stop-expired-request"},
                "Result": "ok",
            },
        )

        payload = stop_voice_v2_provider_session(
            bootstrap_id="bootstrap-123",
            app_id="rtc-app",
            room_id="pv-room-123",
            task_id="pv-task-123",
            reason="provider_session_expired",
        )

        assert payload["stop_reason"] == "provider_session_expired"
        assert "超过 TTL" in payload["next_action"]

    def test_stop_voice_v2_provider_session_supports_remote_media_wait_timeout_reason(self, monkeypatch):
        monkeypatch.setenv("VOICE_V2_VOLC_AK", "ak")
        monkeypatch.setenv("VOICE_V2_VOLC_SK", "sk")
        monkeypatch.setattr(
            "services.voice_v2_session._post_signed_openapi_json",
            lambda **_: {
                "ResponseMetadata": {"RequestId": "stop-remote-media-timeout-request"},
                "Result": "ok",
            },
        )

        payload = stop_voice_v2_provider_session(
            bootstrap_id="bootstrap-123",
            app_id="rtc-app",
            room_id="pv-room-123",
            task_id="pv-task-123",
            reason="remote_media_wait_timeout",
        )

        assert payload["stop_reason"] == "remote_media_wait_timeout"
        assert "远端媒体等待超时" in payload["next_action"]

    def test_stop_voice_v2_provider_session_supports_rtc_join_timeout_reason(self, monkeypatch):
        monkeypatch.setenv("VOICE_V2_VOLC_AK", "ak")
        monkeypatch.setenv("VOICE_V2_VOLC_SK", "sk")
        monkeypatch.setattr(
            "services.voice_v2_session._post_signed_openapi_json",
            lambda **_: {
                "ResponseMetadata": {"RequestId": "stop-rtc-join-timeout-request"},
                "Result": "ok",
            },
        )

        payload = stop_voice_v2_provider_session(
            bootstrap_id="bootstrap-123",
            app_id="rtc-app",
            room_id="pv-room-123",
            task_id="pv-task-123",
            reason="rtc_join_timeout",
        )

        assert payload["stop_reason"] == "rtc_join_timeout"
        assert "RTC joinRoom 超时" in payload["next_action"]

    def test_trigger_voice_v2_finish_speech_recognition_calls_update_voice_chat(self, monkeypatch):
        monkeypatch.setenv("VOICE_V2_VOLC_AK", "ak")
        monkeypatch.setenv("VOICE_V2_VOLC_SK", "sk")
        captured_calls: list[tuple[str, str, dict[str, object]]] = []

        def fake_post_signed_openapi_json(*, action: str, version: str, payload: dict[str, object], now: datetime):
            captured_calls.append((action, version, payload))
            assert now.tzinfo is UTC
            return {
                "ResponseMetadata": {"RequestId": "update-voice-chat-request"},
                "Result": {},
            }

        monkeypatch.setattr(
            "services.voice_v2_session._post_signed_openapi_json",
            fake_post_signed_openapi_json,
        )

        payload = trigger_voice_v2_finish_speech_recognition(
            bootstrap_id="bootstrap-123",
            app_id="rtc-app",
            room_id="pv-room-123",
            task_id="pv-task-123",
        )

        assert payload["status"] == "triggered"
        assert payload["command"] == "FinishSpeechRecognition"
        assert payload["update_voice_chat_request_id"] == "update-voice-chat-request"
        assert "FinishSpeechRecognition" in payload["next_action"]
        assert captured_calls == [
            (
                "UpdateVoiceChat",
                "2024-12-01",
                {
                    "AppId": "rtc-app",
                    "RoomId": "pv-room-123",
                    "TaskId": "pv-task-123",
                    "Command": "FinishSpeechRecognition",
                },
            ),
        ]


class TestVoiceV2Router:
    def _build_client(self, tmp_path: Path, monkeypatch, db_name: str) -> TestClient:
        for env_name in (
            "VOICE_V2_VOLC_AK",
            "VOICE_V2_VOLC_SK",
            "VOICE_V2_VOLC_RTC_APP_ID",
            "VOICE_V2_VOLC_RTC_APP_KEY",
            "VOICE_V2_VOLC_VOICE_CHAT_CONFIG_JSON",
        ):
            if env_name not in os.environ:
                monkeypatch.setenv(env_name, "")
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

    def test_voice_v2_session_route_requires_auth(self, tmp_path, monkeypatch):
        with self._build_client(tmp_path, monkeypatch, "voice-v2-auth.db") as client:
            response = client.get("/api/voice-v2/session")

        _reset_shared_db()
        limiter._storage.reset()
        assert response.status_code == 401

    def test_voice_v2_session_route_returns_bootstrap_for_authenticated_session(self, tmp_path, monkeypatch):
        monkeypatch.setenv("VOICE_V2_ENABLED", "1")

        with self._build_client(tmp_path, monkeypatch, "voice-v2-success.db") as client:
            login = client.post("/api/auth/login", json={"password": "secret-pass"})
            assert login.status_code == 200

            response = client.get("/api/voice-v2/session")

        _reset_shared_db()
        limiter._storage.reset()
        assert response.status_code == 200
        assert response.json() == {
            "phase": "phase_1_bootstrap",
            "feature_flag": True,
            "provider": "doubao",
            "session_mode": "rtc_voice_chat",
            "browser_policy": {
                "system_browser_required": True,
                "embedded_browser_support": "degraded",
            },
            "memory_policy": {
                "ownership": "pervault",
                "bridge_mode": "not_connected",
                "write_enabled": False,
            },
            "connection_policy": {
                "lifecycle": "server_bootstrap",
                "status": "blocked",
                "blocked_reasons": ["provider bootstrap 配置未齐"],
                "missing_config": [
                    "VOICE_V2_VOLC_AK",
                    "VOICE_V2_VOLC_SK",
                    "VOICE_V2_VOLC_RTC_APP_ID",
                    "VOICE_V2_VOLC_RTC_APP_KEY",
                    "VOICE_V2_VOLC_VOICE_CHAT_CONFIG_JSON",
                ],
                "required_config": [
                    "VOICE_V2_VOLC_AK",
                    "VOICE_V2_VOLC_SK",
                    "VOICE_V2_VOLC_RTC_APP_ID",
                    "VOICE_V2_VOLC_RTC_APP_KEY",
                    "VOICE_V2_VOLC_VOICE_CHAT_CONFIG_JSON",
                ],
            },
            "provider_config": {
                "rtc_app_id_configured": False,
                "rtc_app_key_configured": False,
                "account_credentials_configured": False,
                "voice_chat_configured": False,
                "voice_chat_config_keys": [],
            },
            "legacy_fallback_url": "/voice",
            "next_step": "下一步是接 RTC token / StartVoiceChat，接通真实媒体链路后再开始 read-only memory bridge。",
        }

    def test_voice_v2_prepare_route_returns_prepared_shell_session(self, tmp_path, monkeypatch):
        monkeypatch.setenv("VOICE_V2_ENABLED", "1")
        monkeypatch.setenv("VOICE_V2_VOLC_AK", "ak")
        monkeypatch.setenv("VOICE_V2_VOLC_SK", "sk")
        monkeypatch.setenv("VOICE_V2_VOLC_RTC_APP_ID", "rtc-app")
        monkeypatch.setenv("VOICE_V2_VOLC_RTC_APP_KEY", "rtc-key")
        monkeypatch.setenv(
            "VOICE_V2_VOLC_VOICE_CHAT_CONFIG_JSON",
            '{"AgentConfig": {"WelcomeMessage": "hi"}, "ASRConfig": {"Provider": "volc"}}',
        )

        monkeypatch.setattr(
            "services.voice_v2_session._post_signed_openapi_json",
            lambda **_: {
                "ResponseMetadata": {"RequestId": "start-voice-chat-request"},
                "Result": "ok",
            },
        )

        with self._build_client(tmp_path, monkeypatch, "voice-v2-prepare.db") as client:
            login = client.post("/api/auth/login", json={"password": "secret-pass"})
            assert login.status_code == 200

            response = client.post("/api/voice-v2/session/prepare")

        _reset_shared_db()
        limiter._storage.reset()
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "prepared"
        assert body["provider"] == "doubao"
        assert body["session_mode"] == "rtc_voice_chat"
        assert body["app_id"] == "rtc-app"
        assert body["room_id"].startswith("pv-room-")
        assert body["user_id"].isdigit()
        assert body["task_id"].startswith("pv-task-")
        assert body["rtc_app_token"].startswith("001rtc-app")
        assert body["rtc_token_request_id"] is None
        assert body["start_voice_chat_request_id"] == "start-voice-chat-request"
        assert body["provider_session_status"] == "started"
        assert body["voice_chat_config_keys"] == ["AgentConfig", "Config"]

    def test_voice_v2_prepare_route_fails_when_provider_config_missing(self, tmp_path, monkeypatch):
        monkeypatch.setenv("VOICE_V2_ENABLED", "1")

        with self._build_client(tmp_path, monkeypatch, "voice-v2-prepare-fail.db") as client:
            login = client.post("/api/auth/login", json={"password": "secret-pass"})
            assert login.status_code == 200

            response = client.post("/api/voice-v2/session/prepare")

        _reset_shared_db()
        limiter._storage.reset()
        assert response.status_code == 503
        assert "provider bootstrap 未就绪" in response.json()["detail"]

    def test_voice_v2_prepare_route_fails_when_voice_chat_config_shape_invalid(self, tmp_path, monkeypatch):
        monkeypatch.setenv("VOICE_V2_ENABLED", "1")
        monkeypatch.setenv("VOICE_V2_VOLC_AK", "ak")
        monkeypatch.setenv("VOICE_V2_VOLC_SK", "sk")
        monkeypatch.setenv("VOICE_V2_VOLC_RTC_APP_ID", "rtc-app")
        monkeypatch.setenv("VOICE_V2_VOLC_RTC_APP_KEY", "rtc-key")
        monkeypatch.setenv(
            "VOICE_V2_VOLC_VOICE_CHAT_CONFIG_JSON",
            '{"AgentConfig": {"WelcomeMessage": "hi"}}',
        )

        with self._build_client(tmp_path, monkeypatch, "voice-v2-prepare-invalid-config.db") as client:
            login = client.post("/api/auth/login", json={"password": "secret-pass"})
            assert login.status_code == 200

            response = client.post("/api/voice-v2/session/prepare")

        _reset_shared_db()
        limiter._storage.reset()
        assert response.status_code == 503
        assert "缺少最小必需段：ASRConfig" in response.json()["detail"]

    def test_voice_v2_prepare_route_surfaces_provider_errors(self, tmp_path, monkeypatch):
        monkeypatch.setenv("VOICE_V2_ENABLED", "1")
        monkeypatch.setenv("VOICE_V2_VOLC_AK", "ak")
        monkeypatch.setenv("VOICE_V2_VOLC_SK", "sk")
        monkeypatch.setenv("VOICE_V2_VOLC_RTC_APP_ID", "rtc-app")
        monkeypatch.setenv("VOICE_V2_VOLC_RTC_APP_KEY", "rtc-key")
        monkeypatch.setenv(
            "VOICE_V2_VOLC_VOICE_CHAT_CONFIG_JSON",
            '{"AgentConfig": {"WelcomeMessage": "hi"}, "ASRConfig": {"Provider": "volc"}}',
        )
        monkeypatch.setattr(
            "services.voice_v2_session._post_signed_openapi_json",
            lambda **_: (_ for _ in ()).throw(RuntimeError("should not be used")),
        )
        monkeypatch.setattr(
            "services.voice_v2_session._generate_rtc_app_token",
            lambda **_: (_ for _ in ()).throw(VoiceV2ProviderError("bad provider")),
        )

        with self._build_client(tmp_path, monkeypatch, "voice-v2-provider-fail.db") as client:
            login = client.post("/api/auth/login", json={"password": "secret-pass"})
            assert login.status_code == 200

            response = client.post("/api/voice-v2/session/prepare")

        _reset_shared_db()
        limiter._storage.reset()
        assert response.status_code == 502
        assert "bad provider" in response.json()["detail"]

    def test_voice_v2_stop_route_returns_stopped_provider_session(self, tmp_path, monkeypatch):
        monkeypatch.setenv("VOICE_V2_VOLC_AK", "ak")
        monkeypatch.setenv("VOICE_V2_VOLC_SK", "sk")
        monkeypatch.setattr(
            "services.voice_v2_session._post_signed_openapi_json",
            lambda **_: {
                "ResponseMetadata": {"RequestId": "stop-voice-chat-request"},
                "Result": "ok",
            },
        )

        with self._build_client(tmp_path, monkeypatch, "voice-v2-stop.db") as client:
            login = client.post("/api/auth/login", json={"password": "secret-pass"})
            assert login.status_code == 200

            response = client.post(
                "/api/voice-v2/session/stop",
                json={
                    "bootstrap_id": "bootstrap-123",
                    "app_id": "rtc-app",
                    "room_id": "pv-room-123",
                    "task_id": "pv-task-123",
                    "reason": "manual_disconnect",
                },
            )

        _reset_shared_db()
        limiter._storage.reset()
        assert response.status_code == 200
        assert response.json() == {
            "bootstrap_id": "bootstrap-123",
            "provider": "doubao",
            "session_mode": "rtc_voice_chat",
            "status": "stopped",
            "app_id": "rtc-app",
            "room_id": "pv-room-123",
            "task_id": "pv-task-123",
            "issued_at": response.json()["issued_at"],
            "provider_session_status": "stopped",
            "stop_reason": "manual_disconnect",
            "stop_voice_chat_request_id": "stop-voice-chat-request",
            "next_action": "Provider 侧 VoiceChat 已请求停止（原因：用户手动断开）；前端应保持 RTC 已离房并清空本地会话状态。",
        }

    def test_voice_v2_stop_route_defaults_reason_to_unknown(self, tmp_path, monkeypatch):
        monkeypatch.setenv("VOICE_V2_VOLC_AK", "ak")
        monkeypatch.setenv("VOICE_V2_VOLC_SK", "sk")
        monkeypatch.setattr(
            "services.voice_v2_session._post_signed_openapi_json",
            lambda **_: {
                "ResponseMetadata": {"RequestId": "stop-voice-chat-request"},
                "Result": "ok",
            },
        )

        with self._build_client(tmp_path, monkeypatch, "voice-v2-stop-default-reason.db") as client:
            login = client.post("/api/auth/login", json={"password": "secret-pass"})
            assert login.status_code == 200

            response = client.post(
                "/api/voice-v2/session/stop",
                json={
                    "bootstrap_id": "bootstrap-123",
                    "app_id": "rtc-app",
                    "room_id": "pv-room-123",
                    "task_id": "pv-task-123",
                },
            )

        _reset_shared_db()
        limiter._storage.reset()
        assert response.status_code == 200
        body = response.json()
        assert body["stop_reason"] == "unknown"
        assert "未标明原因" in body["next_action"]

    def test_voice_v2_stop_route_accepts_page_hide_reason(self, tmp_path, monkeypatch):
        monkeypatch.setenv("VOICE_V2_VOLC_AK", "ak")
        monkeypatch.setenv("VOICE_V2_VOLC_SK", "sk")
        monkeypatch.setattr(
            "services.voice_v2_session._post_signed_openapi_json",
            lambda **_: {
                "ResponseMetadata": {"RequestId": "stop-page-hide-request"},
                "Result": "ok",
            },
        )

        with self._build_client(tmp_path, monkeypatch, "voice-v2-stop-page-hide.db") as client:
            login = client.post("/api/auth/login", json={"password": "secret-pass"})
            assert login.status_code == 200

            response = client.post(
                "/api/voice-v2/session/stop",
                json={
                    "bootstrap_id": "bootstrap-123",
                    "app_id": "rtc-app",
                    "room_id": "pv-room-123",
                    "task_id": "pv-task-123",
                    "reason": "page_hide",
                },
            )

        _reset_shared_db()
        limiter._storage.reset()
        assert response.status_code == 200
        body = response.json()
        assert body["stop_reason"] == "page_hide"

    def test_voice_v2_stop_route_accepts_provider_session_expired_reason(self, tmp_path, monkeypatch):
        monkeypatch.setenv("VOICE_V2_VOLC_AK", "ak")
        monkeypatch.setenv("VOICE_V2_VOLC_SK", "sk")
        monkeypatch.setattr(
            "services.voice_v2_session._post_signed_openapi_json",
            lambda **_: {
                "ResponseMetadata": {"RequestId": "stop-expired-request"},
                "Result": "ok",
            },
        )

        with self._build_client(tmp_path, monkeypatch, "voice-v2-stop-expired.db") as client:
            login = client.post("/api/auth/login", json={"password": "secret-pass"})
            assert login.status_code == 200

            response = client.post(
                "/api/voice-v2/session/stop",
                json={
                    "bootstrap_id": "bootstrap-123",
                    "app_id": "rtc-app",
                    "room_id": "pv-room-123",
                    "task_id": "pv-task-123",
                    "reason": "provider_session_expired",
                },
            )

        _reset_shared_db()
        limiter._storage.reset()
        assert response.status_code == 200
        body = response.json()
        assert body["stop_reason"] == "provider_session_expired"
        assert "超过 TTL" in body["next_action"]

    def test_voice_v2_stop_route_accepts_remote_media_wait_timeout_reason(self, tmp_path, monkeypatch):
        monkeypatch.setenv("VOICE_V2_VOLC_AK", "ak")
        monkeypatch.setenv("VOICE_V2_VOLC_SK", "sk")
        monkeypatch.setattr(
            "services.voice_v2_session._post_signed_openapi_json",
            lambda **_: {
                "ResponseMetadata": {"RequestId": "stop-remote-media-timeout-request"},
                "Result": "ok",
            },
        )

        with self._build_client(tmp_path, monkeypatch, "voice-v2-stop-remote-media-timeout.db") as client:
            login = client.post("/api/auth/login", json={"password": "secret-pass"})
            assert login.status_code == 200

            response = client.post(
                "/api/voice-v2/session/stop",
                json={
                    "bootstrap_id": "bootstrap-123",
                    "app_id": "rtc-app",
                    "room_id": "pv-room-123",
                    "task_id": "pv-task-123",
                    "reason": "remote_media_wait_timeout",
                },
            )

        _reset_shared_db()
        limiter._storage.reset()
        assert response.status_code == 200
        body = response.json()
        assert body["stop_reason"] == "remote_media_wait_timeout"
        assert "远端媒体等待超时" in body["next_action"]

    def test_voice_v2_stop_route_accepts_rtc_join_timeout_reason(self, tmp_path, monkeypatch):
        monkeypatch.setenv("VOICE_V2_VOLC_AK", "ak")
        monkeypatch.setenv("VOICE_V2_VOLC_SK", "sk")
        monkeypatch.setattr(
            "services.voice_v2_session._post_signed_openapi_json",
            lambda **_: {
                "ResponseMetadata": {"RequestId": "stop-rtc-join-timeout-request"},
                "Result": "ok",
            },
        )

        with self._build_client(tmp_path, monkeypatch, "voice-v2-stop-rtc-join-timeout.db") as client:
            login = client.post("/api/auth/login", json={"password": "secret-pass"})
            assert login.status_code == 200

            response = client.post(
                "/api/voice-v2/session/stop",
                json={
                    "bootstrap_id": "bootstrap-123",
                    "app_id": "rtc-app",
                    "room_id": "pv-room-123",
                    "task_id": "pv-task-123",
                    "reason": "rtc_join_timeout",
                },
            )

        _reset_shared_db()
        limiter._storage.reset()
        assert response.status_code == 200
        body = response.json()
        assert body["stop_reason"] == "rtc_join_timeout"
        assert "RTC joinRoom 超时" in body["next_action"]

    def test_voice_v2_trigger_route_sends_finish_speech_recognition(self, tmp_path, monkeypatch):
        monkeypatch.setenv("VOICE_V2_VOLC_AK", "ak")
        monkeypatch.setenv("VOICE_V2_VOLC_SK", "sk")
        captured_calls: list[tuple[str, str, dict[str, object]]] = []

        def fake_post_signed_openapi_json(*, action: str, version: str, payload: dict[str, object], now: datetime):
            captured_calls.append((action, version, payload))
            return {
                "ResponseMetadata": {"RequestId": "update-voice-chat-request"},
                "Result": "ok",
            }

        monkeypatch.setattr(
            "services.voice_v2_session._post_signed_openapi_json",
            fake_post_signed_openapi_json,
        )

        with self._build_client(tmp_path, monkeypatch, "voice-v2-trigger.db") as client:
            login = client.post("/api/auth/login", json={"password": "secret-pass"})
            assert login.status_code == 200

            response = client.post(
                "/api/voice-v2/session/trigger",
                json={
                    "bootstrap_id": "bootstrap-123",
                    "app_id": "rtc-app",
                    "room_id": "pv-room-123",
                    "task_id": "pv-task-123",
                },
            )

        _reset_shared_db()
        limiter._storage.reset()
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "triggered"
        assert body["command"] == "FinishSpeechRecognition"
        assert body["update_voice_chat_request_id"] == "update-voice-chat-request"
        assert captured_calls == [
            (
                "UpdateVoiceChat",
                "2024-12-01",
                {
                    "AppId": "rtc-app",
                    "RoomId": "pv-room-123",
                    "TaskId": "pv-task-123",
                    "Command": "FinishSpeechRecognition",
                },
            ),
        ]

    def test_voice_v2_stop_route_surfaces_provider_errors(self, tmp_path, monkeypatch):
        monkeypatch.setenv("VOICE_V2_VOLC_AK", "ak")
        monkeypatch.setenv("VOICE_V2_VOLC_SK", "sk")
        monkeypatch.setattr(
            "services.voice_v2_session._stop_voice_chat",
            lambda **_: (_ for _ in ()).throw(VoiceV2ProviderError("stop failed")),
        )

        with self._build_client(tmp_path, monkeypatch, "voice-v2-stop-fail.db") as client:
            login = client.post("/api/auth/login", json={"password": "secret-pass"})
            assert login.status_code == 200

            response = client.post(
                "/api/voice-v2/session/stop",
                json={
                    "bootstrap_id": "bootstrap-123",
                    "app_id": "rtc-app",
                    "room_id": "pv-room-123",
                    "task_id": "pv-task-123",
                    "reason": "rtc_connect_failed",
                },
            )

        _reset_shared_db()
        limiter._storage.reset()
        assert response.status_code == 502
        assert "stop failed" in response.json()["detail"]
