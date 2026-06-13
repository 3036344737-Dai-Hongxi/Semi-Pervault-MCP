import base64
import json
import os
import re
import secrets
import struct
import uuid
import hashlib
import hmac
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from urllib.parse import quote

import httpx


VOICE_V2_PHASE = "phase_1_bootstrap"
VOICE_V2_PROVIDER = "doubao"
VOICE_V2_SESSION_MODE = "rtc_voice_chat"
VOICE_V2_LEGACY_FALLBACK_URL = "/voice"
VOICE_V2_NEXT_STEP = "下一步是接 RTC token / StartVoiceChat，接通真实媒体链路后再开始 read-only memory bridge。"
VOICE_V2_PREPARED_NEXT_ACTION = "RTC token 已用 AppKey 本地生成且 StartVoiceChat 已调用；前端下一步是接 RTC SDK 进房并处理真实媒体事件。"
VOICE_V2_STOP_REASON_LABELS = {
    "manual_disconnect": "用户手动断开",
    "superseded_attempt": "新一轮连接替换旧会话",
    "rtc_connect_failed": "RTC 建连失败后的补偿清理",
    "rtc_join_timeout": "RTC joinRoom 超时后的补偿清理",
    "remote_media_wait_timeout": "RTC 已进房但远端媒体等待超时后的补偿清理",
    "provider_session_expired": "Provider 会话超过 TTL 后的补偿清理",
    "page_hide": "页面离开 / 刷新时的 keepalive 清理",
    "component_unmount": "页面卸载时的补偿清理",
    "unknown": "未标明原因",
}
VOICE_V2_REQUIRED_PROVIDER_CONFIG = (
    "VOICE_V2_VOLC_AK",
    "VOICE_V2_VOLC_SK",
    "VOICE_V2_VOLC_RTC_APP_ID",
    "VOICE_V2_VOLC_RTC_APP_KEY",
    "VOICE_V2_VOLC_VOICE_CHAT_CONFIG_JSON",
)
VOICE_V2_REQUIRED_VOICE_CHAT_CONFIG_SECTIONS = (
    "AgentConfig",
    "ASRConfig",
)
_SAFE_IDENTIFIER_RE = re.compile(r"[^a-zA-Z0-9_@.\-]")
_VOLC_OPENAPI_SERVICE = "rtc"
_VOLC_OPENAPI_REGION = "cn-north-1"
_VOLC_OPENAPI_HOST = "rtc.volcengineapi.com"
_VOLC_START_VOICE_CHAT_VERSION = "2024-12-01"
_VOLC_UPDATE_VOICE_CHAT_VERSION = "2024-12-01"
_VOLC_STOP_VOICE_CHAT_VERSION = "2024-12-01"
_VOLC_CONTENT_TYPE = "application/json"
_VOLC_RTC_TOKEN_VERSION = "001"
_VOLC_RTC_PRIV_PUBLISH_STREAM = 0
_VOLC_RTC_PRIV_PUBLISH_AUDIO_STREAM = 1
_VOLC_RTC_PRIV_PUBLISH_VIDEO_STREAM = 2
_VOLC_RTC_PRIV_PUBLISH_DATA_STREAM = 3
_VOLC_RTC_PRIV_SUBSCRIBE_STREAM = 4


class VoiceV2ConfigurationError(RuntimeError):
    pass


class VoiceV2ProviderError(RuntimeError):
    pass


def _read_flag(name: str, default: bool) -> bool:
    raw_value = os.getenv(name, "1" if default else "0").strip().lower()
    return raw_value in {"1", "true", "yes", "on"}


def _read_env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _read_positive_int(name: str, default: int) -> int:
    raw_value = _read_env(name)
    if not raw_value:
        return default
    try:
        parsed = int(raw_value)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def _sanitize_identifier(raw_value: str, *, fallback: str) -> str:
    sanitized = _SAFE_IDENTIFIER_RE.sub("-", raw_value)[:64].strip("-")
    return sanitized or fallback


def _build_numeric_user_id(raw_value: str) -> str:
    digest = hashlib.sha256(raw_value.encode("utf-8")).hexdigest()
    numeric = int(digest[:15], 16)
    return str(max(numeric, 1))


def _validate_voice_chat_config(parsed_config: object) -> tuple[dict[str, object] | None, list[str]]:
    if not isinstance(parsed_config, dict) or not parsed_config:
        return None, ["VOICE_V2_VOLC_VOICE_CHAT_CONFIG_JSON 需要是非空对象"]

    candidate_config = parsed_config.get("VoiceChat", parsed_config)
    if not isinstance(candidate_config, dict) or not candidate_config:
        return None, ["VOICE_V2_VOLC_VOICE_CHAT_CONFIG_JSON 中 VoiceChat 必须是非空对象"]

    if "Config" in candidate_config:
        return _validate_official_voice_chat_config(candidate_config)

    return _validate_legacy_voice_chat_config(candidate_config)


def _validate_legacy_voice_chat_config(
    parsed_config: dict[str, object],
) -> tuple[dict[str, object] | None, list[str]]:
    blocked_reasons: list[str] = []
    missing_sections = [
        key for key in VOICE_V2_REQUIRED_VOICE_CHAT_CONFIG_SECTIONS if key not in parsed_config
    ]
    if missing_sections:
        blocked_reasons.append(
            "VOICE_V2_VOLC_VOICE_CHAT_CONFIG_JSON 缺少最小必需段："
            + ", ".join(missing_sections)
        )

    invalid_sections = [
        key
        for key in VOICE_V2_REQUIRED_VOICE_CHAT_CONFIG_SECTIONS
        if key in parsed_config and not isinstance(parsed_config[key], dict)
    ]
    if invalid_sections:
        blocked_reasons.append(
            "VOICE_V2_VOLC_VOICE_CHAT_CONFIG_JSON 中以下段必须是对象："
            + ", ".join(invalid_sections)
        )

    if blocked_reasons:
        return None, blocked_reasons

    config_payload = {
        key: value
        for key, value in parsed_config.items()
        if key != "AgentConfig"
    }
    return {
        "AgentConfig": parsed_config["AgentConfig"],
        "Config": config_payload,
    }, []


def _validate_official_voice_chat_config(
    voice_chat_config: dict[str, object],
) -> tuple[dict[str, object] | None, list[str]]:
    blocked_reasons: list[str] = []

    agent_config = voice_chat_config.get("AgentConfig")
    if not isinstance(agent_config, dict):
        blocked_reasons.append("VOICE_V2_VOLC_VOICE_CHAT_CONFIG_JSON 中 AgentConfig 必须是对象")

    config_payload = voice_chat_config.get("Config")
    if not isinstance(config_payload, dict):
        blocked_reasons.append("VOICE_V2_VOLC_VOICE_CHAT_CONFIG_JSON 中 Config 必须是对象")
    elif not isinstance(config_payload.get("ASRConfig"), dict):
        blocked_reasons.append("VOICE_V2_VOLC_VOICE_CHAT_CONFIG_JSON 中 Config.ASRConfig 必须是对象")

    if blocked_reasons:
        return None, blocked_reasons

    return {
        key: value
        for key, value in voice_chat_config.items()
        if key not in {"AppId", "RoomId", "TaskId"}
    }, []


def _build_provider_config_summary() -> tuple[list[str], list[str], bool, list[str]]:
    missing_config = [
        name for name in VOICE_V2_REQUIRED_PROVIDER_CONFIG if not _read_env(name)
    ]
    blocked_reasons: list[str] = []
    voice_chat_configured = False
    voice_chat_keys: list[str] = []

    raw_voice_chat_config = _read_env("VOICE_V2_VOLC_VOICE_CHAT_CONFIG_JSON")
    if raw_voice_chat_config:
        try:
            parsed_config = json.loads(raw_voice_chat_config)
        except json.JSONDecodeError:
            blocked_reasons.append("VOICE_V2_VOLC_VOICE_CHAT_CONFIG_JSON 不是合法 JSON")
        else:
            normalized_config, config_reasons = _validate_voice_chat_config(parsed_config)
            if config_reasons:
                blocked_reasons.extend(config_reasons)
            else:
                voice_chat_configured = True
                voice_chat_keys = sorted(str(key) for key in normalized_config.keys())

    return missing_config, blocked_reasons, voice_chat_configured, voice_chat_keys


def get_voice_v2_feature_flag() -> bool:
    return _read_flag("VOICE_V2_ENABLED", False)


def _read_voice_chat_config() -> dict[str, object]:
    raw_voice_chat_config = _read_env("VOICE_V2_VOLC_VOICE_CHAT_CONFIG_JSON")
    if not raw_voice_chat_config:
        raise VoiceV2ConfigurationError("VOICE_V2_VOLC_VOICE_CHAT_CONFIG_JSON 缺失")

    try:
        parsed_config = json.loads(raw_voice_chat_config)
    except json.JSONDecodeError as exc:
        raise VoiceV2ConfigurationError("VOICE_V2_VOLC_VOICE_CHAT_CONFIG_JSON 不是合法 JSON") from exc

    normalized_config, blocked_reasons = _validate_voice_chat_config(parsed_config)
    if blocked_reasons:
        raise VoiceV2ConfigurationError("；".join(blocked_reasons))

    return normalized_config


def _utc_timestamp(now: datetime) -> tuple[str, str]:
    request_time = now.strftime("%Y%m%dT%H%M%SZ")
    short_date = request_time[:8]
    return request_time, short_date


def _hash_sha256_hex(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _hmac_sha256(key: bytes, value: str) -> bytes:
    return hmac.new(key, value.encode("utf-8"), hashlib.sha256).digest()


def _build_canonical_query(params: dict[str, str]) -> str:
    encoded_pairs = [
        (
            quote(str(key), safe="-_.~"),
            quote(str(value), safe="-_.~"),
        )
        for key, value in sorted(params.items())
    ]
    return "&".join(f"{key}={value}" for key, value in encoded_pairs)


def _build_signed_openapi_headers(
    *,
    action: str,
    version: str,
    body: str,
    now: datetime,
) -> tuple[str, dict[str, str]]:
    host = _read_env("VOICE_V2_VOLC_OPENAPI_HOST", _VOLC_OPENAPI_HOST)
    region = _read_env("VOICE_V2_VOLC_OPENAPI_REGION", _VOLC_OPENAPI_REGION)
    ak = _read_env("VOICE_V2_VOLC_AK")
    sk = _read_env("VOICE_V2_VOLC_SK")
    if not ak or not sk:
        raise VoiceV2ConfigurationError("VOICE_V2_VOLC_AK / VOICE_V2_VOLC_SK 缺失")
    request_time, short_date = _utc_timestamp(now)
    canonical_query = _build_canonical_query(
        {
            "Action": action,
            "Version": version,
        }
    )
    body_hash = _hash_sha256_hex(body)
    canonical_headers = "\n".join(
        [
            f"content-type:{_VOLC_CONTENT_TYPE}",
            f"host:{host}",
            f"x-content-sha256:{body_hash}",
            f"x-date:{request_time}",
        ]
    ) + "\n"
    signed_headers = "content-type;host;x-content-sha256;x-date"
    canonical_request = "\n".join(
        [
            "POST",
            "/",
            canonical_query,
            canonical_headers,
            signed_headers,
            body_hash,
        ]
    )
    credential_scope = f"{short_date}/{region}/{_VOLC_OPENAPI_SERVICE}/request"
    string_to_sign = "\n".join(
        [
            "HMAC-SHA256",
            request_time,
            credential_scope,
            _hash_sha256_hex(canonical_request),
        ]
    )
    signing_key = _hmac_sha256(
        _hmac_sha256(
            _hmac_sha256(sk.encode("utf-8"), short_date),
            region,
        ),
        _VOLC_OPENAPI_SERVICE,
    )
    signing_key = _hmac_sha256(signing_key, "request")
    signature = hmac.new(signing_key, string_to_sign.encode("utf-8"), hashlib.sha256).hexdigest()

    return (
        f"https://{host}/?{canonical_query}",
        {
            "Authorization": (
                "HMAC-SHA256 "
                f"Credential={ak}/{credential_scope}, "
                f"SignedHeaders={signed_headers}, "
                f"Signature={signature}"
            ),
            "Content-Type": _VOLC_CONTENT_TYPE,
            "Host": host,
            "X-Content-Sha256": body_hash,
            "X-Date": request_time,
        },
    )


def _post_signed_openapi_json(
    *,
    action: str,
    version: str,
    payload: dict[str, object],
    now: datetime,
) -> dict[str, object]:
    body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    url, headers = _build_signed_openapi_headers(
        action=action,
        version=version,
        body=body,
        now=now,
    )
    timeout_seconds = float(_read_env("VOICE_V2_VOLC_OPENAPI_TIMEOUT_SECONDS", "10") or "10")

    try:
        with httpx.Client(timeout=timeout_seconds) as client:
            response = client.post(url, content=body.encode("utf-8"), headers=headers)
            response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text.strip() or exc.response.reason_phrase
        raise VoiceV2ProviderError(f"{action} 请求失败：HTTP {exc.response.status_code} {detail}") from exc
    except httpx.HTTPError as exc:
        raise VoiceV2ProviderError(f"{action} 请求失败：{exc}") from exc

    try:
        parsed = response.json()
    except json.JSONDecodeError as exc:
        raise VoiceV2ProviderError(f"{action} 返回了非 JSON 响应") from exc

    if not isinstance(parsed, dict):
        raise VoiceV2ProviderError(f"{action} 返回结构异常")

    metadata = parsed.get("ResponseMetadata")
    if isinstance(metadata, dict):
        error = metadata.get("Error")
        if isinstance(error, dict):
            code = str(error.get("CodeN") or error.get("Code") or "unknown")
            message = str(error.get("Message") or "unknown")
            raise VoiceV2ProviderError(f"{action} 请求失败：{code} {message}")

    return parsed


def _extract_request_id(response_payload: dict[str, object]) -> str | None:
    metadata = response_payload.get("ResponseMetadata")
    if not isinstance(metadata, dict):
        return None
    request_id = metadata.get("RequestId")
    return str(request_id) if request_id else None


def _is_voice_chat_success_result(result: object) -> bool:
    if result == {}:
        return True
    if isinstance(result, str):
        return result in {"ok", "OK"}
    return result is True


def _pack_rtc_uint16(value: int) -> bytes:
    return struct.pack("<H", value)


def _pack_rtc_uint32(value: int) -> bytes:
    return struct.pack("<I", value)


def _pack_rtc_bytes(value: bytes) -> bytes:
    if len(value) > 0xFFFF:
        raise VoiceV2ConfigurationError("RTC token 字段过长，无法按火山 RTC 格式编码")
    return _pack_rtc_uint16(len(value)) + value


def _pack_rtc_string(value: str) -> bytes:
    return _pack_rtc_bytes(value.encode("utf-8"))


def _pack_rtc_privileges(privileges: dict[int, int]) -> bytes:
    if len(privileges) > 0xFFFF:
        raise VoiceV2ConfigurationError("RTC token 权限数量过多，无法按火山 RTC 格式编码")

    chunks = [_pack_rtc_uint16(len(privileges))]
    for privilege, expire_timestamp in sorted(privileges.items()):
        chunks.append(_pack_rtc_uint16(privilege))
        chunks.append(_pack_rtc_uint32(expire_timestamp))
    return b"".join(chunks)


def _build_rtc_token_message(
    *,
    nonce: int,
    issued_at_timestamp: int,
    expires_at_timestamp: int,
    room_id: str,
    rtc_user_id: str,
) -> bytes:
    privileges = {
        _VOLC_RTC_PRIV_PUBLISH_STREAM: expires_at_timestamp,
        _VOLC_RTC_PRIV_PUBLISH_AUDIO_STREAM: expires_at_timestamp,
        _VOLC_RTC_PRIV_PUBLISH_VIDEO_STREAM: expires_at_timestamp,
        _VOLC_RTC_PRIV_PUBLISH_DATA_STREAM: expires_at_timestamp,
        _VOLC_RTC_PRIV_SUBSCRIBE_STREAM: expires_at_timestamp,
    }
    return b"".join(
        [
            _pack_rtc_uint32(nonce),
            _pack_rtc_uint32(issued_at_timestamp),
            _pack_rtc_uint32(expires_at_timestamp),
            _pack_rtc_string(room_id),
            _pack_rtc_string(rtc_user_id),
            _pack_rtc_privileges(privileges),
        ]
    )


def _generate_rtc_app_token(
    *,
    app_id: str,
    app_key: str,
    room_id: str,
    rtc_user_id: str,
    issued_at: datetime,
    expires_at: datetime,
) -> str:
    if not app_id or not app_key:
        raise VoiceV2ConfigurationError("VOICE_V2_VOLC_RTC_APP_ID / VOICE_V2_VOLC_RTC_APP_KEY 缺失")
    if expires_at <= issued_at:
        raise VoiceV2ConfigurationError("RTC token 过期时间必须晚于签发时间")

    issued_at_timestamp = int(issued_at.timestamp())
    expires_at_timestamp = int(expires_at.timestamp())
    nonce = secrets.randbelow(0xFFFFFFFF)
    message = _build_rtc_token_message(
        nonce=nonce,
        issued_at_timestamp=issued_at_timestamp,
        expires_at_timestamp=expires_at_timestamp,
        room_id=room_id,
        rtc_user_id=rtc_user_id,
    )
    signature = hmac.new(app_key.encode("utf-8"), message, hashlib.sha256).digest()
    content = _pack_rtc_bytes(message) + _pack_rtc_bytes(signature)
    return _VOLC_RTC_TOKEN_VERSION + app_id + base64.b64encode(content).decode("ascii")


def _start_voice_chat(
    *,
    app_id: str,
    room_id: str,
    task_id: str,
    rtc_user_id: str,
    agent_user_id: str,
    voice_chat_config: dict[str, object],
    now: datetime,
) -> str | None:
    request_payload = {
        key: deepcopy(value)
        for key, value in voice_chat_config.items()
        if key not in {"AppId", "RoomId", "TaskId"}
    }
    agent_config = request_payload.get("AgentConfig")
    if isinstance(agent_config, dict):
        agent_config["TargetUserId"] = [rtc_user_id]
        agent_config["UserId"] = agent_user_id
    request_payload.update(
        {
            "AppId": app_id,
            "RoomId": room_id,
            "TaskId": task_id,
        }
    )
    response_payload = _post_signed_openapi_json(
        action="StartVoiceChat",
        version=_VOLC_START_VOICE_CHAT_VERSION,
        payload=request_payload,
        now=now,
    )
    result = response_payload.get("Result")
    if not _is_voice_chat_success_result(result):
        raise VoiceV2ProviderError("StartVoiceChat 未返回成功结果")
    return _extract_request_id(response_payload)


def _update_voice_chat_finish_speech_recognition(
    *,
    app_id: str,
    room_id: str,
    task_id: str,
    now: datetime,
) -> str | None:
    response_payload = _post_signed_openapi_json(
        action="UpdateVoiceChat",
        version=_VOLC_UPDATE_VOICE_CHAT_VERSION,
        payload={
            "AppId": app_id,
            "RoomId": room_id,
            "TaskId": task_id,
            "Command": "FinishSpeechRecognition",
        },
        now=now,
    )
    result = response_payload.get("Result")
    if not _is_voice_chat_success_result(result):
        raise VoiceV2ProviderError("UpdateVoiceChat 未返回成功结果")
    return _extract_request_id(response_payload)


def _stop_voice_chat(
    *,
    app_id: str,
    room_id: str,
    task_id: str,
    now: datetime,
) -> str | None:
    response_payload = _post_signed_openapi_json(
        action="StopVoiceChat",
        version=_VOLC_STOP_VOICE_CHAT_VERSION,
        payload={
            "AppId": app_id,
            "RoomId": room_id,
            "TaskId": task_id,
        },
        now=now,
    )
    result = response_payload.get("Result")
    if not _is_voice_chat_success_result(result):
        raise VoiceV2ProviderError("StopVoiceChat 未返回成功结果")
    return _extract_request_id(response_payload)


def _cleanup_failed_prepare_session(
    *,
    app_id: str,
    room_id: str,
    task_id: str,
    now: datetime,
) -> str:
    try:
        stop_request_id = _stop_voice_chat(
            app_id=app_id,
            room_id=room_id,
            task_id=task_id,
            now=now,
        )
    except VoiceV2ProviderError as exc:
        return f"prepare 失败后补偿 StopVoiceChat 也失败：{exc}"

    if stop_request_id:
        return f"prepare 失败后已补偿 StopVoiceChat（request_id={stop_request_id}）"
    return "prepare 失败后已补偿 StopVoiceChat"


def build_voice_v2_session_bootstrap() -> dict[str, object]:
    missing_config, blocked_reasons, voice_chat_configured, voice_chat_keys = _build_provider_config_summary()
    feature_flag = get_voice_v2_feature_flag()

    if not feature_flag:
        blocked_reasons = ["VOICE_V2 feature flag 关闭"] + blocked_reasons
    elif missing_config:
        blocked_reasons = ["provider bootstrap 配置未齐"] + blocked_reasons

    return {
        "phase": VOICE_V2_PHASE,
        "feature_flag": feature_flag,
        "provider": VOICE_V2_PROVIDER,
        "session_mode": VOICE_V2_SESSION_MODE,
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
            "status": "ready" if not blocked_reasons else "blocked",
            "blocked_reasons": blocked_reasons,
            "missing_config": missing_config,
            "required_config": list(VOICE_V2_REQUIRED_PROVIDER_CONFIG),
        },
        "provider_config": {
            "rtc_app_id_configured": bool(_read_env("VOICE_V2_VOLC_RTC_APP_ID")),
            "rtc_app_key_configured": bool(_read_env("VOICE_V2_VOLC_RTC_APP_KEY")),
            "account_credentials_configured": bool(_read_env("VOICE_V2_VOLC_AK"))
            and bool(_read_env("VOICE_V2_VOLC_SK")),
            "voice_chat_configured": voice_chat_configured,
            "voice_chat_config_keys": voice_chat_keys,
        },
        "legacy_fallback_url": VOICE_V2_LEGACY_FALLBACK_URL,
        "next_step": VOICE_V2_NEXT_STEP,
    }


def prepare_voice_v2_provider_session(*, auth_session_id: str | None) -> dict[str, object]:
    bootstrap = build_voice_v2_session_bootstrap()
    connection_policy = bootstrap["connection_policy"]
    if not isinstance(connection_policy, dict):
        raise VoiceV2ConfigurationError("Voice V2 bootstrap 状态异常")

    if connection_policy.get("status") != "ready":
        blocked_reasons = connection_policy.get("blocked_reasons")
        detail = "；".join(blocked_reasons) if isinstance(blocked_reasons, list) else "provider bootstrap 未就绪"
        raise VoiceV2ConfigurationError(f"Voice V2 provider bootstrap 未就绪：{detail}")

    issued_at = datetime.now(UTC)
    ttl_seconds = _read_positive_int("VOICE_V2_SESSION_TTL_SECONDS", 300)
    expires_at = issued_at + timedelta(seconds=ttl_seconds)
    bootstrap_id = str(uuid.uuid4())
    short_id = bootstrap_id.split("-")[0]
    rtc_user_id = _build_numeric_user_id(auth_session_id or bootstrap_id)
    room_prefix = _sanitize_identifier(_read_env("VOICE_V2_VOLC_ROOM_PREFIX", "pv-room"), fallback="pv-room")
    task_prefix = _sanitize_identifier(_read_env("VOICE_V2_VOLC_TASK_PREFIX", "pv-task"), fallback="pv-task")
    provider_config = bootstrap["provider_config"]
    app_id = _read_env("VOICE_V2_VOLC_RTC_APP_ID")
    app_key = _read_env("VOICE_V2_VOLC_RTC_APP_KEY")
    room_id = f"{room_prefix}-{short_id}"
    task_id = f"{task_prefix}-{short_id}"
    agent_prefix = _sanitize_identifier(_read_env("VOICE_V2_VOLC_AGENT_USER_PREFIX", "pv-agent"), fallback="pv-agent")
    agent_user_id = f"{agent_prefix}-{short_id}"
    voice_chat_config = _read_voice_chat_config()
    rtc_app_token = _generate_rtc_app_token(
        app_id=app_id,
        app_key=app_key,
        room_id=room_id,
        rtc_user_id=rtc_user_id,
        issued_at=issued_at,
        expires_at=expires_at,
    )
    try:
        start_voice_chat_request_id = _start_voice_chat(
            app_id=app_id,
            room_id=room_id,
            task_id=task_id,
            rtc_user_id=rtc_user_id,
            agent_user_id=agent_user_id,
            voice_chat_config=voice_chat_config,
            now=issued_at,
        )
    except VoiceV2ProviderError as exc:
        cleanup_detail = _cleanup_failed_prepare_session(
            app_id=app_id,
            room_id=room_id,
            task_id=task_id,
            now=issued_at,
        )
        raise VoiceV2ProviderError(f"{exc}；{cleanup_detail}") from exc

    return {
        "bootstrap_id": bootstrap_id,
        "provider": VOICE_V2_PROVIDER,
        "session_mode": VOICE_V2_SESSION_MODE,
        "status": "prepared",
        "app_id": app_id,
        "room_id": room_id,
        "user_id": rtc_user_id,
        "agent_user_id": agent_user_id,
        "task_id": task_id,
        "issued_at": issued_at.isoformat().replace("+00:00", "Z"),
        "expires_at": expires_at.isoformat().replace("+00:00", "Z"),
        "ttl_seconds": ttl_seconds,
        "rtc_app_token": rtc_app_token,
        "rtc_token_request_id": None,
        "start_voice_chat_request_id": start_voice_chat_request_id,
        "provider_session_status": "started",
        "voice_chat_config_keys": provider_config["voice_chat_config_keys"],
        "next_action": VOICE_V2_PREPARED_NEXT_ACTION,
    }


def trigger_voice_v2_finish_speech_recognition(
    *,
    bootstrap_id: str,
    app_id: str,
    room_id: str,
    task_id: str,
) -> dict[str, object]:
    if not app_id.strip() or not room_id.strip() or not task_id.strip():
        raise VoiceV2ConfigurationError("UpdateVoiceChat 缺少必要的 app_id / room_id / task_id")

    issued_at = datetime.now(UTC)
    update_voice_chat_request_id = _update_voice_chat_finish_speech_recognition(
        app_id=app_id.strip(),
        room_id=room_id.strip(),
        task_id=task_id.strip(),
        now=issued_at,
    )

    return {
        "bootstrap_id": bootstrap_id,
        "provider": VOICE_V2_PROVIDER,
        "session_mode": VOICE_V2_SESSION_MODE,
        "status": "triggered",
        "app_id": app_id.strip(),
        "room_id": room_id.strip(),
        "task_id": task_id.strip(),
        "issued_at": issued_at.isoformat().replace("+00:00", "Z"),
        "command": "FinishSpeechRecognition",
        "update_voice_chat_request_id": update_voice_chat_request_id,
        "next_action": (
            "UpdateVoiceChat 已发送 FinishSpeechRecognition；"
            "前端应继续观察远端用户入房、远端媒体事件和远端音频音量回调。"
        ),
    }


def stop_voice_v2_provider_session(
    *,
    bootstrap_id: str,
    app_id: str,
    room_id: str,
    task_id: str,
    reason: str = "unknown",
) -> dict[str, object]:
    if not app_id.strip() or not room_id.strip() or not task_id.strip():
        raise VoiceV2ConfigurationError("StopVoiceChat 缺少必要的 app_id / room_id / task_id")

    normalized_reason = reason.strip() or "unknown"
    if normalized_reason not in VOICE_V2_STOP_REASON_LABELS:
        normalized_reason = "unknown"

    issued_at = datetime.now(UTC)
    stop_voice_chat_request_id = _stop_voice_chat(
        app_id=app_id.strip(),
        room_id=room_id.strip(),
        task_id=task_id.strip(),
        now=issued_at,
    )

    return {
        "bootstrap_id": bootstrap_id,
        "provider": VOICE_V2_PROVIDER,
        "session_mode": VOICE_V2_SESSION_MODE,
        "status": "stopped",
        "app_id": app_id.strip(),
        "room_id": room_id.strip(),
        "task_id": task_id.strip(),
        "issued_at": issued_at.isoformat().replace("+00:00", "Z"),
        "provider_session_status": "stopped",
        "stop_reason": normalized_reason,
        "stop_voice_chat_request_id": stop_voice_chat_request_id,
        "next_action": (
            f"Provider 侧 VoiceChat 已请求停止（原因：{VOICE_V2_STOP_REASON_LABELS[normalized_reason]}）；"
            "前端应保持 RTC 已离房并清空本地会话状态。"
        ),
    }
