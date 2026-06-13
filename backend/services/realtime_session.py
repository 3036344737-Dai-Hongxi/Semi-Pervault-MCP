import os
import time
import uuid
from typing import Any
from urllib.parse import urlparse, urlunparse

from memory_core.services.llm import AIServiceUnavailableError

DEFAULT_GLM_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
DEFAULT_GLM_REALTIME_MODEL = "glm-realtime"
DEFAULT_GLM_VOICE = "tongtong"
DEFAULT_GLM_INPUT_AUDIO_FORMAT = "pcm16"
DEFAULT_GLM_OUTPUT_AUDIO_FORMAT = "pcm"
DEFAULT_RECOMMENDED_INTERACTION_MODE = "push_to_talk"


def _read_env(name: str, default: str) -> str:
    return os.getenv(name, default).strip() or default


def get_glm_api_key() -> str:
    return os.getenv("VOICECALL_GLM_API_KEY", "").strip()


def get_glm_base_url() -> str:
    return _read_env("VOICECALL_GLM_BASE_URL", DEFAULT_GLM_BASE_URL).rstrip("/")


def get_realtime_model() -> str:
    return _read_env("VOICECALL_GLM_REALTIME_MODEL", DEFAULT_GLM_REALTIME_MODEL)


def get_realtime_voice() -> str:
    return _read_env("VOICECALL_GLM_VOICE", DEFAULT_GLM_VOICE)


def get_input_audio_format() -> str:
    return _read_env("VOICECALL_GLM_INPUT_AUDIO_FORMAT", DEFAULT_GLM_INPUT_AUDIO_FORMAT)


def get_output_audio_format() -> str:
    return _read_env("VOICECALL_GLM_OUTPUT_AUDIO_FORMAT", DEFAULT_GLM_OUTPUT_AUDIO_FORMAT)


def get_transcription_input_audio_format() -> str:
    audio_format = get_input_audio_format().lower()
    if audio_format == "wav":
        return "wav"
    return "pcm"


def get_realtime_instructions() -> str:
    return os.getenv("VOICECALL_REALTIME_INSTRUCTIONS", "").strip()


def get_recommended_interaction_mode() -> str:
    return DEFAULT_RECOMMENDED_INTERACTION_MODE


def ensure_glm_realtime_configured() -> None:
    if get_glm_api_key():
        return
    raise AIServiceUnavailableError(
        "Realtime 会话服务暂不可用，请检查 VOICECALL_GLM_API_KEY 配置"
    )


def build_turn_detection_config() -> dict[str, Any]:
    return {
        "type": "client_vad",
        "create_response": False,
        "interrupt_response": False,
    }


def build_glm_session_config() -> dict[str, Any]:
    config: dict[str, Any] = {
        "model": get_realtime_model(),
        "modalities": ["text", "audio"],
        "voice": get_realtime_voice(),
        "input_audio_format": get_input_audio_format(),
        "output_audio_format": get_output_audio_format(),
        "input_audio_noise_reduction": {
            "type": "near_field",
        },
        "turn_detection": build_turn_detection_config(),
        "beta_fields": {
            "chat_mode": "audio",
        },
    }
    instructions = get_realtime_instructions()
    if instructions:
        config["instructions"] = instructions
    return config


def build_glm_transcription_session_config() -> dict[str, Any]:
    return {
        "input_audio_format": get_transcription_input_audio_format(),
        "input_audio_noise_reduction": {
            "type": "near_field",
        },
        "modalities": ["text", "audio"],
        "turn_detection": build_turn_detection_config(),
    }


def build_browser_websocket_url(*, request_scheme: str, request_host: str) -> str:
    websocket_scheme = "wss" if request_scheme == "https" else "ws"
    return f"{websocket_scheme}://{request_host}/api/realtime/ws"


def build_realtime_session_bootstrap(*, websocket_url: str) -> dict[str, Any]:
    ensure_glm_realtime_configured()
    session_config = build_glm_session_config()
    return {
        "provider": "glm",
        "websocket_url": websocket_url,
        "recommended_interaction_mode": get_recommended_interaction_mode(),
        "session": {
            "model": session_config["model"],
            "voice": session_config["voice"],
            "input_audio_format": session_config["input_audio_format"],
            "output_audio_format": session_config["output_audio_format"],
            "turn_detection": session_config["turn_detection"],
        },
    }


def build_provider_websocket_url() -> str:
    parsed = urlparse(get_glm_base_url())
    scheme = "wss" if parsed.scheme == "https" else "ws"
    path = f"{parsed.path.rstrip('/')}/realtime"
    return urlunparse(parsed._replace(scheme=scheme, path=path, query="", fragment=""))


def build_provider_headers() -> dict[str, str]:
    ensure_glm_realtime_configured()
    return {
        "Authorization": f"Bearer {get_glm_api_key()}",
    }


def build_glm_session_update_event() -> dict[str, Any]:
    return {
        "event_id": str(uuid.uuid4()),
        "client_timestamp": int(time.time() * 1000),
        "type": "session.update",
        "session": build_glm_session_config(),
    }


def build_glm_transcription_session_update_event() -> dict[str, Any]:
    return {
        "event_id": str(uuid.uuid4()),
        "client_timestamp": int(time.time() * 1000),
        "type": "transcription_session.update",
        "session": build_glm_transcription_session_config(),
    }
