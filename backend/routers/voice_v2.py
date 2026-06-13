from typing import Literal

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from services.voice_v2_session import (
    VoiceV2ConfigurationError,
    VoiceV2ProviderError,
    build_voice_v2_session_bootstrap,
    prepare_voice_v2_provider_session,
    stop_voice_v2_provider_session,
    trigger_voice_v2_finish_speech_recognition,
)


router = APIRouter(prefix="/api/voice-v2", tags=["voice-v2"])


class VoiceV2BrowserPolicy(BaseModel):
    system_browser_required: bool
    embedded_browser_support: str


class VoiceV2MemoryPolicy(BaseModel):
    ownership: str
    bridge_mode: str
    write_enabled: bool


class VoiceV2ConnectionPolicy(BaseModel):
    lifecycle: str
    status: str
    blocked_reasons: list[str]
    missing_config: list[str]
    required_config: list[str]


class VoiceV2ProviderConfigSummary(BaseModel):
    rtc_app_id_configured: bool
    rtc_app_key_configured: bool
    account_credentials_configured: bool
    voice_chat_configured: bool
    voice_chat_config_keys: list[str]


class VoiceV2SessionBootstrapResponse(BaseModel):
    phase: str
    feature_flag: bool
    provider: str
    session_mode: str
    browser_policy: VoiceV2BrowserPolicy
    memory_policy: VoiceV2MemoryPolicy
    connection_policy: VoiceV2ConnectionPolicy
    provider_config: VoiceV2ProviderConfigSummary
    legacy_fallback_url: str
    next_step: str


@router.get("/session", response_model=VoiceV2SessionBootstrapResponse)
async def get_voice_v2_session() -> VoiceV2SessionBootstrapResponse:
    payload = build_voice_v2_session_bootstrap()
    return VoiceV2SessionBootstrapResponse(**payload)


class VoiceV2PreparedSessionResponse(BaseModel):
    bootstrap_id: str
    provider: str
    session_mode: str
    status: str
    app_id: str
    room_id: str
    user_id: str
    agent_user_id: str
    task_id: str
    issued_at: str
    expires_at: str
    ttl_seconds: int
    rtc_app_token: str
    rtc_token_request_id: str | None = None
    start_voice_chat_request_id: str | None = None
    provider_session_status: str
    voice_chat_config_keys: list[str]
    next_action: str


@router.post("/session/prepare", response_model=VoiceV2PreparedSessionResponse)
async def prepare_voice_v2_session(request: Request) -> VoiceV2PreparedSessionResponse:
    try:
        payload = await run_in_threadpool(
            prepare_voice_v2_provider_session,
            auth_session_id=getattr(request.state, "auth_session_id", None),
        )
    except VoiceV2ConfigurationError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except VoiceV2ProviderError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc

    return VoiceV2PreparedSessionResponse(**payload)


class VoiceV2TriggerSessionRequest(BaseModel):
    bootstrap_id: str
    app_id: str
    room_id: str
    task_id: str


class VoiceV2TriggeredSessionResponse(BaseModel):
    bootstrap_id: str
    provider: str
    session_mode: str
    status: str
    app_id: str
    room_id: str
    task_id: str
    issued_at: str
    command: str
    update_voice_chat_request_id: str | None = None
    next_action: str


@router.post("/session/trigger", response_model=VoiceV2TriggeredSessionResponse)
async def trigger_voice_v2_session(
    payload: VoiceV2TriggerSessionRequest,
) -> VoiceV2TriggeredSessionResponse:
    try:
        response = await run_in_threadpool(
            trigger_voice_v2_finish_speech_recognition,
            bootstrap_id=payload.bootstrap_id,
            app_id=payload.app_id,
            room_id=payload.room_id,
            task_id=payload.task_id,
        )
    except VoiceV2ConfigurationError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except VoiceV2ProviderError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc

    return VoiceV2TriggeredSessionResponse(**response)


class VoiceV2StopSessionRequest(BaseModel):
    bootstrap_id: str
    app_id: str
    room_id: str
    task_id: str
    reason: Literal[
        "manual_disconnect",
        "superseded_attempt",
        "rtc_connect_failed",
        "rtc_join_timeout",
        "remote_media_wait_timeout",
        "provider_session_expired",
        "page_hide",
        "component_unmount",
        "unknown",
    ] = "unknown"


class VoiceV2StoppedSessionResponse(BaseModel):
    bootstrap_id: str
    provider: str
    session_mode: str
    status: str
    app_id: str
    room_id: str
    task_id: str
    issued_at: str
    provider_session_status: str
    stop_reason: str
    stop_voice_chat_request_id: str | None = None
    next_action: str


@router.post("/session/stop", response_model=VoiceV2StoppedSessionResponse)
async def stop_voice_v2_session(
    payload: VoiceV2StopSessionRequest,
) -> VoiceV2StoppedSessionResponse:
    try:
        response = await run_in_threadpool(
            stop_voice_v2_provider_session,
            bootstrap_id=payload.bootstrap_id,
            app_id=payload.app_id,
            room_id=payload.room_id,
            task_id=payload.task_id,
            reason=payload.reason,
        )
    except VoiceV2ConfigurationError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except VoiceV2ProviderError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc

    return VoiceV2StoppedSessionResponse(**response)
