import asyncio
import json
import logging

import websockets
from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect, WebSocketException, status
from fastapi.websockets import WebSocketState
from pydantic import BaseModel

from routers.auth import SESSION_COOKIE, get_active_session
from memory_core.services.llm import AIServiceUnavailableError
from services.realtime_session import (
    build_browser_websocket_url,
    build_glm_session_update_event,
    build_glm_transcription_session_update_event,
    build_provider_headers,
    build_provider_websocket_url,
    build_realtime_session_bootstrap,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/realtime", tags=["realtime"])

PROVIDER_PING_INTERVAL_SECONDS = 30
PROVIDER_PING_TIMEOUT_SECONDS = 120
PROVIDER_CLOSE_TIMEOUT_SECONDS = 5


class RealtimeTurnDetectionSummary(BaseModel):
    type: str
    create_response: bool
    interrupt_response: bool


class RealtimeSessionSummary(BaseModel):
    model: str
    voice: str
    input_audio_format: str
    output_audio_format: str
    turn_detection: RealtimeTurnDetectionSummary


class RealtimeSessionBootstrapResponse(BaseModel):
    provider: str
    websocket_url: str
    recommended_interaction_mode: str
    session: RealtimeSessionSummary


def _request_host(request: Request) -> str:
    return request.headers.get("x-forwarded-host", request.url.netloc)


def _read_json_object(raw_message: str) -> dict[str, object] | None:
    try:
        payload = json.loads(raw_message)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _log_browser_event(raw_message: str) -> None:
    payload = _read_json_object(raw_message)
    if payload is None:
        return

    event_type = payload.get("type")
    if event_type == "input_audio_buffer.commit":
        logger.info("realtime relay: browser committed the current input turn")
    elif event_type == "response.create":
        logger.info("realtime relay: browser requested a new model response")
    elif event_type == "response.cancel":
        logger.info("realtime relay: browser requested to cancel the current response")
    elif event_type == "input_audio_buffer.clear":
        logger.info("realtime relay: browser cleared the pending input buffer")


def _log_provider_event(raw_message: str) -> None:
    payload = _read_json_object(raw_message)
    if payload is None:
        return

    event_type = payload.get("type")
    if event_type == "session.updated":
        logger.info("realtime relay: provider session updated")
    elif event_type == "response.created":
        logger.info("realtime relay: provider created a response")
    elif event_type == "response.cancelled":
        logger.info("realtime relay: provider cancelled the current response")
    elif event_type == "response.done":
        response = payload.get("response")
        status = response.get("status") if isinstance(response, dict) else None
        logger.info("realtime relay: provider finished a response status=%s", status or "unknown")
    elif event_type == "error":
        error = payload.get("error")
        code = error.get("code") if isinstance(error, dict) else None
        message = error.get("message") if isinstance(error, dict) else None
        logger.warning(
            "realtime relay: provider emitted an error code=%s message=%s",
            code or "unknown",
            message or "unknown",
        )


async def _send_browser_error(websocket: WebSocket, *, code: str, message: str) -> None:
    if websocket.application_state != WebSocketState.CONNECTED:
        return
    try:
        await websocket.send_json(
            {
                "type": "error",
                "error": {
                    "type": "relay_error",
                    "code": code,
                    "message": message,
                },
            }
        )
    except RuntimeError:
        logger.debug("browser websocket closed before relay error could be sent")


async def _close_browser_socket(websocket: WebSocket) -> None:
    if websocket.application_state == WebSocketState.DISCONNECTED:
        return
    try:
        await websocket.close()
    except RuntimeError:
        logger.debug("browser websocket was already closing")


async def _forward_browser_messages(websocket: WebSocket, provider_ws) -> None:
    while True:
        message = await websocket.receive()
        if message["type"] == "websocket.disconnect":
            raise WebSocketDisconnect(code=message.get("code", 1000))

        text = message.get("text")
        if text is None:
            await _send_browser_error(
                websocket,
                code="unsupported_browser_frame",
                message="Realtime relay 当前只支持 JSON 文本帧",
            )
            continue

        _log_browser_event(text)
        await provider_ws.send(text)


async def _forward_provider_messages(provider_ws, websocket: WebSocket) -> None:
    async for message in provider_ws:
        if isinstance(message, bytes):
            await websocket.send_bytes(message)
            continue
        _log_provider_event(message)
        await websocket.send_text(message)


@router.get("/session", response_model=RealtimeSessionBootstrapResponse)
async def get_session(request: Request) -> RealtimeSessionBootstrapResponse:
    websocket_url = build_browser_websocket_url(
        request_scheme=request.url.scheme,
        request_host=_request_host(request),
    )
    try:
        payload = build_realtime_session_bootstrap(websocket_url=websocket_url)
    except AIServiceUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return RealtimeSessionBootstrapResponse(**payload)


@router.websocket("/ws")
async def realtime_websocket_relay(websocket: WebSocket) -> None:
    token = websocket.cookies.get(SESSION_COOKIE, "")
    session = await get_active_session(token, touch=True)
    if session is None:
        raise WebSocketException(code=status.WS_1008_POLICY_VIOLATION)

    await websocket.accept()

    try:
        async with websockets.connect(
            build_provider_websocket_url(),
            additional_headers=build_provider_headers(),
            max_size=None,
            ping_interval=PROVIDER_PING_INTERVAL_SECONDS,
            ping_timeout=PROVIDER_PING_TIMEOUT_SECONDS,
            close_timeout=PROVIDER_CLOSE_TIMEOUT_SECONDS,
        ) as provider_ws:
            await provider_ws.send(
                json.dumps(build_glm_session_update_event(), ensure_ascii=False)
            )
            await provider_ws.send(
                json.dumps(
                    build_glm_transcription_session_update_event(),
                    ensure_ascii=False,
                )
            )

            forward_tasks = {
                asyncio.create_task(_forward_browser_messages(websocket, provider_ws)),
                asyncio.create_task(_forward_provider_messages(provider_ws, websocket)),
            }

            done, pending = await asyncio.wait(
                forward_tasks,
                return_when=asyncio.FIRST_COMPLETED,
            )

            for task in pending:
                task.cancel()
            for task in pending:
                try:
                    await task
                except asyncio.CancelledError:
                    pass

            for task in done:
                exc = task.exception()
                if exc is None or isinstance(exc, WebSocketDisconnect):
                    continue
                raise exc
    except AIServiceUnavailableError as exc:
        await _send_browser_error(
            websocket,
            code="voicecall_glm_unavailable",
            message=str(exc),
        )
    except websockets.exceptions.ConnectionClosed as exc:
        logger.warning(
            "realtime relay: provider websocket closed code=%s reason=%s",
            exc.code,
            exc.reason or "unknown",
        )
        await _send_browser_error(
            websocket,
            code="voicecall_glm_relay_closed",
            message="Realtime relay 已断开，请重新建立会话。",
        )
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("realtime relay failed")
        await _send_browser_error(
            websocket,
            code="voicecall_glm_relay_failed",
            message="Realtime relay 暂时不可用，请稍后重试",
        )
    finally:
        await _close_browser_socket(websocket)
