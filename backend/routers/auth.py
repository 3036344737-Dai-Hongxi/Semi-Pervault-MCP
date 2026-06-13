import hashlib
import hmac
import logging
import os
import secrets
import uuid
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from memory_core.database import get_db
from services.rate_limit import limiter

logger = logging.getLogger(__name__)

SESSION_COOKIE = "pv_session"
_SESSION_SECRET_ENV = "SESSION_SECRET"
_SESSION_MAX_AGE_DAYS = 30
_SESSION_MAX_AGE_SECONDS = _SESSION_MAX_AGE_DAYS * 86400
_AUTH_LOGIN_RATE_LIMIT = os.getenv("AUTH_LOGIN_RATE_LIMIT", "5/minute")
_WEAK_SESSION_SECRET_VALUES = {
    "",
    "changeme",
    "change-me",
    "change_this",
    "change-this",
    "change-this-to-a-random-secret-before-deploy",
    "your-random-secret-at-least-32-chars",
}

router = APIRouter(prefix="/api/auth", tags=["auth"])


class LoginRequest(BaseModel):
    password: str


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _sqlite_datetime(value: datetime) -> str:
    normalized = value.astimezone(timezone.utc).replace(microsecond=0)
    return normalized.strftime("%Y-%m-%d %H:%M:%S")


def _is_secure_context() -> bool:
    """Return True when SECURE_COOKIES=1 (i.e. running behind HTTPS in production)."""
    return os.getenv("SECURE_COOKIES", "0") != "0"


def _get_session_secret() -> str:
    return os.getenv(_SESSION_SECRET_ENV, "").strip()


def _get_auth_password() -> str:
    return os.getenv("AUTH_PASSWORD", "").strip() or os.getenv("API_KEY", "").strip()


def _is_weak_session_secret(secret: str) -> bool:
    normalized = secret.strip()
    return len(normalized) < 32 or normalized.lower() in _WEAK_SESSION_SECRET_VALUES


def validate_auth_configuration() -> None:
    secret = _get_session_secret()
    if not secret:
        logger.warning("auth configuration: SESSION_SECRET missing")
        return
    if not _is_weak_session_secret(secret):
        return

    message = (
        "SESSION_SECRET is weak; use a random secret with at least 32 characters"
    )
    if _is_secure_context():
        raise RuntimeError(message)
    logger.warning("%s (development mode warning)", message)


def _hash_session_token(token: str) -> str:
    secret = _get_session_secret()
    digest = hashlib.sha256()
    digest.update(secret.encode("utf-8"))
    digest.update(b":")
    digest.update(token.encode("utf-8"))
    return digest.hexdigest()


def _make_session_token() -> str:
    return secrets.token_urlsafe(32)


def _request_ip(request: Request) -> str | None:
    return request.client.host if request.client else None


def _request_user_agent(request: Request) -> str | None:
    return request.headers.get("user-agent")


async def revoke_session_token(token: str) -> None:
    if not token:
        return

    db = await get_db()
    token_hash = _hash_session_token(token)
    try:
        await db.execute(
            """UPDATE auth_sessions
               SET revoked_at = COALESCE(revoked_at, datetime('now'))
               WHERE token_hash = ?""",
            (token_hash,),
        )
        await db.commit()
    except Exception:
        await db.rollback()
        logger.exception("auth: failed to revoke session token")
    finally:
        await db.close()


async def create_session(request: Request) -> str:
    db = await get_db()
    token = _make_session_token()
    token_hash = _hash_session_token(token)
    now = _utc_now()
    expires_at = now + timedelta(days=_SESSION_MAX_AGE_DAYS)

    try:
        await db.execute(
            """INSERT INTO auth_sessions
               (id, token_hash, created_at, expires_at, last_seen_at, ip_address, user_agent)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                str(uuid.uuid4()),
                token_hash,
                _sqlite_datetime(now),
                _sqlite_datetime(expires_at),
                _sqlite_datetime(now),
                _request_ip(request),
                _request_user_agent(request),
            ),
        )
        await db.commit()
        return token
    except Exception:
        await db.rollback()
        logger.exception("auth: failed to create session")
        raise
    finally:
        await db.close()


async def _touch_session_last_seen(session_id: str) -> None:
    touch_db = await get_db()
    try:
        await touch_db.execute(
            """UPDATE auth_sessions
               SET last_seen_at = datetime('now')
               WHERE id = ?""",
            (session_id,),
        )
        await touch_db.commit()
    except Exception:
        await touch_db.rollback()
        logger.exception("auth: failed to update last_seen_at session_id=%s", session_id)
    finally:
        await touch_db.close()


async def get_active_session(
    token: str,
    *,
    touch: bool = True,
):
    if not token:
        return None

    secret = _get_session_secret()
    if not secret or _is_weak_session_secret(secret):
        return None

    read_db = await get_db(read_only=True)
    token_hash = _hash_session_token(token)
    try:
        cursor = await read_db.execute(
            """SELECT id, created_at, expires_at, last_seen_at, revoked_at, ip_address, user_agent
               FROM auth_sessions
               WHERE token_hash = ?
                 AND revoked_at IS NULL
                 AND expires_at > datetime('now')
               LIMIT 1""",
            (token_hash,),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        session = dict(row)
    finally:
        await read_db.close()

    if touch:
        await _touch_session_last_seen(session["id"])

    return session


async def verify_session_token(token: str) -> bool:
    """Return True when *token* maps to a valid, unexpired, non-revoked session."""
    return await get_active_session(token, touch=True) is not None


@router.post("/login")
@limiter.limit(_AUTH_LOGIN_RATE_LIMIT)
async def login(request: Request, req: LoginRequest) -> Response:
    auth_password = _get_auth_password()
    if not auth_password:
        return JSONResponse(status_code=503, content={"detail": "服务端未配置鉴权密码"})

    session_secret = _get_session_secret()
    if not session_secret:
        return JSONResponse(status_code=503, content={"detail": "服务端未配置 SESSION_SECRET"})
    if _is_weak_session_secret(session_secret):
        return JSONResponse(status_code=503, content={"detail": "SESSION_SECRET 配置不安全"})

    if not hmac.compare_digest(req.password.encode(), auth_password.encode()):
        logger.warning("auth: failed login attempt ip=%s", _request_ip(request))
        return JSONResponse(status_code=401, content={"detail": "认证失败"})

    token = await create_session(request)
    secure = _is_secure_context()

    response = JSONResponse(content={"ok": True})
    response.set_cookie(
        key=SESSION_COOKIE,
        value=token,
        max_age=_SESSION_MAX_AGE_SECONDS,
        httponly=True,
        secure=secure,
        samesite="strict",
        path="/",
    )
    logger.info("auth: login successful secure=%s ip=%s", secure, _request_ip(request))
    return response


@router.post("/logout")
async def logout(request: Request) -> Response:
    token = request.cookies.get(SESSION_COOKIE, "")
    await revoke_session_token(token)

    response = JSONResponse(content={"ok": True})
    response.delete_cookie(key=SESSION_COOKIE, path="/")
    return response


@router.get("/status")
async def status(request: Request) -> dict:
    token = request.cookies.get(SESSION_COOKIE, "")
    return {"authenticated": await verify_session_token(token)}
