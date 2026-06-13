from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# 必须在 import 其余 memory_core 模块之前锚定数据库位置。
# 默认 ~/.pervault/data.db（多宿主共享）；首次启动自动从旧位置 backend/data.db
# 复制迁移（旧库保留作备份，失败回退旧路径）。显式设置 PERVAULT_DB_PATH 可覆盖。
if "PERVAULT_DB_PATH" not in os.environ:
    from memory_core.db_location import ensure_db_location

    os.environ["PERVAULT_DB_PATH"] = str(
        ensure_db_location(Path(__file__).parent / "data.db")
    )

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from memory_core.exceptions import MemoryCoreError, MemoryNotFoundError
from memory_core.runtime import MemoryRuntime
from routers import voice, voice_v2, memory, graph, chat, auth, system, realtime, core
from routers.auth import SESSION_COOKIE, get_active_session, validate_auth_configuration
from services.rate_limit import limiter


@asynccontextmanager
async def lifespan(app: FastAPI):
    validate_auth_configuration()
    runtime = MemoryRuntime()
    app.state.memory_runtime = runtime
    await runtime.start()
    try:
        yield
    finally:
        await runtime.stop()


app = FastAPI(title="Memory Graph API", version="0.3.0", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# memory_core 领域异常 → HTTP 状态码（内核不再抛 HTTPException）
@app.exception_handler(MemoryNotFoundError)
async def memory_not_found_handler(request: Request, exc: MemoryNotFoundError):
    return JSONResponse(status_code=404, content={"detail": str(exc)})


@app.exception_handler(MemoryCoreError)
async def memory_core_error_handler(request: Request, exc: MemoryCoreError):
    return JSONResponse(status_code=500, content={"detail": str(exc)})

_raw_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000")
ALLOWED_ORIGINS = [o.strip() for o in _raw_origins.split(",") if o.strip()]

# CORS 规范禁止 allow_credentials=True 与通配符 * 同时使用，浏览器会拒绝响应
_allow_credentials = "*" not in ALLOWED_ORIGINS
if not _allow_credentials:
    logging.getLogger("uvicorn.error").warning(
        "ALLOWED_ORIGINS 含通配符 '*'，已自动关闭 allow_credentials —— "
        "cookie 会话将无法跨源工作。生产环境请改用精确 origin 白名单。"
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Paths that do not require an authenticated session
_AUTH_PUBLIC_PATHS: frozenset[str] = frozenset({
    "/api/health",
    "/api/auth/login",
    "/api/auth/logout",
    "/api/auth/status",
})


@app.middleware("http")
async def verify_session(request: Request, call_next):
    path = request.url.path
    request.state.auth_session_id = None
    if request.method == "OPTIONS":
        return await call_next(request)
    if not path.startswith("/api/") or path in _AUTH_PUBLIC_PATHS:
        return await call_next(request)

    token = request.cookies.get(SESSION_COOKIE, "")
    session = await get_active_session(token, touch=True)
    if session is None:
        return JSONResponse(
            status_code=401,
            content={"detail": "未登录或会话已过期"},
        )
    request.state.auth_session_id = session["id"]
    return await call_next(request)


app.include_router(auth.router)
app.include_router(core.router)
app.include_router(realtime.router)
app.include_router(voice_v2.router)
app.include_router(voice.router)
app.include_router(memory.router)
app.include_router(graph.router)
app.include_router(chat.router)
app.include_router(system.router)


@app.get("/api/health")
async def health():
    return {"status": "ok"}
