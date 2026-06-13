import asyncio
import json
from pathlib import Path
import sys
from unittest.mock import AsyncMock, patch

import aiosqlite
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from starlette.requests import Request

import memory_core.database as database
from memory_core.database import ensure_auth_sessions_schema, ensure_data_export_log_schema
from memory_core.models import MemoryExportRequest
from routers import auth
from routers.auth import LoginRequest
from services.rate_limit import limiter


def _request(
    *,
    path: str,
    method: str = "GET",
    cookie: str | None = None,
    auth_session_id: str | None = None,
) -> Request:
    headers: list[tuple[bytes, bytes]] = [(b"user-agent", b"pytest")]
    if cookie:
        headers.append((b"cookie", cookie.encode("utf-8")))
    scope = {
        "type": "http",
        "method": method,
        "path": path,
        "headers": headers,
        "client": ("testclient", 50000),
        "server": ("testserver", 80),
        "scheme": "http",
        "state": {"auth_session_id": auth_session_id},
    }
    return Request(scope)


async def _auth_db():
    db = await aiosqlite.connect(":memory:")
    db.row_factory = aiosqlite.Row
    await ensure_auth_sessions_schema(db)
    await ensure_data_export_log_schema(db)
    await db.execute(
        """CREATE TABLE memory_items (
               id TEXT PRIMARY KEY,
               content TEXT,
               created_at TEXT DEFAULT '2026-04-17 10:00:00'
           )"""
    )
    await db.execute(
        """CREATE TABLE structured_facts (
               id TEXT PRIMARY KEY,
               memory_id TEXT,
               kind TEXT,
               subject TEXT,
               predicate TEXT,
               object TEXT,
               status TEXT,
               created_at TEXT DEFAULT '2026-04-17 10:00:00'
           )"""
    )
    await db.execute(
        """CREATE TABLE graph_nodes (
               id TEXT PRIMARY KEY,
               type TEXT,
               label TEXT,
               properties TEXT DEFAULT '{}',
               weight REAL DEFAULT 1.0,
               source_memory_count INTEGER DEFAULT 0,
               created_at TEXT DEFAULT '2026-04-17 10:00:00',
               last_seen_at TEXT DEFAULT '2026-04-17 10:00:00'
           )"""
    )
    await db.execute(
        """CREATE TABLE graph_edges (
               id TEXT PRIMARY KEY,
               source_id TEXT,
               target_id TEXT,
               relation TEXT,
               weight REAL DEFAULT 1.0,
               source_memory_id TEXT,
               created_at TEXT DEFAULT '2026-04-17 10:00:00'
           )"""
    )
    await db.execute(
        """CREATE TABLE user_persona (
               id TEXT PRIMARY KEY,
               trait_key TEXT,
               trait_value TEXT,
               confidence REAL DEFAULT 0.8,
               evidence_count INTEGER DEFAULT 1,
               source_memory_ids TEXT DEFAULT '[]',
               last_updated TEXT DEFAULT '2026-04-17 10:00:00',
               created_at TEXT DEFAULT '2026-04-17 10:00:00'
           )"""
    )
    await db.execute(
        """CREATE TABLE memory_reflection (
               id TEXT PRIMARY KEY,
               insight TEXT,
               source_memory_ids TEXT DEFAULT '[]',
               importance REAL DEFAULT 8.0,
               created_at TEXT DEFAULT '2026-04-17 10:00:00'
           )"""
    )
    await db.execute(
        """CREATE TABLE preference_revision_log (
               id TEXT PRIMARY KEY,
               persona_id TEXT,
               old_value TEXT,
               new_value TEXT,
               trigger TEXT,
               created_at TEXT DEFAULT '2026-04-17 10:00:00'
           )"""
    )
    await db.commit()
    return db


class TestAuthConfig:
    def test_auth_password_prefers_auth_password(self, monkeypatch):
        monkeypatch.setenv("AUTH_PASSWORD", "new-secret")
        monkeypatch.setenv("API_KEY", "legacy-secret")

        assert auth._get_auth_password() == "new-secret"

    def test_validate_auth_configuration_warns_in_dev(self, monkeypatch, caplog):
        monkeypatch.setenv("SESSION_SECRET", "too-short")
        monkeypatch.setenv("SECURE_COOKIES", "0")

        auth.validate_auth_configuration()

        assert "SESSION_SECRET is weak" in caplog.text

    def test_validate_auth_configuration_raises_in_secure_context(self, monkeypatch):
        monkeypatch.setenv("SESSION_SECRET", "too-short")
        monkeypatch.setenv("SECURE_COOKIES", "1")

        with pytest.raises(RuntimeError):
            auth.validate_auth_configuration()


class TestAuthSessions:
    async def test_create_and_revoke_session_flow(self, monkeypatch):
        db = await _auth_db()
        original_close = db.close
        db.close = AsyncMock()
        monkeypatch.setenv("SESSION_SECRET", "x" * 32)
        request = _request(path="/api/auth/login", method="POST")
        try:
            with patch("routers.auth.get_db", new=AsyncMock(return_value=db)):
                token = await auth.create_session(request)
                session = await auth.get_active_session(token)
                assert session is not None
                assert session["revoked_at"] is None

                await auth.revoke_session_token(token)
                revoked = await auth.get_active_session(token, touch=False)
                assert revoked is None
        finally:
            await original_close()

    async def test_login_unwrapped_returns_cookie_and_persists_session(self, monkeypatch):
        db = await _auth_db()
        original_close = db.close
        db.close = AsyncMock()
        monkeypatch.setenv("AUTH_PASSWORD", "secret-pass")
        monkeypatch.setenv("SESSION_SECRET", "x" * 32)
        request = _request(path="/api/auth/login", method="POST")
        login_impl = getattr(auth.login, "__wrapped__", auth.login)
        try:
            with patch("routers.auth.get_db", new=AsyncMock(return_value=db)):
                response = await login_impl(request, LoginRequest(password="secret-pass"))
                row = await (
                    await db.execute("SELECT COUNT(*) AS cnt FROM auth_sessions")
                ).fetchone()
        finally:
            await original_close()

        assert response.status_code == 200
        assert "pv_session=" in response.headers["set-cookie"]
        assert row["cnt"] == 1

    async def test_login_unwrapped_wrong_password_returns_401(self, monkeypatch):
        db = await _auth_db()
        monkeypatch.setenv("AUTH_PASSWORD", "secret-pass")
        monkeypatch.setenv("SESSION_SECRET", "x" * 32)
        request = _request(path="/api/auth/login", method="POST")
        login_impl = getattr(auth.login, "__wrapped__", auth.login)
        try:
            response = await login_impl(request, LoginRequest(password="wrong-pass"))
        finally:
            await db.close()

        assert response.status_code == 401
        assert json.loads(response.body)["detail"] == "认证失败"

    async def test_status_true_for_active_session_and_false_after_logout(self, monkeypatch):
        db = await _auth_db()
        original_close = db.close
        db.close = AsyncMock()
        monkeypatch.setenv("AUTH_PASSWORD", "secret-pass")
        monkeypatch.setenv("SESSION_SECRET", "x" * 32)
        request = _request(path="/api/auth/login", method="POST")
        try:
            with patch("routers.auth.get_db", new=AsyncMock(return_value=db)):
                token = await auth.create_session(request)
                status_request = _request(
                    path="/api/auth/status",
                    cookie=f"{auth.SESSION_COOKIE}={token}",
                )
                before = await auth.status(status_request)
                assert before["authenticated"] is True

                logout_request = _request(
                    path="/api/auth/logout",
                    method="POST",
                    cookie=f"{auth.SESSION_COOKIE}={token}",
                )
                await auth.logout(logout_request)

                after = await auth.status(status_request)
                assert after["authenticated"] is False
        finally:
            await original_close()

    async def test_creating_new_session_does_not_revoke_existing_session(self, monkeypatch):
        db = await _auth_db()
        original_close = db.close
        db.close = AsyncMock()
        monkeypatch.setenv("AUTH_PASSWORD", "secret-pass")
        monkeypatch.setenv("SESSION_SECRET", "x" * 32)
        request = _request(path="/api/auth/login", method="POST")
        try:
            with patch("routers.auth.get_db", new=AsyncMock(return_value=db)):
                first_token = await auth.create_session(request)
                second_token = await auth.create_session(request)

                first_status_request = _request(
                    path="/api/auth/status",
                    cookie=f"{auth.SESSION_COOKIE}={first_token}",
                )
                second_status_request = _request(
                    path="/api/auth/status",
                    cookie=f"{auth.SESSION_COOKIE}={second_token}",
                )

                first_status = await auth.status(first_status_request)
                second_status = await auth.status(second_status_request)
                row = await (
                    await db.execute("SELECT COUNT(*) AS cnt FROM auth_sessions WHERE revoked_at IS NULL")
                ).fetchone()
        finally:
            await original_close()

        assert first_status["authenticated"] is True
        assert second_status["authenticated"] is True
        assert row["cnt"] == 2

    def test_protected_route_allows_valid_session_cookie(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AUTH_PASSWORD", "secret-pass")
        monkeypatch.setenv("SESSION_SECRET", "x" * 32)
        monkeypatch.setenv("SECURE_COOKIES", "0")
        monkeypatch.setenv("CONSOLIDATION_SCHEDULER_ENABLED", "0")
        monkeypatch.setenv("WEIGHT_DECAY_SCHEDULER_ENABLED", "0")
        monkeypatch.setenv("SLEEP_AGENT_ENABLED", "0")
        monkeypatch.setattr(database, "DB_PATH", Path(tmp_path) / "auth-protected-route.db")
        _reset_shared_db()
        limiter._storage.reset()

        sys.modules.pop("main", None)
        import main

        try:
            with TestClient(main.app) as client:
                unauth = client.get("/api/system/jobs/summary")
                assert unauth.status_code == 401

                login = client.post("/api/auth/login", json={"password": "secret-pass"})
                assert login.status_code == 200

                authed = client.get("/api/system/jobs/summary")
                assert authed.status_code == 200
                assert "jobs" in authed.json()
        finally:
            _reset_shared_db()
            limiter._storage.reset()


class TestExportGovernance:
    async def test_export_requires_explicit_confirmation(self):
        limiter = object()
        request = _request(
            path="/api/memory/export",
            method="POST",
            auth_session_id="session-1",
        )
        sys_modules_main = __import__("sys").modules
        sys_modules_main.setdefault("main", type("M", (), {"limiter": limiter})())
        from routers.memory import export_memories

        with pytest.raises(HTTPException) as exc_info:
            await export_memories(request, MemoryExportRequest(confirm_export=False))

        assert exc_info.value.status_code == 400

    async def test_export_records_completed_audit_log(self):
        limiter = __import__("types").SimpleNamespace(limit=lambda _rule: (lambda func: func))
        __import__("sys").modules.setdefault("main", __import__("types").SimpleNamespace(limiter=limiter))
        from routers.memory import export_memories

        db = await _auth_db()
        await db.execute(
            "INSERT INTO memory_items (id, content) VALUES (?, ?)",
            ("mem-1", "hello"),
        )
        await db.commit()
        request = _request(
            path="/api/memory/export",
            method="POST",
            auth_session_id="session-1",
        )
        try:
            with patch("routers.memory.get_shared_db", new=AsyncMock(return_value=db)):
                response = await export_memories(
                    request,
                    MemoryExportRequest(confirm_export=True),
                )
                audit_row = await (
                    await db.execute(
                        "SELECT status, auth_session_id, memory_count FROM data_export_log"
                    )
                ).fetchone()
        finally:
            await db.close()

        assert response.status_code == 200
        assert audit_row["status"] == "completed"
        assert audit_row["auth_session_id"] == "session-1"
        assert audit_row["memory_count"] == 1

    async def test_export_records_failed_audit_log(self):
        limiter = __import__("types").SimpleNamespace(limit=lambda _rule: (lambda func: func))
        __import__("sys").modules.setdefault("main", __import__("types").SimpleNamespace(limiter=limiter))
        from routers.memory import export_memories

        class _FailingDb:
            def __init__(self):
                self.rows = []

            async def execute(self, sql, params=()):
                if sql.startswith("SELECT * FROM memory_items"):
                    raise RuntimeError("boom")
                if sql.startswith("INSERT INTO data_export_log"):
                    self.rows.append(params)
                    return AsyncMock()
                return AsyncMock()

            async def commit(self):
                return None

            async def rollback(self):
                return None

        request = _request(
            path="/api/memory/export",
            method="POST",
            auth_session_id="session-1",
        )
        db = _FailingDb()
        try:
            with patch("routers.memory.get_shared_db", new=AsyncMock(return_value=db)):
                with pytest.raises(RuntimeError):
                    await export_memories(
                        request,
                        MemoryExportRequest(confirm_export=True),
                    )
        finally:
            pass

        assert db.rows
        assert db.rows[0][2] == "failed"


def _reset_shared_db():
    asyncio.run(database.close_shared_db())


class TestAuthLoginRateLimit:
    def test_login_rate_limit_hits_429(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AUTH_PASSWORD", "secret-pass")
        monkeypatch.setenv("SESSION_SECRET", "x" * 32)
        monkeypatch.setenv("SECURE_COOKIES", "0")
        monkeypatch.setenv("CONSOLIDATION_SCHEDULER_ENABLED", "0")
        monkeypatch.setenv("WEIGHT_DECAY_SCHEDULER_ENABLED", "0")
        monkeypatch.setenv("SLEEP_AGENT_ENABLED", "0")
        monkeypatch.setattr(database, "DB_PATH", Path(tmp_path) / "security.db")
        _reset_shared_db()
        limiter._storage.reset()

        sys.modules.pop("main", None)
        import main

        try:
            with TestClient(main.app) as client:
                responses = [
                    client.post("/api/auth/login", json={"password": "secret-pass"})
                    for _ in range(6)
                ]
        finally:
            _reset_shared_db()
            limiter._storage.reset()

        assert [response.status_code for response in responses[:5]] == [200, 200, 200, 200, 200]
        assert responses[5].status_code == 429
