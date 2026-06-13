"""Loopback Core API（/core/*）测试：token 鉴权 + 存→搜闭环。

全程不依赖外部 LLM/Embedding：recall 显式传 intent 走关键词/FTS 降级路径。
"""

import os
import sys
import tempfile
import types
import uuid
from pathlib import Path

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

sys.modules.setdefault(
    "main",
    types.SimpleNamespace(
        limiter=types.SimpleNamespace(limit=lambda _rule: (lambda func: func))
    ),
)

from memory_core import database  # noqa: E402
from memory_core.database import init_db  # noqa: E402
from routers import core  # noqa: E402

TOKEN = "test-core-token-abc123"


@pytest.fixture
async def client(monkeypatch):
    tmp = Path(tempfile.mkdtemp(prefix="pervault-core-test-"))
    monkeypatch.setattr(database, "DB_PATH", tmp / "core.db")
    monkeypatch.setenv("PERVAULT_CORE_TOKEN", TOKEN)
    # 重置共享连接，确保用本测试的 DB
    if database._shared_db is not None:
        await database.close_shared_db()
    await init_db()

    app = FastAPI()
    app.include_router(core.router)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://core.test") as c:
        yield c
    if database._shared_db is not None:
        await database.close_shared_db()


def _auth():
    return {"X-Pervault-Token": TOKEN}


async def test_health_requires_token(client):
    resp = await client.get("/core/health")
    assert resp.status_code == 401

    resp = await client.get("/core/health", headers={"X-Pervault-Token": "wrong"})
    assert resp.status_code == 401

    resp = await client.get("/core/health", headers=_auth())
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


async def test_store_then_recall_roundtrip(client):
    marker = f"mcp-roundtrip-{uuid.uuid4().hex[:8]}"
    resp = await client.post(
        "/core/memory",
        headers=_auth(),
        json={"content": f"今天确认了 {marker} 方案可行", "source": "mcp"},
    )
    assert resp.status_code == 200, resp.text
    item = resp.json()
    assert item["id"]
    assert "source:mcp" in item["tags"]

    resp = await client.get(
        "/core/recall",
        headers=_auth(),
        params={"q": marker, "intent": "generic"},
    )
    assert resp.status_code == 200, resp.text
    results = resp.json()["results"]
    assert any(marker in (r["content"] or "") for r in results), results


async def test_store_enqueues_background_jobs(client):
    resp = await client.post(
        "/core/memory",
        headers=_auth(),
        json={"content": "记一下：明天要给 pervault 写周报", "source": "mcp"},
    )
    assert resp.status_code == 200
    memory_id = resp.json()["id"]

    db = await database.get_db(read_only=True)
    try:
        cursor = await db.execute(
            "SELECT COUNT(*) AS cnt FROM background_jobs WHERE payload_json LIKE ?",
            (f"%{memory_id}%",),
        )
        row = await cursor.fetchone()
        assert row["cnt"] >= 1, "写入必须留下持久化富化任务（进程死掉不丢）"
    finally:
        await db.close()


async def test_recall_rejects_invalid_intent(client):
    resp = await client.get(
        "/core/recall", headers=_auth(), params={"q": "x", "intent": "bogus"}
    )
    assert resp.status_code == 400


async def test_update_memory_roundtrip(client):
    resp = await client.post(
        "/core/memory",
        headers=_auth(),
        json={"content": "初版内容", "source": "mcp"},
    )
    memory_id = resp.json()["id"]

    resp = await client.patch(
        f"/core/memory/{memory_id}",
        headers=_auth(),
        json={"content": "修订后的内容 v2"},
    )
    assert resp.status_code == 200
    assert resp.json()["content"] == "修订后的内容 v2"


async def test_stats_and_persona_and_reflections_shape(client):
    for path, key in (
        ("/core/stats", "total_memories"),
        ("/core/persona", "traits"),
        ("/core/reflections", "reflections"),
    ):
        resp = await client.get(path, headers=_auth())
        assert resp.status_code == 200, path
        assert key in resp.json()


async def test_why_requires_token(client):
    resp = await client.get("/core/why", params={"q": "爬山"})
    assert resp.status_code == 401


async def test_why_returns_evidence_chain(client):
    # 先经 Core API 存入一条会触发规则法事实抽取的偏好记忆
    resp = await client.post(
        "/core/memory",
        headers=_auth(),
        json={"content": "我更喜欢爬山，周末常去黄山", "source": "test"},
    )
    assert resp.status_code == 200
    memory_id = resp.json()["id"]

    resp = await client.get("/core/why", headers=_auth(), params={"q": "爬山"})
    assert resp.status_code == 200
    data = resp.json()

    # 至少 supporting_memories 必须追溯到那条原始记忆
    ids = [m["memory_id"] for m in data["supporting_memories"]]
    assert memory_id in ids
    # 规则法 structured_facts 生成的 fact 信念应在场（无 LLM 路径）
    facts = [b for b in data["beliefs"] if b["type"] == "fact"]
    assert facts, f"应有规则法事实信念: {data['beliefs']}"
    assert facts[0]["evidence"][0]["memory_id"] == memory_id


async def test_why_no_match_is_empty_not_error(client):
    resp = await client.get("/core/why", headers=_auth(), params={"q": "量子物理"})
    assert resp.status_code == 200
    assert resp.json()["beliefs"] == []
