"""工具 → Core API 映射测试（mock _call，不需要运行中的 daemon）。"""

import server


class CallRecorder:
    def __init__(self):
        self.calls = []

    async def __call__(self, method, path, **kwargs):
        self.calls.append((method, path, kwargs))
        return {"ok": True}


async def test_memory_store_maps_to_post_core_memory(monkeypatch):
    rec = CallRecorder()
    monkeypatch.setattr(server, "_call", rec)
    await server.memory_store("记住这件事", tags=["work"])
    method, path, kw = rec.calls[0]
    assert (method, path) == ("POST", "/core/memory")
    assert kw["json"] == {"content": "记住这件事", "tags": ["work"], "source": "mcp"}


async def test_memory_search_passes_intent_only_when_set(monkeypatch):
    rec = CallRecorder()
    monkeypatch.setattr(server, "_call", rec)
    await server.memory_search("我的项目进展", intent="project_query", limit=5)
    _, path, kw = rec.calls[0]
    assert path == "/core/recall"
    assert kw["params"] == {"q": "我的项目进展", "limit": 5, "intent": "project_query"}

    await server.memory_search("随便查查")
    _, _, kw2 = rec.calls[1]
    assert "intent" not in kw2["params"]


async def test_remaining_tools_map_correctly(monkeypatch):
    rec = CallRecorder()
    monkeypatch.setattr(server, "_call", rec)
    await server.memory_graph("Pervault")
    await server.memory_update("mem-1", "新内容")
    await server.persona_get(limit=3)
    await server.reflections_list(limit=2)
    await server.memory_why("爬山")
    await server.memory_stats()

    assert [(m, p) for m, p, _ in rec.calls] == [
        ("GET", "/core/graph"),
        ("PATCH", "/core/memory/mem-1"),
        ("GET", "/core/persona"),
        ("GET", "/core/reflections"),
        ("GET", "/core/why"),
        ("GET", "/core/stats"),
    ]
    why_kwargs = rec.calls[4][2]
    assert why_kwargs["params"] == {"q": "爬山"}


async def test_missing_token_returns_friendly_error(monkeypatch, tmp_path):
    monkeypatch.setenv("PERVAULT_HOME", str(tmp_path))  # 空目录 → 无 token 文件
    monkeypatch.delenv("PERVAULT_CORE_TOKEN", raising=False)
    result = await server._call("GET", "/core/health")
    assert "error" in result
    assert "daemon" in result["error"]


def test_read_token_env_precedence(monkeypatch, tmp_path):
    monkeypatch.setenv("PERVAULT_HOME", str(tmp_path))
    (tmp_path / "core_token").write_text("file-token")
    monkeypatch.setenv("PERVAULT_CORE_TOKEN", "env-token")
    assert server._read_token() == "env-token"

    monkeypatch.delenv("PERVAULT_CORE_TOKEN")
    assert server._read_token() == "file-token"
