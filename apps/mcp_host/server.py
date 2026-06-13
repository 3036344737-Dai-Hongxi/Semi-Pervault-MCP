"""Pervault 记忆 MCP 服务器（薄桥接）。

架构约束（docs/derivative/01-架构改造方案.md D9）：
本进程**不开数据库、不跑后台循环**，所有调用经 loopback 转发给常驻
Pervault daemon（RUN.command 启动的后端）。
进程随 MCP 客户端起落均无影响——富化任务持久化在 daemon 的作业队列里。
"""

from __future__ import annotations

import os
from pathlib import Path

import httpx
from mcp.server.fastmcp import FastMCP

DAEMON_URL = os.getenv("PERVAULT_DAEMON_URL", "http://127.0.0.1:8000")

_DAEMON_HINT = (
    "无法连接 Pervault daemon（{url}）。"
    "请先在项目目录运行 RUN.command 启动记忆服务。"
)

mcp = FastMCP(
    "pervault-memory",
    instructions=(
        "Pervault 是用户的本地私有长期记忆库（四层记忆 + 知识图谱）。"
        "当用户提到值得长期记住的事实、偏好、决定或进展时，用 memory_store 写入；"
        "当需要回忆用户相关背景时，用 memory_search / persona_get 查询。"
        "所有数据都在用户本机，不会上云。"
    ),
)


def _read_token() -> str:
    env_token = os.getenv("PERVAULT_CORE_TOKEN", "").strip()
    if env_token:
        return env_token
    base = os.getenv("PERVAULT_HOME")
    home = Path(base) if base else Path.home() / ".pervault"
    try:
        return (home / "core_token").read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return ""


async def _call(method: str, path: str, **kwargs) -> dict:
    token = _read_token()
    if not token:
        return {"error": _DAEMON_HINT.format(url=DAEMON_URL) + "（core_token 不存在）"}
    try:
        async with httpx.AsyncClient(
            base_url=DAEMON_URL,
            headers={"X-Pervault-Token": token},
            timeout=30.0,
            # loopback 调用绝不走系统代理（HTTP_PROXY 等环境变量会把
            # 127.0.0.1 路由进代理，返回 502 假象）
            trust_env=False,
        ) as client:
            resp = await client.request(method, path, **kwargs)
    except (httpx.ConnectError, httpx.ConnectTimeout):
        return {"error": _DAEMON_HINT.format(url=DAEMON_URL)}
    if resp.status_code >= 400:
        return {"error": f"daemon 返回 {resp.status_code}: {resp.text[:300]}"}
    return resp.json()


@mcp.tool()
async def memory_store(content: str, tags: list[str] | None = None) -> dict:
    """把一条值得长期记住的信息写入用户的本地记忆库。

    适合：用户的偏好、决定、项目进展、人际事件、重要事实。
    写入后 daemon 会在后台异步完成分类、重要度评分、画像提取等富化。
    """
    return await _call(
        "POST",
        "/core/memory",
        json={"content": content, "tags": tags or [], "source": "mcp"},
    )


@mcp.tool()
async def memory_search(
    query: str, intent: str | None = None, limit: int = 10
) -> dict:
    """在用户的本地记忆库中检索相关记忆（混合检索：事实/全文/向量/图谱）。

    intent 可选，显式指定可跳过意图识别：task_query / persona_query /
    project_query / preference_query / people_query / summary_query / generic。
    """
    params: dict = {"q": query, "limit": limit}
    if intent:
        params["intent"] = intent
    return await _call("GET", "/core/recall", params=params)


@mcp.tool()
async def memory_graph(query: str) -> dict:
    """查询与主题相关的知识图谱上下文（实体与关系）。"""
    return await _call("GET", "/core/graph", params={"q": query})


@mcp.tool()
async def memory_update(memory_id: str, content: str) -> dict:
    """修订一条已存在的记忆内容（memory_id 来自 memory_search 结果）。"""
    return await _call(
        "PATCH", f"/core/memory/{memory_id}", json={"content": content}
    )


@mcp.tool()
async def persona_get(limit: int = 10) -> dict:
    """获取系统沉淀的用户稳定画像（高置信特征），适合作为长期背景注入。"""
    return await _call("GET", "/core/persona", params={"limit": limit})


@mcp.tool()
async def reflections_list(limit: int = 10) -> dict:
    """获取睡眠整理 agent 生成的高阶洞察（跨记忆的归纳）。"""
    return await _call("GET", "/core/reflections", params={"limit": limit})


@mcp.tool()
async def memory_why(query: str) -> dict:
    """解释「为什么记忆库这么认为」：返回信念的完整证据链。

    对某个主题（如“爬山”“项目A”），返回系统持有的相关信念
    （人设特征 / 结构化事实 / 高阶洞察），每条信念附带：
    来源原始记忆、收录时的五维准入打分、置信度与修正审计日志。
    适合在用户问「你为什么觉得我…」或需要核实记忆可信度时调用。
    """
    return await _call("GET", "/core/why", params={"q": query})


@mcp.tool()
async def memory_stats() -> dict:
    """记忆库概览：总量、今日新增、准入分层分布。"""
    return await _call("GET", "/core/stats")


if __name__ == "__main__":
    mcp.run()
