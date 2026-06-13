"""真实 E2E：stdio MCP 客户端 → mcp_host 桥接 → 运行中的 daemon → SQLite。

前置：daemon 已在 127.0.0.1:8000 运行（RUN.command）。
手动运行：uv run python tests/e2e_roundtrip.py

验证点：
1. MCP 握手 + 8 个工具可见
2. memory_store 存一条带随机标记的记忆
3. memory_search 搜回同一条（generic intent，无 LLM 依赖路径）
4. 富化任务已持久化到 daemon 的 background_jobs（桥接进程死掉不丢）
5. 重开一个全新 MCP 会话仍能搜到（桥接无状态）
"""

import asyncio
import json
import sqlite3
import sys
import uuid
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SERVER_DIR = Path(__file__).resolve().parent.parent
MARKER = f"e2e-{uuid.uuid4().hex[:10]}"

SERVER_PARAMS = StdioServerParameters(
    command=str(Path.home() / ".local" / "bin" / "uv"),
    args=["run", "python", "server.py"],
    cwd=str(SERVER_DIR),
)


def _tool_payload(result) -> dict:
    text = result.content[0].text
    return json.loads(text)


async def session_one() -> str:
    async with stdio_client(SERVER_PARAMS) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            names = sorted(t.name for t in tools.tools)
            # 防漂移：期望集合从 server 模块的工具注册表派生（单一真源），
            # 而非手抄清单——加/删工具时本断言自动跟变，不会再出现「测试漏了
            # memory_why 却没人发现」的情况（test-1）。
            import server as mcp_server

            expected = sorted(t.name for t in await mcp_server.mcp.list_tools())
            assert names == expected, f"tools mismatch: live={names} registered={expected}"
            assert "memory_why" in names, "memory_why 工具缺失"
            print(f"[1] 握手成功，{len(names)} 个工具可见: {names}")

            stored = _tool_payload(
                await session.call_tool(
                    "memory_store",
                    {"content": f"E2E 验证记忆 {MARKER}：MCP 链路打通", "tags": ["e2e"]},
                )
            )
            assert "id" in stored, f"store failed: {stored}"
            print(f"[2] memory_store 成功: id={stored['id']}")

            found = _tool_payload(
                await session.call_tool(
                    "memory_search", {"query": MARKER, "intent": "generic"}
                )
            )
            hits = [r for r in found["results"] if MARKER in (r["content"] or "")]
            assert hits, f"search failed: {found}"
            print(f"[3] memory_search 搜回成功: {len(hits)} 条命中")

            return stored["id"]


def check_jobs_persisted(memory_id: str) -> None:
    db_path = Path.home() / ".pervault" / "data.db"
    conn = sqlite3.connect(db_path)
    try:
        count = conn.execute(
            "SELECT COUNT(*) FROM background_jobs WHERE payload_json LIKE ?",
            (f"%{memory_id}%",),
        ).fetchone()[0]
    finally:
        conn.close()
    assert count >= 1, "background_jobs 中没有该记忆的富化任务"
    print(f"[4] 富化任务已持久化: {count} 条 job（桥接进程死掉也不丢）")


async def session_two() -> None:
    async with stdio_client(SERVER_PARAMS) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            found = _tool_payload(
                await session.call_tool(
                    "memory_search", {"query": MARKER, "intent": "generic"}
                )
            )
            hits = [r for r in found["results"] if MARKER in (r["content"] or "")]
            assert hits, "新会话搜不到旧记忆"
            print("[5] 全新 MCP 会话仍能搜到（桥接无状态，记忆在 daemon）")


async def main() -> None:
    memory_id = await session_one()
    check_jobs_persisted(memory_id)
    await session_two()
    print(f"\n✅ E2E 全部通过 marker={MARKER}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except AssertionError as exc:
        print(f"\n❌ E2E 失败: {exc}")
        sys.exit(1)
