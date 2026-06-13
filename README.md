# Pervault Memory MCP — 本地优先的个人记忆 MCP 服务器

> 把你的私有长期记忆（四层记忆 + 知识图谱）作为 MCP 工具暴露给 Claude Desktop / Cursor 等客户端。数据全部留在本机 SQLite，不上云。

## 这是什么

Pervault 是一个本地优先的个人记忆引擎：你把值得长期记住的事实 / 偏好 / 决定 / 进展写进去，它负责分类、重要度评分、画像提炼、知识图谱抽取与混合检索；之后任何 MCP 客户端都能在对话里「记住你、被问回来」。

本仓库是从 Pervault 主项目拆出的 **MCP 产品线**，由三部分组成：

- `apps/mcp_host/` — MCP 服务器（薄桥接）。只依赖 `httpx` + `mcp`，通过 loopback 把 8 个记忆工具转发给常驻 daemon，自身不开数据库、不跑后台。
- `backend/` — 常驻记忆 daemon（FastAPI）。`/core/*` 接口的提供者，也是 `data.db` 的**唯一写入方**。
- `packages/memory_core/` — 记忆内核（SQLite schema、领域模型、写入管线、混合检索、知识图谱、画像 / 洞察、后台整理 agent），provider 无关、**不依赖任何 Web 框架**（由 import 边界测试守卫）。

```
MCP 客户端 ──stdio──> apps/mcp_host/server.py ──127.0.0.1:8000 + X-Pervault-Token──> backend daemon ──> memory_core ──> SQLite
```

## 提供的工具（8 个）

| 工具 | 说明 |
|---|---|
| `memory_store` | 写入一条值得长期记住的信息 |
| `memory_search` | 混合检索（事实 / 全文 / 向量 / 图谱） |
| `memory_graph` | 查询主题相关的知识图谱上下文 |
| `memory_update` | 修订一条已存在的记忆 |
| `persona_get` | 获取沉淀的用户稳定画像 |
| `reflections_list` | 获取睡眠整理 agent 生成的高阶洞察 |
| `memory_why` | 返回某条信念的完整证据链（来源 + 准入打分 + 审计） |
| `memory_stats` | 记忆库概览统计 |

## 快速开始

### 前置条件

- Python ≥ 3.11
- [uv](https://docs.astral.sh/uv/getting-started/)

### 1. 启动记忆 daemon

双击项目根目录的 `RUN.command`，或手动：

```bash
cd backend
cp .env.example .env        # 首次：按需填 OPENAI_API_KEY 等（留空也能跑基础功能）
uv run python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

daemon 仅绑定 `127.0.0.1`，不对外网开放。首次请求会在 `~/.pervault/core_token` 生成本地配对 token（权限 0600）。数据库默认 `~/.pervault/data.db`，可用环境变量 `PERVAULT_DB_PATH` 覆盖。

### 2. 配置 MCP 客户端

Claude Desktop（`~/Library/Application Support/Claude/claude_desktop_config.json`）：

```json
{
  "mcpServers": {
    "pervault-memory": {
      "command": "<uv 绝对路径，可用 `which uv` 查看，如 ~/.local/bin/uv>",
      "args": [
        "--directory", "<项目绝对路径>/apps/mcp_host",
        "run", "python", "server.py"
      ]
    }
  }
}
```

重启客户端后即可使用上面 8 个工具。更多细节见 [`apps/mcp_host/README.md`](apps/mcp_host/README.md)。

## 配置（backend/.env）

权威模板见 [`backend/.env.example`](backend/.env.example)。关键项：

| 变量 | 说明 |
|---|---|
| `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `LLM_MODEL` | LLM（意图路由、写入富化） |
| `GEMINI_API_KEY` / `EMBEDDING_MODEL` / `EMBEDDING_BASE_URL` / `EMBEDDING_DIM` | 向量检索的 embedding（维度须与模型一致） |
| `PERVAULT_DB_PATH` | 数据库位置，默认 `~/.pervault/data.db` |

未配置 embedding 时，向量检索会自动降级到全文 / LIKE，仍可正常使用。

## 开发与测试

```bash
# 记忆内核（最大一套，含 import 边界守卫）
cd packages/memory_core && uv run python -m pytest -q

# 后端 daemon
cd backend && uv run python -m pytest tests/ -q

# MCP 桥接单测
cd apps/mcp_host && uv run python -m pytest -q --ignore=tests/e2e_roundtrip.py
```

## 架构与设计文档

- 架构总纲：[`docs/derivative/01-架构改造方案.md`](docs/derivative/01-架构改造方案.md)
- 四层记忆架构：[`docs/plan/four-memory-architecture-integration-plan.md`](docs/plan/four-memory-architecture-integration-plan.md)
- 内核边界与抽离细节：[`packages/memory_core/README.md`](packages/memory_core/README.md)

## 许可证与隐私

MIT，见 [LICENSE](LICENSE)。记忆数据默认仅存于本机 SQLite；若部署到服务器，请自行处理 HTTPS、访问控制与备份。
