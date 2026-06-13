# CLAUDE.md — Pervault 工作规则

仓库级约定见 `AGENTS.md`（先读它）；本文件定义**工作流铁律**，两者冲突时以本文件为准。
对用户回复一律用中文。

## 项目速览

- 本地优先的 AI 记忆系统：语音/文字 → 记忆沉淀（SQLite 四层记忆+图谱）→ 混合检索 → 聊天/画像/洞察。
- 本仓库：从 Pervault 主项目拆出的 MCP 产品线——`apps/mcp_host/`（MCP 薄桥接）· `backend/`（常驻记忆 daemon，FastAPI + SQLite，唯一写 `data.db`）· `packages/memory_core/`（记忆内核，无 Web 依赖）。
- 当前主题：按 `docs/derivative/01-架构改造方案.md` 把记忆内核抽成 `memory_core`，第一条产品线是 MCP 记忆服务器。

## 铁律 1：闭环验证（最重要）

**任何任务、任何计划步骤，必须以「可验证的检查」收尾，确认 working 才算完成。**

- 做 plan 时，每一步都必须写明它的验证方式（跑哪个测试 / 哪条命令 / 看到什么输出算过）。没有验证方式的步骤不是合格的步骤。
- 改完代码必须立刻跑对应检查，**看到真实通过输出**才能进入下一步。
- 禁止在没跑测试的情况下说「完成」「应该可以了」。汇报时必须附上测试/检查的真实结果（如 `182 passed`）。
- 测试失败 = 任务未完成。如实报告失败输出，不准掩盖、不准跳过。

❌ 不合格：「memory_service 改完了，应该没问题。」
✅ 合格：「memory_service 改完，跑了后端测试：`182 passed`；检索回归脚本退出码 0。遗留风险：未覆盖中文 FTS 边界 case。」

## 验证命令（按改动范围选最窄的）

> **环境前置**：工具链 uv 0.11.20（`~/.local/bin`）。基线结果：memory_core `290 passed`、backend `141 passed`、mcp_host `5 passed`。
> 注意：本地部署用的 `backend/.env` 已生成（cookie 鉴权、AI key 留空、定时任务关闭），数据库在 `~/.pervault/data.db`。

| 改了什么 | 必须跑 |
|---|---|
| `packages/memory_core/` 内核代码 | `cd packages/memory_core && env PYTHONPYCACHEPREFIX=/tmp/pervault-pycache uv run python -m pytest tests/ -q`（含 import 边界守卫），若 backend 调用面受影响再跑 backend 套件 |
| `backend/` 任何代码 | `cd backend && env PYTHONPYCACHEPREFIX=/tmp/pervault-pycache uv run python -m pytest tests/ -q` |
| `apps/mcp_host/` 桥接代码 | `cd apps/mcp_host && uv run python -m pytest -q --ignore=tests/e2e_roundtrip.py` |
| 检索/召回逻辑 | 后端测试 + `cd backend && uv run python scripts/regression_retrieval.py`（退出码 0 才算过） |
| 数据库 schema | 后端测试 + 用旧库副本验证迁移幂等、可回滚 |
| 只改文档 | 通读一遍自查事实正确，无需跑测试 |

## 铁律 2：plan 先行，小步走

- 动手前先给出小计划：本步改什么、**不改什么**、验收标准是什么。
- 一次只做一个内聚的小改动（小 diff），改完即验证，形成可回滚的检查点。
- 禁止一口气跨多个模块堆大改动；禁止顺手重构、顺手清理无关代码。

## 铁律 3：高风险动作先停下问用户

以下动作必须先获用户明确同意，且一次同意只覆盖一次：

- 删除/迁移数据库或任何用户数据
- 修改对外 API 契约
- `git commit` / `git push` / 发布类操作
- 启动重型后台任务（Workflow / deep-research / 多 agent 编排）——默认禁止，除非用户明确要求

## 架构边界（衍生产品方向）

- 衍生路线的总规格是 `docs/derivative/01-架构改造方案.md`（v2），按其 Phase 推进，不跳步。
- 硬不变量：**只有 daemon（后端进程）写 `data.db`**；未来的 `memory_core` 包不得 import `fastapi`/`routers`/前端。
- SQLite-first、本地优先：不引入 Redis/Celery/外部数据库等基础设施。
- `voice_v2` / VoiceVR 相关代码是旁置遗留，不动、不接、不修。
- 后台调度（consolidation / decay / sleep agent / jobs worker）已在 `main.py` lifespan 中管理，加新循环前先看现状。

## 汇报格式

每完成一步，汇报必须包含：① 改了哪些文件；② 跑了什么验证、真实输出是什么；③ 已知风险或遗留问题。三者缺一不可。

> 注：CLAUDE.md 规则属于「强约定」而非硬拦截。若日后需要 100% 强制（如禁止未跑测试就提交），可加 Claude Code hooks（PreToolUse）实现，目前暂不配置。
