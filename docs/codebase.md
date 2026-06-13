# Codebase Overview

## 项目总体定位
一句话：这是一个本地优先的 AI 记忆系统，围绕“语音输入 -> 记忆沉淀 -> 搜索召回 -> 轻量知识图谱”这条主链路构建。

## 当前架构风格
- 前后端分离：前端是 `Next.js App Router + React + Tailwind + shadcn/ui`，后端是 `FastAPI`。
- 单体后端：所有 API、数据库初始化、AI 服务接入都在一个 Python 服务里完成，没有微服务拆分。
- 本地优先数据层：核心数据存在 `SQLite`，不依赖外部数据库服务。
- 搜索方案：记忆检索基于 `SQLite FTS5`，并对中文查询增加了 `LIKE` 兜底。
- 图谱方案：图谱节点和边直接存在 SQLite 的 `graph_nodes` / `graph_edges`，不引入图数据库。
- 可视化方案：前端使用 `Cytoscape.js` 渲染图谱。
- AI 接入方式：通过 OpenAI 兼容 SDK 调用外部模型服务，当前通过环境变量控制 `OPENAI_API_KEY`、`OPENAI_BASE_URL`、`LLM_MODEL`。

## 目录概览

```text
backend/                FastAPI 后端
  database.py           SQLite schema 与连接
  main.py               FastAPI 入口
  models.py             Pydantic 模型
  routers/              HTTP 路由
  services/             LLM / ASR / 图谱抽取服务

frontend/src/           Next.js 前端源码
  app/                  页面路由
  components/           业务组件与基础 UI 组件
  hooks/                自定义 hooks
  lib/                  API 封装与工具
  types/                TypeScript 类型

docs/                   项目文档
```

## 后端文件说明

### 入口与数据库
- `backend/main.py`：FastAPI 入口，负责读取环境变量、初始化数据库、注册 `voice` / `memory` / `graph` 路由，并提供健康检查接口。
- `backend/database.py`：定义 SQLite 数据库路径、建表 SQL、FTS5 触发器、图谱表与索引，以及 `get_db()` / `init_db()`。
- `backend/models.py`：集中定义所有 Pydantic 请求/响应模型，包括语音、记忆、图谱节点、图谱边、图谱详情等结构。

### 路由层
- `backend/routers/voice.py`：处理语音上传与澄清接口，负责文件类型/大小校验、语音记录存在性校验、调用 ASR 和 clarify 服务并更新 `voice_records`。
- `backend/routers/memory.py`：处理记忆写入、记忆搜索和统计接口，负责写入 `memory_items`、写入时计算 `kind/task_status/emotion_score/consolidated`、控制自动图谱抽取范围，以及中文/英文搜索分流和 FTS 语法错误保护。
- `backend/routers/graph.py`：处理图谱抽取、子图查询、节点详情查询，并负责图谱节点/边的落库、去重和来源追溯。

### 服务层
- `backend/services/llm.py`：封装通用 LLM 客户端初始化、clarify prompt 和响应解析逻辑，并把外部模型服务错误统一映射为可解释异常。
- `backend/services/whisper.py`：封装音频转写调用，输入音频文件路径，输出转写文本和置信度。
- `backend/services/graph_extract.py`：封装图谱抽取 prompt、节点/边白名单校验、label 规范化和 LLM JSON 输出解析。
- `backend/services/consolidation.py`：最小离线整合服务。扫描 `consolidated=false` 且 `kind` 为 `project_update/relationship_event/preference` 的记忆，复用现有规则法抽取/补充 `structured_facts`，成功后标记 `consolidated=true`。手动执行入口为 `backend/scripts/run_consolidation.py`。

## 前端文件说明

### 页面层 `frontend/src/app`
- `frontend/src/app/layout.tsx`：全局布局文件，负责挂载侧边栏、主内容区和全局字体/样式。
- `frontend/src/app/page.tsx`：首页仪表盘，展示记忆统计、快捷入口，并可打开录音弹窗。
- `frontend/src/app/memory/page.tsx`：记忆页，负责加载记忆列表、搜索、标签筛选和“提取图谱”按钮交互。
- `frontend/src/app/graph/page.tsx`：图谱页，负责搜索、类型筛选、加载子图、点击节点后展示详情抽屉。
- `frontend/src/app/chat/page.tsx`：聊天页，目前主要是基于 mock 数据的对话 UI 骨架。
- `frontend/src/app/settings/page.tsx`：设置页，目前主要是模型、语言、账户信息的静态/本地状态界面。
- `frontend/src/app/globals.css`：全局样式与设计令牌定义，包含 Tailwind 主题变量和基础配色。

### 业务组件层 `frontend/src/components`
- `frontend/src/components/sidebar.tsx`：左侧导航栏组件，负责页面导航和当前路由高亮。
- `frontend/src/components/page-header.tsx`：页面头部通用组件，统一标题、副标题和右上角用户区布局。
- `frontend/src/components/voice-recorder-dialog.tsx`：语音录制主流程弹窗，串联录音、上传、clarify、保存记忆等前端状态机。
- `frontend/src/components/graph-canvas.tsx`：Cytoscape 封装组件，把后端返回的节点/边渲染为可交互图谱。

### 基础 UI 组件层 `frontend/src/components/ui`
- `frontend/src/components/ui/button.tsx`：统一按钮样式封装。
- `frontend/src/components/ui/card.tsx`：统一卡片容器封装。
- `frontend/src/components/ui/badge.tsx`：统一标签/徽标封装。
- `frontend/src/components/ui/input.tsx`：统一输入框封装。
- `frontend/src/components/ui/dialog.tsx`：通用弹窗封装，供录音弹窗使用。
- `frontend/src/components/ui/sheet.tsx`：通用抽屉封装，供图谱节点详情使用。
- `frontend/src/components/ui/scroll-area.tsx`：通用滚动容器封装。
- `frontend/src/components/ui/separator.tsx`：通用分隔线封装。
- `frontend/src/components/ui/avatar.tsx`：通用头像封装。

### Hook / 工具 / 类型层
- `frontend/src/hooks/use-audio-recorder.ts`：封装浏览器麦克风录音能力，输出音频 Blob、录音时长和录音状态。
- `frontend/src/lib/api.ts`：统一封装前端到后端的 HTTP 请求，包括语音、记忆、图谱三组 API。
- `frontend/src/lib/utils.ts`：提供 `cn()` 这类基础工具函数，用于合并 Tailwind class。
- `frontend/src/lib/mock-data.ts`：提供首页/聊天等页面的 mock 数据，当前主链路已经更多依赖真实 API。
- `frontend/src/types/index.ts`：集中定义前端业务类型，包括记忆、图谱节点、图谱边、图谱详情、聊天会话等。

## 两条核心业务链路

### 1. 语音 -> 记忆
1. 用户在前端 `VoiceRecorderDialog` 中开始录音。
2. `useAudioRecorder()` 产出音频 Blob。
3. 前端调用 `uploadVoice()` 上传音频。
4. 后端 `POST /api/voice/upload` 调用 `services/whisper.py` 做 ASR，并把结果写入 `voice_records`。
5. 前端继续调用 `clarifyVoice()`。
6. 后端 `POST /api/voice/clarify` 调用 `services/llm.py` 判断是 `clear` 还是 `unclear`。
7. 若 `clear`，前端允许保存为记忆；若 `unclear`，前端展示补充说明输入框。
8. 前端调用 `storeMemory()`。
9. 后端 `POST /api/memory/store` 把最终文本写入 `memory_items`，并通过触发器同步进 `memory_fts`。
10. 写入时会按规则补齐 `kind`、`task_status`、`emotion_score`、`consolidated`。
11. 只有 `kind=project_update` 和 `kind=relationship_event` 的记忆会默认触发自动图谱抽取；其他类型只保留原始记忆、structured facts 和向量索引。

### 2. 记忆 -> 图谱
1. 用户在记忆页点击某条记忆上的“提取图谱”按钮。
2. 前端调用 `extractGraph(memoryId, content)`。
3. 后端 `POST /api/graph/extract` 校验 `memory_item_id` 是否存在。
4. `services/graph_extract.py` 调用 LLM，要求返回严格 JSON 的节点与边。
5. `routers/graph.py` 对节点和边做写入、去重、权重更新和来源绑定。
6. 图谱写入到 `graph_nodes` / `graph_edges`。
7. 图谱页调用 `GET /api/graph/subgraph` 获取子图。
8. `GraphCanvas` 用 Cytoscape 把节点和边画出来。
9. 点击节点后，前端调用 `GET /api/graph/node/{node_id}`，右侧抽屉展示节点详情、关系和来源记忆。

## 数据结构概览

### 语音与记忆
- `voice_records`：保存原始转写、澄清文本、状态、置信度。
- `memory_items`：保存最终记忆内容、标签、来源语音记录，以及 `kind`、任务生命周期字段 `task_status`、情绪标记 `emotion_score`、整合标记 `consolidated`。
- `memory_fts`：保存全文搜索索引，由 SQLite 触发器维护。

### 图谱
- `graph_nodes`：保存节点 `id/type/label/properties/weight/source_memory_count` 等信息。
- `graph_edges`：保存边 `source_id/target_id/relation/weight/source_memory_id` 等信息。

## API 边界

### 语音 API
- `POST /api/voice/upload`
- `POST /api/voice/clarify`

### 记忆 API
- `POST /api/memory/store`
- `GET /api/memory/search`
- `GET /api/memory/stats`

### 图谱 API
- `POST /api/graph/extract`
- `GET /api/graph/subgraph`
- `GET /api/graph/node/{node_id}`

## Consolidation 与 Retrieval 可观测性

- `backend/services/retrieval.py` 的每条检索结果附带 `_source` 内部标签（`structured_fact` / `hybrid` / `graph` / `pattern` / `recent` / `boot_fact` / `boot_kind`），用于调试来源构成，不影响对外 API 契约。
- `retrieve_context` 出口日志输出 `retrieval result intent=... total=... composition={...}`，可在 uvicorn 日志中查看每次检索的来源分布。
- `get_boot_context` 出口日志输出 `composition={...}`，区分 structured_facts 通道和 kind 通道的贡献。
- `routers/chat.py` 的 prompt 日志增加 `source_composition=... boot_composition=...`，可追踪每次对话使用了多少 structured_fact 来源。
- `services/consolidation.py` 的 `run_once` 结束时输出汇总日志：`scanned / processed / facts_added / skipped / errors / kind_distribution`。
- `backend/scripts/run_consolidation.py`：手动运行一次整合，终端输出完整统计。
- `backend/scripts/eval_retrieval.py`：最小检索评估脚本，对指定 query 输出 intent、source 构成和详细来源列表。可在 consolidation 前后分别运行以对比效果。用法：`uv run python scripts/eval_retrieval.py` 或 `uv run python scripts/eval_retrieval.py --queries "我喜欢什么" "我在做什么项目"`。

## Retrieval Regression Suite

- `backend/scripts/golden_queries.py`：定义 10 条 golden query，覆盖 `project_query`（3 条）、`preference_query`（3 条）、`summary_query`（2 条）、`task_query`（2 条）。每条 query 声明 `expected_intent` 和需要执行的 checks 列表。
- `backend/scripts/regression_retrieval.py`：regression runner，加载 golden queries，执行检索，对每条 check 输出 PASS / FAIL / SKIP。用法：`uv run python scripts/regression_retrieval.py`（加 `--verbose` 看 source 详情）。退出码 0 = 无 FAIL，退出码 1 = 有 FAIL。
- 判定规则：
  - `intent_match`：`detect_query_intent(query)` 必须等于 `expected_intent`，否则 FAIL。所有 query 都执行。
  - `fact_present`：sources 非空时 composition 中应包含 `structured_fact`；sources 为空或 DB 中无对应 fact 时 SKIP。用于 preference_query 和 summary_query。
  - `fact_priority`：sources 非空且含 `structured_fact` 时，第一条 source 应来自 `structured_fact`；否则 SKIP。用于 project_query。
  - `no_closed_tasks`：对 task_query 返回的 sources 反查 `memory_items.task_status`，若发现 `done` 或 `expired` 状态则 FAIL。用于 task_query。
- FAIL 只在确定 regression 时出现；SKIP 表示当前 DB 数据不足以触发该规则，不算 regression。

## 当前已知薄弱点 / TODO
- clarify 真实分流仍偏“宽松”：模糊样本有时会被判成 `clear`，说明保守澄清能力还不够强。
- 图谱重复提取时存在语义漂移：同一条 memory 多次抽取，可能因为 LLM 表达不稳定而产生近义节点或不同类型节点。
- 图谱搜索当前更偏“严格子图裁剪”：例如按 `Pervault` 搜索时可能只剩一个节点、没有上下文边，展示效果偏孤立。
- 图谱节点粒度还不完全稳定：例如 `我`、`前端`、`自动异步队列` 这类节点在白名单内，但未必总是最理想的实体粒度。
- 聊天检索已新增 `task_query` 路由：任务类问题优先只看 `task_status=open`，非任务问题默认排除 `done/expired` 的任务噪音，Boot Context 中的任务也只保留 open。
- 聊天页和设置页目前仍以 UI 骨架 / mock 为主，不是当前已完成主链路的一部分。
- ASR 服务兼容性依赖外部上游；当前项目支持 OpenAI 兼容客户端，但音频能力是否可用取决于所接入服务。

## 适合外部 AI 的一句话摘要
这是一个用 `Next.js + FastAPI + SQLite + FTS5 + Cytoscape` 实现的本地优先 AI 记忆系统：前端负责录音、记忆列表和图谱展示，后端负责语音转写、LLM 澄清、记忆存储、全文搜索和轻量知识图谱抽取，所有核心数据都保存在 SQLite 中。
