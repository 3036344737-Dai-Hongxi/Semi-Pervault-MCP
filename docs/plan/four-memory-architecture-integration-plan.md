# 四记忆架构集成计划

> 本文档已按当前代码现状更新。当前代码已经完成检索拆分、cookie session 鉴权、Graph pending 审核、权重衰减、聊天会话历史、Whisper 独立配置、以及应用生命周期内的共享 SQLite 连接。后续实现应以这些现状为基础，不再按旧文档里的 `store_memory()`、单体 `retrieval.py` 或每次请求 `get_db()/close()` 写法推进。

## 现状 vs 目标

### 现有架构

```text
memory_items
  原始/情节记忆主表。已有 kind、task_status、emotion_score、consolidated、weight、last_referenced_at。

structured_facts
  从 memory_items 提炼出的结构化事实。当前 consolidation 和检索已经使用。

graph_nodes / graph_edges
  轻量知识图谱。已有 pending/confirmed 状态、possible_duplicate_of、confirm/reject API。

vec_items
  sqlite-vec 向量表。由 memory_service 后台索引。

chat_messages
  会话消息表。当前 chat 已读取同 session 近 10 条历史作为 working context。

weight_decay.py
  已有指数衰减任务，以及被召回记忆的 weight reset。

consolidation.py
  周期性后台整合：memory_items -> structured_facts / graph，并可修正部分 kind。

retrieval_*.py
  检索层已经拆分：
  - retrieval_constants.py
  - retrieval_intent.py
  - retrieval_primitives.py
  - retrieval_context.py
  - retrieval_boot.py
  - graph_retrieval.py

database.py
  已有 get_shared_db() / close_shared_db()。Web 路由和后台服务优先使用共享连接；consolidation 的 read_only 快照和脚本入口可继续使用独立 get_db()。
```

### 当前缺失

- Persona/Reflection 管理 UI：当前已可通过聊天与导出使用，尚未提供单独管理页。
- 旧 `structured_facts` 到 Persona 的批量迁移：当前仍避免自动迁移，防止短期 preference 被误提升为长期画像。

---

## 总体执行顺序

建议按“最小可验证闭环”推进，而不是一次做完整五阶段：

| 状态 | 阶段 | 名称 | 优先级 | 目标 |
|------|------|------|--------|------|
| ✅ 已完成 | 0 | Schema 与契约准备 | 最高 | 已补 memory_items、Admission、Persona、Reflection、Revision schema 与模型/API 契约 |
| ✅ 已完成 | 1 | Importance + 检索重排 | 高 | 已新增 importance 后台评分，并接入 hybrid 检索 generative_score 重排 |
| ✅ 已完成 | 2 | Admission 分层 | 高 | 已新增后台准入评分、admission log、聊天检索隔离与记忆库状态展示 |
| ✅ 已完成 | 3 | Persona 独立层 | 中高 | 已新增 user_persona、后台画像提取、persona_query 检索与 Boot Context 注入 |
| ✅ 已完成 | 4 | Sleep Agent | 中 | 已新增日级整理、Persona Refresh、Reflection 生成与 Boot Context 注入 |
| ✅ 已完成 | 5 | PAHF 双通道纠偏 | 中 | 已支持低置信澄清、自然语言纠偏、Persona 修订与审计日志 |

### 最近完成记录

- ✅ `memory_items` 已新增 `importance`、`admission_score`、`admission_tier` 字段及兼容旧库的迁移检查。
- ✅ `MemoryItem`、后端 `row_to_item()`、前端 `MemoryItemAPI` 已同步新增字段。
- ✅ `score_importance_with_llm()` 已加入 `services/llm.py`，范围为 `1.0-10.0`，失败回退默认 `5.0`。
- ✅ `_update_importance_in_background()` 已加入 `services/memory_service.py`，并挂到 memory store、chat 显式记录、chat 自动沉淀路径。
- ✅ `retrieval_primitives.py` 的 hybrid 检索已带出 `importance`，并用 `generative_score = recency + importance + relevance` 重排。
- ✅ 验证通过：后端测试 `182 passed`，前端 `tsc --noEmit` 通过。
- ✅ `memory_admission_log` 已加入数据库 schema，并由 `init_db()` 兼容创建。
- ✅ `score_admission_with_llm()` 与 `services/memory_admission.py` 已加入，按 utility/confidence/novelty/recency/type_prior 计算 `standard` / `low_value`。
- ✅ `_score_memory_admission_in_background()` 已接入 memory store、chat 显式记录、chat 自动沉淀路径；评分失败时保持默认 `standard`。
- ✅ 聊天检索、Boot Context、结构化事实召回、图谱上下文、consolidation 扫描已默认过滤 `low_value`。
- ✅ 记忆库页面已显示 admission tier 与 score，管理视图仍保留全部记忆。
- ✅ `user_persona` 已加入数据库 schema，并由 `init_db()` 兼容创建。
- ✅ `services/persona_service.py` 已加入，支持稳定画像 LLM 提取、trait_key 校验、confidence 阈值和 upsert 合并。
- ✅ `_score_memory_admission_in_background()` 已在 `standard` 评分成功后触发 Persona 提取；`low_value` 不生成 Persona。
- ✅ 检索层已新增 `persona_query`，Persona 问题优先召回 `user_persona`，空表时回退到现有记忆检索。
- ✅ Boot Context 已加入最多 3 条高置信 Persona，记忆导出已包含 `user_persona`。
- ✅ `memory_reflection` 与 `preference_revision_log` 已加入数据库 schema，并由 `init_db()` 兼容创建。
- ✅ `ChatResponse.needs_clarification` 已预留，默认 `False`，现有聊天响应保持兼容。
- ✅ `MemoryReflection` 与 `PreferenceRevisionLog` 模型契约已加入，导出已包含 reflection / revision log。
- ✅ `services/sleep_agent.py` 已加入，提供 `run_sleep_agent_once()` 与 `run_sleep_agent_periodically()`。
- ✅ Topic Regroup 已扫描最近 24 小时 standard 且 `importance >= 6.0` 的记忆，并把主题聚类作为 Reflection 输入。
- ✅ Persona Refresh 已扫描最近 7 天 high-importance standard 记忆，复用 Persona 提取与 upsert；同一 source memory 重扫不会虚增 `evidence_count`。
- ✅ Reflection 生成已按最近 24 小时 `importance >= 7.0` 且累计重要性阈值写入 `memory_reflection`，并包含重复洞察去重。
- ✅ Boot Context 已加入最多 2 条 high-importance Reflection，聊天 prompt 会显示“长期洞察”分组。
- ✅ 验证通过：Sleep Agent / Persona / Boot Context 相邻测试 `29 passed`。
- ✅ 检索意图已新增 `correction_intent`，支持识别“你记错了”“不对”“其实我”等自然语言纠偏。
- ✅ `services/memory_revision.py` 已加入，支持 Persona 纠偏解析、写入、低置信 Persona 查询与澄清问题生成。
- ✅ 用户纠偏会更新或新增 `user_persona`，并写入 `preference_revision_log` 审计记录。
- ✅ 聊天路径已接入 correction flow；纠偏成功时短路普通回答，不确定时返回 `needs_clarification=True`。
- ✅ 普通聊天已接入低置信 Persona 澄清，确认问题会附在回答末尾。
- ✅ 验证通过：PAHF / correction intent / chat 相邻测试 `63 passed`。

---

## 阶段 0：Schema 与代码契约准备

### 目标

先补数据库结构与模型字段，不改变行为。这样后续每一步都能小步接入。

### 数据库改动

在 `backend/database.py` 中扩展 `BASE_SCHEMA_SQL` 与迁移函数。已有表的列通过 `ensure_memory_items_schema()` 追加，新增表可直接放入 schema。

```sql
ALTER TABLE memory_items ADD COLUMN admission_score REAL DEFAULT NULL;
ALTER TABLE memory_items ADD COLUMN admission_tier TEXT DEFAULT 'standard';
ALTER TABLE memory_items ADD COLUMN importance REAL DEFAULT 5.0;

CREATE TABLE IF NOT EXISTS memory_admission_log (
    id TEXT PRIMARY KEY,
    memory_id TEXT,
    raw_content TEXT NOT NULL,
    score_utility REAL,
    score_confidence REAL,
    score_novelty REAL,
    score_recency REAL,
    score_type_prior REAL,
    total_score REAL,
    admitted INTEGER NOT NULL DEFAULT 1,
    tier TEXT NOT NULL DEFAULT 'standard',
    created_at DATETIME DEFAULT (datetime('now')),
    FOREIGN KEY (memory_id) REFERENCES memory_items(id)
);

CREATE TABLE IF NOT EXISTS user_persona (
    id TEXT PRIMARY KEY,
    trait_key TEXT NOT NULL,
    trait_value TEXT NOT NULL,
    confidence REAL DEFAULT 0.8,
    evidence_count INTEGER DEFAULT 1,
    source_memory_ids TEXT DEFAULT '[]',
    last_updated DATETIME DEFAULT (datetime('now')),
    created_at DATETIME DEFAULT (datetime('now'))
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_user_persona_key
    ON user_persona(trait_key);

CREATE TABLE IF NOT EXISTS memory_reflection (
    id TEXT PRIMARY KEY,
    insight TEXT NOT NULL,
    source_memory_ids TEXT DEFAULT '[]',
    importance REAL DEFAULT 8.0,
    created_at DATETIME DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS preference_revision_log (
    id TEXT PRIMARY KEY,
    persona_id TEXT,
    old_value TEXT,
    new_value TEXT,
    trigger TEXT,
    created_at DATETIME DEFAULT (datetime('now')),
    FOREIGN KEY (persona_id) REFERENCES user_persona(id)
);
```

### 代码契约

- `models.py`
  - `MemoryItem` 增加 `admission_score: float | None`、`admission_tier: str`、`importance: float`。
  - `ChatResponse` 后续阶段可增加 `needs_clarification: bool = False`，保持前端兼容。

- `services/memory_service.py`
  - 当前统一写入口是 `create_memory_item()`，不是旧文档里的 `store_memory()`。
  - `row_to_item()` 要同步读取新增字段。

- `database.py`
  - Web 路由和后台服务继续使用 `get_shared_db()`。
  - `get_db(read_only=True)` 仍保留给 consolidation 的只读快照和脚本使用。

### 验证

- `uv run python -m pytest tests/ -q`
- 手动启动后检查 `init_db()` 不会破坏已有 `data.db`。

---

## 阶段 1：Importance + Generative Agents 检索重排

### 目标

在现有 hybrid 检索结果上叠加 `recency + importance + relevance`，让近期且重要的记忆排得更靠前。

### 当前代码切入点

- 写入侧：`services/memory_service.py:create_memory_item()`
- 后台任务入口：
  - `routers/memory.py:store_memory()`
  - `routers/chat.py` 的显式记录路径和异步 side effects
- 检索侧：`services/retrieval_primitives.py:_retrieve_hybrid_context()`

### 新增函数

在 `services/memory_service.py` 或新建 `services/memory_importance.py`：

```python
async def _update_importance_in_background(memory_item_id: str, content: str) -> None:
    score = await asyncio.wait_for(score_importance_with_llm(content), timeout=30.0)
    db = await get_shared_db()
    await db.execute(
        "UPDATE memory_items SET importance = ? WHERE id = ?",
        (score, memory_item_id),
    )
    await db.commit()
```

建议与 `_update_emotion_score_in_background()` 风格一致：超时跳过、异常记录日志、失败时 rollback。

### 检索重排

当前 `_retrieve_hybrid_context()` 已经生成 `final_score`。不要使用旧文档里的 `item["score"]` 字段。

```python
def _generative_score(item: dict) -> float:
    recency = _recency_score(item["created_at"])
    importance = float(item.get("importance") or 5.0) / 10.0
    relevance = float(item.get("final_score") or 0.0)
    return 0.3 * recency + 0.4 * importance + 0.3 * relevance
```

需要让 `_retrieve_hybrid_keyword_candidates()` / `_retrieve_hybrid_semantic_candidates()` 查询 `importance`，并在 `_row_to_brief(..., include_kind=True)` 或候选 dict 里带出。

### 注意事项

- `weight` 与 `importance` 不合并：
  - `weight`：被召回后重置、随时间衰减，偏“访问热度”。
  - `importance`：内容本身重要性，偏“语义价值”。
- `importance` 默认 5.0，避免旧数据排序异常。

### 验证

- 构造两条同主题记忆：一条旧且低 importance，一条新且高 importance。
- 同一 query 下，新且高 importance 的结果应排在前面。
- 后端测试全绿。

---

## 阶段 2：Admission 分层

### 目标

给记忆打价值分层：`standard` 参与检索，`low_value` 默认不参与检索，`rejected` 仅进入 admission log。

### 当前代码切入点

当前 `create_memory_item()` 是统一写入口。为了不阻塞响应，第一版采用“先写入、后台评分、下次检索生效”的方式。

旧文档写“写入前准入过滤”，但这会把 embedding 和 LLM 评分放进用户请求关键路径，和当前代码的低延迟风格不一致。当前更适合：

1. 同步写入 `memory_items`，默认 `admission_tier='standard'`、`admission_score=NULL`。
2. 后台任务 `_score_memory_admission_in_background()` 计算五维分。
3. 若低于阈值，回写 `admission_tier='low_value'`。
4. 检索默认排除 `low_value/rejected`，但导出和管理页仍可看到。

### 新增文件

`services/memory_admission.py`

```python
@dataclass
class AdmissionScore:
    utility: float
    confidence: float
    novelty: float
    recency: float
    type_prior: float
    total: float
    tier: str


async def compute_admission_score(content: str, kind: str, db) -> AdmissionScore:
    utility = await _score_utility(content)
    confidence = await _score_confidence(content)
    novelty = await _score_novelty(content, db)
    recency = 1.0
    type_prior = _score_type_prior(kind)
    total = 0.3 * utility + 0.2 * confidence + 0.2 * novelty + 0.1 * recency + 0.2 * type_prior
    tier = "low_value" if total < 0.35 else "standard"
    return AdmissionScore(utility, confidence, novelty, recency, type_prior, total, tier)
```

### 检索改动

在 `retrieval_primitives.py` 的记忆查询 SQL 默认加入：

```sql
AND COALESCE(admission_tier, 'standard') = 'standard'
```

要覆盖：

- hybrid keyword
- hybrid semantic 取回 ids 后的 memory_items 查询
- recent kind
- recent pattern
- graph memory recall
- boot context
- summary recent high value

管理页 `/api/memory/search` 可以先不过滤，或者增加 `include_low_value=false` 参数。产品上建议记忆库默认显示全部，聊天检索默认只用 standard。

### 验证

- 低价值闲聊被回写为 `low_value` 后，不再进入聊天 sources。
- 记忆库仍能看到该条记忆及其 tier。

---

## 阶段 3：Persona 独立层

### 目标

把稳定用户特征从 `structured_facts(kind='preference')` 中抽离到 `user_persona`，用于长期、可纠偏的用户画像。

### 当前对应关系

| AdaMem 层 | 当前代码 | 改法 |
|-----------|----------|------|
| Working Memory | `chat_messages` 最近 N 条 | 保留，不新建表 |
| Episodic Memory | `memory_items` | 保留，增加 admission / importance |
| Persona Memory | 部分 `structured_facts(preference/fact)` | 新增 `user_persona` |
| Graph Memory | `graph_nodes` / `graph_edges` | 保留 |

### 迁移策略

第一版不要自动大规模迁移旧数据。更稳的策略：

1. Sleep Agent 的 Persona Refresh 从近期高 importance 记忆中生成/更新 `user_persona`。
2. 对旧 `structured_facts(kind='preference')` 只做只读参考。
3. 等 Persona 层稳定后，再写一次可回滚迁移脚本，把高置信 preference fact 标记为 `migrated`。

原因：当前 `structured_facts` 的 preference 可能是短期偏好，不一定都是稳定 persona。

### Query Intent 改造

- `services/retrieval_constants.py`
  - `QueryIntent` 增加 `"persona_query"`。
  - 增加 persona query patterns，例如“我的习惯”“我的风格”“我通常”“我是什么样的人”。

- `services/retrieval_intent.py`
  - `_detect_query_intent_keyword()` 加 persona 判断。
  - LLM intent mapping 增加 persona label。

- `services/retrieval_context.py`
  - 新增 `_retrieve_persona_context(query, db)`。
  - `retrieve_context()` 中路由 `persona_query`。

### Persona 检索

```python
async def _retrieve_persona_context(query: str, db) -> list[dict]:
    cursor = await db.execute(
        """SELECT id, trait_key, trait_value, confidence, evidence_count, last_updated
           FROM user_persona
           WHERE confidence >= 0.4
           ORDER BY confidence DESC, evidence_count DESC, last_updated DESC
           LIMIT ?""",
        (MAX_LAYER_RESULTS,),
    )
```

第一版可以先不过度语义匹配，直接按 confidence/evidence_count 取高价值 Persona。后续再加 trait_key 关键词过滤。

### 验证

- `user_persona` 有独立行。
- 询问“你觉得我的沟通风格是什么？”时，能从 Persona 表召回，而不是只靠 FTS 或 structured_facts。

---

## 阶段 4：Sleep Agent

### 目标

在现有 `consolidation.py` 之外新增日级 Sleep Agent，负责长期层的重组和演化。

### 实现状态

已完成最小可用闭环：`services/sleep_agent.py` 独立运行，不与 consolidation 混合；Topic Regroup 第一版不落库，只作为 Reflection 输入；Persona Refresh 会处理重复扫描和冲突降置信；Reflection 会写入 `memory_reflection` 并注入 Boot Context。

### 当前代码切入点

- 新建 `services/sleep_agent.py`
- `main.py` lifespan 注册 `run_sleep_agent_periodically()`
- DB 连接使用 `get_shared_db()`，不要在服务内反复打开/关闭普通连接。
- 如果某一步需要只读快照，可像 `consolidation.py` 一样显式使用 `get_db(read_only=True)`。

### 新增服务结构

```python
async def run_sleep_agent_once() -> SleepAgentResult:
    db = await get_shared_db()
    result = SleepAgentResult()
    await _topic_regroup(db, result)
    await _persona_refresh(db, result)
    await _generate_reflections(db, result)
    return result


async def run_sleep_agent_periodically() -> None:
    await asyncio.sleep(SLEEP_AGENT_STARTUP_DELAY_SECONDS)
    while True:
        await run_sleep_agent_once()
        await asyncio.sleep(SLEEP_AGENT_INTERVAL_SECONDS)
```

### 任务 1：Topic Regroup

- 查最近 24 小时 `admission_tier='standard'` 且 `importance >= 6.0` 的 episodic 记忆。
- 用 LLM 聚类主题。
- 第一版 topic 不落库，只作为 reflection prompt 的结构化输入。
- 不删除/覆盖原始记忆，也不修改 `memory_items.consolidated`，避免影响 fact/graph consolidation。

### 任务 2：Persona Refresh

- 扫描过去 7 天 `importance >= 7.0` 的 memory_items。
- LLM 判断是否产生稳定 trait。
- 若支持旧 trait：提高 `evidence_count` / `confidence`。
- 若同一 source memory 重复扫描：不重复增加 `evidence_count`。
- 若矛盾：降低旧 trait `confidence` 并记录日志，不覆盖旧值；revision log 留给阶段 5 用户纠偏。

### 任务 3：Reflection 生成

触发条件：

```sql
SELECT SUM(importance) AS total_importance
FROM memory_items
WHERE created_at > datetime('now', '-1 day')
  AND importance >= 7.0
  AND COALESCE(admission_tier, 'standard') = 'standard'
```

若 `total_importance >= 50`，调用 LLM 生成 1-3 条 reflection，写入 `memory_reflection`。

### 调度

`main.py`：

```python
sleep_task: asyncio.Task | None = None
sleep_enabled = os.getenv("SLEEP_AGENT_ENABLED", "1") != "0"
if sleep_enabled:
    sleep_task = asyncio.create_task(run_sleep_agent_periodically())
```

退出时与 consolidation/weight_decay 一样 cancel + suppress `CancelledError`。

### 验证

- 后台日志出现 Sleep Agent summary。
- `memory_reflection` 有新增行。
- `user_persona` 的 evidence_count/confidence 会随新证据变化。
- Boot Context 出现“长期洞察”分组。

---

## 阶段 5：PAHF 双通道纠偏

### 目标

让系统在不确定时主动问，让用户纠正时能真实修订 Persona。

### 实现状态

已完成最小可用闭环：`correction_intent` 能识别自然语言纠偏；`services/memory_revision.py` 负责解析、匹配、更新 `user_persona` 并写入 `preference_revision_log`；聊天路径在纠偏成功时直接确认，在不确定时返回澄清；普通聊天会对相关低置信 Persona 附加确认问题。

### 前置通道：低置信度澄清

切入点：`routers/chat.py`，在 `retrieve_context()` 后、`answer_with_context()` 前。

```python
low_confidence_personas = await get_low_confidence_personas(
    query=message,
    db=db,
    threshold=0.6,
)
if low_confidence_personas:
    clarification = await generate_persona_clarification(message, low_confidence_personas)
    context = f"{context}\n\n【需要确认的用户画像】\n{clarification}"
```

第一版不强制阻断回答，只把澄清问题附在回答里，并返回 `ChatResponse.needs_clarification=True` / `clarification_question` 供后续 UI 使用。

### 后置通道：用户纠偏

- `retrieval_constants.py`
  - `QueryIntent` 增加 `"correction_intent"`。
- `retrieval_intent.py`
  - 增加关键词：“不对”“你记错了”“不是这样的”“我改变主意了”“以后不要这样记”。
- 新建 `services/memory_revision.py`

```python
async def revise_persona(
    *,
    trait_key: str,
    new_value: str,
    old_value: str | None,
    trigger: str,
    db,
) -> None:
    # 1. 更新或新增 user_persona
    # 2. 保留旧值到 revision log，不直接删除
    # 3. 写 preference_revision_log
```

### 验证

- 用户说“你记错了，我不喜欢被催”后：
  - 相关 Persona confidence 下降或 value 更新。
  - `preference_revision_log` 有记录。
  - 后续回答不再继续使用旧 Persona。

---

## 文件改动汇总

| 文件 | 类型 | 改动内容 |
|------|------|---------|
| `backend/database.py` | 修改 | 新增 memory_items 列、admission/persona/reflection/revision 表；继续使用共享 DB 生命周期 |
| `backend/models.py` | 修改 | `MemoryItem` 新增 admission/importance 字段；`ChatResponse` 后续可加 `needs_clarification` |
| `backend/services/memory_service.py` | 修改 | `create_memory_item()` 后台挂 admission/importance/persona 相关任务 |
| `backend/services/memory_admission.py` | 新建 | Admission 五维评分与 tier 判定 |
| `backend/services/memory_importance.py` | 可选新建 | Importance LLM 评分；也可并入 `memory_service.py` |
| `backend/services/retrieval_primitives.py` | 修改 | 查询带出 importance/admission_tier；hybrid 结果做 generative rerank |
| `backend/services/retrieval_constants.py` | 修改 | 新增 persona/correction intent 常量与权重 |
| `backend/services/retrieval_intent.py` | 修改 | 新增 persona/correction 检测 |
| `backend/services/retrieval_context.py` | 修改 | 新增 `_retrieve_persona_context()` 并接入路由 |
| `backend/services/retrieval_boot.py` | 修改 | Boot Context 可加入高置信 Persona 和 Reflection |
| `backend/services/sleep_agent.py` | 新建 | Topic Regroup、Persona Refresh、Reflection 生成 |
| `backend/services/memory_revision.py` | 新建 | PAHF 后置纠偏 |
| `backend/routers/chat.py` | 修改 | 低置信度澄清、纠偏入口、继续使用同 session history |
| `backend/main.py` | 修改 | 注册 Sleep Agent 调度任务，退出时关闭任务 |
| `frontend/src/types/index.ts` | 修改 | 如启用 `needs_clarification`，同步 Chat API 类型 |
| `frontend/src/app/chat/page.tsx` | 可选修改 | 澄清回复样式、纠偏反馈入口 |

---

## 关键设计决策

### 1. 不新建 Episodic 独立表

`memory_items` 已经承担 episodic 记忆角色。新增表会破坏现有检索、图谱、导出和前端列表，收益不足。

### 2. Persona 独立表，但不急着迁移旧 preference

`structured_facts` 是事实三元组；Persona 需要 confidence、evidence_count、revision log。两者语义不同。第一版先让 Sleep Agent 生成 Persona，旧 preference facts 只读参考，后续再迁移。

### 3. Admission 第一版采用后台评分

当前项目风格是用户请求快速返回，LLM/embedding 任务后台执行。因此 Admission 不应一开始就阻塞写入。低价值内容可以先写入，再回写 tier，并在后续检索中排除。

### 4. Sleep Agent 与 consolidation 分开

`consolidation.py` 负责 memory -> fact/graph 的提升；Sleep Agent 负责 persona/reflection/topic 的长期演化。调度、失败隔离和日志都应分开。

### 5. PAHF 第一版不强制阻断回答

先把澄清问题作为回答的一部分，降低前端改动和交互风险。等 Persona 层稳定，再用 `needs_clarification` 做专门 UI。

---

## 验证里程碑

| 阶段 | 可验证行为 |
|------|------------|
| 阶段 0 | 数据库迁移后旧数据可读；后端测试通过 |
| 阶段 1 | 同类 query 下，高 importance 且近期的记忆排序更靠前 |
| 阶段 2 | low_value 记忆不进入聊天检索 sources，但仍可在记忆库查看 |
| 阶段 3 | `user_persona` 有独立行；persona_query 能从该表召回 |
| 阶段 4 | Sleep Agent 日志出现 summary；`memory_reflection` 出现 insight |
| 阶段 5 | 用户纠偏后 Persona confidence/value 改变，revision log 有记录 |
