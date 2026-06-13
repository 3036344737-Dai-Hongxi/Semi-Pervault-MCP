# memory-core

Pervault 的本地优先个人记忆内核，从原 `backend/` 抽离而来。包含：

- `memory_core/database.py` — SQLite schema、迁移、连接管理（FTS5 + sqlite-vec）
- `memory_core/models.py` — Pydantic 领域模型
- `memory_core/exceptions.py` — 领域异常（替代 Web 框架异常）
- `memory_core/services/` — 记忆写入管线、混合检索、知识图谱、画像/洞察、后台整理 agent

硬性边界：本包**不得**依赖 `fastapi` / `slowapi` 等 Web 框架（由 `tests/test_import_boundary.py` 守卫）。

数据库位置由环境变量 `PERVAULT_DB_PATH` 控制；未设置时默认 `./data.db`（相对当前工作目录）。

测试：`uv run pytest`
