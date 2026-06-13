"""MemoryRuntime：内核后台循环的统一启动/停止句柄。

daemon 宿主（当前是 backend/main.py 的 lifespan）通过它管理内核生命周期，
保证「后台循环只在一个进程里跑」这条硬不变量有单一入口可控。

环境变量开关与既有行为一致：
- CONSOLIDATION_SCHEDULER_ENABLED（默认 1）
- WEIGHT_DECAY_SCHEDULER_ENABLED（默认 1）
- SLEEP_AGENT_ENABLED（默认 1）
- background jobs worker 始终启动（与原 main.py 行为一致）
"""

from __future__ import annotations

import asyncio
from contextlib import suppress
import logging
import os

from memory_core.database import close_shared_db, init_db
from memory_core.services.background_jobs import run_background_jobs_worker
from memory_core.services.consolidation import run_periodically
from memory_core.services.memory_service import register_memory_pipeline_job_handlers
from memory_core.services.sleep_agent import run_sleep_agent_periodically
from memory_core.services.weight_decay import run_decay_periodically

logger = logging.getLogger(__name__)


def _flag_enabled(env_name: str) -> bool:
    return os.getenv(env_name, "1") != "0"


def _log_if_failed(name: str):
    """done callback：后台循环若因非取消异常退出（如启动期 get_db 抛错），
    记录下来而非静默消失（logic-3）。"""

    def _cb(task: "asyncio.Task") -> None:
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.error("MemoryRuntime loop '%s' crashed: %r", name, exc)

    return _cb


class MemoryRuntime:
    def __init__(self) -> None:
        self._tasks: dict[str, asyncio.Task] = {}
        self._started = False

    @property
    def started(self) -> bool:
        return self._started

    @property
    def task_names(self) -> list[str]:
        return sorted(self._tasks)

    async def start(self) -> None:
        if self._started:
            return
        await init_db()
        register_memory_pipeline_job_handlers()

        if _flag_enabled("CONSOLIDATION_SCHEDULER_ENABLED"):
            self._tasks["consolidation"] = asyncio.create_task(run_periodically())
        if _flag_enabled("WEIGHT_DECAY_SCHEDULER_ENABLED"):
            self._tasks["weight_decay"] = asyncio.create_task(run_decay_periodically())
        if _flag_enabled("SLEEP_AGENT_ENABLED"):
            self._tasks["sleep_agent"] = asyncio.create_task(run_sleep_agent_periodically())
        self._tasks["background_jobs"] = asyncio.create_task(run_background_jobs_worker())

        for name, task in self._tasks.items():
            task.add_done_callback(_log_if_failed(name))

        self._started = True
        logger.info("MemoryRuntime started loops=%s", self.task_names)

    async def stop(self) -> None:
        if not self._started:
            return
        for name, task in self._tasks.items():
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
            logger.info("MemoryRuntime loop stopped: %s", name)
        self._tasks.clear()
        await close_shared_db()
        self._started = False
        logger.info("MemoryRuntime stopped")
