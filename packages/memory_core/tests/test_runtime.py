"""MemoryRuntime 生命周期测试。"""

import asyncio

import pytest

from memory_core import database
from memory_core.runtime import MemoryRuntime


@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "runtime-test.db")
    yield


async def test_start_launches_all_loops_by_default(tmp_db, monkeypatch):
    for flag in (
        "CONSOLIDATION_SCHEDULER_ENABLED",
        "WEIGHT_DECAY_SCHEDULER_ENABLED",
        "SLEEP_AGENT_ENABLED",
    ):
        monkeypatch.delenv(flag, raising=False)

    runtime = MemoryRuntime()
    await runtime.start()
    try:
        assert runtime.started
        assert runtime.task_names == [
            "background_jobs",
            "consolidation",
            "sleep_agent",
            "weight_decay",
        ]
    finally:
        await runtime.stop()
    assert not runtime.started
    assert runtime.task_names == []


async def test_env_flags_disable_schedulers_but_not_jobs_worker(tmp_db, monkeypatch):
    monkeypatch.setenv("CONSOLIDATION_SCHEDULER_ENABLED", "0")
    monkeypatch.setenv("WEIGHT_DECAY_SCHEDULER_ENABLED", "0")
    monkeypatch.setenv("SLEEP_AGENT_ENABLED", "0")

    runtime = MemoryRuntime()
    await runtime.start()
    try:
        assert runtime.task_names == ["background_jobs"]
    finally:
        await runtime.stop()


async def test_start_is_idempotent(tmp_db, monkeypatch):
    monkeypatch.setenv("CONSOLIDATION_SCHEDULER_ENABLED", "0")
    monkeypatch.setenv("WEIGHT_DECAY_SCHEDULER_ENABLED", "0")
    monkeypatch.setenv("SLEEP_AGENT_ENABLED", "0")

    runtime = MemoryRuntime()
    await runtime.start()
    first_tasks = dict(runtime._tasks)
    await runtime.start()  # 第二次应是 no-op
    try:
        assert runtime._tasks == first_tasks
    finally:
        await runtime.stop()


async def test_stop_without_start_is_safe(tmp_db):
    runtime = MemoryRuntime()
    await runtime.stop()  # 不应抛异常
    assert not runtime.started


async def test_stop_cancels_running_tasks(tmp_db, monkeypatch):
    monkeypatch.setenv("CONSOLIDATION_SCHEDULER_ENABLED", "0")
    monkeypatch.setenv("WEIGHT_DECAY_SCHEDULER_ENABLED", "0")
    monkeypatch.setenv("SLEEP_AGENT_ENABLED", "0")

    runtime = MemoryRuntime()
    await runtime.start()
    worker_task = runtime._tasks["background_jobs"]
    assert not worker_task.done()
    await runtime.stop()
    # worker 是无限循环，正常情况下只可能因取消而结束——强断言「确实被取消」，
    # 而非弱化的 `cancelled() or done()`（done() 对任何已结束任务恒真，会放过
    # 「worker 因 bug 提前正常退出、根本没响应取消」的情形）。
    assert worker_task.cancelled()
