"""busy_timeout 并发写行为验证。

不变量：第二个写连接在写锁被占时应等待重试（busy_timeout），
而不是立即抛出 sqlite3.OperationalError: database is locked。
"""

import asyncio

import pytest

from memory_core import database
from memory_core.database import get_db


@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "busy-test.db")
    yield


async def test_configure_db_sets_busy_timeout(tmp_db):
    db = await get_db()
    try:
        cursor = await db.execute("PRAGMA busy_timeout")
        row = await cursor.fetchone()
        assert row[0] == 5000
    finally:
        await db.close()


async def test_second_writer_waits_instead_of_failing_immediately(tmp_db):
    writer_a = await get_db()
    writer_b = await get_db()
    try:
        await writer_a.execute("CREATE TABLE IF NOT EXISTS t (v TEXT)")
        await writer_a.commit()

        # A 显式抢占写锁
        await writer_a.execute("BEGIN IMMEDIATE")
        await writer_a.execute("INSERT INTO t (v) VALUES ('a')")

        async def write_b():
            # busy_timeout 生效时：阻塞等待 A 释放，而不是立即报错
            await writer_b.execute("INSERT INTO t (v) VALUES ('b')")
            await writer_b.commit()

        b_task = asyncio.create_task(write_b())
        await asyncio.sleep(0.3)
        assert not b_task.done(), "B 不应立即失败，应在等待写锁"

        await writer_a.commit()  # 释放写锁
        await asyncio.wait_for(b_task, timeout=10)

        cursor = await writer_a.execute("SELECT count(*) FROM t")
        row = await cursor.fetchone()
        assert row[0] == 2
    finally:
        await writer_a.close()
        await writer_b.close()
