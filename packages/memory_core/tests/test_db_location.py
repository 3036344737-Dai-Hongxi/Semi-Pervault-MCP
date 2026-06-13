"""DB 位置解析与一次性迁移测试（全部在临时目录演练，不碰真实 HOME）。"""

import sqlite3
from pathlib import Path

from memory_core import db_location
from memory_core.db_location import ensure_db_location, migrate_legacy_db


def _make_sqlite(path: Path, marker: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE m (v TEXT)")
    conn.execute("INSERT INTO m VALUES (?)", (marker,))
    conn.commit()
    conn.close()


def _read_marker(path: Path) -> str:
    conn = sqlite3.connect(path)
    try:
        return conn.execute("SELECT v FROM m").fetchone()[0]
    finally:
        conn.close()


def test_fresh_when_no_legacy(tmp_path):
    legacy = tmp_path / "old" / "data.db"
    target = tmp_path / "home" / ".pervault" / "data.db"
    assert migrate_legacy_db(legacy, target) == "fresh"
    assert target.parent.is_dir()
    assert not target.exists()


def test_migrates_and_keeps_legacy_as_backup(tmp_path):
    legacy = tmp_path / "old" / "data.db"
    target = tmp_path / "home" / ".pervault" / "data.db"
    _make_sqlite(legacy, "hello")

    assert migrate_legacy_db(legacy, target) == "migrated"
    assert _read_marker(target) == "hello"
    assert legacy.exists(), "旧库必须原样保留作为备份"
    assert _read_marker(legacy) == "hello"


def test_idempotent_second_run_does_not_overwrite(tmp_path):
    legacy = tmp_path / "old" / "data.db"
    target = tmp_path / "home" / ".pervault" / "data.db"
    _make_sqlite(legacy, "v1")
    assert migrate_legacy_db(legacy, target) == "migrated"

    # 旧库后续变化不应再影响目标（迁移只发生一次）
    conn = sqlite3.connect(legacy)
    conn.execute("UPDATE m SET v = 'v2'")
    conn.commit()
    conn.close()

    assert migrate_legacy_db(legacy, target) == "already"
    assert _read_marker(target) == "v1"


def test_failure_rolls_back_and_falls_back_to_legacy(tmp_path, monkeypatch):
    legacy = tmp_path / "old" / "data.db"
    target = tmp_path / "home" / ".pervault" / "data.db"
    _make_sqlite(legacy, "keep-me")

    def boom(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(db_location.shutil, "copy2", boom)

    assert migrate_legacy_db(legacy, target) == "failed"
    assert not target.exists(), "失败后不得留下残缺目标文件"

    resolved = ensure_db_location(legacy, target)
    assert resolved == legacy, "失败时应回退旧路径保证应用可用"
    assert _read_marker(legacy) == "keep-me"


def test_corrupt_copy_detected_and_rolled_back(tmp_path, monkeypatch):
    legacy = tmp_path / "old" / "data.db"
    target = tmp_path / "home" / ".pervault" / "data.db"
    _make_sqlite(legacy, "x")
    monkeypatch.setattr(db_location, "_verify_sqlite_ok", lambda _p: False)

    assert migrate_legacy_db(legacy, target) == "failed"
    assert not target.exists()


def test_ensure_returns_target_on_success(tmp_path):
    legacy = tmp_path / "old" / "data.db"
    target = tmp_path / "home" / ".pervault" / "data.db"
    _make_sqlite(legacy, "ok")
    assert ensure_db_location(legacy, target) == target
    # 再跑一次幂等
    assert ensure_db_location(legacy, target) == target
