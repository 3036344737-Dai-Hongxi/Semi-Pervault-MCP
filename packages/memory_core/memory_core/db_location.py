"""数据库默认位置与一次性迁移（P1）。

目标：所有宿主共享 `~/.pervault/data.db`。
迁移策略为**复制式**：旧库（如 backend/data.db）原样保留，天然充当备份；
任何失败都回退到旧路径继续运行，绝不让应用因迁移瘫痪。

宿主用法（见 backend/main.py）：仅当 PERVAULT_DB_PATH 未设置时调用
`ensure_db_location(legacy_path)`，把返回值写入该环境变量。
"""

from __future__ import annotations

import logging
from pathlib import Path
import shutil
import sqlite3

logger = logging.getLogger(__name__)

# SQLite 运行文件后缀。复制主库 + WAL 即可保证数据完整（WAL 含未 checkpoint 的写）；
# -shm 是 WAL 的共享内存索引、纯缓存，由 SQLite 在下次打开时按 WAL 重建——复制它反而
# 可能与新 -wal 时刻不一致，故不复制（logic-4）。
_SQLITE_SUFFIXES = ("", "-wal")


def default_home_db_path() -> Path:
    return Path.home() / ".pervault" / "data.db"


def _verify_sqlite_ok(db_path: Path) -> bool:
    try:
        conn = sqlite3.connect(db_path)
        try:
            row = conn.execute("PRAGMA quick_check").fetchone()
            return bool(row) and row[0] == "ok"
        finally:
            conn.close()
    except sqlite3.Error:
        logger.exception("migrated db failed quick_check: %s", db_path)
        return False


def migrate_legacy_db(legacy_path: Path, target_path: Path) -> str:
    """把旧库复制到新位置。幂等；旧库永不修改/删除。

    返回状态：
    - "already"  目标已存在，无操作
    - "fresh"    无旧库，仅确保目标目录存在
    - "migrated" 复制成功且完整性校验通过
    - "failed"   复制或校验失败，已清理残留（调用方应回退旧路径）
    """
    if target_path.exists():
        return "already"

    target_path.parent.mkdir(parents=True, exist_ok=True)

    if not legacy_path.exists():
        return "fresh"

    copied: list[Path] = []
    try:
        for suffix in _SQLITE_SUFFIXES:
            src = Path(str(legacy_path) + suffix)
            if src.exists():
                dst = Path(str(target_path) + suffix)
                shutil.copy2(src, dst)
                copied.append(dst)
        if not _verify_sqlite_ok(target_path):
            raise RuntimeError("quick_check failed")
        logger.info("db migrated: %s -> %s (legacy kept as backup)", legacy_path, target_path)
        return "migrated"
    except Exception:
        logger.exception("db migration failed, rolling back partial copy")
        for dst in copied:
            try:
                dst.unlink(missing_ok=True)
            except OSError:
                logger.exception("failed to remove partial file: %s", dst)
        return "failed"


def ensure_db_location(legacy_path: Path, target_path: Path | None = None) -> Path:
    """返回宿主应使用的 DB 路径（默认 ~/.pervault/data.db），按需执行一次性迁移。

    失败回退：迁移失败时返回旧路径，应用保持原行为运行。
    """
    target = target_path or default_home_db_path()
    status = migrate_legacy_db(legacy_path, target)
    if status == "failed":
        logger.warning("falling back to legacy db path: %s", legacy_path)
        return legacy_path
    return target
