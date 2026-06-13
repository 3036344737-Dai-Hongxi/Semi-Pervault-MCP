"""本地宿主间共享的 Core API token（单用户本地信任模型）。

daemon 启动时确保 token 存在；MCP 桥接等本地客户端读取同一文件完成鉴权。
token 是「这台机器上的这个用户」的唯一安全边界，文件权限必须 0600。
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
import secrets
import stat

logger = logging.getLogger(__name__)

_TOKEN_ENV = "PERVAULT_CORE_TOKEN"
_TOKEN_FILENAME = "core_token"


def core_token_path() -> Path:
    base = os.getenv("PERVAULT_HOME")
    home = Path(base) if base else Path.home() / ".pervault"
    return home / _TOKEN_FILENAME


def _tighten_perms(path: Path) -> None:
    """把 token 文件权限单调收敛到 0600；group/other 可读时纠正并告警。

    防御「文件已存在但权限被改松」（备份恢复、手改、历史 TOCTOU 残留）的情形——
    core token 是唯一安全边界，权限只能越来越紧。
    """
    mode = stat.S_IMODE(os.stat(path).st_mode)
    if mode & 0o077:
        os.chmod(path, 0o600)
        logger.warning("tightened core token perms %o -> 0600: %s", mode, path)


def read_or_create_core_token() -> str:
    """读取本地 core token；不存在则原子创建（0600）。环境变量可覆盖（测试/特殊部署用）。"""
    env_token = os.getenv(_TOKEN_ENV, "").strip()
    if env_token:
        return env_token

    path = core_token_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    # 目录也收紧到仅属主可访问（与 data.db 同处一目录）
    try:
        os.chmod(path.parent, 0o700)
    except OSError:
        logger.debug("could not chmod token dir: %s", path.parent)

    if path.exists():
        token = path.read_text(encoding="utf-8").strip()
        if token:
            _tighten_perms(path)
            return token
        logger.warning("core token file empty, regenerating: %s", path)

    token = secrets.token_hex(32)
    # 原子创建：O_CREAT 的 mode 受 umask 约束只会更紧（0o600 无 group/other 位，
    # umask 不会加位），文件自创建起即 0600，无 0644→0600 的世界可读窗口。
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, token.encode("utf-8"))
    finally:
        os.close(fd)
    _tighten_perms(path)  # 兜底：文件先前已存在（空）时 O_CREAT 不改权限
    logger.info("core token generated: %s", path)
    return token
