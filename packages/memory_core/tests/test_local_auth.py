"""core token 生成/读取/权限测试（PERVAULT_HOME 指向临时目录，不碰真实 HOME）。"""

import os
import stat

from memory_core.local_auth import core_token_path, read_or_create_core_token


def test_creates_token_with_0600(tmp_path, monkeypatch):
    monkeypatch.setenv("PERVAULT_HOME", str(tmp_path))
    monkeypatch.delenv("PERVAULT_CORE_TOKEN", raising=False)

    token = read_or_create_core_token()
    path = core_token_path()

    assert len(token) == 64  # token_hex(32)
    assert path.read_text().strip() == token
    # 自创建起即无 group/other 位（不存在 0644→0600 的世界可读窗口）
    assert stat.S_IMODE(os.stat(path).st_mode) & 0o077 == 0


def test_token_created_0600_even_under_loose_umask(tmp_path, monkeypatch):
    monkeypatch.setenv("PERVAULT_HOME", str(tmp_path))
    monkeypatch.delenv("PERVAULT_CORE_TOKEN", raising=False)
    old = os.umask(0)  # 最宽松 umask，模拟最坏情况
    try:
        read_or_create_core_token()
    finally:
        os.umask(old)
    assert stat.S_IMODE(os.stat(core_token_path()).st_mode) & 0o077 == 0


def test_existing_loose_perms_are_tightened_on_read(tmp_path, monkeypatch):
    monkeypatch.setenv("PERVAULT_HOME", str(tmp_path))
    monkeypatch.delenv("PERVAULT_CORE_TOKEN", raising=False)

    first = read_or_create_core_token()
    path = core_token_path()
    os.chmod(path, 0o644)  # 模拟权限被改松（备份恢复/手改）

    second = read_or_create_core_token()  # 读时应收敛权限
    assert second == first
    assert stat.S_IMODE(os.stat(path).st_mode) & 0o077 == 0


def test_reads_existing_token_stable(tmp_path, monkeypatch):
    monkeypatch.setenv("PERVAULT_HOME", str(tmp_path))
    monkeypatch.delenv("PERVAULT_CORE_TOKEN", raising=False)

    first = read_or_create_core_token()
    second = read_or_create_core_token()
    assert first == second


def test_env_override_wins(tmp_path, monkeypatch):
    monkeypatch.setenv("PERVAULT_HOME", str(tmp_path))
    monkeypatch.setenv("PERVAULT_CORE_TOKEN", "env-token-abc")

    assert read_or_create_core_token() == "env-token-abc"
    assert not core_token_path().exists(), "env 覆盖时不应落盘"


def test_empty_file_regenerates(tmp_path, monkeypatch):
    monkeypatch.setenv("PERVAULT_HOME", str(tmp_path))
    monkeypatch.delenv("PERVAULT_CORE_TOKEN", raising=False)
    path = core_token_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("")

    token = read_or_create_core_token()
    assert len(token) == 64
    assert path.read_text().strip() == token
