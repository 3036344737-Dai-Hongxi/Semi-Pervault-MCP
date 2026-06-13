#!/bin/zsh
# Pervault Memory Daemon：启动常驻记忆内核，供 MCP（Claude Desktop / Cursor）使用。
# 仅绑定 127.0.0.1，不对外网开放。

set -e
cd "$(dirname "$0")/backend"

export PATH="$HOME/.local/bin:$PATH"

if ! command -v uv >/dev/null 2>&1; then
  echo "缺少 uv，请先安装：https://docs.astral.sh/uv/getting-started/"
  exit 1
fi

echo "启动 Pervault Memory Daemon (http://127.0.0.1:8000) ..."
echo "数据库：${PERVAULT_DB_PATH:-$HOME/.pervault/data.db}"
exec uv run python -m uvicorn main:app --host 127.0.0.1 --port 8000
