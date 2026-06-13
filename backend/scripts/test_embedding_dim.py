import asyncio
import json
import math
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

load_dotenv(BACKEND_DIR / ".env")

from memory_core.services.llm import embed_text, get_embedding_dim  # noqa: E402


async def main() -> None:
    vector = await embed_text("你好，我在做 Pervault 项目")
    expected_dim = get_embedding_dim()
    norm = math.sqrt(sum(x * x for x in vector))

    print(f"dim = {len(vector)}")
    print(f"head = {json.dumps(vector[:5], ensure_ascii=False)}")
    print(f"norm = {norm}")

    if len(vector) != expected_dim:
        raise SystemExit(
            f"Embedding dimension mismatch: expected {expected_dim}, got {len(vector)}"
        )
    if abs(norm - 1.0) >= 1e-5:
        raise SystemExit(f"Embedding norm mismatch: expected 1.0, got {norm}")

    print("OK")


if __name__ == "__main__":
    asyncio.run(main())
