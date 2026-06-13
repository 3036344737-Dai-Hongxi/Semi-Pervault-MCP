"""memory_core 领域异常。

内核不依赖 Web 框架，因此不抛 HTTPException；由各适配层（如 FastAPI 宿主）
把这些异常映射为各自协议的错误（404 / 500 等）。
"""


class MemoryCoreError(Exception):
    """内核通用错误基类。适配层默认映射为 500。"""


class MemoryNotFoundError(MemoryCoreError):
    """目标记忆不存在。适配层默认映射为 404。"""


class StorageError(MemoryCoreError):
    """写入/更新后无法读回等存储一致性错误。适配层默认映射为 500。"""
