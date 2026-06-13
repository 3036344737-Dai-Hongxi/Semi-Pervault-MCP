import hashlib
import os

from memory_core.services.memory_policy import normalize_query_key


def sensitive_debug_logs_enabled() -> bool:
    return os.getenv("ENABLE_SENSITIVE_DEBUG_LOGS", "0") != "0"


def maybe_sensitive_preview(value: str | None, *, limit: int = 200) -> str:
    text = (value or "").strip()
    if not text:
        return "<empty>"
    if sensitive_debug_logs_enabled():
        return text[:limit]
    return "<redacted>"


def text_fingerprint(value: str | None) -> str:
    normalized = normalize_query_key(value or "")
    if not normalized:
        return "empty"
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:12]
