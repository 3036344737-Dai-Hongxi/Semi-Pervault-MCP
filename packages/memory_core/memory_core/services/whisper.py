import asyncio
import io
import math
import os
from pathlib import Path
from openai import AsyncOpenAI
from memory_core.services.llm import raise_ai_service_error

_whisper_client: AsyncOpenAI | None = None


def _get_whisper_client() -> AsyncOpenAI:
    global _whisper_client
    if _whisper_client is None:
        _whisper_client = AsyncOpenAI(
            api_key=os.getenv("WHISPER_API_KEY") or os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("WHISPER_BASE_URL", "https://api.openai.com/v1"),
        )
    return _whisper_client


async def transcribe(audio_path: Path) -> tuple[str, float]:
    """Call Whisper API. Returns (transcript, confidence)."""
    c = _get_whisper_client()
    # Read the file in a thread pool to avoid blocking the async event loop
    file_bytes = await asyncio.to_thread(audio_path.read_bytes)
    file_obj = io.BytesIO(file_bytes)
    file_obj.name = audio_path.name  # SDK uses name to infer audio format
    try:
        result = await c.audio.transcriptions.create(
            model=os.getenv("WHISPER_MODEL", "whisper-large-v2"),
            file=file_obj,
            response_format="verbose_json",
            language="zh",
        )
    except Exception as exc:
        import logging
        logging.getLogger("uvicorn.error").error("Whisper API error: %s %s", type(exc).__name__, exc)
        raise_ai_service_error(exc, "ASR 服务")
    text = result.text or ""
    avg_confidence = 1.0
    if hasattr(result, "segments") and result.segments:
        probs = [
            s.get("avg_logprob", 0) if isinstance(s, dict) else getattr(s, "avg_logprob", 0)
            for s in result.segments
        ]
        if probs:
            avg_confidence = round(min(1.0, max(0.0, math.exp(sum(probs) / len(probs)))), 3)
    return text, avg_confidence
