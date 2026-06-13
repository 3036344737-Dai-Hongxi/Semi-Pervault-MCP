import uuid
import tempfile
from pathlib import Path
from fastapi import APIRouter, Request, UploadFile, File, HTTPException
from memory_core.models import VoiceUploadResponse, ClarifyRequest, ClarifyResponse
from memory_core.database import get_db
from memory_core.services.whisper import transcribe
from memory_core.services.llm import AIServiceUnavailableError, clarify_transcript
from services.rate_limit import limiter

router = APIRouter(prefix="/api/voice", tags=["voice"])

ALLOWED_TYPES = {"audio/webm", "audio/mp3", "audio/mpeg", "audio/m4a", "audio/mp4",
                 "audio/x-m4a", "video/webm", "audio/wav", "audio/ogg"}
MAX_UPLOAD_BYTES = 25 * 1024 * 1024


@router.post("/upload", response_model=VoiceUploadResponse)
@limiter.limit("10/minute")
async def upload_voice(request: Request, file: UploadFile = File(...)):
    # 快速预检：Content-Length 超标直接拒绝，避免将超大文件读入内存
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"文件过大，最大支持 25 MB")

    content_type = file.content_type or ""
    base_type = content_type.split(";")[0].strip()
    if base_type not in ALLOWED_TYPES:
        raise HTTPException(400, f"不支持的音频格式: {content_type}")

    record_id = str(uuid.uuid4())

    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        # 兜底检查（Content-Length 可伪造）
        raise HTTPException(
            413,
            f"文件过大，最大支持 25 MB，当前约 {len(data) // 1024 // 1024} MB",
        )

    suffix = Path(file.filename or "audio.webm").suffix or ".webm"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        tmp_path = Path(tmp.name)

    try:
        try:
            transcript, confidence = await transcribe(tmp_path)
        except AIServiceUnavailableError as exc:
            raise HTTPException(503, str(exc)) from exc
    finally:
        tmp_path.unlink(missing_ok=True)

    db = await get_db()
    try:
        await db.execute(
            """INSERT INTO voice_records (id, raw_transcript, confidence, status)
               VALUES (?, ?, ?, 'raw')""",
            (record_id, transcript, confidence),
        )
        await db.commit()
    except Exception:
        await db.rollback()
        raise
    finally:
        await db.close()

    return VoiceUploadResponse(id=record_id, transcript=transcript, confidence=confidence)


@router.post("/clarify", response_model=ClarifyResponse)
async def clarify_voice(req: ClarifyRequest):
    read_db = await get_db(read_only=True)
    try:
        cur = await read_db.execute(
            "SELECT id FROM voice_records WHERE id = ?", (req.voice_record_id,)
        )
        if await cur.fetchone() is None:
            raise HTTPException(404, "语音记录不存在")
    finally:
        await read_db.close()

    try:
        result = await clarify_transcript(req.raw_transcript)
    except AIServiceUnavailableError as exc:
        raise HTTPException(503, str(exc)) from exc

    status = result.get("status", "clear")
    normalized = result.get("normalized_text")
    question = result.get("question")

    write_db = await get_db()
    try:
        if status == "clear" and normalized:
            await write_db.execute(
                """UPDATE voice_records
                   SET normalized_text = ?, status = 'clarified',
                       updated_at = datetime('now')
                   WHERE id = ?""",
                (normalized, req.voice_record_id),
            )
        else:
            await write_db.execute(
                """UPDATE voice_records
                   SET status = 'unclear', updated_at = datetime('now')
                   WHERE id = ?""",
                (req.voice_record_id,),
            )
        await write_db.commit()
    except Exception:
        await write_db.rollback()
        raise
    finally:
        await write_db.close()

    return ClarifyResponse(status=status, normalized_text=normalized, question=question)
