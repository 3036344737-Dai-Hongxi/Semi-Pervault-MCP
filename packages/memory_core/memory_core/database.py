import asyncio
import aiosqlite
import json
import logging
import os
from pathlib import Path
import re

import sqlite_vec

# 数据库位置由宿主通过 PERVAULT_DB_PATH 指定（如 backend/main.py 锚定到 backend/data.db）；
# 未设置时回退到当前工作目录，便于内核独立使用。
DB_PATH = Path(os.getenv("PERVAULT_DB_PATH", "data.db"))
VECTOR_DIMENSIONS = 768
logger = logging.getLogger(__name__)
_shared_db: aiosqlite.Connection | None = None
_shared_db_lock = asyncio.Lock()

BASE_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS voice_records (
    id TEXT PRIMARY KEY,
    raw_transcript TEXT,
    normalized_text TEXT,
    status TEXT DEFAULT 'raw',
    confidence REAL,
    created_at DATETIME DEFAULT (datetime('now')),
    updated_at DATETIME DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS memory_items (
    id TEXT PRIMARY KEY,
    voice_record_id TEXT,
    content TEXT,
    content_version INTEGER NOT NULL DEFAULT 1,
    tags TEXT DEFAULT '[]',
    kind TEXT DEFAULT 'other',
    task_status TEXT,
    emotion_score REAL DEFAULT 0.0,
    consolidated INTEGER DEFAULT 0,
    importance REAL DEFAULT 5.0,
    admission_score REAL DEFAULT NULL,
    admission_tier TEXT DEFAULT 'standard',
    weight REAL DEFAULT 1.0,
    last_referenced_at DATETIME,
    created_at DATETIME DEFAULT (datetime('now')),
    FOREIGN KEY (voice_record_id) REFERENCES voice_records(id)
);

CREATE TABLE IF NOT EXISTS structured_facts (
    id TEXT PRIMARY KEY,
    memory_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    subject TEXT NOT NULL DEFAULT '',
    predicate TEXT NOT NULL DEFAULT '',
    object TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'accepted',
    created_at DATETIME DEFAULT (datetime('now')),
    FOREIGN KEY (memory_id) REFERENCES memory_items(id)
);

CREATE INDEX IF NOT EXISTS idx_structured_facts_kind_status_created
    ON structured_facts(kind, status, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_structured_facts_memory
    ON structured_facts(memory_id);

CREATE UNIQUE INDEX IF NOT EXISTS idx_structured_facts_memory_fact
    ON structured_facts(memory_id, kind, subject, predicate, object);

CREATE INDEX IF NOT EXISTS idx_structured_facts_match
    ON structured_facts(kind, subject, predicate, status, created_at DESC);

CREATE TABLE IF NOT EXISTS memory_admission_log (
    id TEXT PRIMARY KEY,
    memory_id TEXT,
    raw_content TEXT NOT NULL,
    score_utility REAL,
    score_confidence REAL,
    score_novelty REAL,
    score_recency REAL,
    score_type_prior REAL,
    total_score REAL,
    admitted INTEGER NOT NULL DEFAULT 1,
    tier TEXT NOT NULL DEFAULT 'standard',
    created_at DATETIME DEFAULT (datetime('now')),
    FOREIGN KEY (memory_id) REFERENCES memory_items(id)
);

CREATE INDEX IF NOT EXISTS idx_memory_admission_log_memory_created
    ON memory_admission_log(memory_id, created_at DESC);

CREATE TABLE IF NOT EXISTS user_persona (
    id TEXT PRIMARY KEY,
    trait_key TEXT NOT NULL,
    trait_value TEXT NOT NULL,
    confidence REAL DEFAULT 0.8,
    evidence_count INTEGER DEFAULT 1,
    source_memory_ids TEXT DEFAULT '[]',
    last_updated DATETIME DEFAULT (datetime('now')),
    created_at DATETIME DEFAULT (datetime('now'))
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_user_persona_key
    ON user_persona(trait_key);

CREATE INDEX IF NOT EXISTS idx_user_persona_confidence_updated
    ON user_persona(confidence DESC, evidence_count DESC, last_updated DESC);

CREATE TABLE IF NOT EXISTS memory_reflection (
    id TEXT PRIMARY KEY,
    insight TEXT NOT NULL,
    source_memory_ids TEXT DEFAULT '[]',
    insight_dedupe_key TEXT DEFAULT '',
    source_memory_fingerprint TEXT DEFAULT '[]',
    importance REAL DEFAULT 8.0,
    created_at DATETIME DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_memory_reflection_importance_created
    ON memory_reflection(importance DESC, created_at DESC);

CREATE TABLE IF NOT EXISTS preference_revision_log (
    id TEXT PRIMARY KEY,
    persona_id TEXT,
    old_value TEXT,
    new_value TEXT,
    trigger TEXT,
    created_at DATETIME DEFAULT (datetime('now')),
    FOREIGN KEY (persona_id) REFERENCES user_persona(id)
);

CREATE INDEX IF NOT EXISTS idx_preference_revision_log_persona_created
    ON preference_revision_log(persona_id, created_at DESC);

CREATE TABLE IF NOT EXISTS auth_sessions (
    id TEXT PRIMARY KEY,
    token_hash TEXT NOT NULL,
    created_at DATETIME DEFAULT (datetime('now')),
    expires_at DATETIME NOT NULL,
    last_seen_at DATETIME DEFAULT (datetime('now')),
    revoked_at DATETIME,
    ip_address TEXT,
    user_agent TEXT
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_auth_sessions_token_hash
    ON auth_sessions(token_hash);

CREATE INDEX IF NOT EXISTS idx_auth_sessions_active
    ON auth_sessions(revoked_at, expires_at DESC);

CREATE TABLE IF NOT EXISTS data_export_log (
    id TEXT PRIMARY KEY,
    auth_session_id TEXT,
    status TEXT NOT NULL,
    requested_at DATETIME DEFAULT (datetime('now')),
    client_ip TEXT,
    user_agent TEXT,
    memory_count INTEGER,
    fact_count INTEGER,
    persona_count INTEGER,
    reflection_count INTEGER,
    revision_count INTEGER,
    graph_node_count INTEGER,
    graph_edge_count INTEGER,
    FOREIGN KEY (auth_session_id) REFERENCES auth_sessions(id)
);

CREATE INDEX IF NOT EXISTS idx_data_export_log_requested
    ON data_export_log(requested_at DESC);

CREATE INDEX IF NOT EXISTS idx_data_export_log_session_requested
    ON data_export_log(auth_session_id, requested_at DESC);

CREATE TABLE IF NOT EXISTS background_jobs (
    id TEXT PRIMARY KEY,
    job_type TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    origin TEXT NOT NULL DEFAULT 'pipeline',
    origin_run_id TEXT,
    payload_json TEXT NOT NULL,
    dedupe_key TEXT NOT NULL,
    attempt_count INTEGER NOT NULL DEFAULT 0,
    max_attempts INTEGER NOT NULL DEFAULT 3,
    available_at DATETIME DEFAULT (datetime('now')),
    started_at DATETIME,
    finished_at DATETIME,
    last_error TEXT,
    created_at DATETIME DEFAULT (datetime('now')),
    updated_at DATETIME DEFAULT (datetime('now')),
    lease_expires_at DATETIME,
    heartbeat_at DATETIME,
    lease_token TEXT,
    terminal_reason TEXT
);

CREATE INDEX IF NOT EXISTS idx_background_jobs_status_available
    ON background_jobs(status, available_at ASC, created_at ASC);

CREATE INDEX IF NOT EXISTS idx_background_jobs_dedupe_created
    ON background_jobs(dedupe_key, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_background_jobs_lease_status
    ON background_jobs(status, lease_expires_at ASC);

CREATE TABLE IF NOT EXISTS scheduler_run_log (
    id TEXT PRIMARY KEY,
    scheduler_name TEXT NOT NULL,
    status TEXT NOT NULL,
    started_at DATETIME DEFAULT (datetime('now')),
    finished_at DATETIME,
    summary_json TEXT,
    error_count INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_scheduler_run_log_name_started
    ON scheduler_run_log(scheduler_name, started_at DESC);

CREATE TABLE IF NOT EXISTS sleep_agent_checkpoint (
    stage_name TEXT PRIMARY KEY,
    checkpoint_created_at DATETIME,
    last_run_id TEXT,
    updated_at DATETIME DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS chat_messages (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    needs_clarification INTEGER NOT NULL DEFAULT 0,
    clarification_question TEXT,
    created_at DATETIME DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_chat_messages_session_created
    ON chat_messages(session_id, created_at DESC);

CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
    content,
    tags,
    content='memory_items',
    content_rowid='rowid'
);

CREATE TRIGGER IF NOT EXISTS memory_ai AFTER INSERT ON memory_items BEGIN
    INSERT INTO memory_fts(rowid, content, tags)
    VALUES (new.rowid, new.content, new.tags);
END;

CREATE TRIGGER IF NOT EXISTS memory_ad AFTER DELETE ON memory_items BEGIN
    INSERT INTO memory_fts(memory_fts, rowid, content, tags)
    VALUES ('delete', old.rowid, old.content, old.tags);
END;

CREATE TRIGGER IF NOT EXISTS memory_au AFTER UPDATE ON memory_items BEGIN
    INSERT INTO memory_fts(memory_fts, rowid, content, tags)
    VALUES ('delete', old.rowid, old.content, old.tags);
    INSERT INTO memory_fts(rowid, content, tags)
    VALUES (new.rowid, new.content, new.tags);
END;

-- Stage 3: graph tables

CREATE TABLE IF NOT EXISTS graph_nodes (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    label TEXT NOT NULL,
    properties TEXT DEFAULT '{}',
    weight REAL DEFAULT 1.0,
    source_memory_count INTEGER DEFAULT 0,
    created_at DATETIME DEFAULT (datetime('now')),
    last_seen_at DATETIME DEFAULT (datetime('now'))
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_graph_nodes_type_label
    ON graph_nodes(type, label);

CREATE TABLE IF NOT EXISTS graph_edges (
    id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    relation TEXT NOT NULL,
    weight REAL DEFAULT 1.0,
    source_memory_id TEXT,
    created_at DATETIME DEFAULT (datetime('now')),
    FOREIGN KEY (source_id) REFERENCES graph_nodes(id),
    FOREIGN KEY (target_id) REFERENCES graph_nodes(id),
    FOREIGN KEY (source_memory_id) REFERENCES memory_items(id)
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_graph_edges_src_tgt_rel
    ON graph_edges(source_id, target_id, relation);
"""

VECTOR_SCHEMA_SQL = f"""
CREATE VIRTUAL TABLE IF NOT EXISTS vec_items USING vec0(
    ref_id TEXT PRIMARY KEY,
    ref_type TEXT PARTITION KEY,
    embedding FLOAT[{VECTOR_DIMENSIONS}]
);
"""


async def load_sqlite_vec(db: aiosqlite.Connection) -> bool:
    try:
        await db.enable_load_extension(True)
        await db.load_extension(sqlite_vec.loadable_path())
        await db.enable_load_extension(False)
        return True
    except Exception:
        logger.exception("Failed to load sqlite-vec extension")
        try:
            await db.enable_load_extension(False)
        except Exception:
            pass
        return False


async def _configure_db(db: aiosqlite.Connection, *, read_only: bool) -> aiosqlite.Connection:
    db.row_factory = aiosqlite.Row
    if not read_only:
        await db.execute("PRAGMA journal_mode=WAL")
    # 写锁竞争时等待重试而不是立即抛 SQLITE_BUSY（多连接/多进程访问的前提）
    await db.execute("PRAGMA busy_timeout=5000")
    await db.execute("PRAGMA foreign_keys=ON")
    db.sqlite_vec_loaded = await load_sqlite_vec(db)
    return db


async def get_db(*, read_only: bool = False) -> aiosqlite.Connection:
    if read_only:
        db = await aiosqlite.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    else:
        db = await aiosqlite.connect(DB_PATH)
    return await _configure_db(db, read_only=read_only)


async def get_shared_db() -> aiosqlite.Connection:
    """返回进程级共享单连接（aiosqlite 内部串行化语句）。

    用途：高频只读检索路径复用连接、省去反复建连开销。注意（conc-6）：
    这是单条连接，调用方**不得在其上开显式长事务或做需要隔离的批量写**——
    那会阻塞所有共享读端点。写路径请用独立的 get_db()（见 routers 各写端点）。
    """
    global _shared_db
    if _shared_db is not None:
        return _shared_db

    async with _shared_db_lock:
        if _shared_db is None:
            db = await aiosqlite.connect(DB_PATH)
            _shared_db = await _configure_db(db, read_only=False)
            logger.info("Shared SQLite connection opened")
        return _shared_db


async def close_shared_db() -> None:
    global _shared_db
    async with _shared_db_lock:
        if _shared_db is not None:
            await _shared_db.close()
            _shared_db = None
            logger.info("Shared SQLite connection closed")


async def _table_exists(db: aiosqlite.Connection, table_name: str) -> bool:
    cursor = await db.execute(
        """SELECT 1
           FROM sqlite_master
           WHERE type = 'table' AND name = ?""",
        (table_name,),
    )
    row = await cursor.fetchone()
    return row is not None


async def _table_has_column(
    db: aiosqlite.Connection, table_name: str, column_name: str
) -> bool:
    cursor = await db.execute(f"PRAGMA table_info({table_name})")
    rows = await cursor.fetchall()
    return any(row["name"] == column_name for row in rows)


async def _memory_items_has_column(
    db: aiosqlite.Connection, column_name: str
) -> bool:
    return await _table_has_column(db, "memory_items", column_name)


async def _graph_nodes_has_column(
    db: aiosqlite.Connection, column_name: str
) -> bool:
    return await _table_has_column(db, "graph_nodes", column_name)


async def ensure_graph_nodes_schema(db: aiosqlite.Connection) -> None:
    if not await _table_exists(db, "graph_nodes"):
        return

    if not await _graph_nodes_has_column(db, "status"):
        await db.execute(
            "ALTER TABLE graph_nodes ADD COLUMN status TEXT NOT NULL DEFAULT 'confirmed'"
        )
        logger.info("Added graph_nodes.status column")

    if not await _graph_nodes_has_column(db, "possible_duplicate_of"):
        await db.execute(
            "ALTER TABLE graph_nodes ADD COLUMN possible_duplicate_of TEXT"
        )
        logger.info("Added graph_nodes.possible_duplicate_of column")

    await db.execute(
        "CREATE INDEX IF NOT EXISTS idx_graph_nodes_status ON graph_nodes(status)"
    )


async def ensure_memory_items_schema(db: aiosqlite.Connection) -> None:
    if not await _table_exists(db, "memory_items"):
        return

    if not await _memory_items_has_column(db, "kind"):
        await db.execute("ALTER TABLE memory_items ADD COLUMN kind TEXT DEFAULT 'other'")
        logger.info("Added memory_items.kind column")

    if not await _memory_items_has_column(db, "task_status"):
        await db.execute("ALTER TABLE memory_items ADD COLUMN task_status TEXT")
        logger.info("Added memory_items.task_status column")

    if not await _memory_items_has_column(db, "emotion_score"):
        await db.execute("ALTER TABLE memory_items ADD COLUMN emotion_score REAL DEFAULT 0.0")
        logger.info("Added memory_items.emotion_score column")

    if not await _memory_items_has_column(db, "consolidated"):
        await db.execute("ALTER TABLE memory_items ADD COLUMN consolidated INTEGER DEFAULT 0")
        logger.info("Added memory_items.consolidated column")

    if not await _memory_items_has_column(db, "importance"):
        await db.execute("ALTER TABLE memory_items ADD COLUMN importance REAL DEFAULT 5.0")
        logger.info("Added memory_items.importance column")

    if not await _memory_items_has_column(db, "admission_score"):
        await db.execute("ALTER TABLE memory_items ADD COLUMN admission_score REAL DEFAULT NULL")
        logger.info("Added memory_items.admission_score column")

    if not await _memory_items_has_column(db, "admission_tier"):
        await db.execute("ALTER TABLE memory_items ADD COLUMN admission_tier TEXT DEFAULT 'standard'")
        logger.info("Added memory_items.admission_tier column")

    if not await _memory_items_has_column(db, "content_version"):
        await db.execute("ALTER TABLE memory_items ADD COLUMN content_version INTEGER NOT NULL DEFAULT 1")
        logger.info("Added memory_items.content_version column")

    await db.execute(
        """UPDATE memory_items
           SET kind = 'other'
           WHERE kind IS NULL OR TRIM(kind) = ''"""
    )
    await db.execute(
        """UPDATE memory_items
           SET task_status = 'open'
           WHERE kind = 'task'
             AND (
                 task_status IS NULL
                 OR TRIM(task_status) = ''
                 OR task_status NOT IN ('open', 'in_progress', 'done', 'expired')
             )"""
    )
    await db.execute(
        """UPDATE memory_items
           SET task_status = NULL
           WHERE kind != 'task'
             AND task_status IS NOT NULL
             AND TRIM(task_status) = ''"""
    )
    await db.execute(
        """UPDATE memory_items
           SET emotion_score = 0.0
           WHERE emotion_score IS NULL"""
    )
    await db.execute(
        """UPDATE memory_items
           SET consolidated = 0
           WHERE consolidated IS NULL"""
    )
    await db.execute(
        """UPDATE memory_items
           SET importance = 5.0
           WHERE importance IS NULL"""
    )
    await db.execute(
        """UPDATE memory_items
           SET admission_tier = 'standard'
           WHERE admission_tier IS NULL OR TRIM(admission_tier) = ''"""
    )
    await db.execute(
        """UPDATE memory_items
           SET content_version = 1
           WHERE content_version IS NULL OR content_version < 1"""
    )
    await db.execute(
        """CREATE INDEX IF NOT EXISTS idx_memory_items_kind_created
           ON memory_items(kind, created_at DESC)"""
    )
    await db.execute(
        """CREATE INDEX IF NOT EXISTS idx_memory_items_kind_task_status_created
           ON memory_items(kind, task_status, created_at DESC)"""
    )
    await db.execute(
        """CREATE INDEX IF NOT EXISTS idx_memory_items_consolidated_created
           ON memory_items(consolidated, created_at ASC)"""
    )


async def ensure_memory_admission_schema(db: aiosqlite.Connection) -> None:
    await db.execute(
        """CREATE TABLE IF NOT EXISTS memory_admission_log (
               id TEXT PRIMARY KEY,
               memory_id TEXT,
               raw_content TEXT NOT NULL,
               score_utility REAL,
               score_confidence REAL,
               score_novelty REAL,
               score_recency REAL,
               score_type_prior REAL,
               total_score REAL,
               admitted INTEGER NOT NULL DEFAULT 1,
               tier TEXT NOT NULL DEFAULT 'standard',
               created_at DATETIME DEFAULT (datetime('now')),
               FOREIGN KEY (memory_id) REFERENCES memory_items(id)
           )"""
    )
    await db.execute(
        """CREATE INDEX IF NOT EXISTS idx_memory_admission_log_memory_created
           ON memory_admission_log(memory_id, created_at DESC)"""
    )


async def ensure_user_persona_schema(db: aiosqlite.Connection) -> None:
    await db.execute(
        """CREATE TABLE IF NOT EXISTS user_persona (
               id TEXT PRIMARY KEY,
               trait_key TEXT NOT NULL,
               trait_value TEXT NOT NULL,
               confidence REAL DEFAULT 0.8,
               evidence_count INTEGER DEFAULT 1,
               source_memory_ids TEXT DEFAULT '[]',
               last_updated DATETIME DEFAULT (datetime('now')),
               created_at DATETIME DEFAULT (datetime('now'))
           )"""
    )
    await db.execute(
        """CREATE UNIQUE INDEX IF NOT EXISTS idx_user_persona_key
           ON user_persona(trait_key)"""
    )
    await db.execute(
        """CREATE INDEX IF NOT EXISTS idx_user_persona_confidence_updated
           ON user_persona(confidence DESC, evidence_count DESC, last_updated DESC)"""
    )


def _normalize_reflection_insight_dedupe_key(text: str) -> str:
    return re.sub(r"[，。！？、,.!?\s]", "", (text or "").strip().lower())


def _normalize_reflection_source_memory_fingerprint(source_memory_ids: list[str]) -> str:
    normalized_ids = sorted(
        {
            source_id.strip()
            for source_id in source_memory_ids
            if isinstance(source_id, str) and source_id.strip()
        }
    )
    return json.dumps(normalized_ids, ensure_ascii=False, separators=(",", ":"))


def _reflection_source_memory_fingerprint_from_json(value: str | None) -> str:
    if not value:
        return "[]"
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return "[]"
    if not isinstance(parsed, list):
        return "[]"
    return _normalize_reflection_source_memory_fingerprint(parsed)


async def ensure_memory_reflection_schema(db: aiosqlite.Connection) -> None:
    await db.execute(
        """CREATE TABLE IF NOT EXISTS memory_reflection (
               id TEXT PRIMARY KEY,
               insight TEXT NOT NULL,
               source_memory_ids TEXT DEFAULT '[]',
               insight_dedupe_key TEXT DEFAULT '',
               source_memory_fingerprint TEXT DEFAULT '[]',
               importance REAL DEFAULT 8.0,
               created_at DATETIME DEFAULT (datetime('now'))
           )"""
    )
    if not await _table_has_column(db, "memory_reflection", "insight_dedupe_key"):
        await db.execute(
            "ALTER TABLE memory_reflection ADD COLUMN insight_dedupe_key TEXT DEFAULT ''"
        )
        logger.info("Added memory_reflection.insight_dedupe_key column")
    if not await _table_has_column(db, "memory_reflection", "source_memory_fingerprint"):
        await db.execute(
            "ALTER TABLE memory_reflection ADD COLUMN source_memory_fingerprint TEXT DEFAULT '[]'"
        )
        logger.info("Added memory_reflection.source_memory_fingerprint column")

    cursor = await db.execute(
        """SELECT id, insight, source_memory_ids
           FROM memory_reflection
           WHERE insight_dedupe_key IS NULL
              OR TRIM(insight_dedupe_key) = ''
              OR source_memory_fingerprint IS NULL
              OR TRIM(source_memory_fingerprint) = ''"""
    )
    rows = await cursor.fetchall()
    for row in rows:
        await db.execute(
            """UPDATE memory_reflection
               SET insight_dedupe_key = ?,
                   source_memory_fingerprint = ?
               WHERE id = ?""",
            (
                _normalize_reflection_insight_dedupe_key(str(row["insight"] or "")),
                _reflection_source_memory_fingerprint_from_json(
                    row["source_memory_ids"]
                ),
                row["id"],
            ),
        )

    await db.execute(
        """CREATE INDEX IF NOT EXISTS idx_memory_reflection_importance_created
           ON memory_reflection(importance DESC, created_at DESC)"""
    )
    await db.execute(
        """CREATE INDEX IF NOT EXISTS idx_memory_reflection_insight_dedupe_key
           ON memory_reflection(insight_dedupe_key)"""
    )
    await db.execute(
        """CREATE INDEX IF NOT EXISTS idx_memory_reflection_source_fingerprint
           ON memory_reflection(source_memory_fingerprint, created_at DESC)"""
    )


async def ensure_preference_revision_schema(db: aiosqlite.Connection) -> None:
    await db.execute(
        """CREATE TABLE IF NOT EXISTS preference_revision_log (
               id TEXT PRIMARY KEY,
               persona_id TEXT,
               old_value TEXT,
               new_value TEXT,
               trigger TEXT,
               created_at DATETIME DEFAULT (datetime('now')),
               FOREIGN KEY (persona_id) REFERENCES user_persona(id)
           )"""
    )
    await db.execute(
        """CREATE INDEX IF NOT EXISTS idx_preference_revision_log_persona_created
           ON preference_revision_log(persona_id, created_at DESC)"""
    )


async def ensure_auth_sessions_schema(db: aiosqlite.Connection) -> None:
    await db.execute(
        """CREATE TABLE IF NOT EXISTS auth_sessions (
               id TEXT PRIMARY KEY,
               token_hash TEXT NOT NULL,
               created_at DATETIME DEFAULT (datetime('now')),
               expires_at DATETIME NOT NULL,
               last_seen_at DATETIME DEFAULT (datetime('now')),
               revoked_at DATETIME,
               ip_address TEXT,
               user_agent TEXT
           )"""
    )
    await db.execute(
        """CREATE UNIQUE INDEX IF NOT EXISTS idx_auth_sessions_token_hash
           ON auth_sessions(token_hash)"""
    )
    await db.execute(
        """CREATE INDEX IF NOT EXISTS idx_auth_sessions_active
           ON auth_sessions(revoked_at, expires_at DESC)"""
    )


async def ensure_data_export_log_schema(db: aiosqlite.Connection) -> None:
    await db.execute(
        """CREATE TABLE IF NOT EXISTS data_export_log (
               id TEXT PRIMARY KEY,
               auth_session_id TEXT,
               status TEXT NOT NULL,
               requested_at DATETIME DEFAULT (datetime('now')),
               client_ip TEXT,
               user_agent TEXT,
               memory_count INTEGER,
               fact_count INTEGER,
               persona_count INTEGER,
               reflection_count INTEGER,
               revision_count INTEGER,
               graph_node_count INTEGER,
               graph_edge_count INTEGER,
               FOREIGN KEY (auth_session_id) REFERENCES auth_sessions(id)
           )"""
    )
    await db.execute(
        """CREATE INDEX IF NOT EXISTS idx_data_export_log_requested
           ON data_export_log(requested_at DESC)"""
    )
    await db.execute(
        """CREATE INDEX IF NOT EXISTS idx_data_export_log_session_requested
           ON data_export_log(auth_session_id, requested_at DESC)"""
    )


async def ensure_background_jobs_schema(db: aiosqlite.Connection) -> None:
    await db.execute(
        """CREATE TABLE IF NOT EXISTS background_jobs (
               id TEXT PRIMARY KEY,
               job_type TEXT NOT NULL,
               status TEXT NOT NULL DEFAULT 'pending',
               origin TEXT NOT NULL DEFAULT 'pipeline',
               origin_run_id TEXT,
               payload_json TEXT NOT NULL,
               dedupe_key TEXT NOT NULL,
               attempt_count INTEGER NOT NULL DEFAULT 0,
               max_attempts INTEGER NOT NULL DEFAULT 3,
               available_at DATETIME DEFAULT (datetime('now')),
               started_at DATETIME,
               finished_at DATETIME,
               last_error TEXT,
               created_at DATETIME DEFAULT (datetime('now')),
               updated_at DATETIME DEFAULT (datetime('now')),
               lease_expires_at DATETIME,
               heartbeat_at DATETIME,
               lease_token TEXT,
               terminal_reason TEXT
           )"""
    )
    if not await _table_has_column(db, "background_jobs", "terminal_reason"):
        await db.execute("ALTER TABLE background_jobs ADD COLUMN terminal_reason TEXT")
        logger.info("Added background_jobs.terminal_reason column")
    if not await _table_has_column(db, "background_jobs", "origin"):
        await db.execute("ALTER TABLE background_jobs ADD COLUMN origin TEXT NOT NULL DEFAULT 'pipeline'")
        logger.info("Added background_jobs.origin column")
    if not await _table_has_column(db, "background_jobs", "origin_run_id"):
        await db.execute("ALTER TABLE background_jobs ADD COLUMN origin_run_id TEXT")
        logger.info("Added background_jobs.origin_run_id column")
    await db.execute(
        """CREATE INDEX IF NOT EXISTS idx_background_jobs_status_available
           ON background_jobs(status, available_at ASC, created_at ASC)"""
    )
    await db.execute(
        """CREATE INDEX IF NOT EXISTS idx_background_jobs_dedupe_created
           ON background_jobs(dedupe_key, created_at DESC)"""
    )
    await db.execute(
        """CREATE INDEX IF NOT EXISTS idx_background_jobs_lease_status
           ON background_jobs(status, lease_expires_at ASC)"""
    )


async def ensure_scheduler_run_log_schema(db: aiosqlite.Connection) -> None:
    await db.execute(
        """CREATE TABLE IF NOT EXISTS scheduler_run_log (
               id TEXT PRIMARY KEY,
               scheduler_name TEXT NOT NULL,
               status TEXT NOT NULL,
               started_at DATETIME DEFAULT (datetime('now')),
               finished_at DATETIME,
               summary_json TEXT,
               error_count INTEGER NOT NULL DEFAULT 0
           )"""
    )
    await db.execute(
        """CREATE INDEX IF NOT EXISTS idx_scheduler_run_log_name_started
           ON scheduler_run_log(scheduler_name, started_at DESC)"""
    )


async def ensure_sleep_agent_checkpoint_schema(db: aiosqlite.Connection) -> None:
    await db.execute(
        """CREATE TABLE IF NOT EXISTS sleep_agent_checkpoint (
               stage_name TEXT PRIMARY KEY,
               checkpoint_created_at DATETIME,
               last_run_id TEXT,
               updated_at DATETIME DEFAULT (datetime('now'))
           )"""
    )


async def ensure_chat_messages_schema(db: aiosqlite.Connection) -> None:
    await db.execute(
        """CREATE TABLE IF NOT EXISTS chat_messages (
               id TEXT PRIMARY KEY,
               session_id TEXT NOT NULL,
               role TEXT NOT NULL,
               content TEXT NOT NULL,
               needs_clarification INTEGER NOT NULL DEFAULT 0,
               clarification_question TEXT,
               created_at DATETIME DEFAULT (datetime('now'))
           )"""
    )
    if not await _table_has_column(db, "chat_messages", "needs_clarification"):
        await db.execute(
            "ALTER TABLE chat_messages ADD COLUMN needs_clarification INTEGER NOT NULL DEFAULT 0"
        )
        logger.info("Added chat_messages.needs_clarification column")
    if not await _table_has_column(db, "chat_messages", "clarification_question"):
        await db.execute(
            "ALTER TABLE chat_messages ADD COLUMN clarification_question TEXT"
        )
        logger.info("Added chat_messages.clarification_question column")
    await db.execute(
        """UPDATE chat_messages
           SET needs_clarification = 0
           WHERE needs_clarification IS NULL"""
    )
    await db.execute(
        """CREATE INDEX IF NOT EXISTS idx_chat_messages_session_created
           ON chat_messages(session_id, created_at DESC)"""
    )


async def init_db():
    db = await get_db()
    try:
        await db.executescript(BASE_SCHEMA_SQL)
        await ensure_memory_items_schema(db)
        await ensure_memory_admission_schema(db)
        await ensure_user_persona_schema(db)
        await ensure_memory_reflection_schema(db)
        await ensure_preference_revision_schema(db)
        await ensure_auth_sessions_schema(db)
        await ensure_data_export_log_schema(db)
        await ensure_background_jobs_schema(db)
        await ensure_scheduler_run_log_schema(db)
        await ensure_sleep_agent_checkpoint_schema(db)
        await ensure_chat_messages_schema(db)
        await ensure_graph_nodes_schema(db)
        if getattr(db, "sqlite_vec_loaded", False):
            await db.executescript(VECTOR_SCHEMA_SQL)
        await db.commit()
    finally:
        await db.close()
