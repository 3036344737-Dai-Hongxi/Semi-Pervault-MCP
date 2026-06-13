import json
import logging
import uuid
from difflib import SequenceMatcher

from memory_core.database import get_db
from memory_core.models import GraphEdge, GraphNode
from memory_core.services.graph_extract import extract_graph
from memory_core.services.memory_policy import normalize_query_key

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dedup detection constants
# ---------------------------------------------------------------------------

# Honorific suffixes to strip from person labels (longest first to avoid
# partial-match issues: strip "总监" before "总").
_PERSON_SUFFIXES: tuple[str, ...] = (
    "总监", "总裁", "经理", "主任", "老师", "先生", "女士", "阿姨",
    "总", "师", "哥", "姐", "叔",
)
_PERSON_PREFIXES: tuple[str, ...] = ("老", "小", "大")

# Project name suffixes to strip.
_PROJECT_SUFFIXES: tuple[str, ...] = ("项目", "project", "plan", "计划")

# Minimum length (in Python str chars) a dedup_key must have to be eligible
# for comparison.  Prevents single-character keys from generating noise.
_DEDUP_MIN_KEY_LEN: int = 2

# SequenceMatcher similarity threshold for R3.
_SIMILARITY_THRESHOLD: float = 0.85

# R3 only applies to labels this short or shorter (measured in str chars,
# so roughly 6 CJK characters or 6 ASCII characters).
_SIMILARITY_MAX_LABEL_LEN: int = 6


# ---------------------------------------------------------------------------
# Dedup helpers (pure functions, no I/O)
# ---------------------------------------------------------------------------


def _compute_dedup_key(label: str, node_type: str) -> str:
    """Return a normalised key used for duplicate candidate comparison.

    Applies type-specific honorific / suffix stripping on top of the shared
    ``normalize_query_key`` (strip punct, collapse whitespace, lowercase).
    """
    key = normalize_query_key(label)

    if node_type == "person":
        stripped = False
        for suffix in _PERSON_SUFFIXES:
            if key.endswith(suffix) and len(key) - len(suffix) >= _DEDUP_MIN_KEY_LEN:
                key = key[: -len(suffix)]
                stripped = True
                break
        if not stripped:
            for prefix in _PERSON_PREFIXES:
                if (
                    key.startswith(prefix)
                    and len(key) - len(prefix) >= _DEDUP_MIN_KEY_LEN
                ):
                    key = key[len(prefix):]
                    break

    elif node_type == "project":
        for suffix in _PROJECT_SUFFIXES:
            if key.endswith(suffix) and len(key) - len(suffix) >= _DEDUP_MIN_KEY_LEN:
                key = key[: -len(suffix)]
                break

    return key


def _is_duplicate_candidate(
    new_key: str,
    existing_key: str,
    new_label: str,
    existing_label: str,
) -> bool:
    """Return True when new_key is a likely duplicate of existing_key.

    Rules (applied in order, all same-type only):
      R1  Exact dedup-key match (catches same entity with different titles).
      R3  SequenceMatcher ≥ 0.85 for short labels (catches minor typos in
          short English or CJK names).  Only applied when both original labels
          are ≤ _SIMILARITY_MAX_LABEL_LEN chars long.
    """
    if not new_key or not existing_key:
        return False

    # R1: exact normalised-key equality
    if new_key == existing_key:
        return True

    # R3: similarity for short labels only
    if (
        len(new_label) <= _SIMILARITY_MAX_LABEL_LEN
        and len(existing_label) <= _SIMILARITY_MAX_LABEL_LEN
        and SequenceMatcher(None, new_key, existing_key).ratio() >= _SIMILARITY_THRESHOLD
    ):
        return True

    return False


async def _find_duplicate_candidate(
    db, label: str, node_type: str
) -> str | None:
    """Search for an existing *confirmed* node that may be a duplicate of *label*.

    Returns the ``id`` of the first matching node, or ``None`` if no candidate
    is found.

    Falls back to ``None`` on any DB or logic error so that the calling code
    always gets a safe result.
    """
    try:
        new_key = _compute_dedup_key(label, node_type)
        if len(new_key) < _DEDUP_MIN_KEY_LEN:
            return None

        cursor = await db.execute(
            "SELECT id, label FROM graph_nodes WHERE type = ? AND status = 'confirmed'",
            (node_type,),
        )
        rows = await cursor.fetchall()

        for row in rows:
            existing_label: str = row["label"]
            # Skip if it's the exact same label — the ON CONFLICT path handles that.
            if existing_label == label:
                continue
            existing_key = _compute_dedup_key(existing_label, node_type)
            if len(existing_key) < _DEDUP_MIN_KEY_LEN:
                continue
            if _is_duplicate_candidate(new_key, existing_key, label, existing_label):
                return row["id"]

        return None
    except Exception:
        logger.exception(
            "dedup detection failed for label=%r type=%s, defaulting to confirmed",
            label,
            node_type,
        )
        return None


# ---------------------------------------------------------------------------
# Row → model helpers
# ---------------------------------------------------------------------------


def _row_to_node(row) -> GraphNode:
    props_raw = row["properties"] or "{}"
    try:
        props = json.loads(props_raw)
    except (json.JSONDecodeError, TypeError):
        props = {}
    row_keys = set(row.keys())
    return GraphNode(
        id=row["id"],
        type=row["type"],
        label=row["label"],
        properties=props,
        weight=row["weight"],
        source_memory_count=row["source_memory_count"],
        created_at=row["created_at"],
        last_seen_at=row["last_seen_at"],
        status=row["status"] if "status" in row_keys else "confirmed",
        possible_duplicate_of=(
            row["possible_duplicate_of"] if "possible_duplicate_of" in row_keys else None
        ),
    )


def _row_to_edge(row) -> GraphEdge:
    return GraphEdge(
        id=row["id"],
        source_id=row["source_id"],
        target_id=row["target_id"],
        relation=row["relation"],
        weight=row["weight"],
        source_memory_id=row["source_memory_id"],
        created_at=row["created_at"],
    )


# ---------------------------------------------------------------------------
# Core graph persistence
# ---------------------------------------------------------------------------


async def persist_graph(
    db, nodes: list[dict], edges: list[dict], memory_item_id: str
) -> tuple[list[GraphNode], list[GraphEdge]]:
    """Persist extracted nodes and edges to the database.

    For each node:
    - If a row with the same (type, label) already exists → bump stats only,
      leave status/possible_duplicate_of unchanged.
    - If it is a genuinely new node → run duplicate-candidate detection and
      set status = 'pending' (+ possible_duplicate_of) when a candidate is
      found, otherwise status = 'confirmed'.

    The ON CONFLICT unique index on (type, label) is preserved as a safety net
    for concurrent writes, but the primary path is now an explicit
    SELECT-before-INSERT so we can apply dedup logic only on new nodes.
    """
    label_to_id: dict[str, str] = {}
    persisted_nodes: list[GraphNode] = []

    for node in nodes:
        node_type = node["type"]
        label = node["label"]

        # ── Phase 1: check whether this node already exists ──────────────
        cur = await db.execute(
            "SELECT id FROM graph_nodes WHERE type = ? AND label = ?",
            (node_type, label),
        )
        existing = await cur.fetchone()

        if existing:
            # Existing node: bump counters; do NOT touch status/dedup fields.
            cur = await db.execute(
                """UPDATE graph_nodes
                   SET weight = weight + 1,
                       source_memory_count = source_memory_count + 1,
                       last_seen_at = datetime('now')
                   WHERE type = ? AND label = ?
                   RETURNING *""",
                (node_type, label),
            )
            row = await cur.fetchone()
        else:
            # ── Phase 2: new node → dedup detection ──────────────────────
            node_id = str(uuid.uuid4())
            duplicate_id = await _find_duplicate_candidate(db, label, node_type)
            status = "pending" if duplicate_id else "confirmed"

            cur = await db.execute(
                """INSERT INTO graph_nodes
                   (id, type, label, source_memory_count, status, possible_duplicate_of)
                   VALUES (?, ?, ?, 1, ?, ?)
                   RETURNING *""",
                (node_id, node_type, label, status, duplicate_id),
            )
            row = await cur.fetchone()
            logger.info(
                "graph node inserted id=%s label=%r type=%s status=%s "
                "possible_duplicate_of=%s",
                node_id,
                label,
                node_type,
                status,
                duplicate_id,
            )

        if row is None:
            continue
        label_to_id[label] = row["id"]
        persisted_nodes.append(_row_to_node(row))

    persisted_edges: list[GraphEdge] = []
    for edge in edges:
        source_id = label_to_id.get(edge["source"])
        target_id = label_to_id.get(edge["target"])
        if not source_id or not target_id:
            continue

        edge_id = str(uuid.uuid4())
        cur = await db.execute(
            """INSERT INTO graph_edges (id, source_id, target_id, relation, source_memory_id)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(source_id, target_id, relation) DO UPDATE SET
                   weight = weight + 1,
                   source_memory_id = excluded.source_memory_id
               RETURNING *""",
            (edge_id, source_id, target_id, edge["relation"], memory_item_id),
        )
        row = await cur.fetchone()
        if row is None:
            continue
        persisted_edges.append(_row_to_edge(row))

    return persisted_nodes, persisted_edges


async def extract_and_store_graph(
    memory_item_id: str,
    content: str,
    *,
    db=None,
) -> tuple[list[GraphNode], list[GraphEdge]]:
    extracted = await extract_graph(content)

    owns_db = db is None
    if db is None:
        db = await get_db()
    try:
        nodes, edges = await persist_graph(
            db, extracted["nodes"], extracted["edges"], memory_item_id
        )
        await db.commit()
        return nodes, edges
    except Exception:
        await db.rollback()
        logger.exception("Failed to extract/store graph for memory %s", memory_item_id)
        raise
    finally:
        if owns_db:
            await db.close()
