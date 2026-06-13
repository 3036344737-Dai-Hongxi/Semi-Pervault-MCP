"""Weight decay service for memory_items.

Two responsibilities:
1. Periodic batch decay: apply exponential decay to all memory weights based on
   time elapsed since last reference (or creation if never referenced).
2. Reference reset: when a memory is surfaced to the user via chat retrieval,
   reset its weight to 1.0 and stamp last_referenced_at = now().

Decay formula (absolute, idempotent):
    w_new = MAX(1.0 * EXP(-ln(2) / H * D), W_min)

where:
    H = half-life in days  (env: WEIGHT_DECAY_HALF_LIFE_DAYS, default 30)
    D = julianday('now') - julianday(COALESCE(last_referenced_at, created_at))
    W_min = floor weight   (env: WEIGHT_DECAY_MIN_WEIGHT, default 0.01)

Because the formula is anchored to the reference/creation timestamp rather than
the previous weight value, running the same batch multiple times yields the same
result (no error accumulation).
"""

import asyncio
import logging
import os

from memory_core.database import get_db

logger = logging.getLogger(__name__)

# ── Configuration (read once at import time so they can be patched in tests) ──

_HALF_LIFE_DAYS = float(os.getenv("WEIGHT_DECAY_HALF_LIFE_DAYS", "30"))
_MIN_WEIGHT = float(os.getenv("WEIGHT_DECAY_MIN_WEIGHT", "0.01"))
_INTERVAL_SECONDS = int(os.getenv("WEIGHT_DECAY_INTERVAL_SECONDS", "3600"))
_STARTUP_DELAY_SECONDS = int(os.getenv("WEIGHT_DECAY_STARTUP_DELAY_SECONDS", "60"))

# ln(2) ≈ 0.6931471805599453
_LN2 = 0.6931471805599453

# Protect memories created within the last day from being decayed immediately.
_NEW_MEMORY_GRACE_DAYS = 1


async def decay_weights_once() -> int:
    """Apply exponential decay to all eligible memory_items weights.

    Skips memories created within the last _NEW_MEMORY_GRACE_DAYS days to avoid
    decaying brand-new entries on the very first scheduler run.

    Returns the number of rows updated.
    """
    half_life = max(_HALF_LIFE_DAYS, 0.1)
    min_weight = max(_MIN_WEIGHT, 0.0)
    decay_rate = _LN2 / half_life

    db = await get_db()
    try:
        cursor = await db.execute(
            """
            UPDATE memory_items
            SET weight = ROUND(
                MAX(
                    EXP(:decay_rate * -(julianday('now') - julianday(COALESCE(last_referenced_at, created_at)))),
                    :min_weight
                ),
                4
            )
            WHERE created_at < datetime('now', :grace_offset)
            """,
            {
                "decay_rate": decay_rate,
                "min_weight": min_weight,
                "grace_offset": f"-{_NEW_MEMORY_GRACE_DAYS} days",
            },
        )
        updated = cursor.rowcount or 0
        await db.commit()
        logger.info(
            "weight decay applied rows=%s half_life_days=%.1f min_weight=%.4f",
            updated,
            half_life,
            min_weight,
        )
        return updated
    except Exception:
        await db.rollback()
        logger.exception("weight decay failed, rolled back")
        return 0
    finally:
        await db.close()


async def reset_referenced_weights(memory_ids: list[str]) -> None:
    """Reset weight to 1.0 and stamp last_referenced_at for referenced memories.

    Called as a background task after retrieve_context returns results.
    Safe to call with duplicates; IDs are de-duplicated by the caller but this
    function also handles them gracefully via IN clause.

    Args:
        memory_ids: List of memory_item IDs that were surfaced to the user.
                    Must be de-duplicated before passing in.
    """
    if not memory_ids:
        return

    placeholders = ",".join("?" for _ in memory_ids)
    db = await get_db()
    try:
        await db.execute(
            f"""
            UPDATE memory_items
            SET weight = 1.0,
                last_referenced_at = datetime('now')
            WHERE id IN ({placeholders})
            """,
            memory_ids,
        )
        await db.commit()
        logger.info(
            "weight reset applied ids=%s",
            memory_ids,
        )
    except Exception:
        await db.rollback()
        logger.exception(
            "weight reset failed for ids=%s, rolled back",
            memory_ids,
        )
    finally:
        await db.close()


async def run_decay_periodically() -> None:
    """Background scheduler loop for periodic weight decay.

    Mirrors the consolidation scheduler pattern: startup delay, then infinite
    loop with configurable interval. Respects asyncio.CancelledError so the
    lifespan shutdown is clean.
    """
    if _STARTUP_DELAY_SECONDS > 0:
        await asyncio.sleep(_STARTUP_DELAY_SECONDS)

    while True:
        try:
            await decay_weights_once()
        except asyncio.CancelledError:
            logger.info("weight decay scheduler cancelled")
            raise
        except Exception:
            logger.exception("weight decay scheduler iteration failed")

        await asyncio.sleep(max(_INTERVAL_SECONDS, 1))
