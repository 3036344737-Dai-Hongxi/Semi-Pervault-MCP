#!/usr/bin/env python3
"""Manual entry-point for a single consolidation pass.

Usage:
    uv run python scripts/run_consolidation.py [--dry-run] [--verbose] [--limit N]
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

import memory_core.database as database
from memory_core.services.consolidation import MemoryReview, run_once


def _print_review(review: MemoryReview) -> None:
    print("-" * 48)
    print(f"memory_id: {review.memory_id}")
    print(f"original_kind: {review.original_kind}")
    print(f"route decision: {review.route_decision}")
    print(f"planned action: {review.planned_action}")
    if review.structured_fact_match_key:
        print(f"structured_fact match key: {review.structured_fact_match_key}")
    if review.graph_candidate_types:
        print(
            "graph candidate types: "
            + ", ".join(review.graph_candidate_types)
        )
    if review.short_reason:
        print(f"reason: {review.short_reason}")
    if review.error:
        print(f"error: {review.error}")


async def main(limit: int, *, dry_run: bool, verbose: bool) -> None:
    if not dry_run:
        await database.init_db()
    result = await run_once(limit=limit, dry_run=dry_run)

    print("=" * 48)
    print("  Consolidation Dry-Run Summary" if dry_run else "  Consolidation Summary")
    print("=" * 48)
    print(f"  mode:                 {'dry-run' if dry_run else 'write'}")
    print(f"  scanned_count:        {result.scanned_count}")
    print(f"  processed:            {result.processed}")
    print(f"  fact_count:           {result.fact_count}")
    print(f"  graph_count:          {result.graph_count}")
    print(f"  noop_count:           {result.noop_count}")
    print(f"  fact_added:           {result.fact_added}")
    print(f"  fact_updated:         {result.fact_updated}")
    print(f"  fact_noop:            {result.fact_noop}")
    print(f"  graph_added:          {result.graph_added}")
    print(f"  graph_noop:           {result.graph_noop}")
    print(f"  noop_marked_candidate:{result.noop_marked_candidate}")
    print(f"  error_count:          {result.error_count}")
    if result.kind_distribution:
        print(f"  kind_distribution: {result.kind_distribution}")
    if result.route_distribution:
        print(f"  route_distribution: {result.route_distribution}")
    if result.processed_ids:
        print(f"  processed_ids: {result.processed_ids}")
    if result.errors:
        print(f"  error_ids: {result.errors}")
    print("=" * 48)

    if verbose and result.reviews:
        print("  Per-Memory Review")
        print("=" * 48)
        for review in result.reviews:
            _print_review(review)
        print("-" * 48)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description="Run one consolidation pass")
    parser.add_argument("--limit", type=int, default=50, help="Max memories per pass")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run full consolidation decision flow without any DB writes",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-memory review details for manual audit",
    )
    args = parser.parse_args()
    asyncio.run(main(args.limit, dry_run=args.dry_run, verbose=args.verbose))
