#!/usr/bin/env python3
"""Minimal retrieval evaluation script.

Runs a set of representative queries through the retrieval pipeline and
prints source composition for each.  Designed to be run before and after
consolidation to compare how structured_facts affect source makeup.

Usage:
    uv run python scripts/eval_retrieval.py
    uv run python scripts/eval_retrieval.py --queries "我喜欢什么" "我在做什么项目"
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
from memory_core.services.retrieval import (
    _source_composition,
    detect_query_intent,
    get_boot_context,
    retrieve_context,
)

CONTENT_PREVIEW_CHARS = 60

DEFAULT_QUERIES = [
    "我在做什么项目",
    "我喜欢什么",
    "我最近都干什么了",
]


def _preview(text: str) -> str:
    if len(text) <= CONTENT_PREVIEW_CHARS:
        return text
    return text[:CONTENT_PREVIEW_CHARS] + "..."


async def evaluate_query(query: str, db) -> dict:
    intent = await detect_query_intent(query)
    sources = await retrieve_context(query, db)

    source_ids = {s["id"] for s in sources}
    source_content_keys = set()
    for s in sources:
        if s.get("content"):
            key = s["content"].strip().lower()
            source_content_keys.add(key)

    boot_items = await get_boot_context(
        db,
        exclude_ids=source_ids,
        exclude_content_keys=source_content_keys,
    )

    return {
        "query": query,
        "intent": intent,
        "source_count": len(sources),
        "source_composition": _source_composition(sources),
        "boot_count": len(boot_items),
        "boot_composition": _source_composition(boot_items),
        "sources": [
            {
                "id": s["id"],
                "_source": s.get("_source", "unknown"),
                "content": _preview(s.get("content", "")),
            }
            for s in sources
        ],
        "boot_items": [
            {
                "id": b["id"],
                "_source": b.get("_source", "unknown"),
                "kind": b.get("kind", ""),
                "content": _preview(b.get("content", "")),
            }
            for b in boot_items
        ],
    }


def print_result(result: dict) -> None:
    print(f"\n{'=' * 60}")
    print(f"  Query:   {result['query']}")
    print(f"  Intent:  {result['intent']}")
    print(f"  Sources: {result['source_count']}  {result['source_composition']}")
    print(f"  Boot:    {result['boot_count']}  {result['boot_composition']}")
    print(f"{'-' * 60}")

    if result["sources"]:
        print("  Sources detail:")
        for i, s in enumerate(result["sources"], 1):
            print(f"    {i}. [{s['_source']}] {s['id'][:8]}… {s['content']}")
    else:
        print("  Sources detail: (none)")

    if result["boot_items"]:
        print("  Boot detail:")
        for i, b in enumerate(result["boot_items"], 1):
            print(
                f"    {i}. [{b['_source']}:{b['kind']}] {b['id'][:8]}… {b['content']}"
            )

    print(f"{'=' * 60}")


async def main(queries: list[str]) -> None:
    await database.init_db()
    db = await database.get_db()
    try:
        print(f"\nEvaluating {len(queries)} queries...")
        for query in queries:
            result = await evaluate_query(query, db)
            print_result(result)
    finally:
        await db.close()

    print("\nDone.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.WARNING, format="%(levelname)s %(name)s %(message)s"
    )
    parser = argparse.ArgumentParser(description="Evaluate retrieval source composition")
    parser.add_argument(
        "--queries",
        nargs="+",
        default=DEFAULT_QUERIES,
        help="Queries to evaluate (default: 3 representative queries)",
    )
    args = parser.parse_args()
    asyncio.run(main(args.queries))
