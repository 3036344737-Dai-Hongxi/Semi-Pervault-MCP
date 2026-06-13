#!/usr/bin/env python3
"""Minimal retrieval regression suite.

Loads golden queries from golden_queries.py, runs each through the
retrieval pipeline, and checks PASS / FAIL / SKIP for every rule.

Usage:
    uv run python scripts/regression_retrieval.py
    uv run python scripts/regression_retrieval.py --verbose

Exit code 0 = all PASS or SKIP, exit code 1 = at least one FAIL.
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

import memory_core.database as database
from scripts.golden_queries import GOLDEN_QUERIES
from memory_core.services.retrieval import (
    _source_composition,
    detect_query_intent,
    retrieve_context,
)

CONTENT_PREVIEW_CHARS = 50

Verdict = str  # "PASS" | "FAIL" | "SKIP"


def _preview(text: str) -> str:
    if len(text) <= CONTENT_PREVIEW_CHARS:
        return text
    return text[:CONTENT_PREVIEW_CHARS] + "…"


# ── Individual check implementations ─────────────────────────


def check_intent_match(
    gq: dict[str, Any], intent: str, sources: list[dict[str, Any]], **_kw: Any
) -> tuple[Verdict, str]:
    expected = gq["expected_intent"]
    if intent == expected:
        return "PASS", f"intent={intent}"
    return "FAIL", f"expected={expected} got={intent}"


def check_fact_present(
    _gq: dict[str, Any], _intent: str, sources: list[dict[str, Any]], **_kw: Any
) -> tuple[Verdict, str]:
    if not sources:
        return "SKIP", "no sources"
    comp = _source_composition(sources)
    count = comp.get("structured_fact", 0)
    if count > 0:
        return "PASS", f"structured_fact={count}"
    return "SKIP", "no structured_fact in DB"


def check_fact_priority(
    _gq: dict[str, Any], _intent: str, sources: list[dict[str, Any]], **_kw: Any
) -> tuple[Verdict, str]:
    if not sources:
        return "SKIP", "no sources"
    comp = _source_composition(sources)
    if comp.get("structured_fact", 0) == 0:
        return "SKIP", "no structured_fact in DB"
    first_source = sources[0].get("_source", "unknown")
    if first_source == "structured_fact":
        return "PASS", "first source is structured_fact"
    return "FAIL", f"first source is {first_source}, expected structured_fact"


async def check_no_closed_tasks(
    _gq: dict[str, Any],
    _intent: str,
    sources: list[dict[str, Any]],
    *,
    db: Any = None,
    **_kw: Any,
) -> tuple[Verdict, str]:
    if not sources:
        return "SKIP", "no sources"
    if db is None:
        return "SKIP", "no db connection"

    source_ids = [s["id"] for s in sources]
    placeholders = ",".join("?" for _ in source_ids)
    cursor = await db.execute(
        f"""SELECT id, task_status
            FROM memory_items
            WHERE id IN ({placeholders})""",
        source_ids,
    )
    rows = await cursor.fetchall()

    closed_ids: list[str] = []
    for row in rows:
        status = row["task_status"]
        if status in ("done", "expired"):
            closed_ids.append(row["id"])

    if closed_ids:
        return "FAIL", f"closed tasks leaked: {closed_ids}"
    statuses = [row["task_status"] or "null" for row in rows]
    return "PASS", f"all task_status ok: {statuses}"


CHECK_REGISTRY: dict[str, Any] = {
    "intent_match": check_intent_match,
    "fact_present": check_fact_present,
    "fact_priority": check_fact_priority,
    "no_closed_tasks": check_no_closed_tasks,
}


# ── Runner ───────────────────────────────────────────────────


async def run_checks(
    gq: dict[str, Any], db: Any
) -> list[dict[str, str]]:
    query = gq["query"]
    expected_intent = gq["expected_intent"]
    intent = await detect_query_intent(query)
    sources = await retrieve_context(query, db)

    results: list[dict[str, str]] = []
    for check_name in gq["checks"]:
        fn = CHECK_REGISTRY.get(check_name)
        if fn is None:
            results.append(
                {"check": check_name, "verdict": "FAIL", "detail": "unknown check"}
            )
            continue

        if asyncio.iscoroutinefunction(fn):
            verdict, detail = await fn(
                gq, intent, sources, db=db
            )
        else:
            verdict, detail = fn(gq, intent, sources, db=db)

        results.append({"check": check_name, "verdict": verdict, "detail": detail})

    return results


def print_query_result(
    gq: dict[str, Any],
    check_results: list[dict[str, str]],
    sources: list[dict[str, Any]],
    verbose: bool = False,
) -> None:
    verdicts = [r["verdict"] for r in check_results]
    has_fail = "FAIL" in verdicts
    status_icon = "FAIL" if has_fail else "PASS"

    print(f"\n  [{status_icon}] {gq['query']}  (expect: {gq['expected_intent']})")
    for r in check_results:
        marker = {"PASS": "  ✓", "FAIL": "  ✗", "SKIP": "  ○"}[r["verdict"]]
        print(f"    {marker} {r['check']}: {r['verdict']}  {r['detail']}")

    if verbose and sources:
        comp = _source_composition(sources)
        print(f"    composition: {comp}")
        for i, s in enumerate(sources[:5], 1):
            tag = s.get("_source", "?")
            print(f"      {i}. [{tag}] {s['id'][:8]}… {_preview(s.get('content', ''))}")


async def main(verbose: bool = False) -> int:
    await database.init_db()
    db = await database.get_db()

    total_pass = 0
    total_fail = 0
    total_skip = 0

    print("=" * 60)
    print("  Retrieval Regression Suite")
    print(f"  Golden queries: {len(GOLDEN_QUERIES)}")
    print("=" * 60)

    try:
        for gq in GOLDEN_QUERIES:
            query = gq["query"]
            sources = await retrieve_context(query, db)
            check_results = await run_checks(gq, db)

            print_query_result(gq, check_results, sources, verbose=verbose)

            for r in check_results:
                if r["verdict"] == "PASS":
                    total_pass += 1
                elif r["verdict"] == "FAIL":
                    total_fail += 1
                else:
                    total_skip += 1
    finally:
        await db.close()

    print(f"\n{'=' * 60}")
    print(f"  Summary: {total_pass} PASS / {total_fail} FAIL / {total_skip} SKIP")
    print(f"{'=' * 60}")

    return 1 if total_fail > 0 else 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.WARNING, format="%(levelname)s %(name)s %(message)s"
    )
    parser = argparse.ArgumentParser(description="Run retrieval regression checks")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show source details per query"
    )
    args = parser.parse_args()
    exit_code = asyncio.run(main(verbose=args.verbose))
    sys.exit(exit_code)
