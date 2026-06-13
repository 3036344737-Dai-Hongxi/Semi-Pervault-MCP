"""Tests for the memory_items.kind backwrite that happens during consolidation.

These tests verify that _apply_memory_decision() issues (or skips) an
UPDATE memory_items SET kind = ? call under the right conditions.

All external I/O is mocked:
  - DB connection is replaced with an AsyncMock so we can inspect calls.
  - _plan_structured_fact, _apply_structured_fact_plan, _mark_memory_consolidated
    are patched to eliminate real DB round-trips.

Tests are grouped by the condition that should gate the UPDATE:
  - Route is not "fact"          → no UPDATE
  - dry_run=True                 → no UPDATE
  - new_kind == original_kind    → no UPDATE (already correct)
  - new_kind == "task"           → no UPDATE (task never backwritten)
  - new_kind is illegal          → no UPDATE (not in _BASE_FACT_KINDS)
  - All conditions pass          → UPDATE is issued with correct args
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from memory_core.services.consolidation import (
    ConsolidationDecision,
    ConsolidationResult,
    FactPlan,
    GraphPlan,
    _apply_memory_decision,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_row(memory_id: str, kind: str | None) -> dict:
    """Create a minimal dict that matches the row interface used by _apply_memory_decision."""

    class Row(dict):
        pass

    return Row(id=memory_id, kind=kind)


def _make_fact_decision(fact_kind: str) -> ConsolidationDecision:
    return ConsolidationDecision(
        route="fact",
        reason="test",
        fact={
            "kind": fact_kind,
            "subject": "user",
            "predicate": "likes",
            "object": "coffee",
        },
    )


def _make_db_mock() -> AsyncMock:
    """Build a minimal async DB mock that records execute() calls."""
    db = AsyncMock()
    cursor = AsyncMock()
    cursor.rowcount = 1
    db.execute = AsyncMock(return_value=cursor)
    db.commit = AsyncMock()
    db.rollback = AsyncMock()
    return db


def _kind_update_calls(db: AsyncMock) -> list:
    """Return all db.execute calls that look like the kind backwrite SQL."""
    return [
        c
        for c in db.execute.call_args_list
        if "UPDATE memory_items SET kind" in str(c)
    ]


# ---------------------------------------------------------------------------
# Backwrite happens
# ---------------------------------------------------------------------------


class TestKindBackwriteOccurs:
    async def test_other_to_preference(self):
        """other → preference: UPDATE must be issued."""
        row = _make_row("mem-1", "other")
        decision = _make_fact_decision("preference")
        result = ConsolidationResult()
        db = _make_db_mock()

        with (
            patch(
                "memory_core.services.consolidation._plan_structured_fact",
                new=AsyncMock(
                    return_value=FactPlan(action="add", match_key="preference|user|likes")
                ),
            ),
            patch(
                "memory_core.services.consolidation._apply_structured_fact_plan",
                new=AsyncMock(return_value="add"),
            ),
            patch(
                "memory_core.services.consolidation._mark_memory_consolidated",
                new=AsyncMock(return_value=None),
            ),
        ):
            await _apply_memory_decision(row, decision, None, result, db, dry_run=False)

        calls = _kind_update_calls(db)
        assert len(calls) == 1, "Expected exactly one kind UPDATE call"
        args = calls[0].args
        assert args[1][0] == "preference", "UPDATE must set kind='preference'"
        assert args[1][1] == "mem-1", "UPDATE must target memory mem-1"

    async def test_other_to_project_update(self):
        row = _make_row("mem-2", "other")
        decision = _make_fact_decision("project_update")
        result = ConsolidationResult()
        db = _make_db_mock()

        with (
            patch(
                "memory_core.services.consolidation._plan_structured_fact",
                new=AsyncMock(
                    return_value=FactPlan(action="add", match_key="project_update|user|likes")
                ),
            ),
            patch(
                "memory_core.services.consolidation._apply_structured_fact_plan",
                new=AsyncMock(return_value="add"),
            ),
            patch(
                "memory_core.services.consolidation._mark_memory_consolidated",
                new=AsyncMock(return_value=None),
            ),
        ):
            await _apply_memory_decision(row, decision, None, result, db, dry_run=False)

        calls = _kind_update_calls(db)
        assert len(calls) == 1
        assert calls[0].args[1][0] == "project_update"

    async def test_other_to_fact(self):
        row = _make_row("mem-3", "other")
        decision = _make_fact_decision("fact")
        result = ConsolidationResult()
        db = _make_db_mock()

        with (
            patch(
                "memory_core.services.consolidation._plan_structured_fact",
                new=AsyncMock(
                    return_value=FactPlan(action="add", match_key="fact|user|likes")
                ),
            ),
            patch(
                "memory_core.services.consolidation._apply_structured_fact_plan",
                new=AsyncMock(return_value="add"),
            ),
            patch(
                "memory_core.services.consolidation._mark_memory_consolidated",
                new=AsyncMock(return_value=None),
            ),
        ):
            await _apply_memory_decision(row, decision, None, result, db, dry_run=False)

        calls = _kind_update_calls(db)
        assert len(calls) == 1
        assert calls[0].args[1][0] == "fact"

    async def test_other_to_relationship_event(self):
        row = _make_row("mem-4", "other")
        decision = _make_fact_decision("relationship_event")
        result = ConsolidationResult()
        db = _make_db_mock()

        with (
            patch(
                "memory_core.services.consolidation._plan_structured_fact",
                new=AsyncMock(
                    return_value=FactPlan(
                        action="add", match_key="relationship_event|user|likes"
                    )
                ),
            ),
            patch(
                "memory_core.services.consolidation._apply_structured_fact_plan",
                new=AsyncMock(return_value="add"),
            ),
            patch(
                "memory_core.services.consolidation._mark_memory_consolidated",
                new=AsyncMock(return_value=None),
            ),
        ):
            await _apply_memory_decision(row, decision, None, result, db, dry_run=False)

        calls = _kind_update_calls(db)
        assert len(calls) == 1
        assert calls[0].args[1][0] == "relationship_event"

    async def test_keyword_misclassification_corrected(self):
        """task → preference: keyword said task but LLM said preference → backwrite."""
        row = _make_row("mem-5", "task")
        decision = _make_fact_decision("preference")
        result = ConsolidationResult()
        db = _make_db_mock()

        with (
            patch(
                "memory_core.services.consolidation._plan_structured_fact",
                new=AsyncMock(
                    return_value=FactPlan(action="add", match_key="preference|user|likes")
                ),
            ),
            patch(
                "memory_core.services.consolidation._apply_structured_fact_plan",
                new=AsyncMock(return_value="add"),
            ),
            patch(
                "memory_core.services.consolidation._mark_memory_consolidated",
                new=AsyncMock(return_value=None),
            ),
        ):
            await _apply_memory_decision(row, decision, None, result, db, dry_run=False)

        calls = _kind_update_calls(db)
        assert len(calls) == 1
        assert calls[0].args[1][0] == "preference"


# ---------------------------------------------------------------------------
# Backwrite must NOT happen
# ---------------------------------------------------------------------------


class TestKindBackwriteSkipped:
    async def test_same_kind_no_update(self):
        """original_kind == fact.kind → no UPDATE."""
        row = _make_row("mem-10", "preference")
        decision = _make_fact_decision("preference")
        result = ConsolidationResult()
        db = _make_db_mock()

        with (
            patch(
                "memory_core.services.consolidation._plan_structured_fact",
                new=AsyncMock(
                    return_value=FactPlan(action="noop", match_key="preference|user|likes")
                ),
            ),
            patch(
                "memory_core.services.consolidation._apply_structured_fact_plan",
                new=AsyncMock(return_value="noop"),
            ),
            patch(
                "memory_core.services.consolidation._mark_memory_consolidated",
                new=AsyncMock(return_value=None),
            ),
        ):
            await _apply_memory_decision(row, decision, None, result, db, dry_run=False)

        assert _kind_update_calls(db) == [], "No UPDATE when kinds are equal"

    async def test_new_kind_task_no_update(self):
        """fact.kind == 'task' → never backwritten."""
        row = _make_row("mem-11", "other")
        decision = ConsolidationDecision(
            route="fact",
            reason="test",
            fact={
                "kind": "task",
                "subject": "user",
                "predicate": "plans",
                "object": "run",
            },
        )
        result = ConsolidationResult()
        db = _make_db_mock()

        with (
            patch(
                "memory_core.services.consolidation._plan_structured_fact",
                new=AsyncMock(
                    return_value=FactPlan(action="add", match_key="task|user|plans")
                ),
            ),
            patch(
                "memory_core.services.consolidation._apply_structured_fact_plan",
                new=AsyncMock(return_value="add"),
            ),
            patch(
                "memory_core.services.consolidation._mark_memory_consolidated",
                new=AsyncMock(return_value=None),
            ),
        ):
            await _apply_memory_decision(row, decision, None, result, db, dry_run=False)

        assert _kind_update_calls(db) == [], "task kind must never be backwritten"

    async def test_dry_run_no_update(self):
        """dry_run=True → no UPDATE even when kinds differ."""
        row = _make_row("mem-12", "other")
        decision = _make_fact_decision("preference")
        result = ConsolidationResult()
        db = _make_db_mock()

        with (
            patch(
                "memory_core.services.consolidation._plan_structured_fact",
                new=AsyncMock(
                    return_value=FactPlan(action="add", match_key="preference|user|likes")
                ),
            ),
            patch(
                "memory_core.services.consolidation._apply_structured_fact_plan",
                new=AsyncMock(return_value="add"),
            ),
            patch(
                "memory_core.services.consolidation._mark_memory_consolidated",
                new=AsyncMock(return_value=None),
            ),
        ):
            await _apply_memory_decision(row, decision, None, result, db, dry_run=True)

        assert _kind_update_calls(db) == [], "dry_run must not write to DB"

    async def test_route_graph_no_update(self):
        """route=graph → no memory_items.kind UPDATE."""
        row = _make_row("mem-13", "other")
        decision = ConsolidationDecision(route="graph", reason="test")
        graph_plan = GraphPlan(action="noop")
        result = ConsolidationResult()
        db = _make_db_mock()

        with (
            patch(
                "memory_core.services.consolidation._apply_graph_plan",
                new=AsyncMock(return_value="noop"),
            ),
            patch(
                "memory_core.services.consolidation._mark_memory_consolidated",
                new=AsyncMock(return_value=None),
            ),
        ):
            await _apply_memory_decision(
                row, decision, graph_plan, result, db, dry_run=False
            )

        assert _kind_update_calls(db) == [], "graph route must not touch memory kind"

    async def test_route_noop_no_update(self):
        """route=noop → no memory_items.kind UPDATE."""
        row = _make_row("mem-14", "other")
        decision = ConsolidationDecision(route="noop", reason="test")
        result = ConsolidationResult()
        db = _make_db_mock()

        with patch(
            "memory_core.services.consolidation._mark_memory_consolidated",
            new=AsyncMock(return_value=None),
        ):
            await _apply_memory_decision(row, decision, None, result, db, dry_run=False)

        assert _kind_update_calls(db) == [], "noop route must not touch memory kind"

    async def test_illegal_new_kind_no_update(self):
        """fact.kind is an illegal value (not in _BASE_FACT_KINDS) → no UPDATE."""
        row = _make_row("mem-15", "other")
        decision = ConsolidationDecision(
            route="fact",
            reason="test",
            fact={
                "kind": "unknown_hallucination",
                "subject": "user",
                "predicate": "x",
                "object": "y",
            },
        )
        result = ConsolidationResult()
        db = _make_db_mock()

        with (
            patch(
                "memory_core.services.consolidation._plan_structured_fact",
                new=AsyncMock(
                    return_value=FactPlan(action="noop", match_key="unknown|user|x")
                ),
            ),
            patch(
                "memory_core.services.consolidation._apply_structured_fact_plan",
                new=AsyncMock(return_value="noop"),
            ),
            patch(
                "memory_core.services.consolidation._mark_memory_consolidated",
                new=AsyncMock(return_value=None),
            ),
        ):
            await _apply_memory_decision(row, decision, None, result, db, dry_run=False)

        assert _kind_update_calls(db) == [], "illegal kind must not be written"

    async def test_new_kind_other_no_update(self):
        """fact.kind='other' is not in _BASE_FACT_KINDS → no UPDATE."""
        row = _make_row("mem-16", "preference")
        decision = ConsolidationDecision(
            route="fact",
            reason="test",
            fact={
                "kind": "other",
                "subject": "user",
                "predicate": "x",
                "object": "y",
            },
        )
        result = ConsolidationResult()
        db = _make_db_mock()

        with (
            patch(
                "memory_core.services.consolidation._plan_structured_fact",
                new=AsyncMock(
                    return_value=FactPlan(action="noop", match_key="other|user|x")
                ),
            ),
            patch(
                "memory_core.services.consolidation._apply_structured_fact_plan",
                new=AsyncMock(return_value="noop"),
            ),
            patch(
                "memory_core.services.consolidation._mark_memory_consolidated",
                new=AsyncMock(return_value=None),
            ),
        ):
            await _apply_memory_decision(row, decision, None, result, db, dry_run=False)

        assert _kind_update_calls(db) == [], "'other' must not be written back"

    async def test_none_kind_treated_as_other_gets_updated(self):
        """row.kind=None is coerced to 'other' → update still applies if new_kind is valid."""
        row = _make_row("mem-17", None)  # None coerced to "other" inside function
        decision = _make_fact_decision("preference")
        result = ConsolidationResult()
        db = _make_db_mock()

        with (
            patch(
                "memory_core.services.consolidation._plan_structured_fact",
                new=AsyncMock(
                    return_value=FactPlan(action="add", match_key="preference|user|likes")
                ),
            ),
            patch(
                "memory_core.services.consolidation._apply_structured_fact_plan",
                new=AsyncMock(return_value="add"),
            ),
            patch(
                "memory_core.services.consolidation._mark_memory_consolidated",
                new=AsyncMock(return_value=None),
            ),
        ):
            await _apply_memory_decision(row, decision, None, result, db, dry_run=False)

        calls = _kind_update_calls(db)
        assert len(calls) == 1, "None kind → coerced to 'other' → should still update"
        assert calls[0].args[1][0] == "preference"
