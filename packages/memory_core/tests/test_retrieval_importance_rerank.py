from datetime import datetime, timezone

import pytest

from memory_core.services.retrieval_primitives import _generative_score, _recency_score


def test_generative_score_prefers_higher_importance_when_relevance_is_equal():
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    low_importance = {
        "created_at": "2026-01-01T00:00:00+00:00",
        "importance": 2.0,
        "final_score": 0.7,
    }
    high_importance = {
        "created_at": "2026-01-01T00:00:00+00:00",
        "importance": 9.0,
        "final_score": 0.7,
    }

    assert _generative_score(high_importance, now=now) > _generative_score(
        low_importance, now=now
    )


def test_generative_score_prefers_recent_memory_when_importance_and_relevance_match():
    now = datetime(2026, 1, 31, tzinfo=timezone.utc)
    old_memory = {
        "created_at": "2025-12-02T00:00:00+00:00",
        "importance": 5.0,
        "final_score": 0.7,
    }
    recent_memory = {
        "created_at": "2026-01-31T00:00:00+00:00",
        "importance": 5.0,
        "final_score": 0.7,
    }

    assert _generative_score(recent_memory, now=now) > _generative_score(
        old_memory, now=now
    )


def test_recency_score_accepts_sqlite_datetime_format():
    now = datetime(2026, 1, 31, tzinfo=timezone.utc)

    assert _recency_score("2026-01-31 00:00:00", now=now) == pytest.approx(1.0)
