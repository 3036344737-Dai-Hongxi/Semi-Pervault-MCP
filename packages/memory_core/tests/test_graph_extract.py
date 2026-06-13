"""Unit tests for graph_extract._validate_and_clean.

Pure synchronous function — zero I/O, zero mocking needed.
We import both the private function and the public whitelists
to keep tests data-driven against the actual constants.
"""

import pytest

from memory_core.services.graph_extract import _validate_and_clean
from memory_core.services.memory_policy import (
    ALLOWED_RELATIONS,
    CONSOLIDATION_NODE_TYPES,
    STORE_PATH_NODE_TYPES,
)

# Convenience aliases used throughout
_STORE_TYPES = set(STORE_PATH_NODE_TYPES)
_CONSOL_TYPES = set(CONSOLIDATION_NODE_TYPES)
_RELS = set(ALLOWED_RELATIONS)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clean_store(raw: dict) -> dict:
    """Call _validate_and_clean with store-path whitelists."""
    return _validate_and_clean(raw, valid_node_types=_STORE_TYPES, valid_relations=_RELS)


def _clean_consol(raw: dict) -> dict:
    """Call _validate_and_clean with consolidation whitelists."""
    return _validate_and_clean(raw, valid_node_types=_CONSOL_TYPES, valid_relations=_RELS)


# ---------------------------------------------------------------------------
# Node validation
# ---------------------------------------------------------------------------


class TestValidateNodes:
    def test_valid_node_passes_through(self):
        raw = {"nodes": [{"label": "Alice", "type": "person"}], "edges": []}
        result = _clean_store(raw)
        assert result["nodes"] == [{"label": "Alice", "type": "person"}]

    def test_invalid_node_type_dropped(self):
        # "event" is not in STORE_PATH_NODE_TYPES
        raw = {"nodes": [{"label": "Kickoff", "type": "event"}], "edges": []}
        result = _clean_store(raw)
        assert result["nodes"] == []

    def test_empty_label_dropped(self):
        raw = {"nodes": [{"label": "", "type": "person"}], "edges": []}
        result = _clean_store(raw)
        assert result["nodes"] == []

    def test_missing_label_key_dropped(self):
        raw = {"nodes": [{"type": "person"}], "edges": []}
        result = _clean_store(raw)
        assert result["nodes"] == []

    def test_whitespace_only_label_dropped(self):
        raw = {"nodes": [{"label": "   ", "type": "project"}], "edges": []}
        result = _clean_store(raw)
        assert result["nodes"] == []

    def test_label_leading_trailing_whitespace_normalized(self):
        raw = {"nodes": [{"label": "  Alice  ", "type": "person"}], "edges": []}
        result = _clean_store(raw)
        assert result["nodes"][0]["label"] == "Alice"

    def test_label_internal_whitespace_collapsed(self):
        raw = {"nodes": [{"label": "Voice  Vault", "type": "project"}], "edges": []}
        result = _clean_store(raw)
        assert result["nodes"][0]["label"] == "Voice Vault"

    def test_duplicate_type_label_deduplicated(self):
        raw = {
            "nodes": [
                {"label": "Alice", "type": "person"},
                {"label": "Alice", "type": "person"},
            ],
            "edges": [],
        }
        result = _clean_store(raw)
        assert len(result["nodes"]) == 1

    def test_same_label_different_types_both_kept(self):
        # "Alice" as person and "Alice" as project are different (type, label) keys
        raw = {
            "nodes": [
                {"label": "Alice", "type": "person"},
                {"label": "Alice", "type": "project"},
            ],
            "edges": [],
        }
        result = _clean_store(raw)
        assert len(result["nodes"]) == 2

    def test_node_type_lowercased_before_check(self):
        # type value is lowercased internally
        raw = {"nodes": [{"label": "Alice", "type": "Person"}], "edges": []}
        result = _clean_store(raw)
        assert result["nodes"] == [{"label": "Alice", "type": "person"}]

    def test_multiple_valid_nodes_all_kept(self):
        raw = {
            "nodes": [
                {"label": "Alice", "type": "person"},
                {"label": "Pervault", "type": "project"},
            ],
            "edges": [],
        }
        result = _clean_store(raw)
        assert len(result["nodes"]) == 2


# ---------------------------------------------------------------------------
# Edge validation
# ---------------------------------------------------------------------------


class TestValidateEdges:
    def _two_node_raw(self, relation: str) -> dict:
        return {
            "nodes": [
                {"label": "Alice", "type": "person"},
                {"label": "Pervault", "type": "project"},
            ],
            "edges": [
                {"source": "Alice", "target": "Pervault", "relation": relation}
            ],
        }

    def test_valid_edge_passes_through(self):
        raw = self._two_node_raw("related_to")
        result = _clean_store(raw)
        assert len(result["edges"]) == 1
        assert result["edges"][0]["relation"] == "related_to"

    def test_invalid_relation_dropped(self):
        raw = self._two_node_raw("knows")
        result = _clean_store(raw)
        assert result["edges"] == []

    def test_relation_lowercased_before_check(self):
        raw = self._two_node_raw("Related_To")
        result = _clean_store(raw)
        assert len(result["edges"]) == 1

    def test_dangling_edge_source_not_in_nodes_dropped(self):
        raw = {
            "nodes": [{"label": "Pervault", "type": "project"}],
            "edges": [
                {"source": "Ghost", "target": "Pervault", "relation": "related_to"}
            ],
        }
        result = _clean_store(raw)
        assert result["edges"] == []

    def test_dangling_edge_target_not_in_nodes_dropped(self):
        raw = {
            "nodes": [{"label": "Alice", "type": "person"}],
            "edges": [
                {"source": "Alice", "target": "Ghost", "relation": "related_to"}
            ],
        }
        result = _clean_store(raw)
        assert result["edges"] == []

    def test_self_loop_dropped(self):
        raw = {
            "nodes": [{"label": "Alice", "type": "person"}],
            "edges": [
                {"source": "Alice", "target": "Alice", "relation": "related_to"}
            ],
        }
        result = _clean_store(raw)
        assert result["edges"] == []

    def test_edge_dropped_when_its_node_was_invalid(self):
        # "event" node is invalid under store-path whitelist → node dropped →
        # edge referencing it should also be dropped
        raw = {
            "nodes": [
                {"label": "Alice", "type": "person"},
                {"label": "Kickoff", "type": "event"},
            ],
            "edges": [
                {"source": "Alice", "target": "Kickoff", "relation": "related_to"}
            ],
        }
        result = _clean_store(raw)
        assert result["nodes"] == [{"label": "Alice", "type": "person"}]
        assert result["edges"] == []


# ---------------------------------------------------------------------------
# Safe degradation / empty input
# ---------------------------------------------------------------------------


class TestSafeDegradation:
    def test_empty_nodes_and_edges(self):
        result = _clean_store({"nodes": [], "edges": []})
        assert result == {"nodes": [], "edges": []}

    def test_missing_nodes_key(self):
        result = _clean_store({"edges": []})
        assert result["nodes"] == []

    def test_missing_edges_key(self):
        result = _clean_store({"nodes": [{"label": "Alice", "type": "person"}]})
        assert result["edges"] == []

    def test_completely_empty_dict(self):
        result = _clean_store({})
        assert result == {"nodes": [], "edges": []}


# ---------------------------------------------------------------------------
# Store-path vs consolidation whitelist difference
# ---------------------------------------------------------------------------


class TestWhitelistDifference:
    def test_idea_valid_for_store_path(self):
        raw = {"nodes": [{"label": "新策略", "type": "idea"}], "edges": []}
        result = _clean_store(raw)
        assert len(result["nodes"]) == 1

    def test_idea_invalid_for_consolidation(self):
        raw = {"nodes": [{"label": "新策略", "type": "idea"}], "edges": []}
        result = _clean_consol(raw)
        assert result["nodes"] == []

    def test_event_valid_for_consolidation(self):
        raw = {"nodes": [{"label": "年度会议", "type": "event"}], "edges": []}
        result = _clean_consol(raw)
        assert len(result["nodes"]) == 1

    def test_event_invalid_for_store_path(self):
        raw = {"nodes": [{"label": "年度会议", "type": "event"}], "edges": []}
        result = _clean_store(raw)
        assert result["nodes"] == []

    def test_task_valid_for_store_path(self):
        raw = {"nodes": [{"label": "完成设计稿", "type": "task"}], "edges": []}
        result = _clean_store(raw)
        assert len(result["nodes"]) == 1

    def test_task_invalid_for_consolidation(self):
        raw = {"nodes": [{"label": "完成设计稿", "type": "task"}], "edges": []}
        result = _clean_consol(raw)
        assert result["nodes"] == []

    def test_person_valid_for_both(self):
        raw = {"nodes": [{"label": "Alice", "type": "person"}], "edges": []}
        assert len(_clean_store(raw)["nodes"]) == 1
        assert len(_clean_consol(raw)["nodes"]) == 1

    def test_project_valid_for_both(self):
        raw = {"nodes": [{"label": "Pervault", "type": "project"}], "edges": []}
        assert len(_clean_store(raw)["nodes"]) == 1
        assert len(_clean_consol(raw)["nodes"]) == 1
