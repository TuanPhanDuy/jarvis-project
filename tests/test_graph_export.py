"""Tests for knowledge-graph export_graph()."""
from __future__ import annotations

from pathlib import Path

import pytest

from jarvis.memory.graph import export_graph, handle_update_knowledge_graph


def _seed(db_path: Path, user_id: str = "shared") -> None:
    """Insert a small test graph."""
    handle_update_knowledge_graph(
        {
            "entities": [
                {"name": "RLHF", "type": "technique", "description": "Reinforcement Learning from Human Feedback"},
                {"name": "PPO", "type": "algorithm", "description": "Proximal Policy Optimisation"},
                {"name": "Anthropic", "type": "organisation", "description": "AI safety company"},
            ],
            "relationships": [
                {"from": "RLHF", "relation": "uses", "to": "PPO"},
                {"from": "Anthropic", "relation": "researches", "to": "RLHF"},
            ],
            "user_id": user_id,
        },
        db_path,
    )


class TestExportGraph:
    def test_empty_for_missing_db(self, tmp_path: Path) -> None:
        result = export_graph(tmp_path / "missing.db")
        assert result == {"nodes": [], "edges": []}

    def test_returns_nodes_and_edges_keys(self, tmp_path: Path) -> None:
        db = tmp_path / "jarvis.db"
        _seed(db)
        result = export_graph(db)
        assert "nodes" in result
        assert "edges" in result

    def test_node_count_matches_entities(self, tmp_path: Path) -> None:
        db = tmp_path / "jarvis.db"
        _seed(db)
        result = export_graph(db)
        assert len(result["nodes"]) == 3

    def test_edge_count_matches_relationships(self, tmp_path: Path) -> None:
        db = tmp_path / "jarvis.db"
        _seed(db)
        result = export_graph(db)
        assert len(result["edges"]) == 2

    def test_node_has_required_fields(self, tmp_path: Path) -> None:
        db = tmp_path / "jarvis.db"
        _seed(db)
        nodes = export_graph(db)["nodes"]
        for node in nodes:
            assert "id" in node
            assert "type" in node
            assert "description" in node

    def test_edge_has_required_fields(self, tmp_path: Path) -> None:
        db = tmp_path / "jarvis.db"
        _seed(db)
        edges = export_graph(db)["edges"]
        for edge in edges:
            assert "source" in edge
            assert "relation" in edge
            assert "target" in edge
            assert "notes" in edge

    def test_node_ids_match_entity_names(self, tmp_path: Path) -> None:
        db = tmp_path / "jarvis.db"
        _seed(db)
        ids = {n["id"] for n in export_graph(db)["nodes"]}
        assert ids == {"RLHF", "PPO", "Anthropic"}

    def test_edges_reference_known_nodes(self, tmp_path: Path) -> None:
        db = tmp_path / "jarvis.db"
        _seed(db)
        result = export_graph(db)
        node_ids = {n["id"] for n in result["nodes"]}
        for edge in result["edges"]:
            assert edge["source"] in node_ids
            assert edge["target"] in node_ids

    def test_limit_caps_node_count(self, tmp_path: Path) -> None:
        db = tmp_path / "jarvis.db"
        _seed(db)
        result = export_graph(db, limit=2)
        assert len(result["nodes"]) <= 2

    def test_edges_excluded_when_endpoint_not_in_node_set(self, tmp_path: Path) -> None:
        db = tmp_path / "jarvis.db"
        _seed(db)
        # limit=1 → only one node; both edges reference nodes not in the set
        result = export_graph(db, limit=1)
        # All returned edges must have both endpoints in the (limited) node set
        node_ids = {n["id"] for n in result["nodes"]}
        for edge in result["edges"]:
            assert edge["source"] in node_ids
            assert edge["target"] in node_ids

    def test_user_isolation(self, tmp_path: Path) -> None:
        db = tmp_path / "jarvis.db"
        _seed(db, user_id="alice")
        # bob has no data; shared data is always included but alice-specific is not returned
        result = export_graph(db, user_id="bob")
        # alice's data (non-shared) should not appear
        node_ids = {n["id"] for n in result["nodes"]}
        # The seed used user_id="alice", not "shared", so bob should see nothing
        assert len(node_ids) == 0

    def test_shared_data_visible_to_any_user(self, tmp_path: Path) -> None:
        db = tmp_path / "jarvis.db"
        _seed(db, user_id="shared")
        result = export_graph(db, user_id="alice")
        assert len(result["nodes"]) == 3
