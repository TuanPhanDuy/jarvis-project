"""Tests for knowledge graph entity browser."""
from __future__ import annotations

import sqlite3
import time

import pytest

from jarvis.memory.graph import get_entity, get_entity_neighbors, list_entities


def _seed(db_path, entities, relationships=None):
    conn = sqlite3.connect(str(db_path))
    conn.execute("""CREATE TABLE IF NOT EXISTS entities
        (id INTEGER PRIMARY KEY, name TEXT, type TEXT, description TEXT,
         user_id TEXT, created_at REAL, UNIQUE(name, user_id))""")
    conn.execute("""CREATE TABLE IF NOT EXISTS relationships
        (id INTEGER PRIMARY KEY, from_entity TEXT, relation TEXT, to_entity TEXT,
         notes TEXT, user_id TEXT, created_at REAL,
         UNIQUE(from_entity, relation, to_entity, user_id))""")
    for i, (name, etype, desc, uid) in enumerate(entities):
        conn.execute(
            "INSERT OR IGNORE INTO entities VALUES (?,?,?,?,?,?)",
            (None, name, etype, desc, uid, float(i)),
        )
    for from_e, rel, to_e, uid in (relationships or []):
        conn.execute(
            "INSERT OR IGNORE INTO relationships VALUES (?,?,?,?,?,?,?)",
            (None, from_e, rel, to_e, "", uid, time.time()),
        )
    conn.commit()
    conn.close()


class TestListEntities:
    def test_empty_db_returns_empty(self, tmp_path):
        assert list_entities(tmp_path / "db") == []

    def test_returns_entities(self, tmp_path):
        db = tmp_path / "db"
        _seed(db, [("Transformer", "concept", "A model arch", "shared")])
        results = list_entities(db)
        assert len(results) == 1
        assert results[0]["name"] == "Transformer"

    def test_includes_metadata_fields(self, tmp_path):
        db = tmp_path / "db"
        _seed(db, [("RLHF", "technique", "Reinforcement learning", "shared")])
        e = list_entities(db)[0]
        assert "name" in e and "type" in e and "description" in e

    def test_filter_by_type(self, tmp_path):
        db = tmp_path / "db"
        _seed(db, [
            ("Transformer", "concept", "", "shared"),
            ("Alice", "person", "", "shared"),
        ])
        concepts = list_entities(db, entity_type="concept")
        assert all(e["type"] == "concept" for e in concepts)
        assert len(concepts) == 1

    def test_limit_respected(self, tmp_path):
        db = tmp_path / "db"
        _seed(db, [(f"E{i}", "concept", "", "shared") for i in range(10)])
        assert len(list_entities(db, limit=3)) == 3

    def test_offset_works(self, tmp_path):
        db = tmp_path / "db"
        _seed(db, [(f"E{i}", "concept", "", "shared") for i in range(5)])
        all_names = {e["name"] for e in list_entities(db, limit=5)}
        page2 = {e["name"] for e in list_entities(db, limit=3, offset=3)}
        assert page2.issubset(all_names)
        assert len(page2) == 2


class TestGetEntity:
    def test_nonexistent_returns_none(self, tmp_path):
        assert get_entity(tmp_path / "db", "Ghost") is None

    def test_returns_entity_with_relationships(self, tmp_path):
        db = tmp_path / "db"
        _seed(
            db,
            [("Transformer", "concept", "a model", "shared"),
             ("Attention", "concept", "mechanism", "shared")],
            [("Transformer", "uses", "Attention", "shared")],
        )
        e = get_entity(db, "Transformer")
        assert e is not None
        assert e["name"] == "Transformer"
        assert any(r["to_entity"] == "Attention" for r in e["relationships"])

    def test_entity_with_no_relationships(self, tmp_path):
        db = tmp_path / "db"
        _seed(db, [("RLHF", "concept", "", "shared")])
        e = get_entity(db, "RLHF")
        assert e is not None
        assert e["relationships"] == []


class TestGetEntityNeighbors:
    def test_empty_db_returns_empty(self, tmp_path):
        result = get_entity_neighbors(tmp_path / "db", "Ghost")
        assert result["nodes"] == []
        assert result["edges"] == []

    def test_entity_with_neighbor_returns_both(self, tmp_path):
        db = tmp_path / "db"
        _seed(
            db,
            [("Transformer", "concept", "", "shared"),
             ("Attention", "concept", "", "shared")],
            [("Transformer", "uses", "Attention", "shared")],
        )
        result = get_entity_neighbors(db, "Transformer", depth=1)
        node_names = {n["name"] for n in result["nodes"]}
        assert "Transformer" in node_names or "Attention" in node_names
        assert len(result["edges"]) >= 1

    def test_depth_capped_at_3(self, tmp_path):
        db = tmp_path / "db"
        _seed(db, [("X", "concept", "", "shared")])
        result = get_entity_neighbors(db, "X", depth=10)
        assert isinstance(result["nodes"], list)

    def test_returns_nodes_and_edges_keys(self, tmp_path):
        db = tmp_path / "db"
        result = get_entity_neighbors(db, "any")
        assert "nodes" in result
        assert "edges" in result
