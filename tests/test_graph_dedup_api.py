"""Tests for knowledge graph deduplication (graph_dedup.py logic)."""
from __future__ import annotations

import sqlite3
import time

import pytest

from jarvis.memory.graph_dedup import (
    _normalize,
    _token_overlap,
    deduplicate_entities,
    find_duplicate_pairs,
    merge_entities,
)


def _seed_entities(db_path, names: list[str], user_id: str = "shared") -> None:
    conn = sqlite3.connect(str(db_path))
    conn.execute("""CREATE TABLE IF NOT EXISTS entities
        (id TEXT, name TEXT, entity_type TEXT, description TEXT,
         user_id TEXT, created_at REAL, updated_at REAL)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS relationships
        (id TEXT, from_entity TEXT, relation TEXT, to_entity TEXT,
         user_id TEXT, created_at REAL)""")
    for i, name in enumerate(names):
        conn.execute(
            "INSERT INTO entities VALUES (?,?,?,?,?,?,?)",
            (f"id-{i}", name, "concept", "", user_id, float(i), float(i)),
        )
    conn.commit()
    conn.close()


def _seed_relationship(db_path, from_e: str, to_e: str, user_id: str = "shared") -> None:
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "INSERT INTO relationships VALUES (?,?,?,?,?,?)",
        (f"r-{from_e}-{to_e}", from_e, "relates_to", to_e, user_id, time.time()),
    )
    conn.commit()
    conn.close()


class TestNormalise:
    def test_lowercase(self):
        assert _normalize("Transformer") == "transformer"

    def test_collapses_whitespace(self):
        assert _normalize("  multi   head  ") == "multi head"


class TestTokenOverlap:
    def test_identical_strings(self):
        assert _token_overlap("attention mechanism", "attention mechanism") == 1.0

    def test_disjoint_strings(self):
        assert _token_overlap("hello world", "foo bar") == 0.0

    def test_partial_overlap(self):
        score = _token_overlap("multi head attention", "multi head")
        assert 0 < score < 1

    def test_empty_string_returns_zero(self):
        assert _token_overlap("", "anything") == 0.0


class TestFindDuplicatePairs:
    def test_no_db_returns_empty(self, tmp_path):
        assert find_duplicate_pairs(tmp_path / "db") == []

    def test_exact_duplicates_found(self, tmp_path):
        db = tmp_path / "db"
        _seed_entities(db, ["Transformer", "transformer"])
        pairs = find_duplicate_pairs(db)
        assert len(pairs) == 1

    def test_similar_names_found_at_threshold(self, tmp_path):
        db = tmp_path / "db"
        # "deep learning model" and "deep learning" share 2/3 tokens → 0.67 Jaccard
        _seed_entities(db, ["deep learning model", "deep learning"])
        pairs = find_duplicate_pairs(db, similarity_threshold=0.6)
        assert len(pairs) == 1

    def test_distinct_names_not_paired(self, tmp_path):
        db = tmp_path / "db"
        _seed_entities(db, ["Transformer", "RNN", "LSTM"])
        pairs = find_duplicate_pairs(db, similarity_threshold=0.85)
        assert pairs == []

    def test_user_id_isolation(self, tmp_path):
        db = tmp_path / "db"
        _seed_entities(db, ["Transformer", "transformer"], user_id="alice")
        _seed_entities(db, ["RLHF"], user_id="bob")
        pairs_alice = find_duplicate_pairs(db, user_id="alice")
        pairs_bob = find_duplicate_pairs(db, user_id="bob")
        assert len(pairs_alice) == 1
        assert len(pairs_bob) == 0


class TestMergeEntities:
    def test_reassigns_relationships(self, tmp_path):
        db = tmp_path / "db"
        _seed_entities(db, ["Transformer", "transformer"])
        _seed_relationship(db, "transformer", "RLHF")
        merge_entities(db, "Transformer", "transformer")
        conn = sqlite3.connect(str(db))
        rows = conn.execute("SELECT from_entity FROM relationships").fetchall()
        conn.close()
        assert all(r[0] == "Transformer" for r in rows)

    def test_removes_duplicate_entity(self, tmp_path):
        db = tmp_path / "db"
        _seed_entities(db, ["Transformer", "transformer"])
        merge_entities(db, "Transformer", "transformer")
        conn = sqlite3.connect(str(db))
        names = [r[0] for r in conn.execute("SELECT name FROM entities").fetchall()]
        conn.close()
        assert "transformer" not in names
        assert "Transformer" in names

    def test_nonexistent_db_returns_minus_one(self, tmp_path):
        result = merge_entities(tmp_path / "db", "A", "B")
        assert result == -1


class TestDeduplicateEntities:
    def test_returns_zero_when_no_duplicates(self, tmp_path):
        db = tmp_path / "db"
        _seed_entities(db, ["Transformer", "RLHF", "Attention"])
        assert deduplicate_entities(db) == 0

    def test_merges_duplicates_and_returns_count(self, tmp_path):
        db = tmp_path / "db"
        _seed_entities(db, ["Transformer", "transformer", "RLHF", "rlhf"])
        merged = deduplicate_entities(db)
        assert merged == 2

    def test_empty_db_returns_zero(self, tmp_path):
        assert deduplicate_entities(tmp_path / "db") == 0
