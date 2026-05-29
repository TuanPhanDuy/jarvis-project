"""Tests for knowledge graph entity deduplication."""
from __future__ import annotations

import pytest

from jarvis.memory.graph_dedup import (
    _token_overlap,
    _normalize,
    find_duplicate_pairs,
    merge_entities,
    deduplicate_entities,
)
from jarvis.memory.graph import _get_conn


def _seed_graph(db_path, entities, relationships=None):
    """Insert entities and optional relationships directly."""
    import time
    conn = _get_conn(db_path)
    for name, etype in entities:
        conn.execute(
            "INSERT OR IGNORE INTO entities (name, type, user_id, created_at) VALUES (?, ?, 'shared', ?)",
            (name, etype, time.time()),
        )
    for from_e, rel, to_e in (relationships or []):
        conn.execute(
            "INSERT OR IGNORE INTO relationships (from_entity, relation, to_entity, user_id, created_at) "
            "VALUES (?, ?, ?, 'shared', ?)",
            (from_e, rel, to_e, time.time()),
        )
    conn.commit()
    conn.close()


class TestHelpers:
    def test_normalize_strips_and_lowercases(self):
        assert _normalize("  RLHF  ") == "rlhf"
        assert _normalize("Large Language Model") == "large language model"

    def test_token_overlap_identical(self):
        assert _token_overlap("RLHF", "RLHF") == 1.0

    def test_token_overlap_no_overlap(self):
        assert _token_overlap("RLHF", "PPO") == 0.0

    def test_token_overlap_partial(self):
        score = _token_overlap("large language model", "large model")
        assert 0 < score < 1.0

    def test_token_overlap_empty(self):
        assert _token_overlap("", "anything") == 0.0


class TestFindDuplicatePairs:
    def test_returns_empty_for_missing_db(self, tmp_path):
        assert find_duplicate_pairs(tmp_path / "no.db") == []

    def test_exact_case_variant_detected(self, tmp_path):
        db = tmp_path / "jarvis.db"
        _seed_graph(db, [("RLHF", "technique"), ("rlhf", "concept")])
        pairs = find_duplicate_pairs(db)
        assert len(pairs) == 1
        assert set(pairs[0]) == {"RLHF", "rlhf"}

    def test_high_overlap_detected(self, tmp_path):
        db = tmp_path / "jarvis.db"
        _seed_graph(db, [
            ("reinforcement learning from human feedback", "technique"),
            ("reinforcement learning from human feedback technique", "concept"),
        ])
        # 5 tokens in common / 6 total → Jaccard ≈ 0.83
        pairs = find_duplicate_pairs(db, similarity_threshold=0.75)
        assert len(pairs) >= 1

    def test_unrelated_entities_not_paired(self, tmp_path):
        db = tmp_path / "jarvis.db"
        _seed_graph(db, [("RLHF", "technique"), ("PPO", "algorithm"), ("Anthropic", "company")])
        pairs = find_duplicate_pairs(db)
        assert pairs == []

    def test_each_entity_in_at_most_one_pair(self, tmp_path):
        """No entity should appear as a duplicate more than once."""
        db = tmp_path / "jarvis.db"
        _seed_graph(db, [("RLHF", "t"), ("rlhf", "t"), ("Rlhf", "t")])
        pairs = find_duplicate_pairs(db)
        duplicates = [p[1] for p in pairs]
        assert len(duplicates) == len(set(duplicates))


class TestMergeEntities:
    def test_relationships_reassigned_to_canonical(self, tmp_path):
        db = tmp_path / "jarvis.db"
        _seed_graph(
            db,
            [("RLHF", "t"), ("rlhf", "t")],
            [("rlhf", "uses", "PPO")],
        )
        merge_entities(db, "RLHF", "rlhf")
        conn = _get_conn(db)
        rows = conn.execute(
            "SELECT from_entity FROM relationships WHERE relation = 'uses'"
        ).fetchall()
        conn.close()
        assert rows[0][0] == "RLHF"

    def test_duplicate_entity_removed(self, tmp_path):
        db = tmp_path / "jarvis.db"
        _seed_graph(db, [("RLHF", "t"), ("rlhf", "t")])
        merge_entities(db, "RLHF", "rlhf")
        conn = _get_conn(db)
        names = [r[0] for r in conn.execute("SELECT name FROM entities WHERE user_id = 'shared'").fetchall()]
        conn.close()
        assert "rlhf" not in names
        assert "RLHF" in names

    def test_returns_negative_for_missing_db(self, tmp_path):
        assert merge_entities(tmp_path / "no.db", "A", "B") < 0


class TestDeduplicateEntities:
    def test_returns_zero_for_no_duplicates(self, tmp_path):
        db = tmp_path / "jarvis.db"
        _seed_graph(db, [("RLHF", "t"), ("PPO", "algo"), ("Anthropic", "org")])
        count = deduplicate_entities(db)
        assert count == 0

    def test_merges_case_duplicate(self, tmp_path):
        db = tmp_path / "jarvis.db"
        _seed_graph(db, [("RLHF", "t"), ("rlhf", "t")])
        merged = deduplicate_entities(db)
        assert merged == 1
        conn = _get_conn(db)
        names = [r[0] for r in conn.execute("SELECT name FROM entities").fetchall()]
        conn.close()
        assert len(names) == 1
