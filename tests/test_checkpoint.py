"""Tests for agent conversation checkpointing."""
from __future__ import annotations

import pytest

from jarvis.agents.checkpoint import (
    delete_checkpoints,
    list_checkpoints,
    load_checkpoint,
    save_checkpoint,
)


class TestSaveCheckpoint:
    def test_returns_nonempty_id(self, tmp_path):
        cp_id = save_checkpoint(tmp_path / "db", "s1", 3, "ResearcherAgent", [])
        assert isinstance(cp_id, str) and len(cp_id) > 0

    def test_persists_messages(self, tmp_path):
        msgs = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]
        cp_id = save_checkpoint(tmp_path / "db", "s1", 1, "ResearcherAgent", msgs)
        cp = load_checkpoint(tmp_path / "db", cp_id)
        assert cp["messages"] == msgs

    def test_persists_turn_count(self, tmp_path):
        cp_id = save_checkpoint(tmp_path / "db", "s1", 7, "CoderAgent", [])
        cp = load_checkpoint(tmp_path / "db", cp_id)
        assert cp["turn_count"] == 7

    def test_persists_agent_type(self, tmp_path):
        cp_id = save_checkpoint(tmp_path / "db", "s1", 1, "PlannerAgent", [])
        cp = load_checkpoint(tmp_path / "db", cp_id)
        assert cp["agent_type"] == "PlannerAgent"

    def test_never_raises_on_bad_path(self, tmp_path):
        result = save_checkpoint(tmp_path / "no" / "sub" / "db", "s", 0, "A", [])
        assert isinstance(result, str)


class TestListCheckpoints:
    def test_empty_session_returns_empty(self, tmp_path):
        assert list_checkpoints(tmp_path / "db", "no-session") == []

    def test_returns_oldest_first(self, tmp_path):
        db = tmp_path / "db"
        save_checkpoint(db, "s1", 5, "A", [])
        save_checkpoint(db, "s1", 2, "A", [])
        save_checkpoint(db, "s1", 8, "A", [])
        cps = list_checkpoints(db, "s1")
        counts = [c["turn_count"] for c in cps]
        assert counts == sorted(counts)

    def test_session_isolation(self, tmp_path):
        db = tmp_path / "db"
        save_checkpoint(db, "sA", 1, "A", [])
        save_checkpoint(db, "sB", 2, "A", [])
        assert len(list_checkpoints(db, "sA")) == 1
        assert len(list_checkpoints(db, "sB")) == 1

    def test_metadata_fields_present(self, tmp_path):
        db = tmp_path / "db"
        save_checkpoint(db, "s1", 3, "ResearcherAgent", [])
        cp = list_checkpoints(db, "s1")[0]
        assert "id" in cp and "session_id" in cp and "turn_count" in cp
        assert "agent_type" in cp and "created_at" in cp


class TestLoadCheckpoint:
    def test_unknown_id_returns_none(self, tmp_path):
        assert load_checkpoint(tmp_path / "db", "nonexistent-id") is None

    def test_returns_full_dict(self, tmp_path):
        db = tmp_path / "db"
        msgs = [{"role": "user", "content": "q"}]
        cp_id = save_checkpoint(db, "s1", 4, "CoderAgent", msgs)
        cp = load_checkpoint(db, cp_id)
        assert cp is not None
        assert cp["id"] == cp_id
        assert cp["session_id"] == "s1"
        assert cp["messages"] == msgs


class TestDeleteCheckpoints:
    def test_deletes_all_for_session(self, tmp_path):
        db = tmp_path / "db"
        save_checkpoint(db, "s1", 1, "A", [])
        save_checkpoint(db, "s1", 2, "A", [])
        deleted = delete_checkpoints(db, "s1")
        assert deleted == 2
        assert list_checkpoints(db, "s1") == []

    def test_isolates_other_sessions(self, tmp_path):
        db = tmp_path / "db"
        save_checkpoint(db, "sA", 1, "A", [])
        save_checkpoint(db, "sB", 1, "A", [])
        delete_checkpoints(db, "sA")
        assert list_checkpoints(db, "sB") == [dict(list_checkpoints(db, "sB")[0])]

    def test_nonexistent_session_returns_zero(self, tmp_path):
        assert delete_checkpoints(tmp_path / "db", "ghost") == 0
