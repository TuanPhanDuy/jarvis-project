"""Tests for episodic memory browser (list, get, boost, delete)."""
from __future__ import annotations

import time

import pytest

from jarvis.memory.episodic import (
    boost_importance,
    delete_episode,
    get_episode,
    list_episodes,
    log_episode,
)


def _seed(db, session_id="s1", user_id="alice", role="user", content="test content"):
    log_episode(db, session_id, role, content, user_id=user_id)


class TestListEpisodes:
    def test_empty_db_returns_empty(self, tmp_path):
        assert list_episodes(tmp_path / "db") == []

    def test_returns_episodes(self, tmp_path):
        db = tmp_path / "db"
        _seed(db)
        results = list_episodes(db)
        assert len(results) == 1

    def test_newest_first(self, tmp_path):
        db = tmp_path / "db"
        _seed(db, content="old")
        _seed(db, content="new")
        results = list_episodes(db)
        assert results[0]["content"] == "new"

    def test_filter_by_user_id(self, tmp_path):
        db = tmp_path / "db"
        _seed(db, user_id="alice")
        _seed(db, user_id="bob")
        results = list_episodes(db, user_id="alice")
        assert all(r["user_id"] == "alice" for r in results)

    def test_filter_by_session_id(self, tmp_path):
        db = tmp_path / "db"
        _seed(db, session_id="sA")
        _seed(db, session_id="sB")
        results = list_episodes(db, session_id="sA")
        assert all(r["session_id"] == "sA" for r in results)

    def test_limit_respected(self, tmp_path):
        db = tmp_path / "db"
        for i in range(10):
            _seed(db, content=f"msg {i}")
        assert len(list_episodes(db, limit=3)) == 3

    def test_includes_importance(self, tmp_path):
        db = tmp_path / "db"
        _seed(db)
        ep = list_episodes(db)[0]
        assert "importance" in ep


class TestGetEpisode:
    def test_nonexistent_returns_none(self, tmp_path):
        assert get_episode(tmp_path / "db", 999) is None

    def test_returns_episode(self, tmp_path):
        db = tmp_path / "db"
        _seed(db, content="specific content")
        ep = list_episodes(db)[0]
        full = get_episode(db, ep["id"])
        assert full is not None
        assert full["content"] == "specific content"


class TestBoostImportance:
    def test_boosts_importance(self, tmp_path):
        db = tmp_path / "db"
        _seed(db)
        ep = list_episodes(db)[0]
        original = ep["importance"]
        boost_importance(db, ep["id"], delta=0.5)
        updated = get_episode(db, ep["id"])
        assert updated["importance"] > original

    def test_boost_delta_applied(self, tmp_path):
        db = tmp_path / "db"
        _seed(db)
        ep = list_episodes(db)[0]
        original = ep["importance"]
        boost_importance(db, ep["id"], delta=0.3)
        updated = get_episode(db, ep["id"])
        assert abs(updated["importance"] - (original + 0.3)) < 0.01

    def test_boost_nonexistent_no_error(self, tmp_path):
        boost_importance(tmp_path / "db", 9999, delta=0.1)


class TestDeleteEpisode:
    def test_deletes_existing(self, tmp_path):
        db = tmp_path / "db"
        _seed(db)
        ep = list_episodes(db)[0]
        assert delete_episode(db, ep["id"]) is True
        assert list_episodes(db) == []

    def test_returns_false_for_nonexistent(self, tmp_path):
        assert delete_episode(tmp_path / "db", 9999) is False

    def test_only_deletes_specified(self, tmp_path):
        db = tmp_path / "db"
        _seed(db, content="keep")
        _seed(db, content="delete")
        eps = list_episodes(db)
        to_delete = next(e for e in eps if e["content"] == "delete")
        delete_episode(db, to_delete["id"])
        remaining = list_episodes(db)
        assert len(remaining) == 1
        assert remaining[0]["content"] == "keep"
