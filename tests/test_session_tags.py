"""Tests for session tagging and full-text search."""
from __future__ import annotations

import pytest

from jarvis.memory.sessions import (
    add_tag,
    get_sessions_by_tag,
    get_tags,
    remove_tag,
    save_session,
    search_sessions,
)


def _seed(db, session_id: str, content: str = "hello world") -> None:
    save_session(db, session_id, [{"role": "user", "content": content}])


class TestAddTag:
    def test_adds_tag_to_existing_session(self, tmp_path):
        db = tmp_path / "db"
        _seed(db, "s1")
        assert add_tag(db, "s1", "research") is True

    def test_returns_false_for_nonexistent_session(self, tmp_path):
        db = tmp_path / "db"
        assert add_tag(db, "ghost", "test") is False

    def test_tag_normalised_to_lowercase(self, tmp_path):
        db = tmp_path / "db"
        _seed(db, "s1")
        add_tag(db, "s1", "RLHF")
        assert "rlhf" in get_tags(db, "s1")

    def test_duplicate_add_is_idempotent(self, tmp_path):
        db = tmp_path / "db"
        _seed(db, "s1")
        add_tag(db, "s1", "ml")
        add_tag(db, "s1", "ml")
        assert get_tags(db, "s1").count("ml") == 1


class TestRemoveTag:
    def test_removes_existing_tag(self, tmp_path):
        db = tmp_path / "db"
        _seed(db, "s1")
        add_tag(db, "s1", "ml")
        assert remove_tag(db, "s1", "ml") is True
        assert "ml" not in get_tags(db, "s1")

    def test_returns_false_for_missing_tag(self, tmp_path):
        db = tmp_path / "db"
        _seed(db, "s1")
        assert remove_tag(db, "s1", "nonexistent") is False


class TestGetTags:
    def test_empty_session_returns_empty(self, tmp_path):
        db = tmp_path / "db"
        _seed(db, "s1")
        assert get_tags(db, "s1") == []

    def test_returns_all_tags_sorted(self, tmp_path):
        db = tmp_path / "db"
        _seed(db, "s1")
        add_tag(db, "s1", "zzz")
        add_tag(db, "s1", "aaa")
        add_tag(db, "s1", "mmm")
        assert get_tags(db, "s1") == ["aaa", "mmm", "zzz"]

    def test_isolation_between_sessions(self, tmp_path):
        db = tmp_path / "db"
        _seed(db, "sA")
        _seed(db, "sB")
        add_tag(db, "sA", "alpha")
        add_tag(db, "sB", "beta")
        assert get_tags(db, "sA") == ["alpha"]
        assert get_tags(db, "sB") == ["beta"]


class TestGetSessionsByTag:
    def test_returns_sessions_with_tag(self, tmp_path):
        db = tmp_path / "db"
        _seed(db, "s1")
        _seed(db, "s2")
        _seed(db, "s3")
        add_tag(db, "s1", "ml")
        add_tag(db, "s3", "ml")
        result = get_sessions_by_tag(db, "ml")
        assert set(result) == {"s1", "s3"}

    def test_no_sessions_returns_empty(self, tmp_path):
        db = tmp_path / "db"
        assert get_sessions_by_tag(db, "notag") == []


class TestSearchSessions:
    def test_finds_content_in_messages(self, tmp_path):
        db = tmp_path / "db"
        save_session(db, "s1", [{"role": "user", "content": "transformer architecture attention"}])
        results = search_sessions(db, "transformer")
        assert any(r["session_id"] == "s1" for r in results)

    def test_empty_query_returns_empty(self, tmp_path):
        db = tmp_path / "db"
        _seed(db, "s1")
        assert search_sessions(db, "") == []
        assert search_sessions(db, "   ") == []

    def test_no_match_returns_empty(self, tmp_path):
        db = tmp_path / "db"
        save_session(db, "s1", [{"role": "user", "content": "reinforcement learning"}])
        assert search_sessions(db, "quantum computing") == []

    def test_tag_filter_applied(self, tmp_path):
        db = tmp_path / "db"
        save_session(db, "s1", [{"role": "user", "content": "RLHF reward model"}])
        save_session(db, "s2", [{"role": "user", "content": "RLHF policy gradient"}])
        add_tag(db, "s1", "important")
        results = search_sessions(db, "RLHF", tag="important")
        ids = [r["session_id"] for r in results]
        assert "s1" in ids
        assert "s2" not in ids

    def test_result_has_snippet_field(self, tmp_path):
        db = tmp_path / "db"
        save_session(db, "s1", [{"role": "user", "content": "constitutional AI from Anthropic"}])
        results = search_sessions(db, "constitutional")
        assert results and "snippet" in results[0]

    def test_multiple_sessions_ranked(self, tmp_path):
        db = tmp_path / "db"
        save_session(db, "s1", [{"role": "user", "content": "attention attention attention"}])
        save_session(db, "s2", [{"role": "user", "content": "one mention of attention"}])
        results = search_sessions(db, "attention")
        ids = [r["session_id"] for r in results]
        assert set(ids) == {"s1", "s2"}
