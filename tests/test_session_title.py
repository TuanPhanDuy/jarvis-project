"""Tests for session auto-title generation and metadata."""
from __future__ import annotations

import pytest

from jarvis.memory.sessions import generate_title, get_metadata, save_session, set_title


class TestGenerateTitle:
    def test_basic_message(self):
        msgs = [{"role": "user", "content": "What is reinforcement learning from human feedback?"}]
        title = generate_title(msgs)
        assert "reinforcement" in title.lower() or "what" in title.lower()

    def test_strips_filler_prefixes(self):
        msgs = [{"role": "user", "content": "Please explain transformer architecture"}]
        title = generate_title(msgs)
        assert "please" not in title.lower()
        assert "explain" in title.lower() or "transformer" in title.lower()

    def test_strips_jarvis_prefix(self):
        msgs = [{"role": "user", "content": "Jarvis, research RLHF for me"}]
        title = generate_title(msgs)
        assert "jarvis" not in title.lower()
        assert "rlhf" in title.lower() or "research" in title.lower()

    def test_truncates_to_ten_words(self):
        long_msg = "Tell me about " + " ".join([f"word{i}" for i in range(20)])
        msgs = [{"role": "user", "content": long_msg}]
        title = generate_title(msgs)
        # Ellipsis at end indicates truncation; at most 10 words + "…"
        words = title.rstrip("…").split()
        assert len(words) <= 10

    def test_adds_ellipsis_when_truncated(self):
        long_msg = " ".join([f"word{i}" for i in range(15)])
        msgs = [{"role": "user", "content": long_msg}]
        title = generate_title(msgs)
        assert title.endswith("…")

    def test_no_ellipsis_for_short_message(self):
        msgs = [{"role": "user", "content": "What is RLHF?"}]
        title = generate_title(msgs)
        assert not title.endswith("…")

    def test_empty_messages_returns_default(self):
        assert generate_title([]) == "Untitled session"

    def test_no_user_message_returns_default(self):
        msgs = [{"role": "assistant", "content": "Hello!"}]
        assert generate_title(msgs) == "Untitled session"

    def test_skips_system_messages(self):
        msgs = [
            {"role": "system", "content": "You are JARVIS."},
            {"role": "user", "content": "Explain attention mechanisms"},
        ]
        title = generate_title(msgs)
        assert "jarvis" not in title.lower()
        assert "attention" in title.lower() or "explain" in title.lower()

    def test_result_is_capitalised(self):
        msgs = [{"role": "user", "content": "what is attention?"}]
        title = generate_title(msgs)
        assert title[0].isupper()


class TestSetTitle:
    def test_sets_title_on_persisted_session(self, tmp_path):
        db = tmp_path / "db"
        save_session(db, "s1", [{"role": "user", "content": "hello"}])
        result = set_title(db, "s1", "My research session")
        assert result is True
        meta = get_metadata(db, "s1")
        assert meta["title"] == "My research session"

    def test_returns_false_for_nonexistent_session(self, tmp_path):
        db = tmp_path / "db"
        assert set_title(db, "ghost", "title") is False

    def test_strips_whitespace_from_title(self, tmp_path):
        db = tmp_path / "db"
        save_session(db, "s1", [])
        set_title(db, "s1", "  padded title  ")
        meta = get_metadata(db, "s1")
        assert meta["title"] == "padded title"


class TestGetMetadata:
    def test_returns_none_for_nonexistent_session(self, tmp_path):
        db = tmp_path / "db"
        assert get_metadata(db, "ghost") is None

    def test_returns_required_fields(self, tmp_path):
        db = tmp_path / "db"
        save_session(db, "s1", [], agent_type="ResearcherAgent", user_id="user-42")
        meta = get_metadata(db, "s1")
        assert meta is not None
        for key in ("session_id", "agent_type", "user_id", "fork_of", "title", "tags",
                    "created_at", "updated_at"):
            assert key in meta

    def test_title_starts_as_none(self, tmp_path):
        db = tmp_path / "db"
        save_session(db, "s1", [])
        assert get_metadata(db, "s1")["title"] is None

    def test_tags_included(self, tmp_path):
        from jarvis.memory.sessions import add_tag
        db = tmp_path / "db"
        save_session(db, "s1", [])
        add_tag(db, "s1", "ml")
        meta = get_metadata(db, "s1")
        assert "ml" in meta["tags"]

    def test_does_not_include_messages(self, tmp_path):
        db = tmp_path / "db"
        save_session(db, "s1", [{"role": "user", "content": "secret"}])
        meta = get_metadata(db, "s1")
        assert "messages" not in meta
