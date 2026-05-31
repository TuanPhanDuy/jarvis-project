"""Tests for prompt version history."""
from __future__ import annotations

import pytest

from jarvis.prompts.history import get_history, get_version, record_version
from jarvis.prompts.overrides import clear_all_overrides, set_override, get_override


@pytest.fixture(autouse=True)
def clean_overrides():
    clear_all_overrides()
    yield
    clear_all_overrides()


class TestRecordVersion:
    def test_returns_version_id(self, tmp_path):
        vid = record_version(tmp_path / "db", "researcher", "my prompt")
        assert isinstance(vid, str) and len(vid) > 0

    def test_persists_prompt(self, tmp_path):
        db = tmp_path / "db"
        vid = record_version(db, "researcher", "version 1")
        v = get_version(db, vid)
        assert v["prompt"] == "version 1"

    def test_stores_agent_type_lowercase(self, tmp_path):
        db = tmp_path / "db"
        vid = record_version(db, "Researcher", "prompt")
        v = get_version(db, vid)
        assert v["agent_type"] == "researcher"

    def test_multiple_versions_stored(self, tmp_path):
        db = tmp_path / "db"
        record_version(db, "researcher", "v1")
        record_version(db, "researcher", "v2")
        assert len(get_history(db, "researcher")) == 2


class TestGetHistory:
    def test_empty_returns_empty(self, tmp_path):
        assert get_history(tmp_path / "db", "researcher") == []

    def test_newest_first(self, tmp_path):
        db = tmp_path / "db"
        record_version(db, "researcher", "v1")
        record_version(db, "researcher", "v2")
        history = get_history(db, "researcher")
        assert history[0]["length_chars"] == len("v2")

    def test_excludes_full_prompt(self, tmp_path):
        db = tmp_path / "db"
        record_version(db, "researcher", "some long prompt text")
        history = get_history(db, "researcher")
        assert "prompt" not in history[0]
        assert "length_chars" in history[0]

    def test_agent_type_isolation(self, tmp_path):
        db = tmp_path / "db"
        record_version(db, "researcher", "r")
        record_version(db, "coder", "c")
        assert len(get_history(db, "researcher")) == 1
        assert len(get_history(db, "coder")) == 1

    def test_required_fields_present(self, tmp_path):
        db = tmp_path / "db"
        record_version(db, "researcher", "text")
        h = get_history(db, "researcher")[0]
        for field in ("version_id", "agent_type", "set_at", "length_chars"):
            assert field in h


class TestGetVersion:
    def test_returns_none_for_unknown(self, tmp_path):
        assert get_version(tmp_path / "db", "nonexistent") is None

    def test_returns_full_prompt(self, tmp_path):
        db = tmp_path / "db"
        vid = record_version(db, "coder", "write clean code")
        v = get_version(db, vid)
        assert v is not None
        assert v["prompt"] == "write clean code"
        assert v["version_id"] == vid


class TestIntegrationWithOverrides:
    def test_set_override_saves_previous_to_history(self, tmp_path, monkeypatch):
        """Simulate what the API does: saves current before overwriting."""
        db = tmp_path / "db"
        set_override("researcher", "first prompt")
        current = get_override("researcher")
        record_version(db, "researcher", current)
        set_override("researcher", "second prompt")
        history = get_history(db, "researcher")
        assert len(history) == 1
        vid = history[0]["version_id"]
        v = get_version(db, vid)
        assert v["prompt"] == "first prompt"
