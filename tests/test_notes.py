"""Tests for session notes / annotations."""
from __future__ import annotations

import pytest

from jarvis.memory.notes import add_note, delete_note, list_notes


class TestAddNote:
    def test_returns_note_dict(self, tmp_path):
        db = tmp_path / "db"
        note = add_note(db, "s1", "interesting finding")
        assert note["content"] == "interesting finding"
        assert note["session_id"] == "s1"
        assert "id" in note and "created_at" in note

    def test_default_author_is_user(self, tmp_path):
        db = tmp_path / "db"
        note = add_note(db, "s1", "content")
        assert note["author"] == "user"

    def test_custom_author_stored(self, tmp_path):
        db = tmp_path / "db"
        note = add_note(db, "s1", "content", author="alice")
        assert note["author"] == "alice"

    def test_empty_content_raises(self, tmp_path):
        with pytest.raises(ValueError):
            add_note(tmp_path / "db", "s1", "")

    def test_whitespace_only_raises(self, tmp_path):
        with pytest.raises(ValueError):
            add_note(tmp_path / "db", "s1", "   ")

    def test_unique_ids(self, tmp_path):
        db = tmp_path / "db"
        n1 = add_note(db, "s1", "note 1")
        n2 = add_note(db, "s1", "note 2")
        assert n1["id"] != n2["id"]


class TestListNotes:
    def test_empty_session_returns_empty(self, tmp_path):
        assert list_notes(tmp_path / "db", "s1") == []

    def test_returns_notes_for_session(self, tmp_path):
        db = tmp_path / "db"
        add_note(db, "s1", "first note")
        add_note(db, "s1", "second note")
        notes = list_notes(db, "s1")
        assert len(notes) == 2

    def test_notes_ordered_oldest_first(self, tmp_path):
        db = tmp_path / "db"
        add_note(db, "s1", "old")
        add_note(db, "s1", "new")
        notes = list_notes(db, "s1")
        assert notes[0]["content"] == "old"
        assert notes[1]["content"] == "new"

    def test_session_isolation(self, tmp_path):
        db = tmp_path / "db"
        add_note(db, "sA", "note for A")
        add_note(db, "sB", "note for B")
        assert len(list_notes(db, "sA")) == 1
        assert list_notes(db, "sA")[0]["content"] == "note for A"

    def test_each_note_has_required_fields(self, tmp_path):
        db = tmp_path / "db"
        add_note(db, "s1", "test")
        note = list_notes(db, "s1")[0]
        for field in ("id", "session_id", "content", "author", "created_at"):
            assert field in note


class TestDeleteNote:
    def test_deletes_existing_note(self, tmp_path):
        db = tmp_path / "db"
        note = add_note(db, "s1", "to be deleted")
        assert delete_note(db, note["id"]) is True
        assert list_notes(db, "s1") == []

    def test_returns_false_for_nonexistent(self, tmp_path):
        assert delete_note(tmp_path / "db", "fake-id") is False

    def test_only_deletes_specified_note(self, tmp_path):
        db = tmp_path / "db"
        n1 = add_note(db, "s1", "keep this")
        n2 = add_note(db, "s1", "delete this")
        delete_note(db, n2["id"])
        remaining = list_notes(db, "s1")
        assert len(remaining) == 1
        assert remaining[0]["id"] == n1["id"]
