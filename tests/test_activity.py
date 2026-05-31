"""Tests for session activity timeline."""
from __future__ import annotations

import sqlite3
import time

import pytest

from jarvis.memory.activity import get_activity


def _seed_audit(db_path, session_id, tool_name, ok=1):
    conn = sqlite3.connect(str(db_path))
    conn.execute("""CREATE TABLE IF NOT EXISTS audit_log
        (id TEXT, session_id TEXT, tool_name TEXT, result_ok INT, duration_ms REAL, timestamp REAL)""")
    conn.execute("INSERT INTO audit_log VALUES (?,?,?,?,?,?)",
                 ("a1", session_id, tool_name, ok, 150.0, time.time()))
    conn.commit()
    conn.close()


def _seed_checkpoint(db_path, session_id):
    conn = sqlite3.connect(str(db_path))
    conn.execute("""CREATE TABLE IF NOT EXISTS agent_checkpoints
        (id TEXT, session_id TEXT, turn_count INT, agent_type TEXT, messages_json TEXT, created_at REAL)""")
    conn.execute("INSERT INTO agent_checkpoints VALUES (?,?,?,?,?,?)",
                 ("cp1", session_id, 3, "ResearcherAgent", "[]", time.time()))
    conn.commit()
    conn.close()


def _seed_note(db_path, session_id):
    conn = sqlite3.connect(str(db_path))
    conn.execute("""CREATE TABLE IF NOT EXISTS session_notes
        (id TEXT, session_id TEXT, content TEXT, author TEXT, created_at REAL)""")
    conn.execute("INSERT INTO session_notes VALUES (?,?,?,?,?)",
                 ("n1", session_id, "important finding", "alice", time.time()))
    conn.commit()
    conn.close()


def _seed_tag(db_path, session_id, tag="ml"):
    conn = sqlite3.connect(str(db_path))
    conn.execute("""CREATE TABLE IF NOT EXISTS session_tags
        (session_id TEXT, tag TEXT, created_at REAL, PRIMARY KEY (session_id, tag))""")
    conn.execute("INSERT OR IGNORE INTO session_tags VALUES (?,?,?)",
                 (session_id, tag, time.time()))
    conn.commit()
    conn.close()


class TestGetActivity:
    def test_empty_session_returns_empty(self, tmp_path):
        result = get_activity(tmp_path / "db", "s1")
        assert result == []

    def test_messages_included(self, tmp_path):
        db = tmp_path / "db"
        messages = [
            {"role": "user", "content": "What is RLHF?"},
            {"role": "assistant", "content": "RLHF stands for..."},
        ]
        result = get_activity(db, "s1", messages=messages)
        types = [e["type"] for e in result]
        assert types.count("message") == 2

    def test_tool_calls_included(self, tmp_path):
        db = tmp_path / "db"
        _seed_audit(db, "s1", "web_search")
        result = get_activity(db, "s1")
        assert any(e["type"] == "tool_call" for e in result)

    def test_checkpoint_included(self, tmp_path):
        db = tmp_path / "db"
        _seed_checkpoint(db, "s1")
        result = get_activity(db, "s1")
        assert any(e["type"] == "checkpoint" for e in result)

    def test_note_included(self, tmp_path):
        db = tmp_path / "db"
        _seed_note(db, "s1")
        result = get_activity(db, "s1")
        assert any(e["type"] == "note" for e in result)

    def test_tag_included(self, tmp_path):
        db = tmp_path / "db"
        _seed_tag(db, "s1", "research")
        result = get_activity(db, "s1")
        assert any(e["type"] == "tag" for e in result)

    def test_all_events_merged(self, tmp_path):
        db = tmp_path / "db"
        _seed_audit(db, "s1", "web_search")
        _seed_checkpoint(db, "s1")
        _seed_note(db, "s1")
        _seed_tag(db, "s1")
        messages = [{"role": "user", "content": "hi"}]
        result = get_activity(db, "s1", messages=messages)
        event_types = {e["type"] for e in result}
        assert {"message", "tool_call", "checkpoint", "note", "tag"} == event_types

    def test_each_event_has_required_fields(self, tmp_path):
        db = tmp_path / "db"
        _seed_audit(db, "s1", "web_search")
        result = get_activity(db, "s1")
        for event in result:
            assert "type" in event
            assert "summary" in event
            assert "detail" in event

    def test_session_isolation(self, tmp_path):
        db = tmp_path / "db"
        _seed_audit(db, "sA", "web_search")
        _seed_audit(db, "sB", "read_url")
        result_a = get_activity(db, "sA")
        result_b = get_activity(db, "sB")
        assert all("web_search" in e["summary"] for e in result_a if e["type"] == "tool_call")
        assert all("read_url" in e["summary"] for e in result_b if e["type"] == "tool_call")

    def test_timestamps_sorted_ascending(self, tmp_path):
        db = tmp_path / "db"
        _seed_audit(db, "s1", "tool_a")
        _seed_checkpoint(db, "s1")
        result = get_activity(db, "s1")
        tss = [e["timestamp"] for e in result if e["timestamp"] is not None]
        assert tss == sorted(tss)

    def test_message_error_tool_call_detail(self, tmp_path):
        db = tmp_path / "db"
        _seed_audit(db, "s1", "web_search", ok=0)
        result = get_activity(db, "s1")
        tc = next(e for e in result if e["type"] == "tool_call")
        assert tc["detail"]["result_ok"] is False
        assert "error" in tc["summary"]
