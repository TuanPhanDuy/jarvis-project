"""Tests for per-agent-turn audit log."""
from __future__ import annotations

import pytest

from jarvis.memory.turns import get_turn_stats, log_turn


class TestLogTurn:
    def test_log_and_retrieve(self, tmp_path):
        db = tmp_path / "db"
        log_turn(db, "sess-1", "ResearcherAgent", "qwen2.5:14b", 100, 200, ["web_search"], 1500.0)
        rows = get_turn_stats(db, session_id="sess-1")
        assert len(rows) == 1
        r = rows[0]
        assert r["session_id"] == "sess-1"
        assert r["agent_type"] == "ResearcherAgent"
        assert r["model"] == "qwen2.5:14b"
        assert r["input_tokens"] == 100
        assert r["output_tokens"] == 200
        assert r["tool_calls"] == ["web_search"]
        assert abs(r["latency_ms"] - 1500.0) < 0.01

    def test_session_filter_isolates_records(self, tmp_path):
        db = tmp_path / "db"
        log_turn(db, "sess-A", "ResearcherAgent", "model", 10, 20, [], 100.0)
        log_turn(db, "sess-B", "CoderAgent", "model", 30, 40, [], 200.0)
        rows_a = get_turn_stats(db, session_id="sess-A")
        rows_b = get_turn_stats(db, session_id="sess-B")
        assert len(rows_a) == 1 and rows_a[0]["agent_type"] == "ResearcherAgent"
        assert len(rows_b) == 1 and rows_b[0]["agent_type"] == "CoderAgent"

    def test_returns_newest_first(self, tmp_path):
        db = tmp_path / "db"
        for i in range(3):
            log_turn(db, "sess-1", f"Agent{i}", "model", i, i, [], float(i))
        rows = get_turn_stats(db)
        # Newest (highest latency/timestamp) should appear first
        latencies = [r["latency_ms"] for r in rows]
        assert latencies == sorted(latencies, reverse=True)

    def test_limit_respected(self, tmp_path):
        db = tmp_path / "db"
        for i in range(10):
            log_turn(db, "sess-1", "Agent", "model", i, i, [], float(i))
        rows = get_turn_stats(db, limit=3)
        assert len(rows) == 3

    def test_no_records_returns_empty_list(self, tmp_path):
        db = tmp_path / "db"
        assert get_turn_stats(db, session_id="nonexistent") == []

    def test_log_turn_never_raises(self, tmp_path):
        # Pass an unwritable path to verify best-effort behavior
        log_turn(tmp_path / "nonexistent" / "sub" / "db", "s", "A", "m", 0, 0, [], 0.0)
