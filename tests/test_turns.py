"""Tests for per-agent-turn audit log."""
from __future__ import annotations

import pytest

from jarvis.memory.turns import get_session_cost, get_turn_stats, log_turn


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


class TestGetSessionCost:
    def test_empty_session_returns_zero_totals(self, tmp_path):
        db = tmp_path / "db"
        result = get_session_cost(db, "no-such-session")
        assert result["session_id"] == "no-such-session"
        assert result["turns"] == []
        assert result["totals"]["turn_count"] == 0
        assert result["totals"]["cost_usd"] == 0.0

    def test_single_turn_sonnet_cost(self, tmp_path):
        db = tmp_path / "db"
        log_turn(db, "s1", "ResearcherAgent", "claude-sonnet-4-6", 1000, 500, [], 100.0)
        result = get_session_cost(db, "s1")
        assert len(result["turns"]) == 1
        t = result["turns"][0]
        assert t["input_tokens"] == 1000
        assert t["output_tokens"] == 500
        assert t["cost_usd"] > 0
        # Sonnet: 1000 * 3/1M + 500 * 15/1M = 0.003 + 0.0075 = 0.0105
        assert abs(t["cost_usd"] - 0.0105) < 1e-6

    def test_totals_aggregate_multiple_turns(self, tmp_path):
        db = tmp_path / "db"
        log_turn(db, "s2", "ResearcherAgent", "claude-haiku-4-5", 1000, 500, [], 50.0)
        log_turn(db, "s2", "ResearcherAgent", "claude-haiku-4-5", 2000, 1000, [], 80.0)
        result = get_session_cost(db, "s2")
        assert result["totals"]["turn_count"] == 2
        assert result["totals"]["input_tokens"] == 3000
        assert result["totals"]["output_tokens"] == 1500
        assert result["totals"]["total_tokens"] == 4500
        assert result["totals"]["cost_usd"] > 0

    def test_unknown_model_falls_back_to_sonnet_pricing(self, tmp_path):
        db = tmp_path / "db"
        log_turn(db, "s3", "CoderAgent", "llama3.2", 1_000_000, 0, [], 100.0)
        result = get_session_cost(db, "s3")
        # Should not raise, should return a cost
        assert result["totals"]["cost_usd"] == 3.0  # 1M * $3/1M

    def test_session_isolation(self, tmp_path):
        db = tmp_path / "db"
        log_turn(db, "sA", "Agent", "claude-sonnet-4-6", 100, 100, [], 10.0)
        log_turn(db, "sB", "Agent", "claude-sonnet-4-6", 200, 200, [], 20.0)
        cost_a = get_session_cost(db, "sA")
        cost_b = get_session_cost(db, "sB")
        assert cost_a["totals"]["turn_count"] == 1
        assert cost_b["totals"]["turn_count"] == 1
        assert cost_a["totals"]["input_tokens"] == 100
        assert cost_b["totals"]["input_tokens"] == 200

    def test_haiku_cheaper_than_sonnet(self, tmp_path):
        db = tmp_path / "db"
        log_turn(db, "haiku", "Agent", "claude-haiku-4-5", 1000, 1000, [], 10.0)
        log_turn(db, "sonnet", "Agent", "claude-sonnet-4-6", 1000, 1000, [], 10.0)
        haiku_cost = get_session_cost(db, "haiku")["totals"]["cost_usd"]
        sonnet_cost = get_session_cost(db, "sonnet")["totals"]["cost_usd"]
        assert haiku_cost < sonnet_cost
