"""Tests for system stats dashboard (api/stats.py)."""
from __future__ import annotations

import pytest

from jarvis.api.stats import get_system_stats


def _populate(db_path):
    """Seed minimal data across tables."""
    import sqlite3, json, time
    conn = sqlite3.connect(str(db_path))
    # persisted_sessions
    conn.execute("""CREATE TABLE IF NOT EXISTS persisted_sessions
        (session_id TEXT PRIMARY KEY, agent_type TEXT, user_id TEXT,
         messages TEXT, fork_of TEXT, title TEXT, created_at REAL, updated_at REAL)""")
    conn.execute("INSERT INTO persisted_sessions VALUES (?,?,?,?,?,?,?,?)",
                 ("s1", "PlannerAgent", "u1", "[]", None, None, time.time(), time.time()))
    # agent_turns
    conn.execute("""CREATE TABLE IF NOT EXISTS agent_turns
        (id TEXT, session_id TEXT, agent_type TEXT, model TEXT,
         input_tokens INT, output_tokens INT, tool_calls_json TEXT,
         latency_ms REAL, timestamp REAL)""")
    conn.execute("INSERT INTO agent_turns VALUES (?,?,?,?,?,?,?,?,?)",
                 ("t1", "s1", "ResearcherAgent", "sonnet", 1000, 500, "[]", 200.0, time.time()))
    # audit_log
    conn.execute("""CREATE TABLE IF NOT EXISTS audit_log
        (id TEXT, session_id TEXT, tool_name TEXT, result_ok INT,
         duration_ms REAL, timestamp REAL)""")
    conn.execute("INSERT INTO audit_log VALUES (?,?,?,?,?,?)",
                 ("a1", "s1", "web_search", 1, 300.0, time.time()))
    conn.execute("INSERT INTO audit_log VALUES (?,?,?,?,?,?)",
                 ("a2", "s1", "web_search", 0, 100.0, time.time()))
    # episodic_memory
    conn.execute("""CREATE TABLE IF NOT EXISTS episodic_memory
        (id TEXT, session_id TEXT, role TEXT, content TEXT, timestamp REAL, importance REAL,
         user_id TEXT, decay_factor REAL)""")
    conn.execute("INSERT INTO episodic_memory VALUES (?,?,?,?,?,?,?,?)",
                 ("e1", "s1", "user", "hello", time.time(), 1.0, "u1", 1.0))
    # kg_entities
    conn.execute("""CREATE TABLE IF NOT EXISTS kg_entities
        (id TEXT, name TEXT, entity_type TEXT, description TEXT,
         user_id TEXT, updated_at REAL)""")
    conn.execute("INSERT INTO kg_entities VALUES (?,?,?,?,?,?)",
                 ("en1", "Transformer", "concept", "...", "u1", time.time()))
    # user_preferences
    conn.execute("""CREATE TABLE IF NOT EXISTS user_preferences
        (user_id TEXT, category TEXT, key TEXT, value TEXT, confidence REAL,
         source TEXT, updated_at REAL)""")
    conn.execute("INSERT INTO user_preferences VALUES (?,?,?,?,?,?,?)",
                 ("u1", "style", "verbosity", "detailed", 0.9, "observed", time.time()))
    conn.commit()
    conn.close()


class TestGetSystemStats:
    def test_returns_required_top_level_keys(self, tmp_path):
        db = tmp_path / "db"
        stats = get_system_stats(db)
        for key in ("generated_at", "sessions", "tokens", "tools", "memory", "jobs", "db_size_bytes"):
            assert key in stats

    def test_empty_db_returns_zeros(self, tmp_path):
        stats = get_system_stats(tmp_path / "db")
        assert stats["sessions"]["active"] == 0
        assert stats["tokens"]["total_input"] == 0
        assert stats["tools"]["total_calls"] == 0
        assert stats["memory"]["episodes"] == 0

    def test_sessions_active_passed_through(self, tmp_path):
        stats = get_system_stats(tmp_path / "db", sessions_active=7)
        assert stats["sessions"]["active"] == 7

    def test_counts_persisted_sessions(self, tmp_path):
        db = tmp_path / "db"
        _populate(db)
        stats = get_system_stats(db)
        assert stats["sessions"]["persisted_total"] == 1

    def test_aggregates_token_counts(self, tmp_path):
        db = tmp_path / "db"
        _populate(db)
        stats = get_system_stats(db)
        assert stats["tokens"]["total_input"] == 1000
        assert stats["tokens"]["total_output"] == 500

    def test_cost_is_positive_when_tokens_nonzero(self, tmp_path):
        db = tmp_path / "db"
        _populate(db)
        stats = get_system_stats(db)
        assert stats["tokens"]["estimated_cost_usd"] > 0

    def test_tool_calls_counted(self, tmp_path):
        db = tmp_path / "db"
        _populate(db)
        stats = get_system_stats(db)
        assert stats["tools"]["total_calls"] == 2
        assert stats["tools"]["total_errors"] == 1
        assert stats["tools"]["error_rate"] == 0.5

    def test_top_10_tools_present(self, tmp_path):
        db = tmp_path / "db"
        _populate(db)
        stats = get_system_stats(db)
        assert len(stats["tools"]["top_10"]) >= 1
        assert stats["tools"]["top_10"][0]["tool_name"] == "web_search"

    def test_memory_counts_populated(self, tmp_path):
        db = tmp_path / "db"
        _populate(db)
        stats = get_system_stats(db)
        assert stats["memory"]["episodes"] == 1
        assert stats["memory"]["graph_entities"] == 1
        assert stats["memory"]["preferences"] == 1

    def test_db_size_nonzero_after_writes(self, tmp_path):
        db = tmp_path / "db"
        _populate(db)
        stats = get_system_stats(db)
        assert stats["db_size_bytes"] > 0

    def test_generated_at_is_recent(self, tmp_path):
        import time
        before = time.time()
        stats = get_system_stats(tmp_path / "db")
        assert stats["generated_at"] >= before
