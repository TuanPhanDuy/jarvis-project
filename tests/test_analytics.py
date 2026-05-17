"""Tests for agent performance analytics."""
from __future__ import annotations

import time

import pytest

from jarvis.memory.analytics import get_agent_performance, _percentile
from jarvis.memory.turns import log_turn


class TestPercentile:
    def test_empty_returns_zero(self):
        assert _percentile([], 50) == 0.0

    def test_single_value(self):
        assert _percentile([100.0], 95) == 100.0

    def test_p50_of_sorted_list(self):
        data = [10.0, 20.0, 30.0, 40.0, 50.0]
        assert _percentile(data, 50) == 30.0

    def test_p95_higher_than_p50(self):
        data = [float(i) for i in range(1, 101)]
        assert _percentile(data, 95) > _percentile(data, 50)


class TestGetAgentPerformance:
    def _write_turns(self, db_path, agent_type, count, latency_ms=100.0):
        for _ in range(count):
            log_turn(
                db_path=db_path,
                session_id="s1",
                agent_type=agent_type,
                model="llama3.2",
                input_tokens=50,
                output_tokens=20,
                tool_calls=[],
                latency_ms=latency_ms,
            )

    def test_returns_empty_for_missing_db(self, tmp_path):
        result = get_agent_performance(tmp_path / "nonexistent.db")
        assert result == []

    def test_returns_per_agent_stats(self, tmp_path):
        db = tmp_path / "jarvis.db"
        self._write_turns(db, "ResearcherAgent", 5, latency_ms=200.0)
        self._write_turns(db, "CoderAgent", 3, latency_ms=150.0)
        result = get_agent_performance(db)
        assert len(result) == 2
        agent_types = {r["agent_type"] for r in result}
        assert "ResearcherAgent" in agent_types
        assert "CoderAgent" in agent_types

    def test_call_count_correct(self, tmp_path):
        db = tmp_path / "jarvis.db"
        self._write_turns(db, "QAAgent", 7)
        result = get_agent_performance(db, agent_type="QAAgent")
        assert len(result) == 1
        assert result[0]["call_count"] == 7

    def test_latency_stats_computed(self, tmp_path):
        db = tmp_path / "jarvis.db"
        self._write_turns(db, "ResearcherAgent", 10, latency_ms=100.0)
        result = get_agent_performance(db, agent_type="ResearcherAgent")
        assert result[0]["avg_latency_ms"] == 100.0
        assert result[0]["p50_latency_ms"] == 100.0
        assert result[0]["p95_latency_ms"] == 100.0

    def test_since_ts_filters_old_records(self, tmp_path):
        db = tmp_path / "jarvis.db"
        self._write_turns(db, "ResearcherAgent", 5)
        future_ts = time.time() + 9999
        result = get_agent_performance(db, since_ts=future_ts)
        assert result == []

    def test_models_used_listed(self, tmp_path):
        db = tmp_path / "jarvis.db"
        self._write_turns(db, "CoderAgent", 3)
        result = get_agent_performance(db, agent_type="CoderAgent")
        assert "llama3.2" in result[0]["models_used"]

    def test_agent_type_filter(self, tmp_path):
        db = tmp_path / "jarvis.db"
        self._write_turns(db, "ResearcherAgent", 3)
        self._write_turns(db, "CoderAgent", 2)
        result = get_agent_performance(db, agent_type="ResearcherAgent")
        assert len(result) == 1
        assert result[0]["agent_type"] == "ResearcherAgent"
