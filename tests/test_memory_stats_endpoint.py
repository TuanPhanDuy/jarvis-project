"""Tests for GET /api/memory/stats and GET /api/analytics/tools endpoints."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _fake_settings(tmp_path: Path) -> MagicMock:
    s = MagicMock()
    s.anthropic_api_key = "test-key"
    s.tavily_api_key = "test-key"
    s.model = "llama3.2"
    s.fast_model = "llama3.2"
    s.max_tokens = 512
    s.max_search_calls = 5
    s.routing_strategy = "always_primary"
    s.allowed_commands = []
    s.reports_dir = tmp_path / "reports"
    s.otel_enabled = False
    s.auth_enabled = False
    s.rate_limit_enabled = False
    s.proactive_enabled = False
    s.peer_enabled = False
    s.api_session_ttl_minutes = 60
    s.memory_retention_days = 90
    s.jwt_secret = "test-secret"
    s.chat_rate_limit = "100/minute"
    s.idle_minutes = 30
    s.agent_turn_timeout_seconds = 120
    s.tool_timeout_seconds = 60
    s.peer_port = 8001
    s.vision_model = "llava:13b"
    return s


@pytest.fixture
def client(tmp_path: Path):
    settings = _fake_settings(tmp_path)
    settings.reports_dir.mkdir(parents=True, exist_ok=True)

    with (
        patch("jarvis.api.server.get_settings", return_value=settings),
        patch("jarvis.config.get_settings", return_value=settings),
        patch("jarvis.scheduler.core.start_scheduler"),
        patch("jarvis.scheduler.core.stop_scheduler"),
        patch("jarvis.tools.registry.build_registry", return_value=([], {})),
    ):
        from fastapi.testclient import TestClient
        from jarvis.api.server import app
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c, settings.reports_dir


# ── GET /api/memory/stats ─────────────────────────────────────────────────────


class TestMemoryStatsEndpoint:
    def test_returns_200(self, client):
        c, _ = client
        resp = c.get("/api/memory/stats")
        assert resp.status_code == 200

    def test_has_required_keys(self, client):
        c, _ = client
        data = c.get("/api/memory/stats").json()
        for key in ("episodes", "feedback", "preferences", "failures"):
            assert key in data

    def test_empty_db_returns_zeros(self, client):
        c, _ = client
        data = c.get("/api/memory/stats").json()
        assert data["episodes"] == 0
        assert data["feedback"] == 0
        assert data["preferences"] == 0
        assert data["failures"] == 0

    def test_missing_db_returns_zeros(self, client):
        c, reports_dir = client
        # Don't create the db file; stats should still return zeros
        data = c.get("/api/memory/stats").json()
        for val in data.values():
            assert val == 0

    def test_counts_match_after_writes(self, client):
        c, reports_dir = client
        db = reports_dir / "jarvis.db"
        from jarvis.memory.episodic import log_episode
        from jarvis.memory.feedback import log_feedback
        log_episode(db, "s1", "user", "hello")
        log_episode(db, "s1", "assistant", "hi")
        log_feedback(db, "s1", "great response", 5)
        data = c.get("/api/memory/stats").json()
        assert data["episodes"] == 2
        assert data["feedback"] == 1


# ── GET /api/analytics/tools ──────────────────────────────────────────────────


class TestToolAnalyticsEndpoint:
    def test_returns_200(self, client):
        c, _ = client
        resp = c.get("/api/analytics/tools")
        assert resp.status_code == 200

    def test_returns_list(self, client):
        c, _ = client
        data = c.get("/api/analytics/tools").json()
        assert isinstance(data, list)

    def test_empty_when_no_audit_log(self, client):
        c, _ = client
        data = c.get("/api/analytics/tools").json()
        assert data == []

    def test_per_tool_entry_after_audit_write(self, client):
        c, reports_dir = client
        db = reports_dir / "jarvis.db"
        from jarvis.security.audit import log_tool_call
        log_tool_call(db, "s1", "web_search", {}, "low", 1, result_ok=1, duration_ms=80.0)
        log_tool_call(db, "s1", "web_search", {}, "low", 1, result_ok=0, duration_ms=120.0)
        data = c.get("/api/analytics/tools").json()
        assert len(data) == 1
        entry = data[0]
        assert entry["tool_name"] == "web_search"
        assert entry["call_count"] == 2
        assert entry["error_count"] == 1

    def test_hours_param_filters_old(self, client):
        c, reports_dir = client
        db = reports_dir / "jarvis.db"
        from jarvis.security.audit import log_tool_call
        log_tool_call(db, "s1", "web_search", {}, "low", 1, result_ok=1, duration_ms=50.0)
        # Pass hours=0 to exclude all past records
        data = c.get("/api/analytics/tools?hours=0").json()
        assert data == []

    def test_required_keys_present(self, client):
        c, reports_dir = client
        db = reports_dir / "jarvis.db"
        from jarvis.security.audit import log_tool_call
        log_tool_call(db, "s1", "delegate_task", {}, "medium", 1, result_ok=1, duration_ms=200.0)
        data = c.get("/api/analytics/tools").json()
        assert len(data) == 1
        for key in ("tool_name", "call_count", "error_count", "error_rate", "avg_latency_ms", "p95_latency_ms"):
            assert key in data[0]


# ── GET /api/failures ─────────────────────────────────────────────────────────


class TestFailuresEndpoint:
    def test_returns_200(self, client):
        c, _ = client
        assert c.get("/api/failures").status_code == 200

    def test_empty_when_no_failures(self, client):
        c, _ = client
        assert c.get("/api/failures").json() == []

    def test_returns_patterns_after_log(self, client):
        c, reports_dir = client
        db = reports_dir / "jarvis.db"
        from jarvis.memory.failures import log_failure
        log_failure(db, "web_search", {}, "connection timeout")
        log_failure(db, "web_search", {}, "connection timeout")
        log_failure(db, "read_file", {}, "file not found")
        data = c.get("/api/failures").json()
        tool_names = {e["tool_name"] for e in data}
        assert "web_search" in tool_names
        assert "read_file" in tool_names

    def test_count_aggregated(self, client):
        c, reports_dir = client
        db = reports_dir / "jarvis.db"
        from jarvis.memory.failures import log_failure
        for _ in range(3):
            log_failure(db, "delegate_task", {}, "timeout")
        data = c.get("/api/failures").json()
        entry = next(e for e in data if e["tool_name"] == "delegate_task")
        assert entry["count"] == 3

    def test_tool_name_filter(self, client):
        c, reports_dir = client
        db = reports_dir / "jarvis.db"
        from jarvis.memory.failures import log_failure
        log_failure(db, "web_search", {}, "err1")
        log_failure(db, "read_file", {}, "err2")
        data = c.get("/api/failures?tool_name=web_search").json()
        assert all(e["tool_name"] == "web_search" for e in data)

    def test_required_keys_present(self, client):
        c, reports_dir = client
        db = reports_dir / "jarvis.db"
        from jarvis.memory.failures import log_failure
        log_failure(db, "some_tool", {}, "boom")
        data = c.get("/api/failures").json()
        for key in ("tool_name", "error_msg", "count"):
            assert key in data[0]


# ── GET /api/budget ───────────────────────────────────────────────────────────


class TestBudgetAllEndpoint:
    def test_returns_200(self, client):
        c, _ = client
        assert c.get("/api/budget").status_code == 200

    def test_empty_when_no_db(self, client):
        c, _ = client
        data = c.get("/api/budget").json()
        assert data == []

    def test_returns_all_users_after_set(self, client):
        c, reports_dir = client
        db = reports_dir / "jarvis.db"
        from jarvis.api.budget import set_budget, record_spend
        set_budget(db, "alice", 10.0)
        set_budget(db, "bob", 5.0)
        record_spend(db, "alice", 2.0)
        data = c.get("/api/budget").json()
        user_ids = {e["user_id"] for e in data}
        assert "alice" in user_ids
        assert "bob" in user_ids

    def test_entry_has_required_keys(self, client):
        c, reports_dir = client
        db = reports_dir / "jarvis.db"
        from jarvis.api.budget import set_budget
        set_budget(db, "carol", 20.0)
        data = c.get("/api/budget").json()
        for key in ("user_id", "monthly_budget_usd", "spent_usd", "remaining_usd", "period"):
            assert key in data[0]


# ── Unit: get_failure_patterns ────────────────────────────────────────────────


class TestGetFailurePatterns:
    def test_empty_for_missing_db(self, tmp_path):
        from jarvis.memory.failures import get_failure_patterns
        result = get_failure_patterns(tmp_path / "missing.db")
        assert result == []

    def test_returns_patterns_sorted_by_count(self, tmp_path):
        from jarvis.memory.failures import get_failure_patterns, log_failure
        db = tmp_path / "jarvis.db"
        for _ in range(5):
            log_failure(db, "a_tool", {}, "err")
        for _ in range(2):
            log_failure(db, "b_tool", {}, "err2")
        result = get_failure_patterns(db)
        assert result[0]["tool_name"] == "a_tool"
        assert result[0]["count"] == 5

    def test_tool_name_filter(self, tmp_path):
        from jarvis.memory.failures import get_failure_patterns, log_failure
        db = tmp_path / "jarvis.db"
        log_failure(db, "tool_x", {}, "boom")
        log_failure(db, "tool_y", {}, "crash")
        result = get_failure_patterns(db, tool_name="tool_x")
        assert len(result) == 1
        assert result[0]["tool_name"] == "tool_x"

    def test_limit_respected(self, tmp_path):
        from jarvis.memory.failures import get_failure_patterns, log_failure
        db = tmp_path / "jarvis.db"
        for i in range(10):
            log_failure(db, f"tool_{i}", {}, "err")
        result = get_failure_patterns(db, limit=3)
        assert len(result) <= 3

    def test_required_keys_in_result(self, tmp_path):
        from jarvis.memory.failures import get_failure_patterns, log_failure
        db = tmp_path / "jarvis.db"
        log_failure(db, "my_tool", {}, "some error")
        result = get_failure_patterns(db)
        assert len(result) == 1
        for key in ("tool_name", "error_msg", "count"):
            assert key in result[0]
