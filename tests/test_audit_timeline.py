"""Tests for audit timeline and slow-tools endpoints."""
from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from jarvis.security.audit import get_session_timeline, get_slow_tools, log_tool_call


@pytest.fixture()
def db(tmp_path: Path) -> Path:
    return tmp_path / "jarvis.db"


def _log(db, session_id="s1", tool="web_search", duration_ms=100.0, result_ok=1):
    log_tool_call(
        db_path=db,
        session_id=session_id,
        tool_name=tool,
        tool_input={"q": "test"},
        risk_level="LOW",
        approved=1,
        approver="auto",
        result_ok=result_ok,
        duration_ms=duration_ms,
    )


# ── get_session_timeline ──────────────────────────────────────────────────────

class TestGetSessionTimeline:
    def test_empty_session_returns_empty(self, db):
        assert get_session_timeline(db, "no-such-session") == []

    def test_returns_ordered_events(self, db):
        _log(db, session_id="s1", tool="web_search", duration_ms=200)
        time.sleep(0.01)
        _log(db, session_id="s1", tool="report_writer", duration_ms=50)
        timeline = get_session_timeline(db, "s1")
        assert len(timeline) == 2
        assert timeline[0]["tool"] == "web_search"
        assert timeline[1]["tool"] == "report_writer"

    def test_entries_have_required_fields(self, db):
        _log(db, session_id="s2", tool="memory_search")
        timeline = get_session_timeline(db, "s2")
        entry = timeline[0]
        for field in ("tool", "timestamp", "duration_ms", "risk_level", "result_ok"):
            assert field in entry

    def test_result_ok_is_bool(self, db):
        _log(db, session_id="s3", result_ok=1)
        _log(db, session_id="s3", result_ok=0)
        timeline = get_session_timeline(db, "s3")
        assert all(isinstance(e["result_ok"], bool) for e in timeline)

    def test_only_returns_events_for_requested_session(self, db):
        _log(db, session_id="alpha", tool="tool_a")
        _log(db, session_id="beta", tool="tool_b")
        timeline = get_session_timeline(db, "alpha")
        assert all(e["tool"] == "tool_a" for e in timeline)

    def test_ordered_oldest_first(self, db):
        for i in range(3):
            _log(db, session_id="order", tool=f"tool_{i}")
            time.sleep(0.005)
        timeline = get_session_timeline(db, "order")
        ts = [e["timestamp"] for e in timeline]
        assert ts == sorted(ts)


# ── get_slow_tools ────────────────────────────────────────────────────────────

class TestGetSlowTools:
    def test_empty_db_returns_empty(self, db):
        assert get_slow_tools(db, threshold_ms=100) == []

    def test_fast_tools_excluded(self, db):
        _log(db, tool="fast_tool", duration_ms=10)
        result = get_slow_tools(db, threshold_ms=5000)
        assert result == []

    def test_slow_tools_included(self, db):
        for _ in range(3):
            _log(db, tool="slow_tool", duration_ms=8000)
        result = get_slow_tools(db, threshold_ms=5000)
        assert len(result) == 1
        assert result[0]["tool_name"] == "slow_tool"

    def test_result_has_required_fields(self, db):
        _log(db, tool="big_tool", duration_ms=9999)
        result = get_slow_tools(db, threshold_ms=100)
        entry = result[0]
        for field in ("tool_name", "avg_duration_ms", "call_count", "max_duration_ms"):
            assert field in entry

    def test_sorted_by_avg_duration_desc(self, db):
        _log(db, tool="slower", duration_ms=9000)
        _log(db, tool="fast_but_above_threshold", duration_ms=6000)
        result = get_slow_tools(db, threshold_ms=5000)
        assert result[0]["avg_duration_ms"] >= result[1]["avg_duration_ms"]

    def test_custom_threshold(self, db):
        _log(db, tool="medium", duration_ms=3000)
        assert get_slow_tools(db, threshold_ms=5000) == []
        assert len(get_slow_tools(db, threshold_ms=2000)) == 1


# ── API endpoints ─────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_auth():
    import jarvis.api.server as _s
    _s._require_auth = None
    yield
    _s._require_auth = None


@pytest.fixture()
def client():
    from jarvis.api.server import app
    return TestClient(app, raise_server_exceptions=False)


class TestTimelineEndpoint:
    def test_returns_200_for_existing_session(self, client, tmp_path):
        with patch_db(tmp_path):
            _log(tmp_path / "jarvis.db", session_id="t1", tool="search")
            resp = client.get("/api/sessions/t1/timeline")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_returns_empty_list_for_unknown_session(self, client, tmp_path):
        with patch_db(tmp_path):
            resp = client.get("/api/sessions/no-such-session/timeline")
        assert resp.status_code == 200
        assert resp.json() == []


class TestSlowToolsEndpoint:
    def test_returns_200(self, client, tmp_path):
        with patch_db(tmp_path):
            resp = client.get("/api/audit/slow-tools")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_custom_threshold_param(self, client, tmp_path):
        with patch_db(tmp_path):
            _log(tmp_path / "jarvis.db", tool="medium", duration_ms=3000)
            resp = client.get("/api/audit/slow-tools?threshold_ms=2000")
        assert resp.status_code == 200
        body = resp.json()
        assert any(t["tool_name"] == "medium" for t in body)

    def test_default_threshold_excludes_fast_tools(self, client, tmp_path):
        with patch_db(tmp_path):
            _log(tmp_path / "jarvis.db", tool="quicktool", duration_ms=100)
            resp = client.get("/api/audit/slow-tools")
        assert resp.status_code == 200
        body = resp.json()
        assert not any(t["tool_name"] == "quicktool" for t in body)


from unittest.mock import patch
from contextlib import contextmanager

@contextmanager
def patch_db(tmp_path):
    from unittest.mock import patch
    import jarvis.api.server as _s
    from unittest.mock import MagicMock
    settings = MagicMock()
    settings.reports_dir = tmp_path
    with patch("jarvis.api.server.get_settings", return_value=settings):
        yield
