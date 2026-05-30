"""Tests for tool registry introspection API."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


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


_FAKE_SCHEMAS = [
    {
        "name": "web_search",
        "description": "Search the web",
        "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}},
    },
    {
        "name": "save_report",
        "description": "Save a report",
        "input_schema": {"type": "object", "properties": {"title": {"type": "string"}}},
    },
]


def _patch_registry(schemas=None):
    return patch(
        "jarvis.api.server.build_registry",
        return_value=(schemas or _FAKE_SCHEMAS, {}),
    )


class TestListTools:
    def test_returns_200(self, client):
        with _patch_registry():
            resp = client.get("/api/tools")
        assert resp.status_code == 200

    def test_returns_list_of_tools(self, client):
        with _patch_registry():
            resp = client.get("/api/tools")
        body = resp.json()
        assert isinstance(body, list)
        assert len(body) == 2

    def test_each_tool_has_required_fields(self, client):
        with _patch_registry():
            resp = client.get("/api/tools")
        for tool in resp.json():
            assert "name" in tool
            assert "description" in tool
            assert "enabled" in tool
            assert "input_schema" in tool

    def test_disabled_plugin_shows_enabled_false(self, client):
        from jarvis.tools import plugin_loader
        plugin_loader._disabled.add("web_search")
        try:
            with _patch_registry():
                resp = client.get("/api/tools")
            tools = {t["name"]: t for t in resp.json()}
            assert tools["web_search"]["enabled"] is False
            assert tools["save_report"]["enabled"] is True
        finally:
            plugin_loader._disabled.discard("web_search")


class TestGetToolDetail:
    def test_existing_tool_returns_schema(self, client):
        with _patch_registry():
            resp = client.get("/api/tools/web_search")
        assert resp.status_code == 200
        assert resp.json()["name"] == "web_search"

    def test_includes_input_schema(self, client):
        with _patch_registry():
            resp = client.get("/api/tools/web_search")
        assert "input_schema" in resp.json()

    def test_unknown_tool_404(self, client):
        with _patch_registry():
            resp = client.get("/api/tools/nonexistent_tool_xyz")
        assert resp.status_code == 404


class TestToolMetrics:
    def test_returns_200(self, client):
        resp = client.get("/api/tools/metrics")
        assert resp.status_code == 200

    def test_returns_list(self, client):
        resp = client.get("/api/tools/metrics")
        assert isinstance(resp.json(), list)

    def test_metrics_from_audit_log(self, client, tmp_path):
        import sqlite3, time as _time
        db = tmp_path / "jarvis.db"
        conn = sqlite3.connect(str(db))
        now = _time.time()
        conn.execute("CREATE TABLE audit_log (tool_name, duration_ms, result_ok, timestamp, session_id, user_id, tool_input, risk_level, approved, approver)")
        conn.execute("INSERT INTO audit_log VALUES ('web_search', 200, 1, ?, 's1', 'u1', '{}', 'LOW', 1, 'auto')", (now,))
        conn.execute("INSERT INTO audit_log VALUES ('web_search', 400, 0, ?, 's1', 'u1', '{}', 'LOW', 1, 'auto')", (now,))
        conn.commit()
        conn.close()
        settings = MagicMock()
        settings.reports_dir = tmp_path
        with patch("jarvis.api.server.get_settings", return_value=settings):
            resp = client.get("/api/tools/metrics?since_hours=1")
        assert resp.status_code == 200
        body = resp.json()
        ws = next((t for t in body if t["tool_name"] == "web_search"), None)
        assert ws is not None
        assert ws["call_count"] == 2
        assert ws["error_rate"] == pytest.approx(0.5)
        assert ws["avg_duration_ms"] == pytest.approx(300.0)
