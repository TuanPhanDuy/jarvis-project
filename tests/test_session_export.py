"""Tests for session export (GET) and import (POST) endpoints."""
from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


_EXPORT_TEST_SESSIONS = {
    "exp-1", "exp-2", "exp-3", "exp-4",
    "import-1", "import-2", "custom-id", "original-id",
    "bad", "rt-source", "rt-restored",
}


@pytest.fixture(autouse=True)
def reset_auth_and_sessions():
    import jarvis.api.server as _s
    _s._require_auth = None
    yield
    _s._require_auth = None
    for sid in list(_EXPORT_TEST_SESSIONS):
        _s._sessions.pop(sid, None)


@pytest.fixture()
def client():
    from jarvis.api.server import app
    return TestClient(app, raise_server_exceptions=False)


def _inject(session_id: str, messages: list | None = None, agent_type: str = "PlannerAgent"):
    from jarvis.api.server import _sessions
    agent = MagicMock()
    agent.__class__.__name__ = agent_type
    agent._turn_tool_calls = []
    agent.get_usage_summary.return_value = {
        "input_tokens": 0, "output_tokens": 0,
        "cache_write_tokens": 0, "cache_read_tokens": 0, "estimated_cost_usd": 0.0,
    }
    agent._approval_gate = None
    _sessions[session_id] = {
        "agent": agent,
        "messages": messages or [],
        "created_at": time.time(),
        "user_id": "alice",
        "approval_gate": None,
    }


class TestExportSession:
    def test_export_existing_session_200(self, client):
        msgs = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]
        _inject("exp-1", msgs)
        resp = client.get("/api/sessions/exp-1/export")
        assert resp.status_code == 200

    def test_export_contains_required_fields(self, client):
        _inject("exp-2", [{"role": "user", "content": "test"}])
        body = client.get("/api/sessions/exp-2/export").json()
        for field in ("session_id", "agent_type", "messages", "exported_at"):
            assert field in body

    def test_export_preserves_messages(self, client):
        msgs = [{"role": "user", "content": "preserved"}, {"role": "assistant", "content": "yes"}]
        _inject("exp-3", msgs)
        body = client.get("/api/sessions/exp-3/export").json()
        assert body["messages"] == msgs

    def test_export_includes_agent_type(self, client):
        _inject("exp-4", [], agent_type="ResearcherAgent")
        body = client.get("/api/sessions/exp-4/export").json()
        assert body["agent_type"] == "ResearcherAgent"

    def test_export_nonexistent_session_404(self, client):
        resp = client.get("/api/sessions/no-such-exp/export")
        assert resp.status_code == 404


class TestImportSession:
    def test_import_creates_session(self, client):
        bundle = {
            "session_id": "import-1",
            "agent_type": "PlannerAgent",
            "messages": [{"role": "user", "content": "restored"}],
        }
        with patch("jarvis.api.server._build_agent_for_session", return_value=MagicMock()), \
             patch("jarvis.api.server._persist_session"):
            resp = client.post("/api/sessions/import", json=bundle)
        assert resp.status_code == 201
        body = resp.json()
        assert body["session_id"] == "import-1"
        assert body["message_count"] == 1

    def test_import_restores_messages_into_session(self, client):
        from jarvis.api.server import _sessions
        bundle = {
            "session_id": "import-2",
            "messages": [{"role": "user", "content": "original"}],
        }
        with patch("jarvis.api.server._build_agent_for_session", return_value=MagicMock()), \
             patch("jarvis.api.server._persist_session"):
            client.post("/api/sessions/import", json=bundle)
        assert "import-2" in _sessions
        assert _sessions["import-2"]["messages"][0]["content"] == "original"

    def test_import_custom_session_id(self, client):
        bundle = {
            "session_id": "original-id",
            "new_session_id": "custom-id",
            "messages": [],
        }
        with patch("jarvis.api.server._build_agent_for_session", return_value=MagicMock()), \
             patch("jarvis.api.server._persist_session"):
            resp = client.post("/api/sessions/import", json=bundle)
        assert resp.json()["session_id"] == "custom-id"

    def test_import_invalid_messages_422(self, client):
        bundle = {"session_id": "bad", "messages": "not-a-list"}
        resp = client.post("/api/sessions/import", json=bundle)
        assert resp.status_code == 422

    def test_export_then_import_roundtrip(self, client):
        msgs = [{"role": "user", "content": "roundtrip"}, {"role": "assistant", "content": "ok"}]
        _inject("rt-source", msgs)

        bundle = client.get("/api/sessions/rt-source/export").json()
        bundle["new_session_id"] = "rt-restored"

        with patch("jarvis.api.server._build_agent_for_session", return_value=MagicMock()), \
             patch("jarvis.api.server._persist_session"):
            import_resp = client.post("/api/sessions/import", json=bundle)

        assert import_resp.status_code == 201
        assert import_resp.json()["message_count"] == 2

        from jarvis.api.server import _sessions
        assert _sessions["rt-restored"]["messages"] == msgs
