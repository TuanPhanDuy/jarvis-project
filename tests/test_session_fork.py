"""Tests for session management API: list, detail, delete, and fork."""
from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


_TEST_SESSION_IDS = {
    "list-test-session", "fields-test", "detail-test", "usage-test", "tools-test",
    "delete-me", "gone-session", "totally-nonexistent",
    "fork-source", "copy-source", "index-source", "dup-source",
    "custom-id-source", "existing-fork-id", "meta-source",
}


@pytest.fixture(autouse=True)
def reset_auth_and_sessions():
    import jarvis.api.server as _s
    _s._require_auth = None
    yield
    _s._require_auth = None
    # Clean up test-specific sessions to avoid polluting other tests
    for sid in list(_TEST_SESSION_IDS):
        _s._sessions.pop(sid, None)


@pytest.fixture()
def client():
    from jarvis.api.server import app
    return TestClient(app, raise_server_exceptions=False)


def _mock_agent(agent_type="PlannerAgent") -> MagicMock:
    m = MagicMock()
    m.__class__.__name__ = agent_type
    m._turn_tool_calls = ["web_search"]
    m.get_usage_summary.return_value = {
        "input_tokens": 5, "output_tokens": 3,
        "cache_write_tokens": 0, "cache_read_tokens": 0,
        "estimated_cost_usd": 0.0,
    }
    m._approval_gate = None
    return m


def _inject_session(session_id: str, messages: list | None = None, **kwargs):
    from jarvis.api.server import _sessions
    _sessions[session_id] = {
        "agent": _mock_agent(),
        "messages": messages or [],
        "created_at": time.time(),
        "user_id": "anonymous",
        "approval_gate": None,
        **kwargs,
    }


class TestListSessions:
    def test_returns_200(self, client):
        resp = client.get("/api/sessions")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_injected_session_appears(self, client):
        from jarvis.api.server import _sessions
        _inject_session("list-test-session")
        resp = client.get("/api/sessions")
        ids = [s["session_id"] for s in resp.json()]
        assert "list-test-session" in ids
        del _sessions["list-test-session"]

    def test_response_has_required_fields(self, client):
        _inject_session("fields-test")
        resp = client.get("/api/sessions")
        entry = next((s for s in resp.json() if s["session_id"] == "fields-test"), None)
        assert entry is not None
        for field in ("session_id", "created_at", "message_count", "user_id", "agent_type"):
            assert field in entry


class TestGetSessionDetail:
    def test_existing_session_200(self, client):
        _inject_session("detail-test", messages=[{"role": "user", "content": "hi"}])
        resp = client.get("/api/sessions/detail-test")
        assert resp.status_code == 200
        body = resp.json()
        assert body["session_id"] == "detail-test"
        assert body["message_count"] == 1

    def test_nonexistent_session_404(self, client):
        resp = client.get("/api/sessions/nonexistent-xyz")
        assert resp.status_code == 404

    def test_detail_includes_usage(self, client):
        _inject_session("usage-test")
        resp = client.get("/api/sessions/usage-test")
        assert "usage" in resp.json()
        assert "input_tokens" in resp.json()["usage"]

    def test_detail_includes_last_turn_tools(self, client):
        _inject_session("tools-test")
        resp = client.get("/api/sessions/tools-test")
        assert "last_turn_tools" in resp.json()


class TestDeleteSession:
    def test_delete_existing_returns_204(self, client):
        _inject_session("delete-me")
        resp = client.delete("/api/sessions/delete-me")
        assert resp.status_code == 204

    def test_deleted_session_no_longer_listed(self, client):
        _inject_session("gone-session")
        client.delete("/api/sessions/gone-session")
        resp = client.get("/api/sessions")
        ids = [s["session_id"] for s in resp.json()]
        assert "gone-session" not in ids

    def test_delete_nonexistent_returns_404(self, client):
        resp = client.delete("/api/sessions/totally-nonexistent")
        assert resp.status_code == 404


class TestForkSession:
    def test_fork_creates_new_session(self, client):
        _inject_session("fork-source", messages=[
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ])
        with patch("jarvis.api.server._build_agent_for_session", return_value=_mock_agent()):
            resp = client.post("/api/sessions/fork-source/fork")
        assert resp.status_code == 201
        body = resp.json()
        assert body["fork_of"] == "fork-source"
        assert body["message_count"] == 2

    def test_fork_copies_messages(self, client):
        msgs = [{"role": "user", "content": "msg1"}, {"role": "assistant", "content": "msg2"}]
        _inject_session("copy-source", messages=msgs)
        with patch("jarvis.api.server._build_agent_for_session", return_value=_mock_agent()):
            resp = client.post("/api/sessions/copy-source/fork")
        new_id = resp.json()["session_id"]
        from jarvis.api.server import _sessions
        assert _sessions[new_id]["messages"] == msgs

    def test_fork_with_message_index(self, client):
        msgs = [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
            {"role": "user", "content": "c"},
        ]
        _inject_session("index-source", messages=msgs)
        with patch("jarvis.api.server._build_agent_for_session", return_value=_mock_agent()):
            resp = client.post("/api/sessions/index-source/fork", json={"message_index": 2})
        assert resp.status_code == 201
        assert resp.json()["message_count"] == 2

    def test_fork_nonexistent_source_404(self, client):
        resp = client.post("/api/sessions/no-such-session/fork")
        assert resp.status_code == 404

    def test_fork_respects_custom_session_id(self, client):
        _inject_session("custom-id-source")
        with patch("jarvis.api.server._build_agent_for_session", return_value=_mock_agent()):
            resp = client.post("/api/sessions/custom-id-source/fork",
                               json={"new_session_id": "my-fork-id"})
        assert resp.status_code == 201
        assert resp.json()["session_id"] == "my-fork-id"

    def test_fork_duplicate_id_409(self, client):
        _inject_session("dup-source")
        _inject_session("existing-fork-id")
        with patch("jarvis.api.server._build_agent_for_session", return_value=_mock_agent()):
            resp = client.post("/api/sessions/dup-source/fork",
                               json={"new_session_id": "existing-fork-id"})
        assert resp.status_code == 409

    def test_fork_metadata_in_detail(self, client):
        _inject_session("meta-source")
        with patch("jarvis.api.server._build_agent_for_session", return_value=_mock_agent()):
            fork_resp = client.post("/api/sessions/meta-source/fork")
        fork_id = fork_resp.json()["session_id"]
        detail = client.get(f"/api/sessions/{fork_id}").json()
        assert detail["fork_of"] == "meta-source"
        assert detail["forked_at"] is not None
