"""Tests for message editing API: PATCH/DELETE/POST /api/sessions/{id}/messages."""
from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

_EDIT_TEST_SESSIONS = {
    "edit-s", "del-s", "ins-s", "neg-s", "oob-s", "empty-s", "missing-s",
}


@pytest.fixture(autouse=True)
def reset_state():
    import jarvis.api.server as _s
    _s._require_auth = None
    yield
    _s._require_auth = None
    for sid in list(_EDIT_TEST_SESSIONS):
        _s._sessions.pop(sid, None)


@pytest.fixture()
def client():
    from jarvis.api.server import app
    return TestClient(app, raise_server_exceptions=False)


def _inject(session_id: str, messages: list):
    from jarvis.api.server import _sessions
    _sessions[session_id] = {
        "agent": MagicMock(),
        "messages": list(messages),
        "created_at": time.time(),
        "user_id": "anonymous",
        "approval_gate": None,
    }


_MSGS = [
    {"role": "user", "content": "first"},
    {"role": "assistant", "content": "second"},
    {"role": "user", "content": "third"},
]


# ── PATCH (edit) ──────────────────────────────────────────────────────────────

class TestEditMessage:
    def test_edit_updates_content(self, client):
        _inject("edit-s", _MSGS)
        with patch("jarvis.api.server._persist_session"):
            resp = client.patch("/api/sessions/edit-s/messages/0",
                                json={"content": "updated first"})
        assert resp.status_code == 200
        assert resp.json()["message"]["content"] == "updated first"

    def test_edit_persists_in_session(self, client):
        from jarvis.api.server import _sessions
        _inject("edit-s", _MSGS)
        with patch("jarvis.api.server._persist_session"):
            client.patch("/api/sessions/edit-s/messages/1", json={"content": "new second"})
        assert _sessions["edit-s"]["messages"][1]["content"] == "new second"

    def test_edit_updates_role(self, client):
        _inject("edit-s", _MSGS)
        with patch("jarvis.api.server._persist_session"):
            resp = client.patch("/api/sessions/edit-s/messages/0",
                                json={"content": "msg", "role": "system"})
        assert resp.json()["message"]["role"] == "system"

    def test_edit_negative_index(self, client):
        _inject("neg-s", _MSGS)
        with patch("jarvis.api.server._persist_session"):
            resp = client.patch("/api/sessions/neg-s/messages/-1",
                                json={"content": "last updated"})
        assert resp.status_code == 200
        assert resp.json()["message"]["content"] == "last updated"

    def test_edit_out_of_range_422(self, client):
        _inject("oob-s", _MSGS)
        resp = client.patch("/api/sessions/oob-s/messages/99", json={"content": "x"})
        assert resp.status_code == 422

    def test_edit_missing_content_422(self, client):
        _inject("edit-s", _MSGS)
        resp = client.patch("/api/sessions/edit-s/messages/0", json={"role": "user"})
        assert resp.status_code == 422

    def test_edit_unknown_session_404(self, client):
        resp = client.patch("/api/sessions/missing-s/messages/0", json={"content": "x"})
        assert resp.status_code == 404

    def test_edit_returns_correct_index(self, client):
        _inject("edit-s", _MSGS)
        with patch("jarvis.api.server._persist_session"):
            resp = client.patch("/api/sessions/edit-s/messages/2", json={"content": "c"})
        assert resp.json()["index"] == 2


# ── DELETE ────────────────────────────────────────────────────────────────────

class TestDeleteMessage:
    def test_delete_removes_message(self, client):
        from jarvis.api.server import _sessions
        _inject("del-s", _MSGS)
        with patch("jarvis.api.server._persist_session"):
            resp = client.delete("/api/sessions/del-s/messages/1")
        assert resp.status_code == 200
        assert len(_sessions["del-s"]["messages"]) == 2

    def test_delete_returns_removed(self, client):
        _inject("del-s", _MSGS)
        with patch("jarvis.api.server._persist_session"):
            resp = client.delete("/api/sessions/del-s/messages/0")
        assert resp.json()["removed"]["content"] == "first"

    def test_delete_returns_new_count(self, client):
        _inject("del-s", _MSGS)
        with patch("jarvis.api.server._persist_session"):
            resp = client.delete("/api/sessions/del-s/messages/0")
        assert resp.json()["message_count"] == 2

    def test_delete_negative_index(self, client):
        from jarvis.api.server import _sessions
        _inject("del-s", _MSGS)
        with patch("jarvis.api.server._persist_session"):
            client.delete("/api/sessions/del-s/messages/-1")
        assert len(_sessions["del-s"]["messages"]) == 2
        assert _sessions["del-s"]["messages"][-1]["content"] == "second"

    def test_delete_out_of_range_422(self, client):
        _inject("del-s", _MSGS)
        resp = client.delete("/api/sessions/del-s/messages/50")
        assert resp.status_code == 422

    def test_delete_unknown_session_404(self, client):
        resp = client.delete("/api/sessions/missing-s/messages/0")
        assert resp.status_code == 404


# ── POST (insert) ─────────────────────────────────────────────────────────────

class TestInsertMessage:
    def test_insert_appends_by_default(self, client):
        from jarvis.api.server import _sessions
        _inject("ins-s", _MSGS)
        with patch("jarvis.api.server._persist_session"):
            resp = client.post("/api/sessions/ins-s/messages",
                               json={"role": "user", "content": "appended"})
        assert resp.status_code == 201
        assert _sessions["ins-s"]["messages"][-1]["content"] == "appended"

    def test_insert_at_position(self, client):
        from jarvis.api.server import _sessions
        _inject("ins-s", _MSGS)
        with patch("jarvis.api.server._persist_session"):
            client.post("/api/sessions/ins-s/messages",
                        json={"role": "system", "content": "injected", "position": 1})
        assert _sessions["ins-s"]["messages"][1]["content"] == "injected"

    def test_insert_returns_index(self, client):
        _inject("ins-s", _MSGS)
        with patch("jarvis.api.server._persist_session"):
            resp = client.post("/api/sessions/ins-s/messages",
                               json={"role": "user", "content": "new"})
        assert resp.json()["index"] == 3  # appended at end

    def test_insert_increments_count(self, client):
        _inject("ins-s", _MSGS)
        with patch("jarvis.api.server._persist_session"):
            resp = client.post("/api/sessions/ins-s/messages",
                               json={"role": "user", "content": "x"})
        assert resp.json()["message_count"] == 4

    def test_insert_empty_content_422(self, client):
        _inject("ins-s", _MSGS)
        resp = client.post("/api/sessions/ins-s/messages", json={"role": "user", "content": ""})
        assert resp.status_code == 422

    def test_insert_unknown_session_404(self, client):
        resp = client.post("/api/sessions/missing-s/messages",
                           json={"role": "user", "content": "x"})
        assert resp.status_code == 404
