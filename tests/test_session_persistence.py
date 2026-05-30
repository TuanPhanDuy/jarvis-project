"""Tests for session persistence: save/load/history."""
from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from jarvis.memory.sessions import (
    delete_persisted_session,
    get_session_history,
    load_sessions,
    save_session,
)


@pytest.fixture()
def db(tmp_path: Path) -> Path:
    return tmp_path / "jarvis.db"


# ── Unit tests ────────────────────────────────────────────────────────────────

class TestSaveSession:
    def test_save_and_load_roundtrip(self, db):
        msgs = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]
        save_session(db, "s1", msgs, agent_type="PlannerAgent", user_id="alice")
        rows = load_sessions(db, ttl_minutes=60)
        assert len(rows) == 1
        assert rows[0]["session_id"] == "s1"
        assert rows[0]["messages"] == msgs

    def test_upsert_updates_messages(self, db):
        save_session(db, "s2", [{"role": "user", "content": "v1"}])
        save_session(db, "s2", [{"role": "user", "content": "v2"}, {"role": "assistant", "content": "ok"}])
        rows = load_sessions(db, ttl_minutes=60)
        assert len(rows) == 1
        assert rows[0]["messages"][0]["content"] == "v2"

    def test_save_never_raises_on_bad_path(self):
        save_session(Path("/nonexistent/path/db.sqlite"), "s3", [])

    def test_fork_of_stored(self, db):
        save_session(db, "fork-s", [], fork_of="source-s")
        rows = load_sessions(db, ttl_minutes=60)
        assert rows[0]["fork_of"] == "source-s"


class TestLoadSessions:
    def test_empty_db_returns_empty(self, db):
        assert load_sessions(db, ttl_minutes=60) == []

    def test_nonexistent_db_returns_empty(self, tmp_path):
        assert load_sessions(tmp_path / "missing.db", ttl_minutes=60) == []

    def test_expired_session_excluded(self, db):
        # Save with a very old updated_at
        import sqlite3
        from jarvis.memory.sessions import _conn
        conn = _conn(db)
        old_ts = time.time() - 7200  # 2 hours ago
        conn.execute(
            "INSERT INTO persisted_sessions (session_id, messages, created_at, updated_at)"
            " VALUES ('old-s', '[]', ?, ?)",
            (old_ts, old_ts),
        )
        conn.commit()
        conn.close()
        # TTL of 60 minutes should exclude a 2-hour-old session
        rows = load_sessions(db, ttl_minutes=60)
        assert not any(r["session_id"] == "old-s" for r in rows)

    def test_recent_session_included(self, db):
        save_session(db, "recent-s", [{"role": "user", "content": "hi"}])
        rows = load_sessions(db, ttl_minutes=60)
        assert any(r["session_id"] == "recent-s" for r in rows)

    def test_multiple_sessions_ordered_newest_first(self, db):
        save_session(db, "a", [])
        time.sleep(0.01)
        save_session(db, "b", [])
        rows = load_sessions(db, ttl_minutes=60)
        ids = [r["session_id"] for r in rows]
        assert ids.index("b") < ids.index("a")


class TestGetSessionHistory:
    def test_returns_messages_newest_first(self, db):
        msgs = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "second"},
            {"role": "user", "content": "third"},
        ]
        save_session(db, "h1", msgs)
        history = get_session_history(db, "h1", limit=10)
        assert history[0]["content"] == "third"

    def test_pagination_limit(self, db):
        msgs = [{"role": "user", "content": f"msg{i}"} for i in range(10)]
        save_session(db, "h2", msgs)
        page = get_session_history(db, "h2", limit=3)
        assert len(page) == 3

    def test_pagination_offset(self, db):
        msgs = [{"role": "user", "content": f"msg{i}"} for i in range(5)]
        save_session(db, "h3", msgs)
        page1 = get_session_history(db, "h3", limit=2, offset=0)
        page2 = get_session_history(db, "h3", limit=2, offset=2)
        assert page1 != page2

    def test_unknown_session_returns_empty(self, db):
        assert get_session_history(db, "no-such", limit=10) == []


class TestDeletePersistedSession:
    def test_delete_existing_returns_true(self, db):
        save_session(db, "del-s", [])
        assert delete_persisted_session(db, "del-s") is True

    def test_delete_missing_returns_false(self, db):
        assert delete_persisted_session(db, "ghost") is False

    def test_deleted_not_in_load(self, db):
        save_session(db, "gone", [])
        delete_persisted_session(db, "gone")
        rows = load_sessions(db, ttl_minutes=60)
        assert not any(r["session_id"] == "gone" for r in rows)


# ── API endpoint ──────────────────────────────────────────────────────────────

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


def _inject(session_id: str, messages: list | None = None):
    from jarvis.api.server import _sessions
    agent = MagicMock()
    agent.__class__.__name__ = "PlannerAgent"
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
        "user_id": "anonymous",
        "approval_gate": None,
    }


class TestHistoryEndpoint:
    def test_in_memory_session_returns_history(self, client):
        msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
        _inject("hist-api-1", msgs)
        resp = client.get("/api/sessions/hist-api-1/history")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)
        assert len(resp.json()) == 2

    def test_persisted_session_returns_history(self, client, tmp_path):
        msgs = [{"role": "user", "content": "persisted message"}]
        with patch("jarvis.api.server.get_settings") as mock_cfg:
            mock_cfg.return_value.reports_dir = tmp_path
            save_session(tmp_path / "jarvis.db", "pers-s", msgs)
            resp = client.get("/api/sessions/pers-s/history")
        assert resp.status_code == 200
        history = resp.json()
        assert any(m["content"] == "persisted message" for m in history)

    def test_unknown_session_404(self, client):
        resp = client.get("/api/sessions/totally-unknown-xyz/history")
        assert resp.status_code == 404

    def test_limit_param_respected(self, client):
        msgs = [{"role": "user", "content": f"msg{i}"} for i in range(10)]
        _inject("hist-limit", msgs)
        resp = client.get("/api/sessions/hist-limit/history?limit=3")
        assert resp.status_code == 200
        assert len(resp.json()) == 3
