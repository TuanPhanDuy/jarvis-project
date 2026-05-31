"""Tests for POST /api/chat/stream (SSE streaming chat)."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

_PATCH_BUDGET = "jarvis.api.budget.check_budget"
_PATCH_EPISODES = "jarvis.api.server._log_episodes"
_PATCH_SPEND = "jarvis.api.server._record_spend"


@pytest.fixture()
def client():
    import jarvis.api.server as _server
    # Reset auth state that may have been set by other tests (e.g. test_security.py)
    _server._require_auth = None
    from jarvis.api.server import app
    yield TestClient(app, raise_server_exceptions=False)
    _server._require_auth = None


def _parse_sse(raw: str) -> list[dict]:
    events = []
    for line in raw.splitlines():
        if line.startswith("data: "):
            events.append(json.loads(line[6:]))
    return events


def _make_mock_agent(reply: str = "Hello!") -> MagicMock:
    agent = MagicMock()
    agent.run_turn.return_value = (reply, [{"role": "assistant", "content": reply}])
    agent.get_usage_summary.return_value = {
        "input_tokens": 10, "output_tokens": 5,
        "cache_write_tokens": 0, "cache_read_tokens": 0,
        "estimated_cost_usd": 0.0,
    }
    agent._approval_gate = None
    agent._before_dispatch = MagicMock()
    return agent


class TestSseChat:
    def _post_stream(self, client, message: str = "hi", session_id: str | None = None):
        payload = {"message": message}
        if session_id:
            payload["session_id"] = session_id
        return client.post("/api/chat/stream", json=payload)

    def test_returns_event_stream_content_type(self, client):
        from jarvis.api.server import _sessions
        sid = "sse-ct-test"
        _sessions[sid] = {
            "agent": _make_mock_agent(),
            "messages": [],
            "user_id": "anonymous",
        }
        with patch(_PATCH_BUDGET), patch(_PATCH_EPISODES), patch(_PATCH_SPEND):
            resp = self._post_stream(client, session_id=sid)
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

    def test_done_event_present(self, client):
        from jarvis.api.server import _sessions
        sid = "sse-done-test"
        _sessions[sid] = {
            "agent": _make_mock_agent("test reply"),
            "messages": [],
            "user_id": "anonymous",
        }
        with patch(_PATCH_BUDGET), patch(_PATCH_EPISODES), patch(_PATCH_SPEND):
            resp = self._post_stream(client, session_id=sid)

        events = _parse_sse(resp.text)
        done_events = [e for e in events if e.get("type") == "done"]
        assert len(done_events) == 1
        assert done_events[0]["session_id"] == sid

    def test_done_event_contains_usage(self, client):
        from jarvis.api.server import _sessions
        sid = "sse-usage-test"
        _sessions[sid] = {
            "agent": _make_mock_agent(),
            "messages": [],
            "user_id": "anonymous",
        }
        with patch(_PATCH_BUDGET), patch(_PATCH_EPISODES), patch(_PATCH_SPEND):
            resp = self._post_stream(client, session_id=sid)

        events = _parse_sse(resp.text)
        done = next(e for e in events if e.get("type") == "done")
        assert "usage" in done
        assert "input_tokens" in done["usage"]

    def test_budget_exceeded_returns_error_event(self, client):
        from jarvis.api.server import _sessions
        from jarvis.api.budget import BudgetExceededError
        sid = "sse-budget-test"
        _sessions[sid] = {
            "agent": _make_mock_agent(),
            "messages": [],
            "user_id": "anonymous",
        }
        with patch(_PATCH_BUDGET, side_effect=BudgetExceededError("u1", 10.0, 15.0)):
            resp = self._post_stream(client, session_id=sid)

        assert resp.status_code == 200
        events = _parse_sse(resp.text)
        error_events = [e for e in events if e.get("type") == "error"]
        assert len(error_events) == 1
        assert "exceeded" in error_events[0]["message"].lower()

    def test_new_session_created_when_no_session_id(self, client):
        mock_agent = _make_mock_agent()
        with patch(_PATCH_BUDGET), patch(_PATCH_EPISODES), patch(_PATCH_SPEND), \
             patch("jarvis.api.server._build_agent_for_session", return_value=mock_agent):
            resp = client.post("/api/chat/stream", json={"message": "hello"})

        assert resp.status_code == 200
        events = _parse_sse(resp.text)
        done_events = [e for e in events if e.get("type") == "done"]
        assert len(done_events) == 1
        assert done_events[0]["session_id"]  # non-empty UUID


class TestSseUsageEvent:
    """Dedicated tests for the usage SSE event emitted before done."""

    def _stream(self, client, sid: str, agent=None):
        from jarvis.api.server import _sessions
        _sessions[sid] = {
            "agent": agent or _make_mock_agent(),
            "messages": [],
            "user_id": "anonymous",
        }
        with patch(_PATCH_BUDGET), patch(_PATCH_EPISODES), patch(_PATCH_SPEND):
            return _parse_sse(client.post("/api/chat/stream",
                                          json={"message": "hi", "session_id": sid}).text)

    def test_usage_event_emitted(self, client):
        events = self._stream(client, "usage-ev-1")
        usage_events = [e for e in events if e.get("type") == "usage"]
        assert len(usage_events) == 1

    def test_usage_event_has_required_fields(self, client):
        events = self._stream(client, "usage-ev-2")
        ev = next(e for e in events if e.get("type") == "usage")
        assert "input_tokens" in ev
        assert "output_tokens" in ev
        assert "cost_usd" in ev
        assert "latency_ms" in ev

    def test_usage_event_before_done(self, client):
        events = self._stream(client, "usage-ev-3")
        types = [e.get("type") for e in events]
        usage_idx = types.index("usage")
        done_idx = types.index("done")
        assert usage_idx < done_idx

    def test_usage_token_values_numeric(self, client):
        agent = _make_mock_agent()
        agent.get_usage_summary.return_value = {
            "input_tokens": 42, "output_tokens": 17,
            "cache_write_tokens": 0, "cache_read_tokens": 0,
            "estimated_cost_usd": 0.001,
        }
        events = self._stream(client, "usage-ev-4", agent=agent)
        ev = next(e for e in events if e.get("type") == "usage")
        assert ev["input_tokens"] == 42
        assert ev["output_tokens"] == 17
        assert ev["cost_usd"] == 0.001
        assert isinstance(ev["latency_ms"], float)
