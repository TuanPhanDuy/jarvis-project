"""Tests for POST /api/chat/batch."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

_PATCH_BUDGET = "jarvis.api.budget.check_budget"
_PATCH_SPEND = "jarvis.api.server._record_spend"
_PATCH_PERSIST = "jarvis.api.server._persist_session"
_PATCH_EPISODES = "jarvis.api.server._log_episodes"


@pytest.fixture()
def client():
    import jarvis.api.server as _server
    _server._require_auth = None
    from jarvis.api.server import app
    yield TestClient(app, raise_server_exceptions=False)
    _server._require_auth = None


def _mock_agent(reply: str = "ok") -> MagicMock:
    agent = MagicMock()
    agent.run_turn.return_value = (reply, [{"role": "assistant", "content": reply}])
    agent.get_usage_summary.return_value = {
        "input_tokens": 5, "output_tokens": 3,
        "cache_write_tokens": 0, "cache_read_tokens": 0,
        "estimated_cost_usd": 0.0,
    }
    agent._approval_gate = None
    agent._before_dispatch = MagicMock()
    return agent


class TestBatchChat:
    def _post(self, client, requests, max_concurrent=None):
        body = {"requests": requests}
        if max_concurrent is not None:
            body["max_concurrent"] = max_concurrent
        return client.post("/api/chat/batch", json=body)

    def test_empty_requests_returns_422(self, client):
        resp = self._post(client, [])
        assert resp.status_code == 422

    def test_missing_requests_key_returns_422(self, client):
        resp = client.post("/api/chat/batch", json={})
        assert resp.status_code == 422

    def test_single_request_succeeds(self, client):
        from jarvis.api.server import _sessions
        sid = "batch-single"
        _sessions[sid] = {
            "agent": _mock_agent("hello"),
            "messages": [],
            "user_id": "anonymous",
        }
        with patch(_PATCH_BUDGET), patch(_PATCH_SPEND), patch(_PATCH_PERSIST):
            resp = self._post(client, [{"session_id": sid, "message": "hi"}])
        assert resp.status_code == 200
        results = resp.json()
        assert len(results) == 1
        assert results[0]["session_id"] == sid
        assert results[0]["response"] == "hello"

    def test_multiple_requests_all_returned(self, client):
        from jarvis.api.server import _sessions
        sids = [f"batch-multi-{i}" for i in range(3)]
        for sid in sids:
            _sessions[sid] = {
                "agent": _mock_agent(f"reply-{sid}"),
                "messages": [],
                "user_id": "anonymous",
            }
        requests = [{"session_id": sid, "message": "test"} for sid in sids]
        with patch(_PATCH_BUDGET), patch(_PATCH_SPEND), patch(_PATCH_PERSIST):
            resp = self._post(client, requests)
        assert resp.status_code == 200
        results = resp.json()
        assert len(results) == 3
        returned_sids = {r["session_id"] for r in results}
        assert returned_sids == set(sids)

    def test_result_has_usage_field(self, client):
        from jarvis.api.server import _sessions
        sid = "batch-usage"
        _sessions[sid] = {
            "agent": _mock_agent(),
            "messages": [],
            "user_id": "anonymous",
        }
        with patch(_PATCH_BUDGET), patch(_PATCH_SPEND), patch(_PATCH_PERSIST):
            resp = self._post(client, [{"session_id": sid, "message": "hi"}])
        result = resp.json()[0]
        assert "usage" in result
        assert "input_tokens" in result["usage"]

    def test_budget_exceeded_returns_error_not_500(self, client):
        from jarvis.api.server import _sessions
        from jarvis.api.budget import BudgetExceededError
        sid = "batch-budget"
        _sessions[sid] = {
            "agent": _mock_agent(),
            "messages": [],
            "user_id": "anonymous",
        }
        with patch(_PATCH_BUDGET, side_effect=BudgetExceededError("u", 5.0, 10.0)):
            resp = self._post(client, [{"session_id": sid, "message": "hi"}])
        assert resp.status_code == 200
        results = resp.json()
        assert "error" in results[0]

    def test_missing_message_returns_error_in_result(self, client):
        with patch(_PATCH_BUDGET), patch(_PATCH_SPEND), patch(_PATCH_PERSIST):
            resp = self._post(client, [{"message": ""}])
        assert resp.status_code == 200
        assert "error" in resp.json()[0]

    def test_max_concurrent_respected(self, client):
        """Sanity check: max_concurrent=1 still processes all items."""
        from jarvis.api.server import _sessions
        sids = [f"batch-seq-{i}" for i in range(2)]
        for sid in sids:
            _sessions[sid] = {
                "agent": _mock_agent(),
                "messages": [],
                "user_id": "anonymous",
            }
        requests = [{"session_id": sid, "message": "hi"} for sid in sids]
        with patch(_PATCH_BUDGET), patch(_PATCH_SPEND), patch(_PATCH_PERSIST):
            resp = self._post(client, requests, max_concurrent=1)
        assert len(resp.json()) == 2
