"""Tests for structured output: run_turn_structured and POST /api/chat/structured."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ── BaseAgent.run_turn_structured ─────────────────────────────────────────────

class _ConcreteAgent:
    """Minimal stand-in so we can call run_turn_structured directly."""
    def get_system_prompt(self): return "test system"
    _model = "test"
    _max_tokens = 512
    _prompt_tokens = 0
    _completion_tokens = 0

    def __init__(self):
        self._router = MagicMock()
        self._router.select.return_value = "test"

    # Borrow the real method
    from jarvis.agents.base_agent import BaseAgent
    run_turn_structured = BaseAgent.run_turn_structured
    _compress_history = BaseAgent._compress_history
    _surface_memory_context = BaseAgent._surface_memory_context
    _context_budget_tokens = BaseAgent._context_budget_tokens
    _settings_flag = BaseAgent._settings_flag
    _estimate_tokens = staticmethod(BaseAgent._estimate_tokens)
    _agent_type_key = BaseAgent._agent_type_key


class TestRunTurnStructured:
    def _agent(self, json_response: dict | str) -> _ConcreteAgent:
        agent = _ConcreteAgent()
        raw = json.dumps(json_response) if isinstance(json_response, dict) else json_response
        mock_resp = MagicMock()
        mock_resp.message.content = raw
        mock_resp.prompt_eval_count = 5
        mock_resp.eval_count = 3
        with patch("jarvis.agents.base_agent.ollama") as mock_ollama:
            mock_ollama.chat.return_value = mock_resp
            agent._mock_ollama = mock_ollama
            agent._mock_resp = mock_resp
        return agent

    def test_returns_parsed_dict(self):
        agent = _ConcreteAgent()
        payload = {"name": "JARVIS", "version": 1}
        mock_resp = MagicMock()
        mock_resp.message.content = json.dumps(payload)
        mock_resp.prompt_eval_count = 10
        mock_resp.eval_count = 5
        with patch("jarvis.agents.base_agent.ollama") as mock_ollama:
            mock_ollama.chat.return_value = mock_resp
            result, updated = agent.run_turn_structured(
                [{"role": "user", "content": "tell me about yourself"}],
                {"type": "object", "properties": {"name": {"type": "string"}}},
            )
        assert result == payload
        assert isinstance(updated, list)

    def test_invalid_json_raises_value_error(self):
        agent = _ConcreteAgent()
        mock_resp = MagicMock()
        mock_resp.message.content = "not json at all"
        mock_resp.prompt_eval_count = 0
        mock_resp.eval_count = 0
        with patch("jarvis.agents.base_agent.ollama") as mock_ollama:
            mock_ollama.chat.return_value = mock_resp
            with pytest.raises(ValueError, match="valid JSON"):
                agent.run_turn_structured([], {"type": "object"})

    def test_format_json_passed_to_ollama(self):
        agent = _ConcreteAgent()
        mock_resp = MagicMock()
        mock_resp.message.content = '{"ok": true}'
        mock_resp.prompt_eval_count = 0
        mock_resp.eval_count = 0
        with patch("jarvis.agents.base_agent.ollama") as mock_ollama:
            mock_ollama.chat.return_value = mock_resp
            agent.run_turn_structured([], {"type": "object"})
            call_kwargs = mock_ollama.chat.call_args[1]
        assert call_kwargs.get("format") == "json"

    def test_updated_messages_appended(self):
        agent = _ConcreteAgent()
        msgs = [{"role": "user", "content": "extract info"}]
        mock_resp = MagicMock()
        mock_resp.message.content = '{"extracted": true}'
        mock_resp.prompt_eval_count = 0
        mock_resp.eval_count = 0
        with patch("jarvis.agents.base_agent.ollama") as mock_ollama:
            mock_ollama.chat.return_value = mock_resp
            _, updated = agent.run_turn_structured(msgs, {"type": "object"})
        assert updated[-1]["role"] == "assistant"

    def test_format_fallback_on_ollama_error(self):
        agent = _ConcreteAgent()
        ok_resp = MagicMock()
        ok_resp.message.content = '{"fallback": true}'
        ok_resp.prompt_eval_count = 0
        ok_resp.eval_count = 0

        call_count = [0]
        def side_effect(**kwargs):
            call_count[0] += 1
            if kwargs.get("format") == "json":
                raise Exception("format not supported")
            return ok_resp

        with patch("jarvis.agents.base_agent.ollama") as mock_ollama:
            mock_ollama.chat.side_effect = side_effect
            result, _ = agent.run_turn_structured([], {"type": "object"})
        assert result == {"fallback": True}
        assert call_count[0] == 2  # first with format, then without


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


def _mock_agent_structured(result: dict) -> MagicMock:
    agent = MagicMock()
    agent.run_turn_structured.return_value = (result, [{"role": "assistant", "content": str(result)}])
    agent.get_usage_summary.return_value = {
        "input_tokens": 5, "output_tokens": 3,
        "cache_write_tokens": 0, "cache_read_tokens": 0,
        "estimated_cost_usd": 0.0,
    }
    agent._approval_gate = None
    agent._before_dispatch = MagicMock()
    return agent


class TestStructuredEndpoint:
    def _post(self, client, message: str, schema: dict, session_id: str | None = None):
        payload = {"message": message, "json_schema": schema}
        if session_id:
            payload["session_id"] = session_id
        return client.post("/api/chat/structured", json=payload)

    def test_returns_200_with_result(self, client):
        from jarvis.api.server import _sessions
        sid = "struct-test-1"
        _sessions[sid] = {
            "agent": _mock_agent_structured({"answer": 42}),
            "messages": [],
            "user_id": "anonymous",
        }
        with patch("jarvis.api.budget.check_budget"), \
             patch("jarvis.api.server._log_episodes"), \
             patch("jarvis.api.server._record_spend"):
            resp = self._post(client, "give me a number", {"type": "object"}, session_id=sid)
        assert resp.status_code == 200
        body = resp.json()
        assert body["result"] == {"answer": 42}
        assert body["session_id"] == sid

    def test_returns_422_when_model_returns_bad_json(self, client):
        from jarvis.api.server import _sessions
        from jarvis.api.budget import BudgetExceededError
        sid = "struct-test-bad"
        agent = MagicMock()
        agent.run_turn_structured.side_effect = ValueError("Model did not return valid JSON")
        agent._approval_gate = None
        agent._before_dispatch = MagicMock()
        _sessions[sid] = {"agent": agent, "messages": [], "user_id": "anonymous"}
        with patch("jarvis.api.budget.check_budget"), \
             patch("jarvis.api.server._log_episodes"), \
             patch("jarvis.api.server._record_spend"):
            resp = self._post(client, "bad", {}, session_id=sid)
        assert resp.status_code == 422

    def test_budget_exceeded_returns_402(self, client):
        from jarvis.api.server import _sessions
        from jarvis.api.budget import BudgetExceededError
        sid = "struct-budget"
        _sessions[sid] = {
            "agent": _mock_agent_structured({}),
            "messages": [],
            "user_id": "anonymous",
        }
        with patch("jarvis.api.budget.check_budget",
                   side_effect=BudgetExceededError("u", 10.0, 20.0)):
            resp = self._post(client, "hi", {}, session_id=sid)
        assert resp.status_code == 402

    def test_new_session_auto_created(self, client):
        mock_agent = _mock_agent_structured({"created": True})
        with patch("jarvis.api.budget.check_budget"), \
             patch("jarvis.api.server._log_episodes"), \
             patch("jarvis.api.server._record_spend"), \
             patch("jarvis.api.server._build_agent_for_session", return_value=mock_agent):
            resp = client.post("/api/chat/structured", json={
                "message": "hello", "json_schema": {"type": "object"},
            })
        assert resp.status_code == 200
        assert resp.json()["session_id"]
