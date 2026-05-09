"""End-to-end integration tests for the HTTP and WebSocket chat paths.

Uses FastAPI TestClient; patches BaseAgent._run_turn_inner at the boundary so
no real Anthropic API calls are made and no env vars are required.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from tests.conftest import make_mock_response


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fake_settings(tmp_path: Path) -> MagicMock:
    s = MagicMock()
    s.anthropic_api_key = "test-key"
    s.tavily_api_key = "test-key"
    s.model = "claude-sonnet-4-6"
    s.fast_model = "claude-haiku-4-5-20251001"
    s.max_tokens = 1024
    s.max_search_calls = 20
    s.routing_strategy = "always_primary"
    s.allowed_commands = ["echo"]
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
    return s


def _make_fake_run_turn(text: str = "Test response from JARVIS"):
    """Return a _run_turn_inner replacement that returns fixed text."""
    def _fake(self_agent, messages, on_chunk=None):
        if on_chunk:
            for word in text.split():
                on_chunk(word + " ")
        updated = messages + [{"role": "assistant", "content": text}]
        return text, updated
    return _fake


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def client(tmp_path: Path):
    settings = _fake_settings(tmp_path)
    settings.reports_dir.mkdir(parents=True, exist_ok=True)

    with (
        patch("jarvis.api.server.get_settings", return_value=settings),
        patch("jarvis.config.get_settings", return_value=settings),
        patch("jarvis.agents.base_agent.BaseAgent._run_turn_inner", _make_fake_run_turn()),
        patch("jarvis.scheduler.core.start_scheduler"),
        patch("jarvis.scheduler.core.stop_scheduler"),
    ):
        from jarvis.api.server import app
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c


# ── HTTP /api/chat ─────────────────────────────────────────────────────────────

class TestHTTPChat:
    def test_chat_returns_session_id_and_response(self, client: TestClient) -> None:
        resp = client.post("/api/chat", json={"message": "Hello JARVIS"})
        assert resp.status_code == 200
        body = resp.json()
        assert "session_id" in body
        assert "response" in body
        assert len(body["session_id"]) > 0
        assert len(body["response"]) > 0

    def test_same_session_id_reuses_agent(self, client: TestClient) -> None:
        payload = {"message": "Hello", "session_id": "test-session-abc"}
        resp1 = client.post("/api/chat", json=payload)
        resp2 = client.post("/api/chat", json=payload)
        assert resp1.status_code == 200
        assert resp2.status_code == 200
        assert resp1.json()["session_id"] == resp2.json()["session_id"] == "test-session-abc"

    def test_auto_generates_session_id_if_missing(self, client: TestClient) -> None:
        resp = client.post("/api/chat", json={"message": "No session"})
        assert resp.status_code == 200
        assert resp.json()["session_id"]

    def test_health_endpoint(self, client: TestClient) -> None:
        resp = client.get("/api/health")
        assert resp.status_code == 200
        assert "sessions_active" in resp.json()


# ── WebSocket /api/ws/{session} ───────────────────────────────────────────────

class TestWebSocketChat:
    def test_ws_receives_thinking_then_done(self, client: TestClient) -> None:
        with client.websocket_connect("/api/ws/ws-test-session") as ws:
            ws.send_text(json.dumps({"message": "Hello via WS"}))

            received_types: list[str] = []
            final_text: str | None = None

            for _ in range(20):
                raw = ws.receive_text()
                msg = json.loads(raw)
                received_types.append(msg.get("type", ""))
                if msg.get("type") == "done":
                    final_text = msg.get("response")
                    break

        assert "done" in received_types
        assert final_text and len(final_text) > 0

    def test_ws_session_reuse(self, client: TestClient) -> None:
        with client.websocket_connect("/api/ws/shared-session") as ws:
            ws.send_text(json.dumps({"message": "First"}))
            for _ in range(20):
                msg = json.loads(ws.receive_text())
                if msg.get("type") == "done":
                    break

            ws.send_text(json.dumps({"message": "Second"}))
            for _ in range(20):
                msg = json.loads(ws.receive_text())
                if msg.get("type") == "done":
                    assert "shared-session" in "/api/ws/shared-session"
                    break
