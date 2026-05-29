"""Tests for POST /api/parallel-map/stream SSE endpoint."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _fake_settings(tmp_path: Path) -> MagicMock:
    s = MagicMock()
    s.anthropic_api_key = "test-key"
    s.tavily_api_key = "test-key"
    s.model = "llama3.2"
    s.fast_model = "llama3.2"
    s.max_tokens = 512
    s.max_search_calls = 5
    s.routing_strategy = "always_primary"
    s.allowed_commands = []
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
    s.agent_turn_timeout_seconds = 120
    s.tool_timeout_seconds = 60
    s.peer_port = 8001
    s.vision_model = "llava:13b"
    return s


@pytest.fixture
def sse_client(tmp_path: Path):
    settings = _fake_settings(tmp_path)
    settings.reports_dir.mkdir(parents=True, exist_ok=True)

    with (
        patch("jarvis.api.server.get_settings", return_value=settings),
        patch("jarvis.config.get_settings", return_value=settings),
        patch("jarvis.scheduler.core.start_scheduler"),
        patch("jarvis.scheduler.core.stop_scheduler"),
        patch("jarvis.tools.registry.build_registry", return_value=([], {})),
    ):
        from fastapi.testclient import TestClient
        from jarvis.api.server import app
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c


def _parse_sse(raw: bytes) -> list[dict]:
    """Parse SSE body into a list of data objects."""
    events = []
    for line in raw.decode().splitlines():
        if line.startswith("data: "):
            events.append(json.loads(line[6:]))
    return events


class TestParallelMapSSE:
    def _fake_run_turn(self, messages, on_chunk=None):
        topic = messages[0]["content"]
        return f"result for {topic[:20]}", messages

    def test_streams_one_event_per_topic(self, sse_client, mock_ollama) -> None:
        with patch("jarvis.agents.researcher.ResearcherAgent.run_turn", self._fake_run_turn):
            resp = sse_client.post("/api/parallel-map/stream", json={
                "task_template": "Research {topic} in depth.",
                "topics": ["RLHF", "transformers"],
                "synthesize": False,
            })

        assert resp.status_code == 200
        events = _parse_sse(resp.content)
        topic_events = [e for e in events if e["type"] == "topic"]
        assert len(topic_events) == 2
        topic_names = {e["topic"] for e in topic_events}
        assert topic_names == {"RLHF", "transformers"}

    def test_done_event_is_last(self, sse_client, mock_ollama) -> None:
        with patch("jarvis.agents.researcher.ResearcherAgent.run_turn", self._fake_run_turn):
            resp = sse_client.post("/api/parallel-map/stream", json={
                "task_template": "Explain {topic}.",
                "topics": ["alpha", "beta"],
                "synthesize": False,
            })

        events = _parse_sse(resp.content)
        assert events[-1]["type"] == "done"

    def test_synthesize_true_emits_synthesis_event(self, sse_client, mock_ollama) -> None:
        with patch("jarvis.agents.researcher.ResearcherAgent.run_turn", self._fake_run_turn):
            resp = sse_client.post("/api/parallel-map/stream", json={
                "task_template": "Research {topic}.",
                "topics": ["RLHF", "LoRA"],
                "synthesize": True,
            })

        events = _parse_sse(resp.content)
        synthesis_events = [e for e in events if e["type"] == "synthesis"]
        assert len(synthesis_events) == 1

    def test_synthesize_false_omits_synthesis_event(self, sse_client, mock_ollama) -> None:
        with patch("jarvis.agents.researcher.ResearcherAgent.run_turn", self._fake_run_turn):
            resp = sse_client.post("/api/parallel-map/stream", json={
                "task_template": "Research {topic}.",
                "topics": ["RLHF", "LoRA"],
                "synthesize": False,
            })

        events = _parse_sse(resp.content)
        synthesis_events = [e for e in events if e["type"] == "synthesis"]
        assert len(synthesis_events) == 0

    def test_unknown_agent_type_returns_422(self, sse_client) -> None:
        resp = sse_client.post("/api/parallel-map/stream", json={
            "task_template": "Research {topic}.",
            "topics": ["RLHF", "transformers"],
            "agent_type": "wizard",
        })
        assert resp.status_code == 422

    def test_missing_placeholder_passthrough(self, sse_client, mock_ollama) -> None:
        with patch("jarvis.agents.researcher.ResearcherAgent.run_turn", self._fake_run_turn):
            resp = sse_client.post("/api/parallel-map/stream", json={
                "task_template": "Research general AI topics.",
                "topics": ["RLHF", "transformers"],
                "synthesize": False,
            })
        # endpoint doesn't validate placeholder — that's the tool's job
        assert resp.status_code == 200

    def test_each_topic_result_contains_result_key(self, sse_client, mock_ollama) -> None:
        with patch("jarvis.agents.researcher.ResearcherAgent.run_turn", self._fake_run_turn):
            resp = sse_client.post("/api/parallel-map/stream", json={
                "task_template": "Explain {topic}.",
                "topics": ["RLHF", "constitutional AI"],
                "synthesize": False,
            })

        events = _parse_sse(resp.content)
        for e in events:
            if e["type"] == "topic":
                assert "result" in e
                assert "topic" in e
