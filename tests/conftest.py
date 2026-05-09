"""Shared pytest fixtures for the JARVIS test suite."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "jarvis.db"


@pytest.fixture
def mock_ollama(monkeypatch):
    """Patch ollama.chat so agent turns return 'ok' without hitting a real model."""
    class _Msg:
        content = "ok"
        tool_calls = None

    class _Resp:
        message = _Msg()
        prompt_eval_count = 5
        eval_count = 10

    monkeypatch.setattr("ollama.chat", lambda **kw: _Resp())
    return _Resp()


@pytest.fixture
def agent_factory(mock_ollama):
    """Build any BaseAgent subclass with empty tool_schemas and tool_registry."""
    def _build(AgentClass, model: str = "llama3.2", max_tokens: int = 512, **kwargs):
        return AgentClass(
            model=model,
            max_tokens=max_tokens,
            tool_schemas=kwargs.pop("tool_schemas", []),
            tool_registry=kwargs.pop("tool_registry", {}),
            **kwargs,
        )
    return _build


@pytest.fixture
def settings_override(monkeypatch, tmp_path):
    """Patch get_settings() to return isolated config pointing at tmp_path."""
    from jarvis.config import Settings

    fake = MagicMock(spec=Settings)
    fake.reports_dir = tmp_path / "reports"
    fake.reports_dir.mkdir(parents=True, exist_ok=True)
    fake.model = "llama3.2"
    fake.max_tokens = 512
    fake.tool_timeout_seconds = 10
    fake.auth_enabled = False
    monkeypatch.setattr("jarvis.config.get_settings", lambda: fake)
    monkeypatch.setattr("jarvis.agents.base_agent.get_settings", lambda: fake, raising=False)
    return fake


def make_mock_response(
    stop_reason: str = "end_turn",
    text: str = "Hello",
    tool_name: str | None = None,
    tool_id: str = "t1",
    tool_input: dict | None = None,
) -> MagicMock:
    """Compatibility helper — builds a mock Anthropic-style response for legacy tests."""
    resp = MagicMock()
    resp.stop_reason = stop_reason
    usage = MagicMock()
    usage.input_tokens = 10
    usage.output_tokens = 20
    usage.cache_read_input_tokens = 0
    usage.cache_creation_input_tokens = 0
    resp.usage = usage
    if stop_reason == "end_turn":
        block = MagicMock()
        block.type = "text"
        block.text = text
        resp.content = [block]
    else:
        block = MagicMock()
        block.type = "tool_use"
        block.id = tool_id
        block.name = tool_name or "save_report"
        block.input = tool_input or {}
        resp.content = [block]
    return resp


@pytest.fixture
def make_researcher(tmp_path: Path, mock_ollama):
    """Factory fixture — returns a ResearcherAgent backed by the Ollama mock."""
    from jarvis.agents.researcher import ResearcherAgent
    from jarvis.tools.registry import build_registry

    def _factory(path: Path | None = None) -> ResearcherAgent:
        reports_dir = (path or tmp_path) / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        schemas, registry = build_registry(reports_dir=reports_dir)
        return ResearcherAgent(
            model="llama3.2",
            max_tokens=1024,
            tool_schemas=schemas,
            tool_registry=registry,
        )

    return _factory
