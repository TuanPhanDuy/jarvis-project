"""Shared pytest fixtures for the JARVIS test suite."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest


def _make_usage_mock(
    input_tokens: int = 10,
    output_tokens: int = 20,
    cache_read: int = 0,
    cache_write: int = 0,
) -> MagicMock:
    usage = MagicMock()
    usage.input_tokens = input_tokens
    usage.output_tokens = output_tokens
    usage.cache_read_input_tokens = cache_read
    usage.cache_creation_input_tokens = cache_write
    return usage


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "jarvis.db"


@pytest.fixture
def mock_anthropic_client() -> MagicMock:
    client = MagicMock()
    response = MagicMock()
    response.stop_reason = "end_turn"
    response.usage = _make_usage_mock()
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = "Hello from JARVIS"
    response.content = [text_block]
    client.messages.create.return_value = response
    return client


@pytest.fixture
def make_researcher(tmp_path: Path, mock_anthropic_client: MagicMock):
    """Factory fixture — call with optional path/client overrides."""
    from jarvis.agents.researcher import ResearcherAgent
    from jarvis.tools.registry import build_registry

    def _factory(
        path: Path | None = None,
        client: MagicMock | None = None,
    ) -> ResearcherAgent:
        reports_dir = (path or tmp_path) / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        schemas, registry = build_registry(
            tavily_api_key="fake-key",
            reports_dir=reports_dir,
        )
        return ResearcherAgent(
            client=client or mock_anthropic_client,
            model="claude-sonnet-4-6",
            max_tokens=1024,
            tool_schemas=schemas,
            tool_registry=registry,
        )

    return _factory


def make_mock_response(
    stop_reason: str = "end_turn",
    text: str = "Hello",
    tool_name: str | None = None,
    tool_id: str = "t1",
    tool_input: dict | None = None,
) -> MagicMock:
    """Build a fully-configured mock Anthropic response."""
    resp = MagicMock()
    resp.stop_reason = stop_reason
    resp.usage = _make_usage_mock()
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
