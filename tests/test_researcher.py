"""Tests for the ResearcherAgent using a mock Anthropic client."""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from jarvis.agents.researcher import ResearcherAgent
from jarvis.tools.registry import build_registry


def _make_agent(tmp_path: Path) -> ResearcherAgent:
    client = MagicMock()
    schemas, registry = build_registry(
        tavily_api_key="fake-key",
        reports_dir=tmp_path / "reports",
    )
    return ResearcherAgent(
        client=client,
        model="claude-sonnet-4-6",
        max_tokens=1024,
        tool_schemas=schemas,
        tool_registry=registry,
    )


class TestResearcherAgent:
    def test_system_prompt_contains_quota(self, tmp_path: Path) -> None:
        agent = _make_agent(tmp_path)
        prompt = agent.get_system_prompt()
        assert "20/20" in prompt

    def test_search_quota_decrements(self, tmp_path: Path) -> None:
        agent = _make_agent(tmp_path)
        agent.on_tool_call("web_search")
        agent.on_tool_call("web_search")
        prompt = agent.get_system_prompt()
        assert "18/20" in prompt

    def test_non_search_tool_does_not_decrement(self, tmp_path: Path) -> None:
        agent = _make_agent(tmp_path)
        agent.on_tool_call("save_report")
        prompt = agent.get_system_prompt()
        assert "20/20" in prompt

    def test_run_turn_end_turn(self, tmp_path: Path) -> None:
        agent = _make_agent(tmp_path)

        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "Hello from JARVIS"

        mock_response = MagicMock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [text_block]

        agent._client.messages.create.return_value = mock_response

        messages = [{"role": "user", "content": "What is RLHF?"}]
        text, updated = agent.run_turn(messages)

        assert text == "Hello from JARVIS"
        assert len(updated) == 2  # user + assistant

    def test_run_turn_tool_use_then_end_turn(self, tmp_path: Path) -> None:
        agent = _make_agent(tmp_path)

        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.id = "tool_1"
        tool_block.name = "save_report"
        tool_block.input = {"title": "T", "content": "c", "topic": "t"}

        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "Report saved."

        tool_response = MagicMock()
        tool_response.stop_reason = "tool_use"
        tool_response.content = [tool_block]

        final_response = MagicMock()
        final_response.stop_reason = "end_turn"
        final_response.content = [text_block]

        agent._client.messages.create.side_effect = [tool_response, final_response]

        messages = [{"role": "user", "content": "Save a report about RLHF."}]
        text, updated = agent.run_turn(messages)

        assert text == "Report saved."
        assert agent._client.messages.create.call_count == 2
