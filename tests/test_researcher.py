"""Tests for the ResearcherAgent using a mock Anthropic client."""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from jarvis.agents.researcher import ResearcherAgent
from jarvis.tools.registry import build_registry
from tests.conftest import make_mock_response


def _make_usage() -> MagicMock:
    u = MagicMock()
    u.input_tokens = 10
    u.output_tokens = 20
    u.cache_read_input_tokens = 0
    u.cache_creation_input_tokens = 0
    return u


def _make_agent(tmp_path: Path) -> ResearcherAgent:
    client = MagicMock()
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    schemas, registry = build_registry(
        tavily_api_key="fake-key",
        reports_dir=reports_dir,
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
        mock_response = make_mock_response(stop_reason="end_turn", text="Hello from JARVIS")
        agent._client.messages.create.return_value = mock_response

        messages = [{"role": "user", "content": "What is RLHF?"}]
        text, updated = agent.run_turn(messages)

        assert text == "Hello from JARVIS"
        assert len(updated) == 2  # user + assistant

    def test_run_turn_tool_use_then_end_turn(self, tmp_path: Path) -> None:
        agent = _make_agent(tmp_path)

        tool_response = make_mock_response(
            stop_reason="tool_use",
            tool_name="save_report",
            tool_id="tool_1",
            tool_input={"title": "T", "content": "c", "topic": "t"},
        )
        final_response = make_mock_response(stop_reason="end_turn", text="Report saved.")

        agent._client.messages.create.side_effect = [tool_response, final_response]

        messages = [{"role": "user", "content": "Save a report about RLHF."}]
        text, updated = agent.run_turn(messages)

        assert text == "Report saved."
        assert agent._client.messages.create.call_count == 2

    def test_get_messages_before_conversation(self, tmp_path: Path) -> None:
        agent = _make_agent(tmp_path)
        msgs = agent.get_messages()
        assert msgs == []

    def test_search_quota_enforced(self, tmp_path: Path) -> None:
        agent = _make_agent(tmp_path)
        agent._max_search_calls = 2
        agent.on_tool_call("web_search")
        agent.on_tool_call("web_search")
        prompt = agent.get_system_prompt()
        assert "0/2" in prompt

    def test_streaming_on_chunk_callback(self, tmp_path: Path) -> None:
        agent = _make_agent(tmp_path)

        chunks_received: list[str] = []

        stream_ctx = MagicMock()
        stream_ctx.__enter__ = MagicMock(return_value=stream_ctx)
        stream_ctx.__exit__ = MagicMock(return_value=False)
        stream_ctx.text_stream = iter(["Hello", " world"])

        final = make_mock_response(stop_reason="end_turn", text="Hello world")
        stream_ctx.get_final_message.return_value = final
        agent._client.messages.stream.return_value = stream_ctx

        messages = [{"role": "user", "content": "Say hello."}]
        text, updated = agent.run_turn(messages, on_chunk=chunks_received.append)

        assert chunks_received == ["Hello", " world"]
        assert "Hello world" in text
        assert agent._client.messages.stream.called
