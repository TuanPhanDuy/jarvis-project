"""Tests for ResearcherAgent using Ollama mock (no real model calls)."""
from __future__ import annotations

import pytest

from jarvis.agents.researcher import ResearcherAgent


class TestResearcherAgent:
    def test_system_prompt_contains_initial_quota(self, make_researcher) -> None:
        agent = make_researcher()
        prompt = agent.get_system_prompt()
        assert "20/20" in prompt

    def test_quota_decrements_on_web_search(self, make_researcher) -> None:
        agent = make_researcher()
        agent.on_tool_call("web_search")
        agent.on_tool_call("web_search")
        assert "18/20" in agent.get_system_prompt()

    def test_non_search_tool_does_not_decrement_quota(self, make_researcher) -> None:
        agent = make_researcher()
        agent.on_tool_call("save_report")
        agent.on_tool_call("read_url")
        assert "20/20" in agent.get_system_prompt()

    def test_get_messages_initially_empty(self, make_researcher) -> None:
        agent = make_researcher()
        assert agent.get_messages() == []

    def test_tool_filter_allows_web_search(self, make_researcher) -> None:
        agent = make_researcher()
        tool_names = {s["name"] for s in agent._tool_schemas}
        assert "web_search" in tool_names

    def test_tool_filter_blocks_run_command(self, make_researcher) -> None:
        agent = make_researcher()
        tool_names = {s["name"] for s in agent._tool_schemas}
        assert "run_command" not in tool_names

    def test_tool_filter_blocks_delegate_task(self, make_researcher) -> None:
        agent = make_researcher()
        tool_names = {s["name"] for s in agent._tool_schemas}
        assert "delegate_task" not in tool_names

    def test_run_turn_returns_string_and_updated_messages(self, make_researcher, mock_ollama) -> None:
        agent = make_researcher()
        messages = [{"role": "user", "content": "What is RLHF?"}]
        text, updated = agent.run_turn(messages)
        assert isinstance(text, str)
        assert len(updated) == 2

    def test_run_turn_appends_assistant_message(self, make_researcher, mock_ollama) -> None:
        agent = make_researcher()
        messages = [{"role": "user", "content": "Hello"}]
        _text, updated = agent.run_turn(messages)
        assert updated[-1]["role"] == "assistant"

    def test_search_quota_enforced_custom_limit(self, tmp_path, mock_ollama) -> None:
        from jarvis.tools.registry import build_registry
        schemas, registry = build_registry(reports_dir=tmp_path / "reports")
        agent = ResearcherAgent(
            model="llama3.2", max_tokens=512,
            tool_schemas=schemas, tool_registry=registry,
            max_search_calls=3,
        )
        agent.on_tool_call("web_search")
        agent.on_tool_call("web_search")
        agent.on_tool_call("web_search")
        assert "0/3" in agent.get_system_prompt()
