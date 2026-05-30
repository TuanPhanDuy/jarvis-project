"""Tests for token-aware context compression in BaseAgent."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from jarvis.agents.base_agent import BaseAgent, _CHARS_PER_TOKEN, _COMPRESS_KEEP_RECENT


class _ConcreteAgent(BaseAgent):
    def get_system_prompt(self) -> str:
        return "test"


def _make_agent(budget: int = 200) -> _ConcreteAgent:
    with patch("jarvis.agents.base_agent.ModelRouter"):
        agent = _ConcreteAgent.__new__(_ConcreteAgent)
        agent._model = "test"
        agent._max_tokens = 512
        agent._tool_schemas = []
        agent._tool_registry = {}
        agent._approval_gate = None
        agent._session_id = ""
        agent._user_id = None
        agent._prompt_tokens = 0
        agent._completion_tokens = 0
        agent._turn_tool_calls = []
        agent._router = MagicMock()
        return agent


def _make_messages(count: int, chars_each: int = 10) -> list[dict]:
    return [{"role": "user" if i % 2 == 0 else "assistant", "content": "x" * chars_each}
            for i in range(count)]


class TestEstimateTokens:
    def test_empty(self):
        assert _ConcreteAgent._estimate_tokens([]) == 0

    def test_single_message(self):
        msgs = [{"role": "user", "content": "a" * 400}]
        assert _ConcreteAgent._estimate_tokens(msgs) == 100

    def test_multi_message(self):
        msgs = [{"role": "user", "content": "a" * 200},
                {"role": "assistant", "content": "b" * 200}]
        assert _ConcreteAgent._estimate_tokens(msgs) == 100

    def test_missing_content(self):
        # missing key → "" (0 chars); None → str(None) = "None" (4 chars = 1 token)
        msgs = [{"role": "user"}, {"role": "assistant", "content": None}]
        assert _ConcreteAgent._estimate_tokens(msgs) == 1


class TestContextBudgetToken:
    def test_reads_from_settings(self):
        agent = _make_agent()
        with patch("jarvis.config.get_settings") as mock_settings:
            mock_settings.return_value.context_budget_tokens = 1234
            assert agent._context_budget_tokens() == 1234

    def test_fallback_on_error(self):
        agent = _make_agent()
        with patch("jarvis.config.get_settings", side_effect=RuntimeError("no config")):
            assert agent._context_budget_tokens() == 4096


class TestCompressHistory:
    def test_no_compression_below_budget(self):
        agent = _make_agent()
        # 5 messages × 10 chars = 50 chars = ~12 tokens, well under default 4096
        msgs = _make_messages(5, chars_each=10)
        with patch.object(agent, "_context_budget_tokens", return_value=4096):
            result = agent._compress_history(msgs)
        assert result == msgs

    def test_compression_triggered_above_budget(self):
        agent = _make_agent()
        # 20 messages × 100 chars = 2000 chars = 500 tokens
        msgs = _make_messages(20, chars_each=100)
        summary_msg = {"role": "system", "content": "[Prior conversation summary]: summary text"}

        with patch.object(agent, "_context_budget_tokens", return_value=400), \
             patch("jarvis.agents.base_agent.ollama") as mock_ollama:
            mock_resp = MagicMock()
            mock_resp.message.content = "summary text"
            mock_ollama.chat.return_value = mock_resp
            result = agent._compress_history(msgs)

        # Should keep only the recent messages + a summary prefix
        assert len(result) == _COMPRESS_KEEP_RECENT + 1
        assert result[0]["role"] == "system"
        assert "[Prior conversation summary]" in result[0]["content"]

    def test_compression_fallback_on_ollama_error(self):
        agent = _make_agent()
        msgs = _make_messages(20, chars_each=100)

        with patch.object(agent, "_context_budget_tokens", return_value=400), \
             patch("jarvis.agents.base_agent.ollama") as mock_ollama:
            mock_ollama.chat.side_effect = RuntimeError("model offline")
            result = agent._compress_history(msgs)

        # Fallback: truncate without summary
        assert len(result) == _COMPRESS_KEEP_RECENT

    def test_budget_exactly_at_threshold_no_compression(self):
        agent = _make_agent()
        # 4 messages × 400 chars = 1600 chars = 400 tokens
        msgs = _make_messages(4, chars_each=400)
        with patch.object(agent, "_context_budget_tokens", return_value=400):
            result = agent._compress_history(msgs)
        assert result == msgs

    def test_compression_preserves_recent_messages(self):
        agent = _make_agent()
        msgs = _make_messages(20, chars_each=100)

        with patch.object(agent, "_context_budget_tokens", return_value=400), \
             patch("jarvis.agents.base_agent.ollama") as mock_ollama:
            mock_resp = MagicMock()
            mock_resp.message.content = "summary"
            mock_ollama.chat.return_value = mock_resp
            result = agent._compress_history(msgs)

        # The recent messages must be exactly the last _COMPRESS_KEEP_RECENT of original
        assert result[1:] == msgs[-_COMPRESS_KEEP_RECENT:]
