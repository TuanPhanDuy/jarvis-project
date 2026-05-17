"""Tests for self-reflection loop and confidence-gated escalation."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from jarvis.agents.base_agent import BaseAgent, _HEDGE_PHRASES


def _make_agent():
    class ConcreteAgent(BaseAgent):
        def get_system_prompt(self):
            return "You are JARVIS."
    return ConcreteAgent(
        model="test-model",
        max_tokens=512,
        tool_schemas=[],
        tool_registry={},
    )


class TestDetectHedges:
    def test_detects_i_think(self):
        assert BaseAgent._detect_hedges("I think this is correct.")

    def test_detects_might_be(self):
        assert BaseAgent._detect_hedges("This might be the answer.")

    def test_detects_not_sure(self):
        assert BaseAgent._detect_hedges("I'm not sure about this claim.")

    def test_confident_response_not_flagged(self):
        assert not BaseAgent._detect_hedges("RLHF uses PPO to optimize reward signals.")

    def test_case_insensitive(self):
        assert BaseAgent._detect_hedges("I'M NOT SURE this is right.")

    def test_all_phrases_in_constant(self):
        for phrase in _HEDGE_PHRASES:
            assert BaseAgent._detect_hedges(f"This answer is {phrase} valid.")


class TestReflect:
    def test_lgtm_returns_original(self):
        agent = _make_agent()
        with patch("ollama.chat") as mock_chat:
            mock_chat.return_value = MagicMock(message=MagicMock(content="LGTM"))
            result = agent._reflect("A" * 200)
        assert result == "A" * 200

    def test_revision_replaces_original(self):
        agent = _make_agent()
        with patch("ollama.chat") as mock_chat:
            mock_chat.return_value = MagicMock(message=MagicMock(content="A better, improved response."))
            result = agent._reflect("short original" * 20)
        assert result == "A better, improved response."

    def test_short_response_skipped(self):
        agent = _make_agent()
        with patch("ollama.chat") as mock_chat:
            result = agent._reflect("Too short.")
        mock_chat.assert_not_called()
        assert result == "Too short."

    def test_ollama_failure_returns_original(self):
        agent = _make_agent()
        original = "X" * 200
        with patch("ollama.chat", side_effect=RuntimeError("timeout")):
            result = agent._reflect(original)
        assert result == original


class TestSettingsFlag:
    def test_reflection_disabled_by_default(self):
        agent = _make_agent()
        with patch("jarvis.config.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(reflection_enabled=False)
            assert agent._settings_flag("reflection_enabled", False) is False

    def test_confidence_gate_enabled_by_default(self):
        agent = _make_agent()
        with patch("jarvis.config.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(confidence_gate_enabled=True)
            assert agent._settings_flag("confidence_gate_enabled", True) is True
