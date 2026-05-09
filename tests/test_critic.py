"""Tests for CriticAgent — critique parsing, score fallbacks, retry logic."""
from __future__ import annotations

from jarvis.agents.critic import CriticAgent, CritiqueResult, _parse_critique, build_critic


class TestParseCritique:
    def test_parse_valid_output(self):
        text = "SCORE: 8\nISSUES: none\nRETRY: no\nREVISED_TASK: none"
        result = _parse_critique(text)
        assert result.score == 8
        assert result.should_retry is False
        assert result.revised_task is None
        assert result.issues == []

    def test_parse_malformed_score_defaults_to_3(self):
        text = "SCORE: bad\nRETRY: no"
        result = _parse_critique(text)
        assert result.score == 3

    def test_parse_missing_retry_defaults_false(self):
        text = "SCORE: 7"
        result = _parse_critique(text)
        assert result.should_retry is False

    def test_parse_retry_yes_with_revised_task(self):
        text = "SCORE: 3\nISSUES: too vague\nRETRY: yes\nREVISED_TASK: do it better"
        result = _parse_critique(text)
        assert result.should_retry is True
        assert result.revised_task == "do it better"

    def test_parse_issues_comma_split(self):
        text = "SCORE: 5\nISSUES: missing detail, wrong format\nRETRY: no\nREVISED_TASK: none"
        result = _parse_critique(text)
        assert "missing detail" in result.issues
        assert "wrong format" in result.issues


class TestBuildCritic:
    def test_build_critic_produces_critic_agent(self):
        critic = build_critic("llama3.2", 1024)
        assert isinstance(critic, CriticAgent)

    def test_build_critic_has_empty_tools(self):
        critic = build_critic("llama3.2", 1024)
        assert critic._tool_schemas == []
        assert critic._tool_registry == {}

    def test_build_critic_caps_max_tokens_at_512(self):
        critic = build_critic("llama3.2", 8096)
        assert critic._max_tokens == 512
