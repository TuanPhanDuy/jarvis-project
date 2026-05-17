"""Tests for CriticAgent — critique parsing, score fallbacks, retry logic."""
from __future__ import annotations

from unittest.mock import patch

import structlog.testing

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


class TestParseCritiqueJson:
    def test_parse_json_valid(self):
        text = '{"score": 4, "issues": ["too brief"], "retry": false, "revised_task": null}'
        result = _parse_critique(text)
        assert result.score == 4
        assert result.issues == ["too brief"]
        assert result.should_retry is False
        assert result.revised_task is None

    def test_parse_json_with_retry(self):
        text = '{"score": 2, "issues": ["no code"], "retry": true, "revised_task": "Write Python code"}'
        result = _parse_critique(text)
        assert result.should_retry is True
        assert result.revised_task == "Write Python code"

    def test_parse_json_embedded_in_prose(self):
        text = 'My assessment:\n{"score": 3, "issues": [], "retry": false, "revised_task": null}\nDone.'
        result = _parse_critique(text)
        assert result.score == 3
        assert result.issues == []

    def test_parse_json_revised_task_with_colons(self):
        """Colons inside revised_task value are preserved (JSON handles this; KV parser would truncate)."""
        text = '{"score": 1, "issues": ["empty"], "retry": true, "revised_task": "Call func(x: int) -> str"}'
        result = _parse_critique(text)
        assert "func(x: int) -> str" in result.revised_task

    def test_parse_json_falls_back_to_kv_on_invalid_json(self):
        text = "SCORE: 5\nISSUES: none\nRETRY: no\nREVISED_TASK: none"
        result = _parse_critique(text)
        assert result.score == 5
        assert result.should_retry is False


class TestCriticLogging:
    def test_critique_logs_warning_on_exception(self):
        critic = build_critic("llama3.2", 512)
        with patch.object(critic, "run_turn", side_effect=RuntimeError("connection failed")):
            with structlog.testing.capture_logs() as cap:
                result = critic.critique("do a task", "some result")
        assert result.score == 3
        assert result.should_retry is False
        events = [e.get("event") for e in cap]
        assert "critic_failed" in events


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
