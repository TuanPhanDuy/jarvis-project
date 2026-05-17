"""Tests for step result auto-summarization."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from jarvis.agents.step_summarizer import (
    summarize_if_large,
    _MAX_INJECT_CHARS,
    _MIN_CHARS_TO_SUMMARIZE,
)


class TestSummarizeIfLarge:
    def test_short_result_returned_unchanged(self):
        short = "A short result."
        assert summarize_if_large(short, "task", "model") == short

    def test_exactly_at_limit_returned_unchanged(self):
        at_limit = "X" * _MAX_INJECT_CHARS
        assert summarize_if_large(at_limit, "task", "model") == at_limit

    def test_large_result_summarized_via_ollama(self):
        long_text = "A" * 3000
        mock_resp = MagicMock()
        mock_resp.message.content = "Concise summary of findings."
        with patch("ollama.chat", return_value=mock_resp):
            result = summarize_if_large(long_text, "research task", "llama3.2")
        assert result.startswith("[Summary]")
        assert "Concise summary" in result

    def test_ollama_failure_falls_back_to_truncation(self):
        long_text = "B" * 3000
        with patch("ollama.chat", side_effect=RuntimeError("timeout")):
            result = summarize_if_large(long_text, "task", "model")
        assert len(result) <= _MAX_INJECT_CHARS + 20  # allow for ellipsis
        assert "truncated" in result

    def test_empty_summary_falls_back_to_truncation(self):
        long_text = "C" * 3000
        mock_resp = MagicMock()
        mock_resp.message.content = ""
        with patch("ollama.chat", return_value=mock_resp):
            result = summarize_if_large(long_text, "task", "model")
        assert "truncated" in result

    def test_below_min_summarize_threshold_returned_unchanged(self):
        # Shorter than _MIN_CHARS_TO_SUMMARIZE but longer than _MAX_INJECT_CHARS
        # is an impossible state; test short text below both thresholds
        short = "X" * (_MIN_CHARS_TO_SUMMARIZE - 1)
        with patch("ollama.chat") as mock_chat:
            result = summarize_if_large(short, "task", "model")
        mock_chat.assert_not_called()
        assert result == short

    def test_step_description_passed_to_prompt(self):
        long_text = "D" * 3000
        mock_resp = MagicMock()
        mock_resp.message.content = "Summary."
        with patch("ollama.chat", return_value=mock_resp) as mock_chat:
            summarize_if_large(long_text, "research RLHF deeply", "model")
        call_kwargs = mock_chat.call_args
        prompt_content = call_kwargs[1]["messages"][0]["content"]
        assert "research RLHF deeply" in prompt_content
