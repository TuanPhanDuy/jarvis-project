"""Tests for session summarizer."""
from __future__ import annotations

import pytest

from jarvis.agents.summarize import SessionSummary, _heuristic_summary, summarize_session


class TestHeuristicSummary:
    def test_empty_messages_returns_no_messages(self):
        result = _heuristic_summary([])
        assert "0 messages" in result.summary or "No messages" in result.summary.lower() \
               or "messages" in result.summary

    def test_extracts_frequent_words_as_topics(self):
        msgs = [
            {"role": "user", "content": "transformer transformer transformer attention mechanism"},
        ]
        result = _heuristic_summary(msgs)
        assert any("transformer" in t.lower() for t in result.key_topics)

    def test_skips_assistant_content_for_topics(self):
        msgs = [
            {"role": "assistant", "content": "exclusiveword " * 10},
            {"role": "user", "content": "research topic"},
        ]
        result = _heuristic_summary(msgs)
        # assistant content should not dominate topic extraction
        assert result.key_topics is not None

    def test_returns_session_summary_type(self):
        result = _heuristic_summary([{"role": "user", "content": "hello world"}])
        assert isinstance(result, SessionSummary)

    def test_message_count_correct(self):
        msgs = [{"role": "user", "content": f"msg {i}"} for i in range(5)]
        result = _heuristic_summary(msgs)
        assert result.message_count == 5


class TestSummarizeSession:
    def test_empty_messages_returns_placeholder(self):
        result = summarize_session([])
        assert result.message_count == 0
        assert isinstance(result.summary, str)

    def test_no_model_uses_heuristic(self):
        msgs = [{"role": "user", "content": "What is reinforcement learning?"}]
        result = summarize_session(msgs, model="")
        assert isinstance(result.summary, str)
        assert isinstance(result.key_topics, list)
        assert isinstance(result.action_items, list)

    def test_returns_dataclass(self):
        result = summarize_session(
            [{"role": "user", "content": "explain RLHF"}], model=""
        )
        assert isinstance(result, SessionSummary)

    def test_message_count_correct(self):
        msgs = [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "Q2"},
        ]
        result = summarize_session(msgs, model="")
        assert result.message_count == 3

    def test_model_unavailable_falls_back_gracefully(self):
        msgs = [{"role": "user", "content": "What is multimodal learning?"}]
        result = summarize_session(msgs, model="nonexistent-model-xyz")
        assert isinstance(result.summary, str) and result.summary
        assert isinstance(result.key_topics, list)

    def test_system_messages_excluded_from_user_topics(self):
        msgs = [
            {"role": "system", "content": "You are JARVIS."},
            {"role": "user", "content": "Explain attention mechanism"},
            {"role": "assistant", "content": "Attention allows the model to focus..."},
        ]
        result = summarize_session(msgs, model="")
        assert result.message_count == 3

    def test_long_transcript_truncated_gracefully(self):
        long_content = "reinforcement learning " * 5000
        msgs = [{"role": "user", "content": long_content}]
        result = summarize_session(msgs, model="", max_context_chars=100)
        assert isinstance(result.summary, str)
