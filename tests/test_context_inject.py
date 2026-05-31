"""Tests for cross-session context injection."""
from __future__ import annotations

import pytest

from jarvis.agents.context_inject import (
    InjectedContext,
    _extract_text,
    build_context_block,
    inject_into_messages,
)


class TestExtractText:
    def test_extracts_user_and_assistant(self):
        msgs = [
            {"role": "user", "content": "What is RLHF?"},
            {"role": "assistant", "content": "RLHF stands for..."},
        ]
        text = _extract_text(msgs, max_chars=1000)
        assert "RLHF" in text
        assert "stands for" in text

    def test_skips_system_and_tool_messages(self):
        msgs = [
            {"role": "system", "content": "SYSTEM SECRET"},
            {"role": "tool", "content": "tool result"},
            {"role": "user", "content": "user query"},
        ]
        text = _extract_text(msgs, max_chars=1000)
        assert "SYSTEM SECRET" not in text
        assert "user query" in text

    def test_respects_max_chars(self):
        msgs = [{"role": "user", "content": "a" * 500}]
        text = _extract_text(msgs, max_chars=100)
        assert len(text) <= 100

    def test_empty_messages_returns_empty(self):
        assert _extract_text([], max_chars=1000) == ""

    def test_list_content_blocks_extracted(self):
        msgs = [{"role": "assistant", "content": [
            {"type": "text", "text": "important insight"}
        ]}]
        text = _extract_text(msgs, max_chars=1000)
        assert "important insight" in text


class TestBuildContextBlock:
    def _msgs(self, content: str) -> list[dict]:
        return [{"role": "user", "content": content}]

    def test_empty_sources_returns_empty(self):
        result = build_context_block([])
        assert result.injected_chars == 0
        assert result.context_block == ""
        assert result.source_count == 0

    def test_single_source_included(self):
        sources = [("sess-abc", self._msgs("Transformer architecture details"))]
        result = build_context_block(sources)
        assert "Transformer architecture" in result.context_block
        assert result.source_count == 1

    def test_label_in_block(self):
        sources = [("s1", self._msgs("content"))]
        result = build_context_block(sources, label="My custom label")
        assert "My custom label" in result.context_block

    def test_session_id_truncated_in_block(self):
        sources = [("abcdef1234567890", self._msgs("content"))]
        result = build_context_block(sources)
        assert "abcdef12" in result.context_block  # first 8 chars

    def test_multiple_sources_both_included(self):
        sources = [
            ("s1", self._msgs("transformer research")),
            ("s2", self._msgs("RLHF discussion")),
        ]
        result = build_context_block(sources)
        assert "transformer" in result.context_block
        assert "RLHF" in result.context_block
        assert result.source_count == 2

    def test_empty_messages_excluded_from_count(self):
        sources = [
            ("s1", []),  # no user/assistant messages
            ("s2", self._msgs("real content")),
        ]
        result = build_context_block(sources)
        assert result.source_count == 1  # only s2 had content

    def test_injected_chars_positive(self):
        sources = [("s1", self._msgs("hello world"))]
        result = build_context_block(sources)
        assert result.injected_chars > 0

    def test_returns_dataclass(self):
        result = build_context_block([])
        assert isinstance(result, InjectedContext)


class TestInjectIntoMessages:
    def test_prepends_system_message(self):
        messages = [{"role": "user", "content": "hi"}]
        result = inject_into_messages(messages, "context block")
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "context block"
        assert result[1] == messages[0]

    def test_empty_block_returns_unchanged(self):
        messages = [{"role": "user", "content": "hi"}]
        result = inject_into_messages(messages, "")
        assert result == messages

    def test_does_not_mutate_input(self):
        messages = [{"role": "user", "content": "hi"}]
        original_len = len(messages)
        inject_into_messages(messages, "context")
        assert len(messages) == original_len

    def test_empty_messages_with_context(self):
        result = inject_into_messages([], "context block")
        assert len(result) == 1
        assert result[0]["role"] == "system"
