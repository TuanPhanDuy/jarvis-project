"""Tests for session replay / regression testing."""
from __future__ import annotations

import pytest

from jarvis.agents.replay import ReplayTurn, _extract_turns, replay_session


class TestExtractTurns:
    def test_empty_messages_returns_empty(self):
        assert _extract_turns([]) == []

    def test_single_pair(self):
        msgs = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        turns = _extract_turns(msgs)
        assert turns == [("Hello", "Hi there")]

    def test_multiple_pairs(self):
        msgs = [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "A2"},
        ]
        turns = _extract_turns(msgs)
        assert turns == [("Q1", "A1"), ("Q2", "A2")]

    def test_user_without_assistant_skipped(self):
        msgs = [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "Q2"},
        ]
        turns = _extract_turns(msgs)
        assert len(turns) == 1

    def test_system_messages_ignored(self):
        msgs = [
            {"role": "system", "content": "You are JARVIS."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        turns = _extract_turns(msgs)
        assert turns == [("Hello", "Hi")]

    def test_tool_use_messages_skipped(self):
        msgs = [
            {"role": "user", "content": "Search for X"},
            {"role": "tool", "content": "search result"},
            {"role": "assistant", "content": "Here is what I found"},
        ]
        turns = _extract_turns(msgs)
        assert turns == [("Search for X", "Here is what I found")]


class TestReplaySession:
    def _make_messages(self, pairs: list[tuple[str, str]]) -> list[dict]:
        msgs = []
        for user, asst in pairs:
            msgs.append({"role": "user", "content": user})
            msgs.append({"role": "assistant", "content": asst})
        return msgs

    def test_dry_run_returns_stubs(self):
        msgs = self._make_messages([("Hello", "Hi"), ("Bye", "Goodbye")])
        results = replay_session(msgs, run_turn_fn=lambda m: ("irrelevant", m), dry_run=True)
        assert all(t.replayed_response == "<stub>" for t in results)

    def test_dry_run_marks_all_changed(self):
        msgs = self._make_messages([("Hello", "Something different")])
        results = replay_session(msgs, run_turn_fn=lambda m: ("irrelevant", m), dry_run=True)
        assert results[0].changed is True  # "<stub>" != "Something different"

    def test_unchanged_response_marked_false(self):
        original = "Hello, I am JARVIS."
        msgs = self._make_messages([("Hi", original)])
        results = replay_session(msgs, run_turn_fn=lambda m: (original, m))
        assert results[0].changed is False

    def test_changed_response_marked_true(self):
        msgs = self._make_messages([("Hi", "old answer")])
        results = replay_session(msgs, run_turn_fn=lambda m: ("new answer", m))
        assert results[0].changed is True

    def test_correct_turn_count(self):
        msgs = self._make_messages([("Q1", "A1"), ("Q2", "A2"), ("Q3", "A3")])
        results = replay_session(msgs, run_turn_fn=lambda m: ("X", m))
        assert len(results) == 3

    def test_turn_indices_sequential(self):
        msgs = self._make_messages([("Q1", "A1"), ("Q2", "A2")])
        results = replay_session(msgs, run_turn_fn=lambda m: ("X", m))
        assert [t.turn_index for t in results] == [0, 1]

    def test_user_message_preserved(self):
        msgs = self._make_messages([("What is RLHF?", "RLHF means...")])
        results = replay_session(msgs, run_turn_fn=lambda m: ("other", m))
        assert results[0].user_message == "What is RLHF?"

    def test_original_response_preserved(self):
        msgs = self._make_messages([("Q", "original answer")])
        results = replay_session(msgs, run_turn_fn=lambda m: ("new answer", m))
        assert results[0].original_response == "original answer"

    def test_error_in_run_turn_captured(self):
        msgs = self._make_messages([("Q", "A")])
        def boom(m):
            raise RuntimeError("model offline")
        results = replay_session(msgs, run_turn_fn=boom)
        assert "<error:" in results[0].replayed_response

    def test_empty_session_returns_empty(self):
        assert replay_session([], run_turn_fn=lambda m: ("x", m)) == []

    def test_context_grows_across_turns(self):
        """Each turn's run_turn_fn receives growing context."""
        received_lengths = []
        def capture(m):
            received_lengths.append(len(m))
            return ("ok", m + [{"role": "assistant", "content": "ok"}])

        msgs = self._make_messages([("Q1", "A1"), ("Q2", "A2")])
        replay_session(msgs, run_turn_fn=capture)
        assert received_lengths[0] < received_lengths[1]
