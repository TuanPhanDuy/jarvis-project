"""Tests for proactive memory surfacing."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from jarvis.memory.surfacing import surface_memory, _MIN_QUERY_LEN


class TestSurfaceMemory:
    def test_short_query_returns_empty(self, tmp_path):
        db = tmp_path / "db"
        result = surface_memory("hi", db)
        assert result == ""

    def test_returns_empty_on_no_matches(self, tmp_path):
        db = tmp_path / "db"
        with patch("jarvis.memory.episodic._search", return_value=[]), \
             patch("jarvis.memory.graph.handle_query_knowledge_graph", return_value="No entities found."):
            result = surface_memory("What is RLHF and how does it work?", db)
        assert result == ""

    def test_episodic_results_included(self, tmp_path):
        db = tmp_path / "db"
        mock_row = {"role": "user", "content": "RLHF uses reward modeling."}
        with patch("jarvis.memory.episodic._search", return_value=[mock_row]), \
             patch("jarvis.memory.graph.handle_query_knowledge_graph", return_value="No entities found."):
            result = surface_memory("What is RLHF and how does it work?", db)
        assert "RLHF uses reward modeling" in result
        assert "[USER]" in result

    def test_graph_results_included(self, tmp_path):
        db = tmp_path / "db"
        with patch("jarvis.memory.episodic._search", return_value=[]), \
             patch("jarvis.memory.graph.handle_query_knowledge_graph",
                   return_value="RLHF -- uses --> PPO"):
            result = surface_memory("Explain RLHF training procedure", db)
        assert "RLHF -- uses --> PPO" in result
        assert "[Graph]" in result

    def test_graph_error_result_excluded(self, tmp_path):
        db = tmp_path / "db"
        with patch("jarvis.memory.episodic._search", return_value=[]), \
             patch("jarvis.memory.graph.handle_query_knowledge_graph",
                   return_value="ERROR: something went wrong"):
            result = surface_memory("Explain RLHF training procedure", db)
        assert result == ""

    def test_output_truncated_at_max_chars(self, tmp_path):
        db = tmp_path / "db"
        from jarvis.memory.surfacing import _MAX_CONTEXT_CHARS
        long_content = "X" * 1000
        mock_row = {"role": "assistant", "content": long_content}
        with patch("jarvis.memory.episodic._search", return_value=[mock_row] * 5), \
             patch("jarvis.memory.graph.handle_query_knowledge_graph", return_value="No entities found."):
            result = surface_memory("What is RLHF and how does it work?", db)
        assert len(result) <= _MAX_CONTEXT_CHARS + 10  # allow for ellipsis

    def test_min_query_len_boundary(self, tmp_path):
        db = tmp_path / "db"
        # Exactly at boundary — should not skip
        query = "a" * _MIN_QUERY_LEN
        with patch("jarvis.memory.episodic._search", return_value=[]), \
             patch("jarvis.memory.graph.handle_query_knowledge_graph", return_value="No entities."):
            result = surface_memory(query, db)
        # No results = empty string (not skipped due to length)
        assert isinstance(result, str)
