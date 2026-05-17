"""Tests for automatic knowledge graph extraction."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from jarvis.agents.graph_extractor import _parse_extraction, extract_graph_from_text


class TestParseExtraction:
    def test_parses_entity_lines(self):
        text = "ENTITY|RLHF|technique\nENTITY|PPO|algorithm"
        entities, rels = _parse_extraction(text)
        assert len(entities) == 2
        assert entities[0] == {"name": "RLHF", "type": "technique"}
        assert entities[1] == {"name": "PPO", "type": "algorithm"}

    def test_parses_relationship_lines(self):
        text = "REL|RLHF|uses|PPO\nREL|RLHF|developed_by|OpenAI"
        entities, rels = _parse_extraction(text)
        assert len(rels) == 2
        assert rels[0] == {"from": "RLHF", "relation": "uses", "to": "PPO"}

    def test_mixed_lines_parsed_correctly(self):
        text = "ENTITY|RLHF|concept\nREL|RLHF|uses|PPO\nSOME OTHER TEXT"
        entities, rels = _parse_extraction(text)
        assert len(entities) == 1
        assert len(rels) == 1

    def test_empty_text_returns_empty(self):
        entities, rels = _parse_extraction("")
        assert entities == []
        assert rels == []

    def test_malformed_entity_skipped(self):
        text = "ENTITY|  |concept\nENTITY|Valid|technique"
        entities, _ = _parse_extraction(text)
        assert len(entities) == 1
        assert entities[0]["name"] == "Valid"

    def test_malformed_rel_skipped(self):
        text = "REL|from||to\nREL|A|uses|B"
        _, rels = _parse_extraction(text)
        assert len(rels) == 1
        assert rels[0]["from"] == "A"


class TestExtractGraphFromText:
    def test_short_text_skips_extraction(self, tmp_path):
        count = extract_graph_from_text("Too short.", tmp_path / "db", "model")
        assert count == 0

    def test_extraction_writes_to_graph(self, tmp_path):
        mock_resp = MagicMock()
        mock_resp.message.content = "ENTITY|RLHF|technique\nREL|RLHF|uses|PPO"
        with patch("ollama.chat", return_value=mock_resp), \
             patch("jarvis.memory.graph.handle_update_knowledge_graph", return_value="OK") as mock_update:
            count = extract_graph_from_text("A" * 200, tmp_path / "db", "model", "user-1")
        mock_update.assert_called_once()
        assert count == 2  # 1 entity + 1 rel

    def test_ollama_failure_returns_zero(self, tmp_path):
        with patch("ollama.chat", side_effect=RuntimeError("down")):
            count = extract_graph_from_text("A" * 200, tmp_path / "db", "model")
        assert count == 0

    def test_empty_extraction_returns_zero(self, tmp_path):
        mock_resp = MagicMock()
        mock_resp.message.content = "nothing useful here"
        with patch("ollama.chat", return_value=mock_resp):
            count = extract_graph_from_text("A" * 200, tmp_path / "db", "model")
        assert count == 0
