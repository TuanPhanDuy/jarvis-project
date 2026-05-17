"""Tests for TrainingDataGenerator."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch


class TestTrainingDataGenerator:
    def _make_generator(self, tmp_path: Path):
        from jarvis.training.data_generator import TrainingDataGenerator
        return TrainingDataGenerator(reports_dir=tmp_path, api_key="test-key")

    def test_get_document_chunks_reads_markdown_files(self, tmp_path):
        (tmp_path / "paper1.md").write_text("# Paper 1\n\n" + "body text " * 100)
        (tmp_path / "paper2.md").write_text("# Paper 2\n\n" + "other text " * 100)

        gen = self._make_generator(tmp_path)
        chunks = gen.get_document_chunks(n=4)
        assert len(chunks) >= 2
        assert all(isinstance(c, tuple) and len(c) == 2 for c in chunks)

    def test_get_document_chunks_skips_short_content(self, tmp_path):
        (tmp_path / "tiny.md").write_text("# Tiny\n\nshort")
        gen = self._make_generator(tmp_path)
        chunks = gen.get_document_chunks(n=10)
        assert all(len(c[0]) > 200 for c in chunks)

    def test_generate_pairs_calls_anthropic_and_parses_json(self, tmp_path):
        from jarvis.training.data_generator import TrainingDataGenerator

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=json.dumps([
            {"question": "What is RLHF?", "answer": "RLHF is..."},
            {"question": "Why use PPO?", "answer": "PPO is stable..."},
        ]))]

        gen = TrainingDataGenerator(reports_dir=tmp_path, api_key="test-key")
        with patch("anthropic.Anthropic") as MockClient:
            MockClient.return_value.messages.create.return_value = mock_response
            pairs = gen.generate_pairs("RLHF trains language models...", "paper.md", n_pairs=2)

        assert len(pairs) == 2
        assert pairs[0].question == "What is RLHF?"
        assert pairs[1].answer == "PPO is stable..."

    def test_generate_pairs_returns_empty_on_api_error(self, tmp_path):
        from jarvis.training.data_generator import TrainingDataGenerator

        gen = TrainingDataGenerator(reports_dir=tmp_path, api_key="test-key")
        with patch("anthropic.Anthropic") as MockClient:
            MockClient.return_value.messages.create.side_effect = RuntimeError("api down")
            pairs = gen.generate_pairs("some text", "paper.md")

        assert pairs == []

    def test_generate_pairs_skips_malformed_entries(self, tmp_path):
        from jarvis.training.data_generator import TrainingDataGenerator

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=json.dumps([
            {"question": "Good Q?", "answer": "Good A"},
            {"bad_key": "no question or answer"},
        ]))]

        gen = TrainingDataGenerator(reports_dir=tmp_path, api_key="test-key")
        with patch("anthropic.Anthropic") as MockClient:
            MockClient.return_value.messages.create.return_value = mock_response
            pairs = gen.generate_pairs("text", "paper.md")

        assert len(pairs) == 1
        assert pairs[0].question == "Good Q?"

    def test_run_writes_jsonl_to_output_path(self, tmp_path):
        from jarvis.training.data_generator import QAPair, TrainingDataGenerator

        (tmp_path / "paper.md").write_text("# Title\n\n" + "body text " * 200)

        gen = TrainingDataGenerator(reports_dir=tmp_path, api_key="test-key")
        fake_pairs = [QAPair(question="Q?", answer="A.", source_file="paper.md")]

        out = tmp_path / "training" / "dataset.jsonl"
        with patch.object(gen, "generate_pairs", return_value=fake_pairs):
            count = gen.run(out_path=out, target_pairs=2, pairs_per_chunk=1)

        assert count >= 1
        assert out.exists()
        line = json.loads(out.read_text().strip().splitlines()[0])
        assert "messages" in line
        assert line["messages"][0]["role"] == "user"

    def test_init_raises_without_api_key(self, tmp_path):
        from jarvis.training.data_generator import TrainingDataGenerator
        import pytest
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
                TrainingDataGenerator(reports_dir=tmp_path, api_key="")
