"""Tests for TrainingDataGenerator (Ollama-backed, no external API key required)."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch


def _ollama_response(pairs: list[dict]) -> dict:
    return {"message": {"content": json.dumps(pairs)}}


class TestTrainingDataGenerator:
    def _make_generator(self, tmp_path: Path):
        from jarvis.training.data_generator import TrainingDataGenerator
        return TrainingDataGenerator(reports_dir=tmp_path, model="qwen2.5:3b")

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

    def test_generate_pairs_calls_ollama_and_parses_json(self, tmp_path):
        gen = self._make_generator(tmp_path)
        mock_resp = _ollama_response([
            {"question": "What is RLHF?", "answer": "RLHF is..."},
            {"question": "Why use PPO?", "answer": "PPO is stable..."},
        ])
        with patch("ollama.chat", return_value=mock_resp):
            pairs = gen.generate_pairs("RLHF trains language models...", "paper.md", n_pairs=2)

        assert len(pairs) == 2
        assert pairs[0].question == "What is RLHF?"
        assert pairs[1].answer == "PPO is stable..."

    def test_generate_pairs_strips_markdown_fences(self, tmp_path):
        gen = self._make_generator(tmp_path)
        inner = json.dumps([{"question": "Q?", "answer": "A."}])
        fenced = f"```json\n{inner}\n```"
        with patch("ollama.chat", return_value={"message": {"content": fenced}}):
            pairs = gen.generate_pairs("text", "paper.md")
        assert len(pairs) == 1

    def test_generate_pairs_returns_empty_on_error(self, tmp_path):
        gen = self._make_generator(tmp_path)
        with patch("ollama.chat", side_effect=RuntimeError("ollama down")):
            pairs = gen.generate_pairs("some text", "paper.md")
        assert pairs == []

    def test_generate_pairs_skips_malformed_entries(self, tmp_path):
        gen = self._make_generator(tmp_path)
        mock_resp = _ollama_response([
            {"question": "Good Q?", "answer": "Good A"},
            {"bad_key": "no question or answer"},
        ])
        with patch("ollama.chat", return_value=mock_resp):
            pairs = gen.generate_pairs("text", "paper.md")
        assert len(pairs) == 1
        assert pairs[0].question == "Good Q?"

    def test_run_writes_jsonl_to_output_path(self, tmp_path):
        from jarvis.training.data_generator import QAPair

        (tmp_path / "paper.md").write_text("# Title\n\n" + "body text " * 200)

        gen = self._make_generator(tmp_path)
        fake_pairs = [QAPair(question="Q?", answer="A.", source_file="paper.md")]
        out = tmp_path / "training" / "dataset.jsonl"

        with patch.object(gen, "generate_pairs", return_value=fake_pairs):
            count = gen.run(out_path=out, target_pairs=2, pairs_per_chunk=1)

        assert count >= 1
        assert out.exists()
        line = json.loads(out.read_text().strip().splitlines()[0])
        assert "messages" in line
        assert line["messages"][0]["role"] == "user"

    def test_run_returns_zero_when_no_chunks(self, tmp_path):
        gen = self._make_generator(tmp_path)
        out = tmp_path / "dataset.jsonl"
        count = gen.run(out_path=out, target_pairs=10)
        assert count == 0
        assert not out.exists()
