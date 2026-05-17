"""Tests for DatasetManager."""
from __future__ import annotations

import json
from pathlib import Path


def _make_record(question: str, answer: str = "answer") -> dict:
    return {"messages": [{"role": "user", "content": question},
                         {"role": "assistant", "content": answer}]}


class TestDatasetManager:
    def _dm(self):
        from jarvis.training.dataset_manager import DatasetManager
        return DatasetManager()

    def test_append_creates_file_and_writes_records(self, tmp_path):
        path = tmp_path / "dataset.jsonl"
        records = [_make_record("Q1?"), _make_record("Q2?")]
        self._dm().append(path, records)

        assert path.exists()
        lines = path.read_text().strip().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["messages"][0]["content"] == "Q1?"

    def test_append_is_additive(self, tmp_path):
        path = tmp_path / "dataset.jsonl"
        dm = self._dm()
        dm.append(path, [_make_record("Q1?")])
        dm.append(path, [_make_record("Q2?")])
        lines = path.read_text().strip().splitlines()
        assert len(lines) == 2

    def test_deduplicate_removes_exact_question_duplicates(self, tmp_path):
        path = tmp_path / "dataset.jsonl"
        records = [_make_record("Same Q?"), _make_record("Same Q?"), _make_record("Different Q?")]
        self._dm().append(path, records)

        removed = self._dm().deduplicate(path)
        assert removed == 1
        lines = path.read_text().strip().splitlines()
        assert len(lines) == 2

    def test_deduplicate_is_case_insensitive(self, tmp_path):
        path = tmp_path / "dataset.jsonl"
        records = [_make_record("What is RLHF?"), _make_record("what is rlhf?")]
        self._dm().append(path, records)
        removed = self._dm().deduplicate(path)
        assert removed == 1

    def test_deduplicate_returns_zero_for_missing_file(self, tmp_path):
        removed = self._dm().deduplicate(tmp_path / "missing.jsonl")
        assert removed == 0

    def test_split_creates_train_and_val_files(self, tmp_path):
        path = tmp_path / "dataset.jsonl"
        records = [_make_record(f"Q{i}?") for i in range(20)]
        self._dm().append(path, records)

        train_path, val_path = self._dm().split(path, val_ratio=0.2)
        train_lines = train_path.read_text().strip().splitlines()
        val_lines = val_path.read_text().strip().splitlines()

        assert len(train_lines) == 16
        assert len(val_lines) == 4
        assert len(train_lines) + len(val_lines) == 20

    def test_split_val_minimum_one(self, tmp_path):
        path = tmp_path / "dataset.jsonl"
        records = [_make_record("Only Q?")]
        self._dm().append(path, records)
        train_path, val_path = self._dm().split(path, val_ratio=0.1)
        val_lines = val_path.read_text().strip().splitlines()
        assert len(val_lines) >= 1

    def test_stats_returns_correct_counts(self, tmp_path):
        path = tmp_path / "dataset.jsonl"
        records = [_make_record("Short?", "Yes."), _make_record("Longer question?", "Longer answer.")]
        self._dm().append(path, records)

        stats = self._dm().stats(path)
        assert stats["count"] == 2
        assert stats["exists"] is True
        assert stats["avg_chars"] > 0
        assert stats["min_chars"] <= stats["avg_chars"] <= stats["max_chars"]

    def test_stats_for_missing_file(self, tmp_path):
        stats = self._dm().stats(tmp_path / "none.jsonl")
        assert stats["count"] == 0
        assert stats["exists"] is False
