"""JSONL training dataset lifecycle management.

Handles appending, deduplication, train/val splitting, and statistics
for the JARVIS fine-tuning dataset.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path

import structlog

log = structlog.get_logger()


def _load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def _save_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _record_key(record: dict) -> str:
    """Stable deduplication key based on first user message content."""
    try:
        msgs = record.get("messages", [])
        user_msg = next((m["content"] for m in msgs if m.get("role") == "user"), "")
        return hashlib.sha256(user_msg.lower().strip().encode()).hexdigest()
    except Exception:
        return hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()


class DatasetManager:
    """Manage a JSONL fine-tuning dataset."""

    def append(self, path: Path, records: list[dict]) -> None:
        """Append records to the dataset file (creates it if missing)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def deduplicate(self, path: Path) -> int:
        """Remove duplicate records in-place. Returns count removed."""
        if not path.exists():
            return 0
        records = _load_jsonl(path)
        seen: set[str] = set()
        unique = []
        for rec in records:
            key = _record_key(rec)
            if key not in seen:
                seen.add(key)
                unique.append(rec)
        removed = len(records) - len(unique)
        if removed > 0:
            _save_jsonl(path, unique)
            log.info("dataset_deduped", removed=removed, kept=len(unique))
        return removed

    def split(
        self,
        path: Path,
        val_ratio: float = 0.1,
        seed: int = 42,
    ) -> tuple[Path, Path]:
        """Split dataset into train/val files. Returns (train_path, val_path)."""
        records = _load_jsonl(path)
        random.seed(seed)
        random.shuffle(records)
        val_n = max(1, math.ceil(len(records) * val_ratio))
        val_records = records[:val_n]
        train_records = records[val_n:]

        train_path = path.parent / "train.jsonl"
        val_path = path.parent / "val.jsonl"
        _save_jsonl(train_path, train_records)
        _save_jsonl(val_path, val_records)

        log.info("dataset_split", train=len(train_records), val=len(val_records))
        return train_path, val_path

    def stats(self, path: Path) -> dict:
        """Return summary statistics for the dataset."""
        if not path.exists():
            return {"count": 0, "path": str(path), "exists": False}
        records = _load_jsonl(path)
        if not records:
            return {"count": 0, "path": str(path), "exists": True}

        lengths = []
        for rec in records:
            total_chars = sum(
                len(m.get("content", ""))
                for m in rec.get("messages", [])
            )
            lengths.append(total_chars)

        return {
            "count": len(records),
            "avg_chars": int(sum(lengths) / len(lengths)),
            "min_chars": min(lengths),
            "max_chars": max(lengths),
            "path": str(path),
            "exists": True,
        }
