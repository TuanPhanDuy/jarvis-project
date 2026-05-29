"""Synthetic training data generator.

Uses the local Ollama model to generate instruction-tuning Q&A pairs from
documents in the JARVIS reports directory. Output is JSONL compatible with
mlx-lm fine-tuning format — no external API key required.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import ollama
import structlog

from jarvis.config import get_settings

log = structlog.get_logger()

_SYSTEM_PROMPT = (
    "You are an expert AI researcher. Given a passage from an AI research document, "
    "generate high-quality instruction-following Q&A pairs. Each pair should:\n"
    "- Ask a specific, answerable question about the content\n"
    "- Provide a detailed, accurate answer grounded in the passage\n"
    "- Cover different aspects: definitions, mechanisms, tradeoffs, applications\n\n"
    "Output ONLY a JSON array of objects with keys 'question' and 'answer'. "
    "No preamble, no markdown fences, no explanation."
)

_USER_TEMPLATE = "Passage:\n{chunk}\n\nGenerate {n_pairs} Q&A pairs as a JSON array."


@dataclass
class QAPair:
    question: str
    answer: str
    source_file: str = ""


class TrainingDataGenerator:
    """Generate instruction-tuning pairs from research documents using local Ollama."""

    def __init__(self, reports_dir: Path, model: str = "") -> None:
        self._reports_dir = reports_dir
        self._model = model or get_settings().model

    def get_document_chunks(self, n: int = 50, chunk_size: int = 2000) -> list[tuple[str, str]]:
        """Return up to n (chunk_text, filename) pairs from markdown reports."""
        chunks: list[tuple[str, str]] = []
        for report in sorted(self._reports_dir.glob("*.md")):
            if len(chunks) >= n:
                break
            try:
                text = report.read_text(encoding="utf-8", errors="ignore")
                body_start = text.find("\n\n") + 2
                body = text[body_start:] if body_start > 2 else text
                for i in range(0, min(len(body), chunk_size * 4), chunk_size):
                    if len(chunks) >= n:
                        break
                    chunk = body[i : i + chunk_size].strip()
                    if len(chunk) > 200:
                        chunks.append((chunk, report.name))
            except Exception:
                continue
        return chunks

    def generate_pairs(self, chunk: str, source_file: str, n_pairs: int = 3) -> list[QAPair]:
        """Ask the local model to generate n_pairs Q&A pairs for a document chunk."""
        try:
            response = ollama.chat(
                model=self._model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": _USER_TEMPLATE.format(
                        chunk=chunk[:2000], n_pairs=n_pairs,
                    )},
                ],
            )
            raw = response["message"]["content"].strip()
            # Strip markdown fences the model might add
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            pairs_data = json.loads(raw)
            return [
                QAPair(question=p["question"], answer=p["answer"], source_file=source_file)
                for p in pairs_data
                if "question" in p and "answer" in p
            ]
        except Exception as exc:
            log.warning("generate_pairs_failed", source=source_file, error=str(exc))
            return []

    def run(self, out_path: Path, target_pairs: int = 500, pairs_per_chunk: int = 3) -> int:
        """Generate training pairs until target_pairs is reached. Appends to out_path (JSONL)."""
        out_path.parent.mkdir(parents=True, exist_ok=True)
        chunks_needed = (target_pairs + pairs_per_chunk - 1) // pairs_per_chunk
        chunks = self.get_document_chunks(n=chunks_needed)

        if not chunks:
            log.warning("no_chunks_found", reports_dir=str(self._reports_dir))
            return 0

        total = 0
        with out_path.open("a", encoding="utf-8") as f:
            for chunk_text, source_file in chunks:
                if total >= target_pairs:
                    break
                for pair in self.generate_pairs(chunk_text, source_file, n_pairs=pairs_per_chunk):
                    f.write(json.dumps({
                        "messages": [
                            {"role": "user", "content": pair.question},
                            {"role": "assistant", "content": pair.answer},
                        ]
                    }, ensure_ascii=False) + "\n")
                    total += 1
                log.info("pairs_generated", total=total, target=target_pairs)

        log.info("data_generation_complete", pairs=total, path=str(out_path))
        return total
