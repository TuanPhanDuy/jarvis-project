"""Synthetic training data generator.

Uses Claude Haiku with prompt caching to generate instruction-tuning Q&A pairs
from documents stored in the JARVIS reports directory. Output is JSONL, one
conversation object per line, compatible with mlx-lm fine-tuning format.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import structlog

log = structlog.get_logger()

_SYSTEM_PROMPT = (
    "You are an expert AI researcher. Given a passage from an AI research document, "
    "generate high-quality instruction-following Q&A pairs. Each pair should:\n"
    "- Ask a specific, answerable question about the content\n"
    "- Provide a detailed, accurate answer grounded in the passage\n"
    "- Cover different aspects: definitions, mechanisms, tradeoffs, applications\n\n"
    "Output ONLY a JSON array of objects with keys 'question' and 'answer'. "
    "No preamble, no markdown fences."
)

_USER_TEMPLATE = (
    "Passage:\n{chunk}\n\n"
    "Generate {n_pairs} Q&A pairs as a JSON array."
)


@dataclass
class QAPair:
    question: str
    answer: str
    source_file: str = ""


class TrainingDataGenerator:
    """Generate instruction-tuning pairs from ChromaDB-indexed research documents."""

    def __init__(self, reports_dir: Path, api_key: str = "") -> None:
        self._reports_dir = reports_dir
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not self._api_key:
            raise ValueError("ANTHROPIC_API_KEY must be set for training data generation")

    def get_document_chunks(self, n: int = 50, chunk_size: int = 2000) -> list[tuple[str, str]]:
        """Return up to n (chunk_text, filename) pairs from markdown reports."""
        chunks: list[tuple[str, str]] = []
        reports = sorted(self._reports_dir.glob("*.md"))
        for report in reports:
            if len(chunks) >= n:
                break
            try:
                text = report.read_text(encoding="utf-8", errors="ignore")
                # Skip the metadata header lines
                body_start = text.find("\n\n") + 2
                body = text[body_start:] if body_start > 2 else text
                # Chunk into non-overlapping windows
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
        """Call Claude Haiku to generate n_pairs Q&A pairs for a document chunk."""
        import anthropic

        client = anthropic.Anthropic(api_key=self._api_key)
        try:
            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=1024,
                system=[{"type": "text", "text": _SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
                messages=[{
                    "role": "user",
                    "content": _USER_TEMPLATE.format(chunk=chunk[:2000], n_pairs=n_pairs),
                }],
            )
            raw = response.content[0].text.strip()
            pairs_data = json.loads(raw)
            return [
                QAPair(
                    question=p["question"],
                    answer=p["answer"],
                    source_file=source_file,
                )
                for p in pairs_data
                if "question" in p and "answer" in p
            ]
        except Exception as exc:
            log.warning("generate_pairs_failed", source=source_file, error=str(exc))
            return []

    def run(
        self,
        out_path: Path,
        target_pairs: int = 500,
        pairs_per_chunk: int = 3,
    ) -> int:
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
                pairs = self.generate_pairs(chunk_text, source_file, n_pairs=pairs_per_chunk)
                for pair in pairs:
                    record = {
                        "messages": [
                            {"role": "user", "content": pair.question},
                            {"role": "assistant", "content": pair.answer},
                        ]
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    total += 1
                log.info("pairs_generated", total=total, target=target_pairs)

        log.info("data_generation_complete", pairs=total, path=str(out_path))
        return total
