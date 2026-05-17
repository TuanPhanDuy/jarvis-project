"""Fine-tuning orchestration via mlx-lm (Apple Silicon Metal).

Runs LoRA fine-tuning on a quantised Qwen2.5 model, then fuses the adapter
and exports a GGUF file for loading into Ollama.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import structlog

log = structlog.get_logger()

_DEFAULT_BASE_MODEL = "mlx-community/Qwen2.5-14B-Instruct-4bit"
_STEPS_PER_EPOCH = 100  # approximate; dataset size / batch_size


class Finetuner:
    """Orchestrate mlx-lm LoRA fine-tuning and GGUF export."""

    def __init__(
        self,
        base_model: str = _DEFAULT_BASE_MODEL,
        adapter_dir: Path | None = None,
    ) -> None:
        self._base_model = base_model
        self._adapter_dir = adapter_dir or Path("reports/training/adapters")

    def train(
        self,
        data_dir: Path,
        epochs: int = 3,
        lora_rank: int = 16,
        batch_size: int = 4,
    ) -> None:
        """Run mlx-lm LoRA fine-tuning. data_dir must contain train.jsonl and val.jsonl."""
        self._adapter_dir.mkdir(parents=True, exist_ok=True)
        iters = epochs * _STEPS_PER_EPOCH

        cmd = [
            sys.executable, "-m", "mlx_lm.lora",
            "--model", self._base_model,
            "--train",
            "--data", str(data_dir),
            "--adapter-path", str(self._adapter_dir),
            "--batch-size", str(batch_size),
            "--lora-layers", str(lora_rank),
            "--iters", str(iters),
        ]
        log.info("finetune_start", model=self._base_model, epochs=epochs, iters=iters)
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"mlx_lm.lora exited with code {result.returncode}")
        log.info("finetune_complete", adapter_dir=str(self._adapter_dir))

    def export_gguf(self, output_path: Path) -> None:
        """Fuse adapter weights and export as GGUF for Ollama loading."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable, "-m", "mlx_lm.fuse",
            "--model", self._base_model,
            "--adapter-path", str(self._adapter_dir),
            "--save-path", str(output_path),
            "--export-gguf",
        ]
        log.info("gguf_export_start", output=str(output_path))
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"mlx_lm.fuse exited with code {result.returncode}")
        log.info("gguf_export_complete", path=str(output_path))
