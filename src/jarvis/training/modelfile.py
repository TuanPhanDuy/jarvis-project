"""Ollama Modelfile builder and model registration.

Generates a Modelfile from a GGUF path and registers it with Ollama
so the fine-tuned model is available via `ollama run jarvis-ft`.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import structlog

log = structlog.get_logger()

_SYSTEM_PROMPT = (
    "You are JARVIS, an AI research assistant specialising in frontier AI systems "
    "including transformers, RLHF, constitutional AI, multimodal systems, and memory "
    "architectures. You are precise, thorough, and cite your reasoning."
)

_MODELFILE_TEMPLATE = """\
FROM {gguf_path}

SYSTEM \"{system_prompt}\"

PARAMETER num_ctx 4096
PARAMETER temperature 0.7
PARAMETER top_p 0.9
"""


def build_modelfile(gguf_path: Path, system_prompt: str = _SYSTEM_PROMPT) -> str:
    """Return the Modelfile content string for the given GGUF."""
    return _MODELFILE_TEMPLATE.format(
        gguf_path=gguf_path.resolve(),
        system_prompt=system_prompt.replace('"', '\\"'),
    )


def register_model(gguf_path: Path, model_name: str, system_prompt: str = _SYSTEM_PROMPT) -> bool:
    """Write a Modelfile and register the fine-tuned model with Ollama.

    Returns True on success.
    """
    if not gguf_path.exists():
        log.error("gguf_not_found", path=str(gguf_path))
        return False

    content = build_modelfile(gguf_path, system_prompt)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".Modelfile", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(content)
        modelfile_path = tmp.name

    log.info("modelfile_written", path=modelfile_path, model=model_name)

    try:
        result = subprocess.run(
            ["ollama", "create", model_name, "-f", modelfile_path],
            capture_output=True,
            text=True,
        )
    finally:
        try:
            os.unlink(modelfile_path)
        except OSError:
            pass

    if result.returncode == 0:
        log.info("model_registered", model=model_name)
        return True

    log.error("model_registration_failed", model=model_name,
               stdout=result.stdout, stderr=result.stderr)
    return False
