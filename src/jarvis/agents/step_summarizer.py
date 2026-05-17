"""Compress large plan step results before injecting as downstream context.

When a step produces thousands of characters, injecting the full output as
context for the next step wastes tokens and degrades quality. This module
summarizes large results via a short LLM call and falls back to truncation
if Ollama is unavailable.
"""
from __future__ import annotations

import structlog

log = structlog.get_logger()

_MAX_INJECT_CHARS = 2000
_MIN_CHARS_TO_SUMMARIZE = 500


def summarize_if_large(result: str, step_description: str, model: str) -> str:
    """Return the result unchanged if short; otherwise return a concise summary."""
    if len(result) <= _MAX_INJECT_CHARS:
        return result
    if len(result) < _MIN_CHARS_TO_SUMMARIZE:
        return result
    try:
        import ollama
        prompt = (
            f"Summarize the key findings from this task output in 3–5 sentences.\n"
            f"Task: {step_description}\n\n"
            f"Output:\n{result[:4000]}"
        )
        resp = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.1, "num_predict": 300},
        )
        summary = (resp.message.content or "").strip()
        if summary:
            log.info("step_result_summarized",
                     step=step_description[:60],
                     original_len=len(result),
                     summary_len=len(summary))
            return f"[Summary] {summary}"
    except Exception as exc:
        log.debug("step_summarization_skipped", error=str(exc))
    return result[:_MAX_INJECT_CHARS] + "… [truncated]"
