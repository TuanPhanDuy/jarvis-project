"""Prompt template loader.

Reads .md files from the prompts/ directory and optionally substitutes
{variable} placeholders using Python's str.format_map().

Usage:
    from jarvis.prompts.loader import load_prompt
    text = load_prompt("researcher")
    text = load_prompt("planner", agent_list="researcher, coder, qa")
"""
from __future__ import annotations

from pathlib import Path

_PROMPTS_DIR = Path(__file__).parent


def load_prompt(name: str, **variables: str) -> str:
    """Load a prompt template by name (without the .md extension).

    Args:
        name: Template filename without extension, e.g. "researcher".
        **variables: Optional substitutions for {placeholder} tokens in the template.

    Returns:
        The prompt string with variables substituted.

    Raises:
        FileNotFoundError: If no template with that name exists.
    """
    path = _PROMPTS_DIR / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if variables:
        text = text.format_map(variables)
    return text
