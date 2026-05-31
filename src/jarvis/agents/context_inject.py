"""Cross-session context injection.

Pulls relevant message content from one or more source sessions and condenses
it into a labelled context block that can be prepended to a target session's
next turn.

This lets an agent carry forward insights from prior research sessions without
loading full conversation histories into the active context.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class InjectedContext:
    context_block: str
    injected_chars: int
    source_count: int


def _extract_text(messages: list[dict], max_chars: int) -> str:
    """Collect user+assistant text from a message list, up to max_chars."""
    parts: list[str] = []
    total = 0
    for m in messages:
        if m.get("role") not in ("user", "assistant"):
            continue
        content = m.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                b.get("text", "") for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            )
        text = str(content).strip()
        if not text:
            continue
        remaining = max_chars - total
        if remaining <= 0:
            break
        chunk = text[:remaining]
        parts.append(chunk)
        total += len(chunk)
    return "\n".join(parts)


def build_context_block(
    sources: list[tuple[str, list[dict]]],
    label: str = "Injected context from prior sessions",
    max_chars_per_source: int = 2000,
) -> InjectedContext:
    """Build a context block from multiple source sessions.

    Args:
        sources:               List of (session_id, messages) tuples.
        label:                 Heading for the injected block.
        max_chars_per_source:  Maximum characters to extract per source session.

    Returns:
        InjectedContext with the assembled block and metrics.
    """
    if not sources:
        return InjectedContext("", 0, 0)

    sections: list[str] = [f"[{label}]"]
    total_chars = 0
    contributing = 0

    for session_id, messages in sources:
        text = _extract_text(messages, max_chars_per_source)
        if not text:
            continue
        sections.append(f"\n--- From session {session_id[:8]} ---\n{text}")
        total_chars += len(text)
        contributing += 1

    if total_chars == 0:
        return InjectedContext("", 0, 0)

    block = "\n".join(sections)
    return InjectedContext(
        context_block=block,
        injected_chars=total_chars,
        source_count=contributing,
    )


def inject_into_messages(
    messages: list[dict],
    context_block: str,
) -> list[dict]:
    """Prepend the context block as a system message to the messages list.

    Returns a new list — does not mutate the input.
    """
    if not context_block:
        return messages
    system_msg = {"role": "system", "content": context_block}
    return [system_msg] + list(messages)
