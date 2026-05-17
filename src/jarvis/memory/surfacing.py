"""Proactive memory surfacing — inject relevant prior context into agent turns.

Called before an agent processes a user message. Searches episodic memory
and knowledge graph for relevant prior results and returns a formatted context
block to prepend to the user turn. Only injects when meaningful results are found.
"""
from __future__ import annotations

from pathlib import Path

import structlog

log = structlog.get_logger()

_MIN_QUERY_LEN = 10   # don't surface for trivially short queries
_MAX_CONTEXT_CHARS = 600


def surface_memory(
    query: str,
    db_path: Path,
    user_id: str | None = None,
) -> str:
    """Return a formatted memory context block for the given query, or empty string.

    Searches:
    1. Episodic memory (FTS5) for the 3 most relevant past exchanges.
    2. Knowledge graph for entities matching key terms in the query.
    """
    if len(query.strip()) < _MIN_QUERY_LEN:
        return ""

    lines: list[str] = []

    # Episodic memory search
    try:
        from jarvis.memory.episodic import _search as ep_search
        rows = ep_search(db_path, query, limit=3, user_id=user_id)
        for row in rows:
            content = str(row["content"])[:200].replace("\n", " ")
            role = str(row["role"]).upper()
            lines.append(f"[{role}]: {content}")
    except Exception:
        pass

    # Knowledge graph: look up the first multi-char word in the query as a seed entity
    try:
        from jarvis.memory.graph import handle_query_knowledge_graph
        words = [w for w in query.split() if len(w) > 4]
        if words:
            result = handle_query_knowledge_graph({"entity": words[0], "depth": 1}, db_path)
            if result and not result.startswith("No ") and not result.startswith("ERROR"):
                lines.append(f"[Graph]: {result[:250]}")
    except Exception:
        pass

    if not lines:
        return ""

    block = "\n".join(lines)
    if len(block) > _MAX_CONTEXT_CHARS:
        block = block[:_MAX_CONTEXT_CHARS] + "…"

    log.debug("memory_surfaced", query_preview=query[:60], lines=len(lines))
    return block
