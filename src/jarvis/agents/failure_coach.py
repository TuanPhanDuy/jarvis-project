"""Inject recent tool-failure warnings into agent system prompts.

When a tool has failed N+ times in the last hour, agents receive an explicit
warning so they can avoid or handle that tool defensively in the next turn.
"""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path

_RECENT_WINDOW_S = 3600   # scan failures from the last hour
_FAILURE_THRESHOLD = 3    # minimum failures before a warning is injected


def get_failure_warnings(db_path: Path | None, threshold: int = _FAILURE_THRESHOLD) -> str:
    """Return a coaching block listing tools with recent repeated failures.

    Returns empty string when there is nothing to report (no db, no failures,
    or fewer failures than the threshold).
    """
    if db_path is None or not db_path.exists():
        return ""
    try:
        conn = sqlite3.connect(str(db_path))
        since = time.time() - _RECENT_WINDOW_S
        rows = conn.execute(
            """
            SELECT tool_name, COUNT(*) AS cnt, MAX(error_msg) AS last_error
            FROM tool_failures
            WHERE timestamp >= ?
            GROUP BY tool_name
            HAVING cnt >= ?
            ORDER BY cnt DESC
            """,
            (since, threshold),
        ).fetchall()
        conn.close()

        if not rows:
            return ""

        lines = ["[TOOL WARNINGS — recent failures in the last hour]"]
        for tool_name, cnt, last_error in rows:
            lines.append(
                f"- {tool_name}: {cnt} failures. "
                f"Last error: {(last_error or '')[:100]}"
            )
        lines.append(
            "Handle these tools defensively: check for errors, use alternatives, "
            "or inform the user if they cannot be relied upon.\n"
        )
        return "\n".join(lines)
    except Exception:
        return ""
