"""Plugin: filesystem_search — search local files by name pattern or content.

Personal data connector: lets JARVIS find files on the local machine.
Configure JARVIS_FS_ROOT to restrict searches to a safe directory.
"""
from __future__ import annotations

import os
from pathlib import Path


def _get_root() -> Path:
    root = os.environ.get("JARVIS_FS_ROOT", "")
    return Path(root).expanduser() if root else Path.home()


def handle(tool_input: dict) -> str:
    try:
        pattern = tool_input.get("pattern", "*").strip()
        contains = tool_input.get("contains", "").strip()
        search_path = tool_input.get("path", "").strip()
        max_results = int(tool_input.get("max_results", 20))

        root = _get_root()
        base = (root / search_path).resolve() if search_path else root

        # Safety: keep within root
        if not str(base).startswith(str(root)):
            return f"ERROR: path '{search_path}' is outside the allowed root '{root}'."

        if not base.exists():
            return f"ERROR: path '{base}' does not exist."

        matches = list(base.rglob(pattern))[:max_results * 3]  # over-fetch for content filter

        if contains:
            filtered = []
            for p in matches:
                if p.is_file():
                    try:
                        if contains.lower() in p.read_text(encoding="utf-8", errors="ignore").lower():
                            filtered.append(p)
                    except Exception:
                        pass
                if len(filtered) >= max_results:
                    break
            matches = filtered
        else:
            matches = [m for m in matches if m.is_file()][:max_results]

        if not matches:
            return f"No files found matching pattern '{pattern}'" + (f" containing '{contains}'" if contains else "") + f" under '{base}'."

        lines = [f"Found {len(matches)} file(s) under '{base}':\n"]
        for p in matches:
            lines.append(f"  {p.relative_to(root)}")
        return "\n".join(lines)
    except Exception as e:
        return f"ERROR: filesystem_search failed — {e}"


SCHEMA: dict = {
    "name": "filesystem_search",
    "description": (
        "Search local files by name pattern and/or content. "
        "Set JARVIS_FS_ROOT env var to restrict to a safe directory (default: home). "
        "Use this to find documents, code, or notes on the local machine."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Glob pattern, e.g. '*.py', '*.md', 'report_*.txt'. Default '*'.",
                "default": "*",
            },
            "contains": {
                "type": "string",
                "description": "Optional: only return files containing this text (case-insensitive).",
            },
            "path": {
                "type": "string",
                "description": "Sub-path within JARVIS_FS_ROOT to search (default: root).",
            },
            "max_results": {
                "type": "integer",
                "description": "Max files to return (default 20).",
                "default": 20,
            },
        },
        "required": [],
    },
}
