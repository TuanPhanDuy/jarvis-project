"""Per-tool runtime configuration.

Stores per-tool overrides for timeout, max_retries, and cache_ttl in SQLite.
BaseAgent._dispatch() reads these at call time; global settings are the fallback.

Configuration precedence (highest to lowest):
  1. Per-tool override (this module)
  2. Global JARVIS_TOOL_TIMEOUT / JARVIS_TOOL_MAX_RETRIES env vars
  3. Hardcoded defaults
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

_DEFAULTS: dict[str, int | None] = {
    "timeout_seconds": None,   # None = use global setting
    "max_retries": None,
    "cache_ttl_seconds": None,
}


def _conn(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=5)
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS tool_config (
            tool_name        TEXT PRIMARY KEY,
            timeout_seconds  INTEGER,
            max_retries      INTEGER,
            cache_ttl_seconds INTEGER
        )
    """)
    conn.commit()
    return conn


def set_tool_config(
    db_path: Path,
    tool_name: str,
    timeout_seconds: int | None = None,
    max_retries: int | None = None,
    cache_ttl_seconds: int | None = None,
) -> None:
    """Upsert per-tool configuration. Pass None to keep the existing value."""
    try:
        conn = _conn(db_path)
        try:
            existing = conn.execute(
                "SELECT * FROM tool_config WHERE tool_name = ?", (tool_name,)
            ).fetchone()
            if existing:
                updates = {}
                if timeout_seconds is not None:
                    updates["timeout_seconds"] = timeout_seconds
                if max_retries is not None:
                    updates["max_retries"] = max_retries
                if cache_ttl_seconds is not None:
                    updates["cache_ttl_seconds"] = cache_ttl_seconds
                if updates:
                    set_clause = ", ".join(f"{k} = ?" for k in updates)
                    conn.execute(
                        f"UPDATE tool_config SET {set_clause} WHERE tool_name = ?",
                        list(updates.values()) + [tool_name],
                    )
            else:
                conn.execute(
                    """INSERT INTO tool_config (tool_name, timeout_seconds, max_retries, cache_ttl_seconds)
                       VALUES (?,?,?,?)""",
                    (tool_name, timeout_seconds, max_retries, cache_ttl_seconds),
                )
            conn.commit()
        finally:
            conn.close()
    except Exception:
        pass


def get_tool_config(db_path: Path, tool_name: str) -> dict:
    """Return effective config for a tool, with None for unset fields."""
    try:
        conn = _conn(db_path)
        try:
            row = conn.execute(
                "SELECT * FROM tool_config WHERE tool_name = ?", (tool_name,)
            ).fetchone()
        finally:
            conn.close()
        if row:
            return {
                "tool_name": row["tool_name"],
                "timeout_seconds": row["timeout_seconds"],
                "max_retries": row["max_retries"],
                "cache_ttl_seconds": row["cache_ttl_seconds"],
            }
    except Exception:
        pass
    return {"tool_name": tool_name, **_DEFAULTS}


def delete_tool_config(db_path: Path, tool_name: str) -> bool:
    """Remove per-tool config, reverting to global defaults. Returns True if existed."""
    try:
        conn = _conn(db_path)
        try:
            cur = conn.execute(
                "DELETE FROM tool_config WHERE tool_name = ?", (tool_name,)
            )
            conn.commit()
        finally:
            conn.close()
        return cur.rowcount > 0
    except Exception:
        return False


def get_tool_timeout(db_path: Path, tool_name: str, global_default: int = 60) -> int:
    """Return effective timeout for a tool, falling back to global_default."""
    cfg = get_tool_config(db_path, tool_name)
    v = cfg.get("timeout_seconds")
    return int(v) if v is not None else global_default


def get_tool_max_retries(db_path: Path, tool_name: str, global_default: int = 2) -> int:
    """Return effective max_retries for a tool, falling back to global_default."""
    cfg = get_tool_config(db_path, tool_name)
    v = cfg.get("max_retries")
    return int(v) if v is not None else global_default
