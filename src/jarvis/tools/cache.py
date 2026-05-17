"""SQLite-backed tool result cache with per-tool TTL.

Cacheable tools are listed in _TOOL_TTLS. Tools absent from the map
are never cached (TTL=0). Cache keys are SHA-256 of (tool_name, sorted input JSON).
Expired entries are evicted opportunistically on each write.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path

import structlog

log = structlog.get_logger()

# Per-tool TTL in seconds. Tools not listed here are never cached.
_TOOL_TTLS: dict[str, int] = {
    "web_search":             3600,    # 1 h — search results change slowly
    "read_url":               86400,   # 24 h — page content
    "query_knowledge_graph":  300,     # 5 min — graph is write-heavy during research
    "search_episodic_memory": 60,      # 1 min — recency matters
}


def _get_conn(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS tool_cache (
            key        TEXT PRIMARY KEY,
            tool_name  TEXT NOT NULL,
            result     TEXT NOT NULL,
            created_at REAL NOT NULL,
            expires_at REAL NOT NULL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_tc_expires ON tool_cache(expires_at)")
    conn.commit()
    return conn


def _cache_key(tool_name: str, tool_input: dict) -> str:
    payload = json.dumps({"tool": tool_name, "input": tool_input}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def get_cached(db_path: Path, tool_name: str, tool_input: dict) -> str | None:
    """Return a cached result if one exists and hasn't expired, else None."""
    if tool_name not in _TOOL_TTLS:
        return None
    try:
        conn = _get_conn(db_path)
        key = _cache_key(tool_name, tool_input)
        row = conn.execute(
            "SELECT result FROM tool_cache WHERE key = ? AND expires_at > ?",
            (key, time.time()),
        ).fetchone()
        conn.close()
        if row:
            log.debug("tool_cache_hit", tool=tool_name)
            return row["result"]
    except Exception:
        pass
    return None


def set_cached(db_path: Path, tool_name: str, tool_input: dict, result: str) -> None:
    """Store a result in the cache. No-op for uncacheable tools or on error."""
    ttl = _TOOL_TTLS.get(tool_name, 0)
    if ttl == 0:
        return
    try:
        now = time.time()
        conn = _get_conn(db_path)
        key = _cache_key(tool_name, tool_input)
        conn.execute(
            """INSERT OR REPLACE INTO tool_cache
               (key, tool_name, result, created_at, expires_at) VALUES (?,?,?,?,?)""",
            (key, tool_name, result, now, now + ttl),
        )
        conn.execute("DELETE FROM tool_cache WHERE expires_at <= ?", (now,))
        conn.commit()
        conn.close()
    except Exception:
        pass
