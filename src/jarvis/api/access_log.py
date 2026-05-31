"""SQLite-backed API request access log.

Records every HTTP request with method, path, status code, latency, and user.
Exposes queryable history for debugging and API usage analytics.
"""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path


def _conn(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=5)
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS access_log (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            method      TEXT    NOT NULL,
            path        TEXT    NOT NULL,
            status_code INTEGER NOT NULL,
            latency_ms  REAL    NOT NULL,
            user_id     TEXT    NOT NULL DEFAULT 'anonymous',
            timestamp   REAL    NOT NULL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_al_ts   ON access_log(timestamp DESC)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_al_path ON access_log(path)")
    conn.commit()
    return conn


def record_request(
    db_path: Path,
    method: str,
    path: str,
    status_code: int,
    latency_ms: float,
    user_id: str = "anonymous",
) -> None:
    """Append one access-log entry. Best-effort — never raises."""
    try:
        conn = _conn(db_path)
        try:
            conn.execute(
                "INSERT INTO access_log (method, path, status_code, latency_ms, user_id, timestamp) "
                "VALUES (?,?,?,?,?,?)",
                (method, path, status_code, round(latency_ms, 2), user_id, time.time()),
            )
            conn.commit()
        finally:
            conn.close()
    except Exception:
        pass


def get_access_log(
    db_path: Path,
    path_filter: str | None = None,
    status_filter: int | None = None,
    since_ts: float | None = None,
    limit: int = 100,
) -> list[dict]:
    """Return recent access-log entries, newest first."""
    try:
        conn = _conn(db_path)
        try:
            clauses, params = [], []
            if path_filter:
                clauses.append("path LIKE ?")
                params.append(f"%{path_filter}%")
            if status_filter is not None:
                clauses.append("status_code = ?")
                params.append(status_filter)
            if since_ts is not None:
                clauses.append("timestamp >= ?")
                params.append(since_ts)
            where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
            params.append(limit)
            rows = conn.execute(
                f"SELECT * FROM access_log {where} ORDER BY timestamp DESC LIMIT ?",
                params,
            ).fetchall()
        finally:
            conn.close()
        return [dict(r) for r in rows]
    except Exception:
        return []


def get_access_log_stats(db_path: Path, since_ts: float | None = None) -> dict:
    """Return aggregated stats: top paths, error rate, avg latency."""
    cutoff = since_ts or 0.0
    try:
        conn = _conn(db_path)
        try:
            total = (conn.execute(
                "SELECT COUNT(*) FROM access_log WHERE timestamp >= ?", (cutoff,)
            ).fetchone() or [0])[0]
            errors = (conn.execute(
                "SELECT COUNT(*) FROM access_log WHERE timestamp >= ? AND status_code >= 400",
                (cutoff,),
            ).fetchone() or [0])[0]
            avg_latency = (conn.execute(
                "SELECT AVG(latency_ms) FROM access_log WHERE timestamp >= ?", (cutoff,)
            ).fetchone() or [0.0])[0] or 0.0
            top_paths = conn.execute(
                "SELECT path, COUNT(*) AS cnt FROM access_log WHERE timestamp >= ? "
                "GROUP BY path ORDER BY cnt DESC LIMIT 10",
                (cutoff,),
            ).fetchall()
            top_errors = conn.execute(
                "SELECT path, status_code, COUNT(*) AS cnt FROM access_log "
                "WHERE timestamp >= ? AND status_code >= 400 "
                "GROUP BY path, status_code ORDER BY cnt DESC LIMIT 10",
                (cutoff,),
            ).fetchall()
        finally:
            conn.close()
        return {
            "total_requests": int(total),
            "error_count": int(errors),
            "error_rate": round(errors / total, 4) if total else 0.0,
            "avg_latency_ms": round(avg_latency, 2),
            "top_paths": [{"path": r["path"], "count": r["cnt"]} for r in top_paths],
            "top_errors": [
                {"path": r["path"], "status_code": r["status_code"], "count": r["cnt"]}
                for r in top_errors
            ],
        }
    except Exception:
        return {"total_requests": 0, "error_count": 0, "error_rate": 0.0,
                "avg_latency_ms": 0.0, "top_paths": [], "top_errors": []}
