"""Database maintenance utilities: stats, vacuum, and targeted pruning."""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path


def get_db_stats(db_path: Path) -> dict:
    """Return table-level row counts and total DB file size."""
    if not db_path.exists():
        return {"tables": [], "db_size_bytes": 0, "db_path": str(db_path)}
    try:
        conn = sqlite3.connect(str(db_path), timeout=5)
        try:
            tables = [
                r[0] for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
                ).fetchall()
            ]
            result = []
            for table in tables:
                try:
                    count = conn.execute(f"SELECT COUNT(*) FROM \"{table}\"").fetchone()[0]
                except Exception:
                    count = -1
                result.append({"name": table, "row_count": count})
        finally:
            conn.close()
        db_size = db_path.stat().st_size
        return {"tables": result, "db_size_bytes": db_size, "db_path": str(db_path)}
    except Exception as exc:
        return {"tables": [], "db_size_bytes": 0, "db_path": str(db_path), "error": str(exc)}


def vacuum_db(db_path: Path) -> dict:
    """Run VACUUM on the database and return size_before, size_after, reclaimed_bytes."""
    if not db_path.exists():
        return {"size_before": 0, "size_after": 0, "reclaimed_bytes": 0}
    size_before = db_path.stat().st_size
    try:
        conn = sqlite3.connect(str(db_path), timeout=30)
        try:
            conn.execute("VACUUM")
        finally:
            conn.close()
        size_after = db_path.stat().st_size
        return {
            "size_before": size_before,
            "size_after": size_after,
            "reclaimed_bytes": max(0, size_before - size_after),
        }
    except Exception as exc:
        return {"size_before": size_before, "size_after": size_before,
                "reclaimed_bytes": 0, "error": str(exc)}


def prune_data(
    db_path: Path,
    target: str,
    older_than_days: int,
) -> dict:
    """Delete rows older than older_than_days from the chosen target table(s).

    target values:
      "turns"       — agent_turns table
      "episodes"    — episodic_memory table
      "audit"       — audit_log table
      "checkpoints" — agent_checkpoints table
      "jobs"        — agent_jobs (done/failed/cancelled only)
      "all"         — all of the above

    Returns {deleted_counts: {table: count}}.
    """
    cutoff = time.time() - older_than_days * 86400
    deleted: dict[str, int] = {}

    _targets = {
        "turns":       ("agent_turns", "timestamp"),
        "episodes":    ("episodic_memory", "timestamp"),
        "audit":       ("audit_log", "timestamp"),
        "checkpoints": ("agent_checkpoints", "created_at"),
        "jobs":        None,  # handled specially
    }

    selected = list(_targets.keys()) if target == "all" else [target]

    if not db_path.exists():
        return {"deleted_counts": {t: 0 for t in selected}}

    conn = sqlite3.connect(str(db_path), timeout=10)
    try:
        for t in selected:
            if t == "jobs":
                try:
                    cur = conn.execute(
                        "DELETE FROM agent_jobs WHERE finished_at < ? "
                        "AND status IN ('done','failed','cancelled')",
                        (cutoff,),
                    )
                    deleted["jobs"] = cur.rowcount
                except Exception:
                    deleted["jobs"] = 0
            elif t in _targets:
                table, ts_col = _targets[t]
                try:
                    cur = conn.execute(
                        f"DELETE FROM \"{table}\" WHERE \"{ts_col}\" < ?", (cutoff,)
                    )
                    deleted[t] = cur.rowcount
                except Exception:
                    deleted[t] = 0
        conn.commit()
    finally:
        conn.close()

    return {"deleted_counts": deleted}
