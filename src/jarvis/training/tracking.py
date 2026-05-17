"""Training run tracker — persists auto-training history in SQLite.

Each run records its type (crawl or finetune), status, timing, and outputs
so the API and scheduler can make decisions about when to re-train.
"""
from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainingRun:
    id: int
    run_type: str          # "crawl" | "finetune" | "pipeline"
    status: str            # "running" | "completed" | "failed"
    started_at: float
    completed_at: float | None
    docs_crawled: int
    pairs_generated: int
    model_name: str
    notes: str


def _conn(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS training_runs (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            run_type        TEXT    NOT NULL,
            status          TEXT    NOT NULL DEFAULT 'running',
            started_at      REAL    NOT NULL,
            completed_at    REAL,
            docs_crawled    INTEGER NOT NULL DEFAULT 0,
            pairs_generated INTEGER NOT NULL DEFAULT 0,
            model_name      TEXT    NOT NULL DEFAULT '',
            notes           TEXT    NOT NULL DEFAULT ''
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_tr_started ON training_runs(started_at)")
    conn.commit()
    return conn


def start_run(db_path: Path, run_type: str) -> int:
    """Insert a new running record; return its ID."""
    conn = _conn(db_path)
    cur = conn.execute(
        "INSERT INTO training_runs (run_type, status, started_at) VALUES (?, 'running', ?)",
        (run_type, time.time()),
    )
    run_id = cur.lastrowid
    conn.commit()
    conn.close()
    return run_id


def complete_run(
    db_path: Path,
    run_id: int,
    *,
    status: str = "completed",
    docs_crawled: int = 0,
    pairs_generated: int = 0,
    model_name: str = "",
    notes: str = "",
) -> None:
    """Mark a run as completed (or failed)."""
    conn = _conn(db_path)
    conn.execute(
        """UPDATE training_runs
           SET status=?, completed_at=?, docs_crawled=?, pairs_generated=?, model_name=?, notes=?
           WHERE id=?""",
        (status, time.time(), docs_crawled, pairs_generated, model_name, notes, run_id),
    )
    conn.commit()
    conn.close()


def get_history(db_path: Path, limit: int = 20) -> list[TrainingRun]:
    """Return the most recent training runs, newest first."""
    conn = _conn(db_path)
    rows = conn.execute(
        "SELECT * FROM training_runs ORDER BY started_at DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [
        TrainingRun(
            id=r["id"],
            run_type=r["run_type"],
            status=r["status"],
            started_at=r["started_at"],
            completed_at=r["completed_at"],
            docs_crawled=r["docs_crawled"],
            pairs_generated=r["pairs_generated"],
            model_name=r["model_name"],
            notes=r["notes"],
        )
        for r in rows
    ]


def get_last_run(db_path: Path, run_type: str | None = None) -> TrainingRun | None:
    """Return the most recent completed run (optionally filtered by type)."""
    conn = _conn(db_path)
    if run_type:
        row = conn.execute(
            "SELECT * FROM training_runs WHERE run_type=? AND status='completed' "
            "ORDER BY completed_at DESC LIMIT 1",
            (run_type,),
        ).fetchone()
    else:
        row = conn.execute(
            "SELECT * FROM training_runs WHERE status='completed' "
            "ORDER BY completed_at DESC LIMIT 1"
        ).fetchone()
    conn.close()
    if row is None:
        return None
    return TrainingRun(
        id=row["id"],
        run_type=row["run_type"],
        status=row["status"],
        started_at=row["started_at"],
        completed_at=row["completed_at"],
        docs_crawled=row["docs_crawled"],
        pairs_generated=row["pairs_generated"],
        model_name=row["model_name"],
        notes=row["notes"],
    )


def count_new_docs_since(db_path: Path, since_ts: float, reports_dir: Path) -> int:
    """Count research reports written after since_ts (based on file mtime)."""
    try:
        count = 0
        for f in reports_dir.glob("research_*.md"):
            if f.stat().st_mtime > since_ts:
                count += 1
        return count
    except Exception:
        return 0
