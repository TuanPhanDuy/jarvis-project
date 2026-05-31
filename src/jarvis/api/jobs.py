"""SQLite-backed async job queue for long-running agent tasks.

Jobs are created via POST /api/jobs and executed in the API's thread pool.
Status is polled via GET /api/jobs/{id}.

Job lifecycle: pending → running → done | failed
"""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path


def _conn(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS agent_jobs (
            id           TEXT PRIMARY KEY,
            agent_type   TEXT NOT NULL DEFAULT 'planner',
            message      TEXT NOT NULL,
            user_id      TEXT NOT NULL DEFAULT 'anonymous',
            status       TEXT NOT NULL DEFAULT 'pending',
            result       TEXT,
            error        TEXT,
            usage_json   TEXT NOT NULL DEFAULT '{}',
            created_at   REAL NOT NULL,
            started_at   REAL,
            finished_at  REAL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_status ON agent_jobs(status, created_at DESC)")
    conn.commit()
    return conn


def create_job(
    db_path: Path,
    message: str,
    agent_type: str = "planner",
    user_id: str = "anonymous",
) -> str:
    """Insert a new pending job. Returns the job ID."""
    job_id = str(uuid.uuid4())
    conn = _conn(db_path)
    try:
        conn.execute(
            """INSERT INTO agent_jobs (id, agent_type, message, user_id, status, created_at)
               VALUES (?,?,?,?,?,?)""",
            (job_id, agent_type, message, user_id, "pending", time.time()),
        )
        conn.commit()
    finally:
        conn.close()
    return job_id


def mark_running(db_path: Path, job_id: str) -> None:
    conn = _conn(db_path)
    try:
        conn.execute(
            "UPDATE agent_jobs SET status='running', started_at=? WHERE id=?",
            (time.time(), job_id),
        )
        conn.commit()
    finally:
        conn.close()


def mark_done(db_path: Path, job_id: str, result: str, usage: dict) -> None:
    conn = _conn(db_path)
    try:
        conn.execute(
            """UPDATE agent_jobs
               SET status='done', result=?, usage_json=?, finished_at=?
               WHERE id=?""",
            (result, json.dumps(usage), time.time(), job_id),
        )
        conn.commit()
    finally:
        conn.close()


def mark_failed(db_path: Path, job_id: str, error: str) -> None:
    conn = _conn(db_path)
    try:
        conn.execute(
            "UPDATE agent_jobs SET status='failed', error=?, finished_at=? WHERE id=?",
            (error, time.time(), job_id),
        )
        conn.commit()
    finally:
        conn.close()


def get_job(db_path: Path, job_id: str) -> dict | None:
    """Return a job record or None if not found."""
    try:
        conn = _conn(db_path)
        try:
            row = conn.execute(
                "SELECT * FROM agent_jobs WHERE id=?", (job_id,)
            ).fetchone()
        finally:
            conn.close()
        if not row:
            return None
        d = dict(row)
        d["usage"] = json.loads(d.pop("usage_json", "{}"))
        return d
    except Exception:
        return None


def list_jobs(
    db_path: Path,
    user_id: str | None = None,
    status: str | None = None,
    limit: int = 50,
) -> list[dict]:
    """List recent jobs, newest first. Optionally filter by user_id or status."""
    try:
        conn = _conn(db_path)
        try:
            where_clauses, params = [], []
            if user_id:
                where_clauses.append("user_id=?")
                params.append(user_id)
            if status:
                where_clauses.append("status=?")
                params.append(status)
            where = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
            params.append(limit)
            rows = conn.execute(
                f"SELECT * FROM agent_jobs {where} ORDER BY created_at DESC LIMIT ?",
                params,
            ).fetchall()
        finally:
            conn.close()
        result = []
        for r in rows:
            d = dict(r)
            d["usage"] = json.loads(d.pop("usage_json", "{}"))
            result.append(d)
        return result
    except Exception:
        return []
