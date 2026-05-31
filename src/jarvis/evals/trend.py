"""Eval trend tracking — persist run summaries to SQLite for regression detection."""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path


def _conn(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=10)
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS eval_runs (
            run_id       TEXT PRIMARY KEY,
            timestamp    REAL NOT NULL,
            total        INTEGER NOT NULL,
            passed       INTEGER NOT NULL,
            failed       INTEGER NOT NULL,
            pass_rate    REAL NOT NULL,
            avg_latency_s REAL NOT NULL DEFAULT 0.0,
            total_cost_usd REAL NOT NULL DEFAULT 0.0,
            avg_judge_score REAL,
            tags         TEXT NOT NULL DEFAULT '[]',
            results_json TEXT NOT NULL DEFAULT '[]'
        );
        CREATE INDEX IF NOT EXISTS idx_eval_ts ON eval_runs(timestamp DESC);
    """)
    conn.commit()
    return conn


def record_run(
    db_path: Path,
    run_id: str,
    total: int,
    passed: int,
    failed: int,
    pass_rate: float,
    avg_latency_s: float = 0.0,
    total_cost_usd: float = 0.0,
    avg_judge_score: float | None = None,
    tags: list[str] | None = None,
    results: list[dict] | None = None,
) -> None:
    """Upsert an eval run summary. Best-effort — never raises."""
    try:
        conn = _conn(db_path)
        try:
            conn.execute(
                """INSERT INTO eval_runs
                       (run_id, timestamp, total, passed, failed, pass_rate,
                        avg_latency_s, total_cost_usd, avg_judge_score, tags, results_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(run_id) DO UPDATE SET
                       pass_rate = excluded.pass_rate,
                       passed    = excluded.passed,
                       failed    = excluded.failed""",
                (
                    run_id,
                    time.time(),
                    total,
                    passed,
                    failed,
                    pass_rate,
                    avg_latency_s,
                    total_cost_usd,
                    avg_judge_score,
                    json.dumps(tags or []),
                    json.dumps(results or []),
                ),
            )
            conn.commit()
        finally:
            conn.close()
    except Exception:
        pass


def get_trend(db_path: Path, last_n: int = 10) -> list[dict]:
    """Return the last N eval run summaries, newest-first."""
    if not db_path.exists():
        return []
    try:
        conn = _conn(db_path)
        try:
            rows = conn.execute(
                "SELECT * FROM eval_runs ORDER BY timestamp DESC LIMIT ?", (last_n,)
            ).fetchall()
        finally:
            conn.close()
        return [
            {
                "run_id": r["run_id"],
                "timestamp": r["timestamp"],
                "total": r["total"],
                "passed": r["passed"],
                "failed": r["failed"],
                "pass_rate": r["pass_rate"],
                "avg_latency_s": r["avg_latency_s"],
                "total_cost_usd": r["total_cost_usd"],
                "avg_judge_score": r["avg_judge_score"],
                "tags": json.loads(r["tags"]),
            }
            for r in rows
        ]
    except Exception:
        return []


def compare_runs(db_path: Path, run_a_id: str, run_b_id: str) -> dict | None:
    """Compare two eval runs and return a structured diff.

    Returns:
        {run_a, run_b, delta_pass_rate,
         improved: [case_ids], regressed: [case_ids],
         unchanged_pass: N, unchanged_fail: N}

    Returns None if either run is not found.
    """
    run_a = get_run(db_path, run_a_id)
    run_b = get_run(db_path, run_b_id)
    if run_a is None or run_b is None:
        return None

    # Build pass/fail maps by case_id
    def _pass_map(results: list[dict]) -> dict[str, bool]:
        return {r["case_id"]: bool(r.get("overall_pass", False)) for r in results}

    map_a = _pass_map(run_a.get("results", []))
    map_b = _pass_map(run_b.get("results", []))
    all_cases = set(map_a) | set(map_b)

    improved, regressed, unchanged_pass, unchanged_fail = [], [], 0, 0
    for case_id in sorted(all_cases):
        a_pass = map_a.get(case_id, False)
        b_pass = map_b.get(case_id, False)
        if not a_pass and b_pass:
            improved.append(case_id)
        elif a_pass and not b_pass:
            regressed.append(case_id)
        elif a_pass and b_pass:
            unchanged_pass += 1
        else:
            unchanged_fail += 1

    return {
        "run_a": {k: v for k, v in run_a.items() if k != "results"},
        "run_b": {k: v for k, v in run_b.items() if k != "results"},
        "delta_pass_rate": round(run_b["pass_rate"] - run_a["pass_rate"], 4),
        "improved": improved,
        "regressed": regressed,
        "unchanged_pass": unchanged_pass,
        "unchanged_fail": unchanged_fail,
    }


def get_run(db_path: Path, run_id: str) -> dict | None:
    """Return full details for a single eval run, including per-case results."""
    if not db_path.exists():
        return None
    try:
        conn = _conn(db_path)
        try:
            row = conn.execute(
                "SELECT * FROM eval_runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        finally:
            conn.close()
        if not row:
            return None
        return {
            "run_id": row["run_id"],
            "timestamp": row["timestamp"],
            "total": row["total"],
            "passed": row["passed"],
            "failed": row["failed"],
            "pass_rate": row["pass_rate"],
            "avg_latency_s": row["avg_latency_s"],
            "total_cost_usd": row["total_cost_usd"],
            "avg_judge_score": row["avg_judge_score"],
            "tags": json.loads(row["tags"]),
            "results": json.loads(row["results_json"]),
        }
    except Exception:
        return None
