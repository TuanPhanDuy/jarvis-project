"""Feedback analyzer — self-improvement loop for JARVIS.

Reads low-rated feedback and recurring tool failures, then uses Claude to
generate actionable improvement suggestions saved as a markdown report.

Usage:
    from jarvis.evals.feedback_analyzer import run_analysis
    run_analysis(db_path, reports_dir, client, fast_model)

Or via CLI:
    jarvis-eval --analyze-feedback
"""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import structlog

log = structlog.get_logger()

_ANALYSIS_PROMPT = """\
You are JARVIS's self-improvement module. Analyze the following quality signals and produce
a concise, actionable improvement report. Focus on root causes and concrete fixes.

## Low-Rated Responses (rating < 3)
{bad_feedback}

## Recurring Tool Failures
{tool_failures}

## Overall Stats
{stats}

Write a markdown report with these sections:
1. **Root Cause Analysis** — what patterns do you see across failures?
2. **Top 3 Actionable Improvements** — specific, implementable changes
3. **Healthy Signals** — what is working well (don't change these)

Keep the report under 600 words. Be direct and specific."""


def _fetch_bad_feedback(db_path: Path, limit: int = 20) -> list[dict]:
    """Return recent low-rated feedback (rating <= 2)."""
    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT rating, comment, session_id, ts
            FROM feedback
            WHERE rating <= 2
            ORDER BY ts DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception:
        return []


def _fetch_failure_patterns(db_path: Path, limit: int = 15) -> list[dict]:
    """Return top recurring tool failure patterns."""
    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT tool_name, error_msg, COUNT(*) AS count
            FROM tool_failures
            GROUP BY tool_name, error_msg
            ORDER BY count DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception:
        return []


def _fetch_stats(db_path: Path) -> dict:
    """Return aggregate feedback statistics."""
    result = {"total_feedback": 0, "avg_rating": 0.0, "total_failures": 0}
    if not db_path.exists():
        return result
    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row

        # Check which tables exist
        tables = {
            r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }

        if "feedback" in tables:
            row = conn.execute(
                "SELECT COUNT(*) as total, AVG(rating) as avg FROM feedback"
            ).fetchone()
            result["total_feedback"] = row["total"] or 0
            result["avg_rating"] = round(row["avg"] or 0, 2)

        if "tool_failures" in tables:
            row2 = conn.execute("SELECT COUNT(*) as n FROM tool_failures").fetchone()
            result["total_failures"] = row2["n"] or 0

        conn.close()
    except Exception:
        pass
    return result


def _format_feedback(items: list[dict]) -> str:
    if not items:
        return "No low-rated responses recorded yet."
    lines = []
    for item in items:
        ts = time.strftime("%Y-%m-%d", time.gmtime(item["ts"]))
        comment = item["comment"] or "(no comment)"
        lines.append(f"- [{ts}] rating={item['rating']}: {comment}")
    return "\n".join(lines)


def _format_failures(items: list[dict]) -> str:
    if not items:
        return "No tool failures recorded yet."
    lines = []
    for item in items:
        lines.append(
            f"- **{item['tool_name']}** (×{item['count']}): {item['error_msg'][:150]}"
        )
    return "\n".join(lines)


def run_analysis(
    db_path: Path,
    reports_dir: Path,
    client,
    fast_model: str,
) -> str:
    """Run feedback analysis and save improvement report. Returns file path or error."""
    bad_feedback = _fetch_bad_feedback(db_path)
    failures = _fetch_failure_patterns(db_path)
    stats = _fetch_stats(db_path)

    if stats["total_feedback"] == 0 and stats["total_failures"] == 0:
        return "No feedback or failure data to analyze yet."

    prompt = _ANALYSIS_PROMPT.format(
        bad_feedback=_format_feedback(bad_feedback),
        tool_failures=_format_failures(failures),
        stats=(
            f"Total feedback entries: {stats['total_feedback']}, "
            f"avg rating: {stats['avg_rating']}/5, "
            f"total tool failures: {stats['total_failures']}"
        ),
    )

    response = client.messages.create(
        model=fast_model,
        max_tokens=1024,
        system=[{
            "type": "text",
            "text": "You are JARVIS's self-improvement module. Be direct and analytical.",
            "cache_control": {"type": "ephemeral"},
        }],
        messages=[{"role": "user", "content": prompt}],
    )
    report_text = response.content[0].text

    # Save report
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / "improvement_suggestions.md"
    date_str = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
    out_path.write_text(
        f"# JARVIS Self-Improvement Analysis\n\n_Generated: {date_str}_\n\n{report_text}\n",
        encoding="utf-8",
    )
    log.info("feedback_analysis_saved", path=str(out_path))
    return str(out_path)
