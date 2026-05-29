"""Per-agent performance analytics derived from the agent_turns audit table.

Computes latency percentiles, token usage, and call counts grouped by agent type.
Exposed via GET /api/analytics/agents.
"""
from __future__ import annotations

import sqlite3
import statistics
from pathlib import Path


def _percentile(data: list[float], pct: int) -> float:
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * pct / 100
    lo = int(k)
    hi = min(lo + 1, len(sorted_data) - 1)
    return sorted_data[lo] + (sorted_data[hi] - sorted_data[lo]) * (k - lo)


def get_agent_performance(
    db_path: Path,
    agent_type: str | None = None,
    since_ts: float | None = None,
    limit: int = 1000,
) -> list[dict]:
    """Return per-agent performance summary: latency percentiles, token usage, call count.

    Each returned dict has keys: agent_type, call_count, avg_latency_ms,
    p50_latency_ms, p95_latency_ms, avg_input_tokens, avg_output_tokens, models_used.
    """
    try:
        if not db_path.exists():
            return []
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row

        where_parts: list[str] = []
        params: list = []
        if agent_type:
            where_parts.append("agent_type = ?")
            params.append(agent_type)
        if since_ts:
            where_parts.append("timestamp >= ?")
            params.append(since_ts)

        where_clause = f"WHERE {' AND '.join(where_parts)}" if where_parts else ""
        params.append(limit)

        rows = conn.execute(
            f"SELECT agent_type, model, latency_ms, input_tokens, output_tokens "
            f"FROM agent_turns {where_clause} ORDER BY timestamp DESC LIMIT ?",
            params,
        ).fetchall()
        conn.close()

        by_agent: dict[str, list[dict]] = {}
        for row in rows:
            at = row["agent_type"] or "unknown"
            by_agent.setdefault(at, []).append(dict(row))

        result = []
        for at, turns in sorted(by_agent.items()):
            latencies = [t["latency_ms"] for t in turns if t["latency_ms"] is not None and t["latency_ms"] > 0]
            in_toks = [t["input_tokens"] for t in turns if t["input_tokens"] is not None]
            out_toks = [t["output_tokens"] for t in turns if t["output_tokens"] is not None]

            result.append({
                "agent_type": at,
                "call_count": len(turns),
                "avg_latency_ms": round(statistics.mean(latencies), 1) if latencies else 0.0,
                "p50_latency_ms": round(statistics.median(latencies), 1) if latencies else 0.0,
                "p95_latency_ms": round(_percentile(latencies, 95), 1) if latencies else 0.0,
                "avg_input_tokens": round(statistics.mean(in_toks), 1) if in_toks else 0.0,
                "avg_output_tokens": round(statistics.mean(out_toks), 1) if out_toks else 0.0,
                "models_used": sorted({t["model"] for t in turns if t["model"]}),
            })
        return result
    except Exception:
        return []


def get_tool_performance(
    db_path: Path,
    since_ts: float | None = None,
    limit: int = 5000,
) -> list[dict]:
    """Return per-tool performance summary derived from audit_log.

    Each dict has: tool_name, call_count, error_count, error_rate,
    avg_latency_ms, p95_latency_ms.
    """
    try:
        if not db_path.exists():
            return []
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row

        where_clause = "WHERE timestamp >= ?" if since_ts else ""
        params: list = [since_ts, limit] if since_ts else [limit]

        rows = conn.execute(
            f"SELECT tool_name, result_ok, duration_ms "
            f"FROM audit_log {where_clause} ORDER BY timestamp DESC LIMIT ?",
            params,
        ).fetchall()
        conn.close()

        by_tool: dict[str, list[dict]] = {}
        for row in rows:
            name = row["tool_name"] or "unknown"
            by_tool.setdefault(name, []).append(dict(row))

        result = []
        for tool_name, calls in sorted(by_tool.items()):
            latencies = [c["duration_ms"] for c in calls if c["duration_ms"] is not None and c["duration_ms"] >= 0]
            errors = sum(1 for c in calls if c["result_ok"] == 0)
            result.append({
                "tool_name": tool_name,
                "call_count": len(calls),
                "error_count": errors,
                "error_rate": round(errors / len(calls), 4) if calls else 0.0,
                "avg_latency_ms": round(statistics.mean(latencies), 1) if latencies else 0.0,
                "p95_latency_ms": round(_percentile(latencies, 95), 1) if latencies else 0.0,
            })
        return result
    except Exception:
        return []
