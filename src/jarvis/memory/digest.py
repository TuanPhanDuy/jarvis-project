"""Research digest: summarise recent JARVIS activity into a readable report.

A digest is generated on demand or by the scheduler. It inspects:
- Recent agent turns (tool calls made, topics from web_search)
- Recently added episodic memory entries
- Recently added knowledge-graph entities

The summary text is produced by Claude (or a fallback template) and stored
in the `research_digests` SQLite table so GET /api/digest/latest is fast.
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
        CREATE TABLE IF NOT EXISTS research_digests (
            id           TEXT PRIMARY KEY,
            period_start REAL NOT NULL,
            period_end   REAL NOT NULL,
            summary      TEXT NOT NULL,
            stats_json   TEXT NOT NULL DEFAULT '{}',
            created_at   REAL NOT NULL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_digest_created ON research_digests(created_at DESC)")
    conn.commit()
    return conn


def _gather_stats(db_path: Path, since: float) -> dict:
    """Collect raw activity data since the given timestamp."""
    stats: dict = {"since": since, "until": time.time()}

    # Agent turns & tool call counts
    try:
        conn = sqlite3.connect(str(db_path), timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT tool_calls_json FROM agent_turns WHERE timestamp >= ?", (since,)
            ).fetchall()
        finally:
            conn.close()
        tool_counter: dict[str, int] = {}
        for r in rows:
            for t in json.loads(r["tool_calls_json"] or "[]"):
                tool_counter[t] = tool_counter.get(t, 0) + 1
        stats["turn_count"] = len(rows)
        stats["tool_calls"] = tool_counter
        stats["top_tools"] = sorted(tool_counter.items(), key=lambda x: -x[1])[:5]
    except Exception:
        stats["turn_count"] = 0
        stats["tool_calls"] = {}
        stats["top_tools"] = []

    # Recent episodic memory entries
    try:
        conn = sqlite3.connect(str(db_path), timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT content FROM episodic_memory WHERE timestamp >= ? ORDER BY timestamp DESC LIMIT 20",
                (since,),
            ).fetchall()
        finally:
            conn.close()
        stats["episodic_count"] = len(rows)
        stats["episodic_snippets"] = [r["content"][:120] for r in rows]
    except Exception:
        stats["episodic_count"] = 0
        stats["episodic_snippets"] = []

    # New knowledge graph entities
    try:
        conn = sqlite3.connect(str(db_path), timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT name, entity_type FROM kg_entities WHERE created_at >= ? ORDER BY created_at DESC LIMIT 30",
                (since,),
            ).fetchall()
        finally:
            conn.close()
        stats["new_entities"] = [{"name": r["name"], "type": r["entity_type"]} for r in rows]
    except Exception:
        stats["new_entities"] = []

    return stats


def _render_summary(stats: dict, use_claude: bool = True) -> str:
    """Generate a human-readable digest summary."""
    if use_claude:
        try:
            return _claude_summary(stats)
        except Exception:
            pass
    return _template_summary(stats)


def _template_summary(stats: dict) -> str:
    from datetime import datetime, timezone
    since_dt = datetime.fromtimestamp(stats["since"], tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    until_dt = datetime.fromtimestamp(stats["until"], tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"## JARVIS Research Digest",
        f"**Period:** {since_dt} → {until_dt}",
        "",
        f"**Agent turns:** {stats['turn_count']}",
    ]
    if stats.get("top_tools"):
        tools_str = ", ".join(f"{t}×{c}" for t, c in stats["top_tools"])
        lines.append(f"**Top tools:** {tools_str}")
    if stats.get("episodic_count"):
        lines.append(f"**New memories:** {stats['episodic_count']}")
    if stats.get("new_entities"):
        entity_names = ", ".join(e["name"] for e in stats["new_entities"][:10])
        lines.append(f"**New KG entities ({len(stats['new_entities'])}):** {entity_names}")
    if stats.get("episodic_snippets"):
        lines += ["", "**Recent activity:**"]
        for snippet in stats["episodic_snippets"][:5]:
            lines.append(f"- {snippet}")
    return "\n".join(lines)


def _claude_summary(stats: dict) -> str:
    import anthropic
    from jarvis.config import get_settings

    settings = get_settings()
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    tool_summary = ", ".join(f"{t}({c})" for t, c in stats.get("top_tools", []))
    entities = ", ".join(e["name"] for e in stats.get("new_entities", [])[:15])
    snippets = "\n".join(f"- {s}" for s in stats.get("episodic_snippets", [])[:8])

    prompt = (
        f"Generate a concise research digest (3-5 sentences) summarising JARVIS AI assistant activity:\n"
        f"- {stats['turn_count']} agent turns\n"
        f"- Tools used: {tool_summary or 'none'}\n"
        f"- New KG entities: {entities or 'none'}\n"
        f"- Recent memory snippets:\n{snippets or '  (none)'}\n\n"
        "Focus on key research topics and notable findings. Be specific and informative."
    )
    resp = client.messages.create(
        model=settings.model,
        max_tokens=400,
        system=[{"type": "text", "text": "You are a research digest writer. Be concise and factual.",
                 "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text.strip()


def generate_digest(
    db_path: Path,
    hours: int = 24,
    use_claude: bool = True,
) -> dict:
    """Generate and persist a research digest covering the last `hours` hours."""
    since = time.time() - hours * 3600
    stats = _gather_stats(db_path, since)
    summary = _render_summary(stats, use_claude=use_claude)

    digest_id = str(uuid.uuid4())
    now = time.time()
    conn = _conn(db_path)
    try:
        conn.execute(
            "INSERT INTO research_digests (id, period_start, period_end, summary, stats_json, created_at) VALUES (?,?,?,?,?,?)",
            (digest_id, since, stats["until"], summary, json.dumps(stats), now),
        )
        conn.commit()
    finally:
        conn.close()

    return {
        "id": digest_id,
        "period_start": since,
        "period_end": stats["until"],
        "summary": summary,
        "stats": stats,
        "created_at": now,
    }


def get_latest_digest(db_path: Path) -> dict | None:
    try:
        conn = _conn(db_path)
        try:
            row = conn.execute(
                "SELECT * FROM research_digests ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
        finally:
            conn.close()
        if not row:
            return None
        d = dict(row)
        d["stats"] = json.loads(d.pop("stats_json", "{}"))
        return d
    except Exception:
        return None


def list_digests(db_path: Path, limit: int = 20) -> list[dict]:
    try:
        conn = _conn(db_path)
        try:
            rows = conn.execute(
                "SELECT id, period_start, period_end, created_at FROM research_digests ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        finally:
            conn.close()
        return [dict(r) for r in rows]
    except Exception:
        return []
