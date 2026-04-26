"""Digital Twin — public API and tool handlers.

Exposes two tools:
  snapshot_system    — capture current system state into the knowledge graph
  query_system_twin  — ask a question about the stored system topology

Also provides take_snapshot() for use by the scheduler.
"""
from __future__ import annotations

import json
import time
from pathlib import Path


def take_snapshot(db_path: Path) -> int:
    """Take a full system snapshot and store in knowledge graph. Returns item count."""
    from jarvis.twin.system_map import take_full_snapshot
    from jarvis.twin.service_graph import store_snapshot_in_graph

    snapshot = take_full_snapshot()
    count = store_snapshot_in_graph(snapshot, db_path)
    return count


def diff_snapshots(old: dict, new: dict) -> str:
    """Return a human-readable diff between two snapshots."""
    lines: list[str] = []

    old_ports = {p["port"] for p in old.get("ports", [])}
    new_ports = {p["port"] for p in new.get("ports", [])}
    added_ports = new_ports - old_ports
    removed_ports = old_ports - new_ports
    if added_ports:
        lines.append(f"New listening ports: {sorted(added_ports)}")
    if removed_ports:
        lines.append(f"Closed ports: {sorted(removed_ports)}")

    old_procs = {p["name"] for p in old.get("processes", [])}
    new_procs = {p["name"] for p in new.get("processes", [])}
    started = new_procs - old_procs
    stopped = old_procs - new_procs
    if started:
        lines.append(f"New processes (top CPU): {sorted(started)}")
    if stopped:
        lines.append(f"Stopped processes: {sorted(stopped)}")

    old_disks = {d["mountpoint"]: d["used_pct"] for d in old.get("disks", [])}
    new_disks = {d["mountpoint"]: d["used_pct"] for d in new.get("disks", [])}
    for mp, pct in new_disks.items():
        old_pct = old_disks.get(mp, pct)
        delta = pct - old_pct
        if abs(delta) >= 5:
            direction = "grew" if delta > 0 else "shrank"
            lines.append(f"Disk {mp} {direction} by {abs(delta):.1f}%")

    return "\n".join(lines) if lines else "No significant changes detected."


def handle_snapshot_system(tool_input: dict, db_path: Path) -> str:
    """Tool handler: take a system snapshot and store in the knowledge graph."""
    try:
        count = take_snapshot(db_path)
        ts = time.strftime("%H:%M UTC", time.gmtime())
        return f"System snapshot taken at {ts}. Stored {count} entities/relationships in the knowledge graph."
    except Exception as e:
        return f"ERROR: snapshot_system failed — {e}"


def handle_query_system_twin(tool_input: dict, db_path: Path) -> str:
    """Tool handler: query the system topology knowledge graph."""
    try:
        from jarvis.memory.graph import handle_query_knowledge_graph

        entity = tool_input.get("entity", "local_system")
        result = handle_query_knowledge_graph({"entity": entity}, db_path)
        if "No knowledge found" in result:
            return (
                "No system topology data found. Run 'snapshot_system' first to capture current state."
            )
        return result
    except Exception as e:
        return f"ERROR: query_system_twin failed — {e}"


SNAPSHOT_SCHEMA: dict = {
    "name": "snapshot_system",
    "description": (
        "Capture the current system topology (running processes, listening ports, "
        "disk mounts, git repos) and store it in the knowledge graph for later querying."
    ),
    "input_schema": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}

QUERY_SCHEMA: dict = {
    "name": "query_system_twin",
    "description": (
        "Query the Digital Twin knowledge graph for system topology information. "
        "Ask about running services, ports, disk usage, or git repositories."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "entity": {
                "type": "string",
                "description": (
                    "Entity to query, e.g. 'local_system', 'port:tcp:8000', 'process:python:1234'. "
                    "Default: 'local_system' (returns all relationships)."
                ),
                "default": "local_system",
            },
        },
        "required": [],
    },
}
