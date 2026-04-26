"""Knowledge graph sync: exchange graph deltas between edge and cloud JARVIS.

On connect, the edge agent:
  1. Exports its local graph changes since last_sync_ts
  2. Publishes to: jarvis/sync/graph/{device_id}
  3. Cloud merges and publishes its delta back to: jarvis/sync/graph/cloud
  4. Edge merges the cloud delta locally

This way both nodes share a converging knowledge graph over time.
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path


def export_delta(db_path: Path, since_ts: float = 0.0) -> dict:
    """Export entities and relationships created after since_ts."""
    if not db_path.exists():
        return {"entities": [], "relationships": [], "since_ts": since_ts, "exported_at": time.time()}

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    entities = [
        {"name": r["name"], "type": r["type"], "description": r["description"]}
        for r in conn.execute(
            "SELECT name, type, description FROM entities WHERE created_at > ?", (since_ts,)
        ).fetchall()
    ]
    relationships = [
        {"from": r["from_entity"], "relation": r["relation"], "to": r["to_entity"], "notes": r["notes"]}
        for r in conn.execute(
            "SELECT from_entity, relation, to_entity, notes FROM relationships WHERE created_at > ?", (since_ts,)
        ).fetchall()
    ]
    conn.close()
    return {
        "entities": entities,
        "relationships": relationships,
        "since_ts": since_ts,
        "exported_at": time.time(),
    }


def import_delta(db_path: Path, delta: dict) -> int:
    """Merge a graph delta into the local DB. Returns count of items merged."""
    from jarvis.memory.graph import handle_update_knowledge_graph
    entities = delta.get("entities", [])
    relationships = delta.get("relationships", [])
    if not entities and not relationships:
        return 0
    handle_update_knowledge_graph(
        {"entities": entities, "relationships": relationships},
        db_path,
    )
    return len(entities) + len(relationships)


def sync_via_mqtt(transport, db_path: Path, last_sync_ts: float = 0.0) -> float:
    """Push local delta to cloud and pull cloud delta. Returns new sync timestamp."""
    delta = export_delta(db_path, since_ts=last_sync_ts)
    if delta["entities"] or delta["relationships"]:
        transport.query(f"__sync_graph__{json.dumps(delta)}")
    return time.time()
