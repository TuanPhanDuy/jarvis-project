"""Service graph builder: converts a system snapshot into knowledge graph entities and relationships.

Transforms raw system_map data into the entity/relationship format used by memory/graph.py,
then stores it under user_id="__twin__" for isolation from user knowledge.
"""
from __future__ import annotations

from pathlib import Path


def snapshot_to_graph_entities(snapshot: dict) -> tuple[list[dict], list[dict]]:
    """Convert a system snapshot into (entities, relationships) for the knowledge graph."""
    entities: list[dict] = []
    relationships: list[dict] = []

    # System node
    entities.append({
        "name": "local_system",
        "type": "system",
        "description": f"Local machine snapshot taken at {snapshot.get('ts', 0):.0f}",
    })

    # Processes
    for proc in snapshot.get("processes", []):
        name = proc.get("name", "unknown")
        pid = proc.get("pid", 0)
        ent_name = f"process:{name}:{pid}"
        entities.append({
            "name": ent_name,
            "type": "process",
            "description": f"{name} (PID {pid}, status: {proc.get('status', '?')})",
        })
        relationships.append({
            "from": "local_system",
            "relation": "runs",
            "to": ent_name,
            "notes": f"cpu%={proc.get('cpu_percent', 0):.1f}",
        })

    # Listening ports
    for port_info in snapshot.get("ports", []):
        port = port_info.get("port", 0)
        proto = port_info.get("proto", "tcp")
        ent_name = f"port:{proto}:{port}"
        entities.append({
            "name": ent_name,
            "type": "port",
            "description": f"Listening {proto.upper()} port {port} on {port_info.get('host', '*')}",
        })
        relationships.append({
            "from": "local_system",
            "relation": "listens_on",
            "to": ent_name,
            "notes": f"pid={port_info.get('pid')}",
        })

    # Disk mounts
    for disk in snapshot.get("disks", []):
        mp = disk.get("mountpoint", "?")
        ent_name = f"disk:{mp}"
        entities.append({
            "name": ent_name,
            "type": "disk",
            "description": (
                f"Disk {mp} ({disk.get('fstype', '?')}) — "
                f"{disk.get('total_gb', 0)}GB total, {disk.get('used_pct', 0)}% used"
            ),
        })
        relationships.append({
            "from": "local_system",
            "relation": "has_mount",
            "to": ent_name,
            "notes": f"used_pct={disk.get('used_pct', 0)}",
        })

    # Git repos
    for repo in snapshot.get("git_repos", []):
        ent_name = f"repo:{repo.get('name', 'unknown')}"
        entities.append({
            "name": ent_name,
            "type": "git_repo",
            "description": f"Git repository at {repo.get('path', '?')}",
        })
        relationships.append({
            "from": "local_system",
            "relation": "has_repo",
            "to": ent_name,
            "notes": repo.get("path", ""),
        })

    return entities, relationships


def store_snapshot_in_graph(snapshot: dict, db_path: Path) -> int:
    """Convert snapshot to graph and persist. Returns count of items stored."""
    from jarvis.memory.graph import handle_update_knowledge_graph

    entities, relationships = snapshot_to_graph_entities(snapshot)
    if not entities and not relationships:
        return 0

    result = handle_update_knowledge_graph(
        {
            "entities": entities,
            "relationships": relationships,
            "user_id": "__twin__",
        },
        db_path,
    )
    return len(entities) + len(relationships)
