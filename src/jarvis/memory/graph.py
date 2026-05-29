"""Knowledge graph: entity-relationship store in SQLite.

Entities are concepts, people, papers, techniques, etc.
Relationships connect entities with typed edges.

DB location: reports_dir/jarvis.db (shared SQLite file).

Example graph after research:
  RLHF --[uses]--> PPO
  RLHF --[developed_by]--> OpenAI
  Constitutional AI --[improves_on]--> RLHF
  Constitutional AI --[developed_by]--> Anthropic
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path


def _get_conn(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS entities (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT    NOT NULL,
            type        TEXT    NOT NULL DEFAULT 'concept',
            description TEXT    NOT NULL DEFAULT '',
            user_id     TEXT    NOT NULL DEFAULT 'shared',
            created_at  REAL    NOT NULL,
            UNIQUE(name, user_id)
        );
        CREATE TABLE IF NOT EXISTS relationships (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            from_entity TEXT    NOT NULL,
            relation    TEXT    NOT NULL,
            to_entity   TEXT    NOT NULL,
            notes       TEXT    NOT NULL DEFAULT '',
            user_id     TEXT    NOT NULL DEFAULT 'shared',
            created_at  REAL    NOT NULL,
            UNIQUE(from_entity, relation, to_entity, user_id)
        );
        CREATE INDEX IF NOT EXISTS idx_ent_name   ON entities(name);
        CREATE INDEX IF NOT EXISTS idx_ent_user   ON entities(user_id);
        CREATE INDEX IF NOT EXISTS idx_rel_from   ON relationships(from_entity);
        CREATE INDEX IF NOT EXISTS idx_rel_to     ON relationships(to_entity);
        CREATE INDEX IF NOT EXISTS idx_rel_user   ON relationships(user_id);
    """)
    conn.commit()
    return conn


def handle_update_knowledge_graph(tool_input: dict, db_path: Path) -> str:
    try:
        entities = tool_input.get("entities", [])
        relationships = tool_input.get("relationships", [])

        if not entities and not relationships:
            return "ERROR: provide at least one entity or relationship to add."

        conn = _get_conn(db_path)
        try:
            now = time.time()
            added_ents, added_rels = 0, 0
            user_id = tool_input.get("user_id", "shared")

            for ent in entities:
                name = ent.get("name", "").strip()
                if not name:
                    continue
                conn.execute(
                    """
                    INSERT INTO entities (name, type, description, user_id, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(name, user_id) DO UPDATE SET
                        type = excluded.type,
                        description = CASE WHEN excluded.description != '' THEN excluded.description ELSE description END
                    """,
                    (name, ent.get("type", "concept"), ent.get("description", ""), user_id, now),
                )
                added_ents += 1

            for rel in relationships:
                frm = rel.get("from", "").strip()
                relation = rel.get("relation", "").strip()
                to = rel.get("to", "").strip()
                if not frm or not relation or not to:
                    continue
                conn.execute(
                    """
                    INSERT OR REPLACE INTO relationships (from_entity, relation, to_entity, notes, user_id, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (frm, relation, to, rel.get("notes", ""), user_id, now),
                )
                added_rels += 1

            conn.commit()
        finally:
            conn.close()
        return f"Knowledge graph updated: {added_ents} entity/entities, {added_rels} relationship(s) added."
    except Exception as e:
        return f"ERROR: update_knowledge_graph failed — {e}"


def _bfs_subgraph(
    conn: sqlite3.Connection,
    seed_entity: str,
    depth: int,
    relation_filter: str | None,
) -> tuple[set[str], list]:
    """BFS from seed entity up to `depth` hops. Returns (visited_names, rel_rows)."""
    seed_rows = conn.execute(
        "SELECT name FROM entities WHERE name LIKE ? LIMIT 10",
        (f"%{seed_entity}%",),
    ).fetchall()
    frontier = {r["name"] for r in seed_rows} or {seed_entity}
    visited: set[str] = set(frontier)
    seen_rels: set[tuple] = set()
    all_rels: list = []

    for _ in range(depth):
        if not frontier or len(visited) > 50:
            break
        placeholders = ",".join("?" * len(frontier))
        args: list = list(frontier) + list(frontier)
        if relation_filter:
            sql = (
                f"SELECT * FROM relationships "
                f"WHERE (from_entity IN ({placeholders}) OR to_entity IN ({placeholders})) "
                f"AND relation = ?"
            )
            args.append(relation_filter)
        else:
            sql = (
                f"SELECT * FROM relationships "
                f"WHERE from_entity IN ({placeholders}) OR to_entity IN ({placeholders})"
            )
        rels = conn.execute(sql, args).fetchall()

        new_frontier: set[str] = set()
        for r in rels:
            key = (r["from_entity"], r["relation"], r["to_entity"])
            if key not in seen_rels:
                seen_rels.add(key)
                all_rels.append(r)
            for name in (r["from_entity"], r["to_entity"]):
                if name not in visited:
                    new_frontier.add(name)
                    visited.add(name)
        frontier = new_frontier

    return visited, all_rels


def handle_query_knowledge_graph(tool_input: dict, db_path: Path) -> str:
    try:
        entity = tool_input.get("entity", "").strip()
        if not entity:
            return "ERROR: entity name is required."

        depth = min(max(int(tool_input.get("depth", 1)), 1), 3)
        relation_filter = tool_input.get("relation_filter", None) or None

        conn = _get_conn(db_path)
        try:
            ent_row = conn.execute(
                "SELECT * FROM entities WHERE name LIKE ? ORDER BY name LIMIT 1", (f"%{entity}%",)
            ).fetchone()

            if depth == 1 and not relation_filter:
                rels = conn.execute(
                    """
                    SELECT * FROM relationships
                    WHERE from_entity LIKE ? OR to_entity LIKE ?
                    ORDER BY from_entity, relation
                    LIMIT 50
                    """,
                    (f"%{entity}%", f"%{entity}%"),
                ).fetchall()
                visited: set[str] = set()
            else:
                visited, rels = _bfs_subgraph(conn, entity, depth, relation_filter)
        finally:
            conn.close()

        if not ent_row and not rels:
            return f"No knowledge found for '{entity}'. Add it with update_knowledge_graph."

        lines = []
        if ent_row:
            lines.append(f"**{ent_row['name']}** ({ent_row['type']})")
            if ent_row["description"]:
                lines.append(ent_row["description"])
            lines.append("")

        if depth > 1 and rels:
            lines.append(f"Subgraph (depth={depth}, {len(visited)} entities, {len(rels)} relationships):")
        elif rels:
            lines.append(f"Relationships ({len(rels)}):")

        for r in rels:
            note = f"  # {r['notes']}" if r["notes"] else ""
            lines.append(f"  {r['from_entity']} --[{r['relation']}]--> {r['to_entity']}{note}")

        return "\n".join(lines)
    except Exception as e:
        return f"ERROR: query_knowledge_graph failed — {e}"


UPDATE_SCHEMA: dict = {
    "name": "update_knowledge_graph",
    "description": (
        "Add entities and relationships to JARVIS's knowledge graph. "
        "Call this after researching a topic to build a long-term map of concepts. "
        "Example: RLHF --[uses]--> PPO, Anthropic --[developed]--> Constitutional AI."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "entities": {
                "type": "array",
                "description": "List of entities to add.",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "type": {"type": "string", "description": "e.g. technique, paper, person, company"},
                        "description": {"type": "string"},
                    },
                    "required": ["name"],
                },
            },
            "relationships": {
                "type": "array",
                "description": "List of relationships to add.",
                "items": {
                    "type": "object",
                    "properties": {
                        "from": {"type": "string", "description": "Source entity name"},
                        "relation": {"type": "string", "description": "Relationship type, e.g. uses, improves_on, developed_by"},
                        "to": {"type": "string", "description": "Target entity name"},
                        "notes": {"type": "string"},
                    },
                    "required": ["from", "relation", "to"],
                },
            },
        },
        "required": [],
    },
}

def export_graph(
    db_path: Path,
    user_id: str = "shared",
    limit: int = 500,
) -> dict:
    """Export the knowledge graph as {nodes, edges} for visualisation.

    Includes entities belonging to ``user_id`` or the shared namespace.
    ``limit`` caps the number of nodes returned; edges are included only when
    both endpoints appear in the node set.
    """
    if not db_path.exists():
        return {"nodes": [], "edges": []}
    try:
        conn = _get_conn(db_path)
        try:
            entity_rows = conn.execute(
                "SELECT name, type, description FROM entities "
                "WHERE user_id IN (?, 'shared') ORDER BY created_at DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
            node_names = {r["name"] for r in entity_rows}

            rel_rows = conn.execute(
                "SELECT from_entity, relation, to_entity, notes FROM relationships "
                "WHERE user_id IN (?, 'shared')",
                (user_id,),
            ).fetchall()
        finally:
            conn.close()

        nodes = [
            {"id": r["name"], "type": r["type"], "description": r["description"]}
            for r in entity_rows
        ]
        edges = [
            {
                "source": r["from_entity"],
                "relation": r["relation"],
                "target": r["to_entity"],
                "notes": r["notes"],
            }
            for r in rel_rows
            if r["from_entity"] in node_names and r["to_entity"] in node_names
        ]
        return {"nodes": nodes, "edges": edges}
    except Exception:
        return {"nodes": [], "edges": []}


def delete_entity(db_path: Path, name: str, user_id: str = "shared") -> bool:
    """Delete an entity and its relationships by name+user_id. Returns True if found."""
    if not db_path.exists():
        return False
    try:
        conn = _get_conn(db_path)
        try:
            cur = conn.execute(
                "DELETE FROM entities WHERE name = ? AND user_id = ?", (name, user_id)
            )
            deleted = cur.rowcount > 0
            if deleted:
                conn.execute(
                    "DELETE FROM relationships WHERE (from_entity = ? OR to_entity = ?) AND user_id = ?",
                    (name, name, user_id),
                )
            conn.commit()
        finally:
            conn.close()
        return deleted
    except Exception:
        return False


def delete_relationship(
    db_path: Path,
    from_entity: str,
    relation: str,
    to_entity: str,
    user_id: str = "shared",
) -> bool:
    """Delete a specific relationship triple. Returns True if found."""
    if not db_path.exists():
        return False
    try:
        conn = _get_conn(db_path)
        try:
            cur = conn.execute(
                "DELETE FROM relationships WHERE from_entity = ? AND relation = ? AND to_entity = ? AND user_id = ?",
                (from_entity, relation, to_entity, user_id),
            )
            deleted = cur.rowcount > 0
            conn.commit()
        finally:
            conn.close()
        return deleted
    except Exception:
        return False


def get_recent_entities(db_path: Path, user_id: str = "shared", limit: int = 20) -> list[str]:
    """Return the names of the most recently added entities in the knowledge graph."""
    try:
        conn = _get_conn(db_path)
        try:
            rows = conn.execute(
                "SELECT name FROM entities WHERE user_id IN (?, 'shared') ORDER BY created_at DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
        finally:
            conn.close()
        return [r["name"] for r in rows]
    except Exception:
        return []


QUERY_SCHEMA: dict = {
    "name": "query_knowledge_graph",
    "description": (
        "Query JARVIS's knowledge graph to find what is known about an entity "
        "and its relationships to other concepts. Use depth=2 or depth=3 to follow "
        "transitive connections across multiple hops."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "entity": {
                "type": "string",
                "description": "Entity name to look up, e.g. 'RLHF' or 'Anthropic'.",
            },
            "depth": {
                "type": "integer",
                "description": (
                    "How many relationship hops to follow. "
                    "1 = direct neighbors only (default). 2-3 = transitive discovery."
                ),
                "default": 1,
                "minimum": 1,
                "maximum": 3,
            },
            "relation_filter": {
                "type": "string",
                "description": (
                    "If given, only follow edges of this relation type, e.g. 'uses' or 'improves_on'. "
                    "Applies at all depths."
                ),
            },
        },
        "required": ["entity"],
    },
}
