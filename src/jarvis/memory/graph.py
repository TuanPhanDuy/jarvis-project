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
        conn.close()
        return f"Knowledge graph updated: {added_ents} entity/entities, {added_rels} relationship(s) added."
    except Exception as e:
        return f"ERROR: update_knowledge_graph failed — {e}"


def handle_query_knowledge_graph(tool_input: dict, db_path: Path) -> str:
    try:
        entity = tool_input.get("entity", "").strip()
        if not entity:
            return "ERROR: entity name is required."

        conn = _get_conn(db_path)

        # Look up the entity
        ent_row = conn.execute(
            "SELECT * FROM entities WHERE name LIKE ? ORDER BY name LIMIT 1", (f"%{entity}%",)
        ).fetchone()

        # Find all relationships involving this entity
        rels = conn.execute(
            """
            SELECT * FROM relationships
            WHERE from_entity LIKE ? OR to_entity LIKE ?
            ORDER BY from_entity, relation
            LIMIT 50
            """,
            (f"%{entity}%", f"%{entity}%"),
        ).fetchall()
        conn.close()

        if not ent_row and not rels:
            return f"No knowledge found for '{entity}'. Add it with update_knowledge_graph."

        lines = []
        if ent_row:
            lines.append(f"**{ent_row['name']}** ({ent_row['type']})")
            if ent_row["description"]:
                lines.append(ent_row["description"])
            lines.append("")

        if rels:
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

QUERY_SCHEMA: dict = {
    "name": "query_knowledge_graph",
    "description": (
        "Query JARVIS's knowledge graph to find what is known about an entity "
        "and its relationships to other concepts."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "entity": {
                "type": "string",
                "description": "Entity name to look up, e.g. 'RLHF' or 'Anthropic'.",
            }
        },
        "required": ["entity"],
    },
}
