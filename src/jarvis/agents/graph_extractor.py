"""Automatic knowledge graph extraction from agent responses.

After a ResearcherAgent produces a final response, this module runs a
lightweight extraction pass to populate entities and relationships without
requiring the agent to explicitly call update_knowledge_graph.

Called from a background daemon thread — zero latency impact on the response.
"""
from __future__ import annotations

from pathlib import Path

import structlog

log = structlog.get_logger()

_EXTRACTION_PROMPT = """\
Extract entities and relationships from the following text.

For each entity output one line: ENTITY|<name>|<type>
For each relationship output one line: REL|<from_entity>|<relation>|<to_entity>

Entity types: concept, technique, paper, person, organization, system, tool
Relation types: uses, improves_on, developed_by, is_part_of, related_to, contrasts_with, trained_on

Rules:
- Extract only clearly stated facts — no inference.
- Maximum 6 entities and 6 relationships.
- Entity names should be concise (1-4 words).
- Output ONLY ENTITY/REL lines — no other text.

Text:
{text}
"""


def _parse_extraction(text: str) -> tuple[list[dict], list[dict]]:
    entities: list[dict] = []
    relationships: list[dict] = []
    for line in text.strip().splitlines():
        parts = line.strip().split("|")
        if parts[0] == "ENTITY" and len(parts) >= 3:
            name, etype = parts[1].strip(), parts[2].strip()
            if name:
                entities.append({"name": name, "type": etype or "concept"})
        elif parts[0] == "REL" and len(parts) >= 4:
            frm, rel, to = parts[1].strip(), parts[2].strip(), parts[3].strip()
            if frm and rel and to:
                relationships.append({"from": frm, "relation": rel, "to": to})
    return entities, relationships


def extract_graph_from_text(
    text: str,
    db_path: Path,
    model: str,
    user_id: str | None = None,
) -> int:
    """Extract entities/relationships from text and upsert them into the knowledge graph.

    Returns the number of items extracted (entities + relationships), or 0 on failure.
    """
    if len(text) < 100:
        return 0
    try:
        import ollama
        resp = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": _EXTRACTION_PROMPT.format(text=text[:3000])}],
            options={"temperature": 0.1, "num_predict": 512},
        )
        entities, relationships = _parse_extraction(resp.message.content or "")
        if not entities and not relationships:
            return 0

        from jarvis.memory.graph import handle_update_knowledge_graph
        tool_input: dict = {
            "entities": entities,
            "relationships": relationships,
            "user_id": user_id or "shared",
        }
        result = handle_update_knowledge_graph(tool_input, db_path)
        count = len(entities) + len(relationships)
        log.info("auto_graph_extracted", entities=len(entities), relationships=len(relationships))
        return count
    except Exception as exc:
        log.debug("auto_graph_extraction_skipped", error=str(exc))
        return 0
