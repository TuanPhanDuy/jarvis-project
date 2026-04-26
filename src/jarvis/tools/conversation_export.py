"""Tool: export_conversation — save the full conversation history as JSON."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


def handle_export_conversation(
    tool_input: dict,
    get_messages: object,  # Callable[[], list[dict]]
    reports_dir: Path,
) -> str:
    try:
        from collections.abc import Callable
        assert callable(get_messages)
        topic = tool_input.get("topic", "conversation")
        slug = topic.lower().replace(" ", "-")[:40]
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        reports_dir.mkdir(parents=True, exist_ok=True)
        filename = reports_dir / f"{timestamp}-{slug}-conversation.json"

        messages = get_messages()
        # Serialize — content blocks may be Anthropic SDK objects, convert to dicts
        serializable = []
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, list):
                content = [
                    (block.model_dump() if hasattr(block, "model_dump") else block)
                    for block in content
                ]
            serializable.append({"role": msg.get("role"), "content": content})

        filename.write_text(
            json.dumps({"exported_at": timestamp, "messages": serializable}, indent=2),
            encoding="utf-8",
        )
        return f"Conversation exported to: {filename}"
    except Exception as e:
        return f"ERROR: export_conversation failed — {e}"


SCHEMA: dict = {
    "name": "export_conversation",
    "description": (
        "Export the full conversation history to a JSON file alongside the research reports. "
        "Useful for debugging, reviewing your research process, or sharing the full context."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "topic": {
                "type": "string",
                "description": "Topic label used in the filename (e.g. 'rlhf-research').",
            },
        },
        "required": ["topic"],
    },
}
