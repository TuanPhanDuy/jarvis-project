from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from jarvis.tools import (
    browser,
    conversation_export,
    delegation,
    memory,
    os_command,
    plan_tool,
    report_writer,
    url_reader,
    web_search,
)
from jarvis.memory import episodic as episodic_memory
from jarvis.memory import failures as tool_failures
from jarvis.memory import feedback as memory_feedback
from jarvis.memory import graph as knowledge_graph
from jarvis.memory import preferences as user_preferences
from jarvis.vision import capture as vision_capture, face as vision_face
from jarvis.tools.plugin_loader import load_plugins


def _save_report_and_index(tool_input: dict, reports_dir: Path) -> str:
    result = report_writer.handle_save_report(tool_input, reports_dir)
    if not result.startswith("ERROR"):
        try:
            saved_path = Path(result.split("Report saved to: ", 1)[1].strip())
            memory.index_new_report(reports_dir, saved_path.name)
        except Exception:
            pass
    return result


def build_registry(
    reports_dir: Path,
    get_messages: Callable[[], list[dict]] | None = None,
    allowed_commands: list[str] | None = None,
    user_id: str = "anonymous",
    vision_model: str = "llava:13b",
) -> tuple[list[dict], dict[str, Callable[[dict], str]]]:
    """Return (tool_schemas, tool_dispatch_map) bound to the given config."""
    _allowed = allowed_commands or []
    db_path = reports_dir / "jarvis.db"
    _uid = user_id or "anonymous"

    schemas = [
        web_search.SCHEMA,
        report_writer.SCHEMA,
        report_writer.UPDATE_SCHEMA,
        url_reader.SCHEMA,
        conversation_export.SCHEMA,
        memory.SCHEMA,
        os_command.SCHEMA,
        browser.SCHEMA,
        # Layered memory
        episodic_memory.SCHEMA,
        knowledge_graph.UPDATE_SCHEMA,
        knowledge_graph.QUERY_SCHEMA,
        tool_failures.SCHEMA,
        memory_feedback.SCHEMA,
        # User preferences
        user_preferences.UPDATE_SCHEMA,
        user_preferences.RECALL_SCHEMA,
        # Vision
        vision_capture.SCHEMA,
        vision_capture.DESCRIBE_SCHEMA,
        vision_face.SCHEMA,
    ]
    registry: dict[str, Callable[[dict], str]] = {
        "web_search":               lambda inp: web_search.handle_web_search(inp),
        "save_report":              lambda inp: _save_report_and_index(inp, reports_dir),
        "update_report":            lambda inp: report_writer.handle_update_report(inp, reports_dir),
        "read_url":                 lambda inp: url_reader.handle_read_url(inp),
        "export_conversation":      lambda inp: conversation_export.handle_export_conversation(
            inp, get_messages or (lambda: []), reports_dir
        ),
        "search_memory":            lambda inp: memory.handle_search_memory(inp, reports_dir),
        "run_command":              lambda inp: os_command.handle_run_command(inp, _allowed),
        "browse":                   lambda inp: browser.handle_browse(inp, screenshots_dir=reports_dir),
        # Layered memory
        "search_episodic_memory":   lambda inp: episodic_memory.handle_search_episodic_memory(inp, db_path),
        "update_knowledge_graph":   lambda inp: knowledge_graph.handle_update_knowledge_graph(inp, db_path),
        "query_knowledge_graph":    lambda inp: knowledge_graph.handle_query_knowledge_graph(inp, db_path),
        "analyze_failures":         lambda inp: tool_failures.handle_analyze_failures(inp, db_path),
        "record_feedback":          lambda inp: memory_feedback.handle_record_feedback(inp, db_path),
        # User preferences
        "update_user_preference":   lambda inp: user_preferences.handle_update_user_preference(inp, db_path, _uid),
        "recall_user_preferences":  lambda inp: user_preferences.handle_recall_user_preferences(inp, db_path, _uid),
        # Vision
        "capture_camera":           lambda inp: vision_capture.handle_capture_camera(inp, reports_dir),
        "describe_scene":           lambda inp: vision_capture.handle_describe_scene(inp, reports_dir, vision_model),
        "recognize_face":           lambda inp: vision_face.handle_recognize_face(inp, reports_dir),
    }

    plugin_schemas, plugin_registry = load_plugins()
    schemas.extend(plugin_schemas)
    registry.update(plugin_registry)

    return schemas, registry


def build_planner_registry(
    base_schemas: list[dict],
    base_registry: dict[str, Callable[[dict], str]],
    model: str,
    max_tokens: int,
    db_path: Path | None = None,
    session_id: str = "",
    user_id: str | None = None,
) -> tuple[list[dict], dict[str, Callable[[dict], str]]]:
    """Extend a base registry with delegate_task and create_plan for PlannerAgent."""
    delegation_handler = delegation.build_delegation_handler(
        model=model,
        max_tokens=max_tokens,
        sub_tool_schemas=base_schemas,
        sub_tool_registry=base_registry,
    )
    plan_handler = plan_tool.build_plan_handler(
        model=model,
        max_tokens=max_tokens,
        sub_tool_schemas=base_schemas,
        sub_tool_registry=base_registry,
        db_path=db_path,
        session_id=session_id,
        user_id=user_id,
    )
    planner_schemas = base_schemas + [delegation.SCHEMA, plan_tool.SCHEMA]
    planner_registry = dict(base_registry)
    planner_registry["delegate_task"] = delegation_handler
    planner_registry["create_plan"] = plan_handler
    return planner_schemas, planner_registry
