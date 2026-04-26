from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import anthropic

from jarvis.tools import (
    browser,
    conversation_export,
    delegation,
    memory,
    os_command,
    plan_tool,
    report_writer,
    team_delegation,
    url_reader,
    web_search,
)
from jarvis.memory import episodic as episodic_memory
from jarvis.memory import failures as tool_failures
from jarvis.memory import feedback as memory_feedback
from jarvis.memory import graph as knowledge_graph
from jarvis.memory import preferences as user_preferences
from jarvis.vision import capture as vision_capture, face as vision_face
from jarvis.twin import main as digital_twin
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
    tavily_api_key: str,
    reports_dir: Path,
    get_messages: Callable[[], list[dict]] | None = None,
    allowed_commands: list[str] | None = None,
    user_id: str = "anonymous",
    anthropic_api_key: str = "",
) -> tuple[list[dict], dict[str, Callable[[dict], str]]]:
    """Return (tool_schemas, tool_dispatch_map) bound to the given config.

    Args:
        tavily_api_key: API key for Tavily web search.
        reports_dir: Directory where reports and exports are saved.
        get_messages: Optional callable that returns the current conversation history.
        allowed_commands: Allowlist of command basenames for run_command tool.
    """
    _allowed = allowed_commands or []
    db_path = reports_dir / "jarvis.db"
    _uid = user_id or "anonymous"

    # ── Built-in tool schemas ─────────────────────────────────────────────────
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
        # Digital Twin
        digital_twin.SNAPSHOT_SCHEMA,
        digital_twin.QUERY_SCHEMA,
    ]
    registry: dict[str, Callable[[dict], str]] = {
        "web_search":               lambda inp: web_search.handle_web_search(inp, tavily_api_key),
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
        "describe_scene":           lambda inp: vision_capture.handle_describe_scene(inp, reports_dir, anthropic_api_key),
        "recognize_face":           lambda inp: vision_face.handle_recognize_face(inp, reports_dir),
        # Digital Twin
        "snapshot_system":          lambda inp: digital_twin.handle_snapshot_system(inp, db_path),
        "query_system_twin":        lambda inp: digital_twin.handle_query_system_twin(inp, db_path),
    }

    # ── Plugin tools (auto-discovered from tools/plugins/) ────────────────────
    plugin_schemas, plugin_registry = load_plugins()
    schemas.extend(plugin_schemas)
    registry.update(plugin_registry)

    return schemas, registry


def build_planner_registry(
    base_schemas: list[dict],
    base_registry: dict[str, Callable[[dict], str]],
    client: anthropic.Anthropic,
    model: str,
    max_tokens: int,
    session_id: str = "",
    user_id: str | None = None,
) -> tuple[list[dict], dict[str, Callable[[dict], str]]]:
    """Extend a base registry with the delegate_task tool for PlannerAgent.

    Sub-agents spawned by the delegation handler receive base_schemas/base_registry
    (no delegate_task), preventing recursive delegation.

    Args:
        base_schemas: Tool schemas from build_registry() — no delegate_task.
        base_registry: Tool handlers from build_registry() — no delegate_task.
        client: Shared Anthropic client passed to spawned sub-agents.
        model: Model name for sub-agents.
        max_tokens: Token budget for sub-agent responses.
        session_id: Session identifier threaded into sub-agents.
        user_id: User identifier threaded into sub-agents.
    """
    from jarvis.config import get_settings

    settings = get_settings()
    db_path = settings.reports_dir / "jarvis.db"

    delegation_handler = delegation.build_delegation_handler(
        client=client,
        model=model,
        max_tokens=max_tokens,
        sub_tool_schemas=base_schemas,
        sub_tool_registry=base_registry,
    )
    plan_handler = plan_tool.build_plan_handler(
        client=client,
        model=model,
        fast_model=settings.fast_model,
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


def build_team_registry(
    base_schemas: list[dict],
    base_registry: dict[str, Callable[[dict], str]],
    client: anthropic.Anthropic,
    model: str,
    max_tokens: int,
) -> tuple[list[dict], dict[str, Callable[[dict], str]]]:
    """Build a registry for TeamAgent in Manager role.

    Hierarchy:
      Manager  → can delegate to team_lead, frontend, backend
      Team Lead → can delegate to frontend, backend
      Frontend / Backend → leaf nodes (base tools only)

    Args:
        base_schemas: Tool schemas from build_registry() — no team delegation.
        base_registry: Tool handlers from build_registry() — no team delegation.
        client: Shared Anthropic client passed to spawned sub-agents.
        model: Model name for sub-agents.
        max_tokens: Token budget for sub-agent responses.
    """
    # Leaf agents: frontend and backend get the base tool set only
    leaf_configs: dict[str, tuple[list[dict], dict[str, Callable[[dict], str]]]] = {
        "frontend": (base_schemas, base_registry),
        "backend": (base_schemas, base_registry),
    }

    # Team Lead can delegate to frontend and backend
    tl_handler = team_delegation.build_team_delegation_handler(
        client=client,
        model=model,
        max_tokens=max_tokens,
        role_configs=leaf_configs,
        allowed_roles=["frontend", "backend"],
    )
    tl_schemas = base_schemas + [team_delegation.SCHEMA]
    tl_registry: dict[str, Callable[[dict], str]] = dict(base_registry)
    tl_registry["delegate_to_team_member"] = tl_handler

    # Manager can delegate to team_lead (with its delegation power), frontend, backend
    manager_configs: dict[str, tuple[list[dict], dict[str, Callable[[dict], str]]]] = {
        "team_lead": (tl_schemas, tl_registry),
        "frontend": (base_schemas, base_registry),
        "backend": (base_schemas, base_registry),
    }
    mgr_handler = team_delegation.build_team_delegation_handler(
        client=client,
        model=model,
        max_tokens=max_tokens,
        role_configs=manager_configs,
        allowed_roles=["team_lead", "frontend", "backend"],
    )
    mgr_schemas = base_schemas + [team_delegation.SCHEMA]
    mgr_registry: dict[str, Callable[[dict], str]] = dict(base_registry)
    mgr_registry["delegate_to_team_member"] = mgr_handler

    return mgr_schemas, mgr_registry
