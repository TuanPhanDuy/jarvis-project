"""TeamAgent — a role-specialised agent for multi-agent team coordination."""
from __future__ import annotations

from collections.abc import Callable

from jarvis.agents.base_agent import BaseAgent
from jarvis.prompts.loader import load_prompt

VALID_ROLES = ("manager", "team_lead", "frontend", "backend")

_ROLE_TOOLS: dict[str, frozenset[str]] = {
    "manager": frozenset({
        "delegate_task", "create_plan",
        "save_report", "update_report",
        "search_episodic_memory", "search_memory",
        "query_knowledge_graph", "update_knowledge_graph",
        "recall_user_preferences", "analyze_failures", "record_feedback",
    }),
    "team_lead": frozenset({
        "delegate_task",
        "save_report", "update_report",
        "search_episodic_memory", "search_memory",
        "query_knowledge_graph",
        "filesystem_search", "git_context",
        "analyze_text", "record_feedback",
    }),
    "frontend": frozenset({
        "execute_python", "filesystem_search", "git_context",
        "analyze_text",
        "save_report", "update_report",
        "search_episodic_memory", "search_memory",
        "query_knowledge_graph", "record_feedback",
    }),
    "backend": frozenset({
        "execute_python", "run_command",
        "database_query", "query_database",
        "filesystem_search", "git_context",
        "save_report", "update_report",
        "search_episodic_memory", "search_memory",
        "query_knowledge_graph", "update_knowledge_graph",
        "record_feedback",
    }),
}


class TeamAgent(BaseAgent):
    def __init__(
        self,
        model: str,
        max_tokens: int,
        tool_schemas: list[dict],
        tool_registry: dict[str, Callable[[dict], str]],
        role: str,
        approval_gate=None,
        session_id: str = "",
        user_id: str | None = None,
    ) -> None:
        if role not in VALID_ROLES:
            raise ValueError(f"Unknown team role '{role}'. Valid: {VALID_ROLES}")
        self._role = role
        allowed = _ROLE_TOOLS[role]
        filtered_schemas = [s for s in tool_schemas if s.get("name") in allowed]
        filtered_registry = {k: v for k, v in tool_registry.items() if k in allowed}
        super().__init__(
            model, max_tokens, filtered_schemas, filtered_registry,
            approval_gate=approval_gate, session_id=session_id, user_id=user_id,
        )

    def get_system_prompt(self) -> str:
        return load_prompt(self._role)
