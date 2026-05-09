"""DevOpsAgent — system diagnostics, shell commands, and infrastructure tasks."""
from __future__ import annotations

from collections.abc import Callable

from jarvis.agents.base_agent import BaseAgent
from jarvis.prompts.loader import load_prompt

_DEVOPS_TOOLS = {
    "system_info",
    "run_command",
    "git_context",
    "filesystem_search",
    "save_report", "update_report",
    "search_memory",
    "query_database",
}


class DevOpsAgent(BaseAgent):
    def __init__(
        self,
        model: str,
        max_tokens: int,
        tool_schemas: list[dict],
        tool_registry: dict[str, Callable[[dict], str]],
        approval_gate=None,
        session_id: str = "",
        user_id: str | None = None,
    ) -> None:
        filtered_schemas = [s for s in tool_schemas if s.get("name") in _DEVOPS_TOOLS]
        filtered_registry = {k: v for k, v in tool_registry.items() if k in _DEVOPS_TOOLS}
        super().__init__(
            model, max_tokens, filtered_schemas, filtered_registry,
            approval_gate=approval_gate, session_id=session_id, user_id=user_id,
        )

    def get_system_prompt(self) -> str:
        return load_prompt("devops")
