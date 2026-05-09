"""DataAnalystAgent — queries, analyses, and visualises structured data."""
from __future__ import annotations

from collections.abc import Callable

from jarvis.agents.base_agent import BaseAgent
from jarvis.prompts.loader import load_prompt

_ANALYST_TOOLS = {
    "query_database",
    "execute_python",
    "filesystem_search",
    "analyze_text",
    "save_report", "update_report",
    "search_memory",
    "ingest_document",
}


class DataAnalystAgent(BaseAgent):
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
        filtered_schemas = [s for s in tool_schemas if s.get("name") in _ANALYST_TOOLS]
        filtered_registry = {k: v for k, v in tool_registry.items() if k in _ANALYST_TOOLS}
        super().__init__(
            model, max_tokens, filtered_schemas, filtered_registry,
            approval_gate=approval_gate, session_id=session_id, user_id=user_id,
        )

    def get_system_prompt(self) -> str:
        return load_prompt("data_analyst")
