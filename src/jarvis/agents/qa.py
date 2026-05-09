"""QAAgent — reviews and tests code, returns a structured verdict."""
from __future__ import annotations

from collections.abc import Callable

from jarvis.agents.base_agent import BaseAgent
from jarvis.prompts.loader import load_prompt

_QA_TOOLS = {"execute_python", "filesystem_search", "git_context", "analyze_text"}


class QAAgent(BaseAgent):
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
        qa_schemas = [s for s in tool_schemas if s.get("name") in _QA_TOOLS]
        qa_registry = {k: v for k, v in tool_registry.items() if k in _QA_TOOLS}
        super().__init__(
            model, max_tokens, qa_schemas, qa_registry,
            approval_gate=approval_gate, session_id=session_id, user_id=user_id,
        )

    def get_system_prompt(self) -> str:
        return load_prompt("qa")
