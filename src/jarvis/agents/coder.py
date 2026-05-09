"""CoderAgent — writes, runs, and explains code."""
from __future__ import annotations

from collections.abc import Callable

from jarvis.agents.base_agent import BaseAgent
from jarvis.prompts.loader import load_prompt


class CoderAgent(BaseAgent):
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
        super().__init__(
            model, max_tokens, tool_schemas, tool_registry,
            approval_gate=approval_gate, session_id=session_id, user_id=user_id,
        )

    def get_system_prompt(self) -> str:
        return load_prompt("coder")
