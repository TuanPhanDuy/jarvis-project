"""QAAgent — specialized sub-agent for code review and testing."""
from __future__ import annotations

from collections.abc import Callable

import anthropic

from jarvis.agents.base_agent import BaseAgent
from jarvis.prompts.loader import load_prompt


class QAAgent(BaseAgent):
    """Sub-agent that reviews code for correctness, edge cases, and quality.

    Used exclusively as a delegation target from PlannerAgent — it has no
    interactive REPL. The planner calls run_turn() with a task and
    receives a structured review as a string.
    """

    def __init__(
        self,
        client: anthropic.Anthropic,
        model: str,
        max_tokens: int,
        tool_schemas: list[dict],
        tool_registry: dict[str, Callable[[dict], str]],
        approval_gate=None,
        session_id: str = "",
        user_id: str | None = None,
    ) -> None:
        super().__init__(
            client, model, max_tokens, tool_schemas, tool_registry,
            approval_gate=approval_gate, session_id=session_id, user_id=user_id,
        )

    def get_system_prompt(self) -> str:
        return load_prompt("qa")
