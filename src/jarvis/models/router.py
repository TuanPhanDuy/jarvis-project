"""Smart model routing: select primary or fast model per agent turn.

Strategy "always_primary" (default): use the primary model for every turn.
Strategy "smart": use fast_model for mid-loop turns (tool results in context),
                  primary model for initial calls and final synthesis.

agent_model_map: optional dict mapping agent type names to model overrides,
                 taking highest precedence over strategy routing.
                 Example: {"coder": "codellama:7b", "researcher": "qwen2.5:14b"}
"""
from __future__ import annotations


class ModelRouter:
    def __init__(
        self,
        primary: str,
        fast: str,
        strategy: str = "always_primary",
        agent_model_map: dict[str, str] | None = None,
    ) -> None:
        self._primary = primary
        self._fast = fast
        self._strategy = strategy
        self._agent_model_map = agent_model_map or {}

    def select(self, messages: list[dict], agent_type: str = "") -> str:
        # Per-agent override has highest precedence
        if agent_type and agent_type in self._agent_model_map:
            return self._agent_model_map[agent_type]
        if self._strategy != "smart":
            return self._primary
        has_tool_results = any(m.get("role") == "tool" for m in messages)
        return self._fast if has_tool_results else self._primary
