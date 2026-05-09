from __future__ import annotations

from collections.abc import Callable

from jarvis.agents.base_agent import BaseAgent
from jarvis.prompts.loader import load_prompt


class ResearcherAgent(BaseAgent):
    def __init__(
        self,
        model: str,
        max_tokens: int,
        tool_schemas: list[dict],
        tool_registry: dict[str, Callable[[dict], str]],
        max_search_calls: int = 20,
        approval_gate=None,
        session_id: str = "",
        user_id: str | None = None,
    ) -> None:
        super().__init__(
            model, max_tokens, tool_schemas, tool_registry,
            approval_gate=approval_gate, session_id=session_id, user_id=user_id,
        )
        self._max_search_calls = max_search_calls
        self._search_calls_used = 0
        self._messages: list[dict] = []

    def get_system_prompt(self) -> str:
        remaining = self._max_search_calls - self._search_calls_used
        quota_note = (
            f"\n\nSearch quota: {remaining}/{self._max_search_calls} web searches remaining this session."
        )
        return load_prompt("researcher") + quota_note

    def on_tool_call(self, name: str) -> None:
        if name == "web_search":
            self._search_calls_used += 1

    def _before_dispatch(self, name: str, tool_input: dict) -> None:
        self.on_tool_call(name)
        super()._before_dispatch(name, tool_input)

    def get_messages(self) -> list[dict]:
        return self._messages

    def run_conversation(
        self,
        on_response: Callable[[str], None],
        on_thinking: Callable[[str], None],
        get_input: Callable[[], str],
        on_chunk: Callable[[str], None] | None = None,
    ) -> None:
        self._messages = []
        while True:
            user_input = get_input()
            if not user_input or user_input.strip().lower() in ("exit", "quit", "bye"):
                break
            self._messages.append({"role": "user", "content": user_input})
            on_thinking("JARVIS is thinking...")
            response_text, self._messages = self.run_turn(self._messages, on_chunk=on_chunk)
            on_response(response_text)

            if len(self._messages) > 40:
                self._messages = self._messages[-20:]
