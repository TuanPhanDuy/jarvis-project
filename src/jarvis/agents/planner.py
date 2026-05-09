from __future__ import annotations

from collections.abc import Callable

from jarvis.agents.base_agent import BaseAgent
from jarvis.prompts.loader import load_prompt


class PlannerAgent(BaseAgent):
    """Orchestrator that delegates to specialist sub-agents when appropriate."""

    def __init__(
        self,
        model: str,
        max_tokens: int,
        tool_schemas: list[dict],
        tool_registry: dict[str, Callable[[dict], str]],
        approval_gate=None,
        session_id: str = "",
        user_id: str | None = None,
        session_count: int = 0,
    ) -> None:
        super().__init__(
            model, max_tokens, tool_schemas, tool_registry,
            approval_gate=approval_gate, session_id=session_id, user_id=user_id,
        )
        self._messages: list[dict] = []
        self._session_count = session_count

    def get_system_prompt(self) -> str:
        base = load_prompt("planner")
        extras: list[str] = []

        if self._user_id and self._user_id != "anonymous":
            try:
                from jarvis.config import get_settings
                from jarvis.memory.preferences import get_preference_context, get_preferences
                from jarvis.memory.personality import get_personality_context
                db_path = get_settings().reports_dir / "jarvis.db"
                prefs = get_preferences(db_path, self._user_id)
                pref_ctx = get_preference_context(db_path, self._user_id)
                personality_ctx = get_personality_context(self._user_id, prefs, self._session_count)
                if personality_ctx:
                    extras.append(personality_ctx)
                if pref_ctx:
                    extras.append(pref_ctx)
            except Exception:
                pass

        return base + ("\n\n" + "\n\n".join(extras) if extras else "")

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
