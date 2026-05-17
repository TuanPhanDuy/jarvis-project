from __future__ import annotations

import threading
from collections.abc import Callable

from jarvis.agents.base_agent import BaseAgent
from jarvis.prompts.loader import load_prompt

_RESEARCHER_TOOLS = {
    "web_search", "read_url", "browse",
    "search_memory", "search_episodic_memory", "query_knowledge_graph",
    "analyze_text", "summarize_youtube",
    "save_report", "update_report",
    "filesystem_search",
}


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
        filtered_schemas = [s for s in tool_schemas if s.get("name") in _RESEARCHER_TOOLS]
        filtered_registry = {k: v for k, v in tool_registry.items() if k in _RESEARCHER_TOOLS}
        super().__init__(
            model, max_tokens, filtered_schemas, filtered_registry,
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

    def run_turn(
        self,
        messages: list[dict],
        on_chunk: Callable[[str], None] | None = None,
    ) -> tuple[str, list[dict]]:
        # Proactive memory surfacing: inject relevant prior context into the user turn
        messages = self._inject_memory_context(messages)
        response, updated = super().run_turn(messages, on_chunk)
        # Auto knowledge graph extraction in background (zero latency impact)
        self._extract_graph_async(response)
        return response, updated

    def _inject_memory_context(self, messages: list[dict]) -> list[dict]:
        if not messages or messages[-1].get("role") != "user":
            return messages
        try:
            from jarvis.config import get_settings
            if not get_settings().proactive_memory_enabled:
                return messages
            from jarvis.memory.surfacing import surface_memory
            db_path = get_settings().reports_dir / "jarvis.db"
            query = str(messages[-1].get("content", ""))
            ctx = surface_memory(query, db_path, self._user_id)
            if ctx:
                augmented = list(messages)
                augmented[-1] = {
                    "role": "user",
                    "content": f"[Relevant memory context]\n{ctx}\n\n[User query]\n{query}",
                }
                return augmented
        except Exception:
            pass
        return messages

    def _extract_graph_async(self, response: str) -> None:
        try:
            from jarvis.config import get_settings
            if not get_settings().auto_graph_extraction:
                return
            db_path = get_settings().reports_dir / "jarvis.db"
            model = self._model
            user_id = self._user_id
        except Exception:
            return
        from jarvis.agents.graph_extractor import extract_graph_from_text
        threading.Thread(
            target=extract_graph_from_text,
            args=(response, db_path, model, user_id),
            daemon=True,
        ).start()

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
