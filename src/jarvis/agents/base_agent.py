from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

import anthropic

from jarvis.telemetry.tracing import get_tracer


class BaseAgent(ABC):
    """Generic agentic loop with tool dispatch.

    Subclasses implement get_system_prompt() and optionally override
    _before_dispatch() to hook into tool execution.
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
        self._client = client
        self._model = model
        self._max_tokens = max_tokens
        self._tool_schemas = tool_schemas
        self._tool_registry = tool_registry
        self._approval_gate = approval_gate
        self._session_id = session_id
        self._user_id = user_id
        self._input_tokens = 0
        self._output_tokens = 0
        self._cache_read_tokens = 0
        self._cache_write_tokens = 0

    @abstractmethod
    def get_system_prompt(self) -> str: ...

    def _before_dispatch(self, name: str, tool_input: dict) -> None:
        """Hook called before each tool is dispatched.

        If an approval_gate is configured and the tool requires approval, this
        blocks until the user approves or the timeout elapses.  Raises
        ToolDeniedException if the user explicitly denies.
        """
        if self._approval_gate and self._approval_gate.requires_approval(name):
            from jarvis.security.approval import ToolDeniedException
            approved = self._approval_gate.check_sync(name, tool_input)
            if not approved:
                raise ToolDeniedException(f"Tool '{name}' was denied by the user.")

    def _make_api_params(self, messages: list[dict]) -> dict:
        return dict(
            model=self._model,
            max_tokens=self._max_tokens,
            system=[
                {
                    "type": "text",
                    "text": self.get_system_prompt(),
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            tools=self._tool_schemas,
            messages=messages,
        )

    def _accumulate_usage(self, usage: object) -> None:
        self._input_tokens += getattr(usage, "input_tokens", 0) or 0
        self._output_tokens += getattr(usage, "output_tokens", 0) or 0
        self._cache_read_tokens += getattr(usage, "cache_read_input_tokens", 0) or 0
        self._cache_write_tokens += getattr(usage, "cache_creation_input_tokens", 0) or 0

    def get_usage_summary(self) -> dict:
        """Return accumulated token counts and estimated cost for this session."""
        input_cost = self._input_tokens * 3.0 / 1_000_000
        output_cost = self._output_tokens * 15.0 / 1_000_000
        cache_write_cost = self._cache_write_tokens * 3.75 / 1_000_000
        cache_read_cost = self._cache_read_tokens * 0.30 / 1_000_000
        return {
            "input_tokens": self._input_tokens,
            "output_tokens": self._output_tokens,
            "cache_write_tokens": self._cache_write_tokens,
            "cache_read_tokens": self._cache_read_tokens,
            "estimated_cost_usd": round(
                input_cost + output_cost + cache_write_cost + cache_read_cost, 6
            ),
        }

    def run_turn(
        self,
        messages: list[dict],
        on_chunk: Callable[[str], None] | None = None,
    ) -> tuple[str, list[dict]]:
        """Send messages to Claude, dispatch tools if needed, return final text.

        Args:
            messages: Conversation history.
            on_chunk: If provided, stream text chunks to this callback as they arrive.

        Returns (response_text, updated_messages_list). Mutates nothing.
        """
        tracer = get_tracer()
        with tracer.start_as_current_span("agent.run_turn") as span:
            span.set_attribute("model", self._model)
            span.set_attribute("messages_count", len(messages))
            return self._run_turn_inner(messages, on_chunk)

    def _run_turn_inner(
        self,
        messages: list[dict],
        on_chunk: Callable[[str], None] | None = None,
    ) -> tuple[str, list[dict]]:
        params = self._make_api_params(messages)

        if on_chunk is not None:
            with self._client.messages.stream(**params) as stream:  # type: ignore[arg-type]
                for chunk in stream.text_stream:
                    on_chunk(chunk)
                response = stream.get_final_message()
        else:
            response = self._client.messages.create(**params)  # type: ignore[arg-type]

        self._accumulate_usage(response.usage)

        if response.stop_reason == "end_turn":
            text = self._extract_text(response.content)
            updated = messages + [{"role": "assistant", "content": response.content}]
            return text, updated

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    try:
                        self._before_dispatch(block.name, block.input)
                        result = self._dispatch(block.name, block.input)
                    except Exception as exc:
                        from jarvis.security.approval import ToolDeniedException
                        from jarvis.security.approval import TOOL_RISK_MAP, RiskLevel
                        if isinstance(exc, ToolDeniedException):
                            result = f"DENIED: {exc}"
                            self._write_audit(
                                block.name, block.input, result_ok=0, duration_ms=0,
                                approved=0, approver=f"user:{self._user_id or 'unknown'}",
                            )
                        else:
                            result = f"ERROR: {exc}"
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result,
                        }
                    )

            updated = messages + [
                {"role": "assistant", "content": response.content},
                {"role": "user", "content": tool_results},
            ]
            return self._run_turn_inner(updated, on_chunk=on_chunk)

        # Unexpected stop reason — surface what we have
        text = self._extract_text(response.content)
        updated = messages + [{"role": "assistant", "content": response.content}]
        return text or f"[Stopped: {response.stop_reason}]", updated

    def _dispatch(self, name: str, tool_input: dict) -> str:
        handler = self._tool_registry.get(name)
        if handler is None:
            return f"ERROR: unknown tool '{name}'"
        import time
        tracer = get_tracer()
        t0 = time.perf_counter()
        with tracer.start_as_current_span(f"tool.{name}") as span:
            span.set_attribute("tool.name", name)
            result = handler(tool_input)
            duration_ms = (time.perf_counter() - t0) * 1000
            result_ok = 0 if result.startswith("ERROR") else 1
            if not result_ok:
                span.set_attribute("tool.error", result)
                self._record_failure(name, tool_input, result)
            needed_approval = self._approval_gate and self._approval_gate.requires_approval(name)
            self._write_audit(
                name, tool_input, result_ok, duration_ms,
                approved=1 if needed_approval else -1,
                approver=f"user:{self._user_id or 'unknown'}" if needed_approval else "auto",
            )
            return result

    def _record_failure(self, name: str, tool_input: dict, error: str) -> None:
        try:
            from jarvis.memory.failures import log_failure
            from jarvis.config import get_settings
            db_path = get_settings().reports_dir / "jarvis.db"
            log_failure(db_path, name, tool_input, error)
        except Exception:
            pass

    def _write_audit(
        self,
        name: str,
        tool_input: dict,
        result_ok: int,
        duration_ms: float,
        approved: int = -1,
        approver: str = "auto",
    ) -> None:
        try:
            from jarvis.security.audit import log_tool_call
            from jarvis.security.approval import TOOL_RISK_MAP, RiskLevel
            from jarvis.config import get_settings
            db_path = get_settings().reports_dir / "jarvis.db"
            risk = TOOL_RISK_MAP.get(name, RiskLevel.LOW).name
            log_tool_call(
                db_path=db_path,
                session_id=self._session_id,
                tool_name=name,
                tool_input=tool_input,
                risk_level=risk,
                approved=approved,
                approver=approver,
                result_ok=result_ok,
                duration_ms=duration_ms,
                user_id=self._user_id,
            )
        except Exception:
            pass

    @staticmethod
    def _extract_text(content: list) -> str:
        parts = [block.text for block in content if hasattr(block, "text")]
        return "\n".join(parts).strip()
