from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Callable

import ollama

from jarvis.telemetry.tracing import get_tracer


class BaseAgent(ABC):
    """Generic agentic loop powered by a local Ollama model.

    Subclasses implement get_system_prompt() and optionally override
    _before_dispatch() to hook into tool execution.
    """

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
        self._model = model
        self._max_tokens = max_tokens
        self._tool_schemas = tool_schemas
        self._tool_registry = tool_registry
        self._approval_gate = approval_gate
        self._session_id = session_id
        self._user_id = user_id
        self._prompt_tokens = 0
        self._completion_tokens = 0

    @abstractmethod
    def get_system_prompt(self) -> str: ...

    def _to_ollama_tools(self) -> list[dict]:
        """Convert tool schemas from internal format to Ollama function-call format."""
        tools = []
        for s in self._tool_schemas:
            tools.append({
                "type": "function",
                "function": {
                    "name": s["name"],
                    "description": s.get("description", ""),
                    "parameters": s.get("input_schema", {"type": "object", "properties": {}}),
                },
            })
        return tools

    def _before_dispatch(self, name: str, tool_input: dict) -> None:
        if self._approval_gate and self._approval_gate.requires_approval(name):
            from jarvis.security.approval import ToolDeniedException
            approved = self._approval_gate.check_sync(name, tool_input)
            if not approved:
                raise ToolDeniedException(f"Tool '{name}' was denied by the user.")

    def get_usage_summary(self) -> dict:
        return {
            "input_tokens": self._prompt_tokens,
            "output_tokens": self._completion_tokens,
            "cache_write_tokens": 0,
            "cache_read_tokens": 0,
            "estimated_cost_usd": 0.0,
        }

    def run_turn(
        self,
        messages: list[dict],
        on_chunk: Callable[[str], None] | None = None,
    ) -> tuple[str, list[dict]]:
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
        system = self.get_system_prompt()
        all_messages = [{"role": "system", "content": system}] + messages
        tools = self._to_ollama_tools()

        if on_chunk is not None:
            # Stream text but collect tool_calls from the final chunk
            full_text = ""
            tool_calls = []
            for chunk in ollama.chat(
                model=self._model,
                messages=all_messages,
                tools=tools or None,
                stream=True,
                options={"num_predict": self._max_tokens},
            ):
                if chunk.message.content:
                    on_chunk(chunk.message.content)
                    full_text += chunk.message.content
                if chunk.message.tool_calls:
                    tool_calls = chunk.message.tool_calls
                if chunk.done:
                    self._prompt_tokens += getattr(chunk, "prompt_eval_count", 0) or 0
                    self._completion_tokens += getattr(chunk, "eval_count", 0) or 0

            if tool_calls:
                updated, tool_msgs = self._execute_tool_calls(messages, full_text, tool_calls)
                return self._run_turn_inner(updated + tool_msgs, on_chunk)

            return full_text, messages + [{"role": "assistant", "content": full_text}]

        else:
            response = ollama.chat(
                model=self._model,
                messages=all_messages,
                tools=tools or None,
                options={"num_predict": self._max_tokens},
            )
            self._prompt_tokens += getattr(response, "prompt_eval_count", 0) or 0
            self._completion_tokens += getattr(response, "eval_count", 0) or 0

            if response.message.tool_calls:
                updated, tool_msgs = self._execute_tool_calls(
                    messages,
                    response.message.content or "",
                    response.message.tool_calls,
                )
                return self._run_turn_inner(updated + tool_msgs, on_chunk)

            text = response.message.content or ""
            return text, messages + [{"role": "assistant", "content": text}]

    def _execute_tool_calls(
        self,
        messages: list[dict],
        assistant_text: str,
        tool_calls: list,
    ) -> tuple[list[dict], list[dict]]:
        """Dispatch all tool calls and return (updated_messages, tool_result_messages)."""
        # Record the assistant turn with tool calls
        assistant_msg: dict = {"role": "assistant", "content": assistant_text}
        if tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "function": {
                        "name": tc.function.name,
                        "arguments": json.loads(tc.function.arguments)
                        if isinstance(tc.function.arguments, str)
                        else tc.function.arguments,
                    }
                }
                for tc in tool_calls
            ]
        updated = messages + [assistant_msg]

        tool_msgs = []
        for tc in tool_calls:
            name = tc.function.name
            args = tc.function.arguments
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            try:
                self._before_dispatch(name, args)
                result = self._dispatch(name, args)
            except Exception as exc:
                from jarvis.security.approval import ToolDeniedException
                if isinstance(exc, ToolDeniedException):
                    result = f"DENIED: {exc}"
                else:
                    result = f"ERROR: {exc}"
            tool_msgs.append({"role": "tool", "content": result})

        return updated, tool_msgs

    def _dispatch(self, name: str, tool_input: dict) -> str:
        handler = self._tool_registry.get(name)
        if handler is None:
            return f"ERROR: unknown tool '{name}'"
        import time
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as _Timeout
        tracer = get_tracer()
        t0 = time.perf_counter()
        with tracer.start_as_current_span(f"tool.{name}") as span:
            span.set_attribute("tool.name", name)
            try:
                tool_timeout = self._tool_timeout_seconds()
                with ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(handler, tool_input)
                    result = future.result(timeout=tool_timeout)
            except _Timeout:
                result = f"ERROR: tool '{name}' timed out after {tool_timeout}s"
            except Exception as exc:
                result = f"ERROR: tool '{name}' raised — {exc}"
            duration_ms = (time.perf_counter() - t0) * 1000
            result_ok = 0 if str(result).startswith("ERROR") else 1
            if not result_ok:
                span.set_attribute("tool.error", result)
                self._record_failure(name, tool_input, result)
            self._record_tool_metric(name, duration_ms / 1000)
            self._write_audit(name, tool_input, result_ok, duration_ms)
            return result

    def _tool_timeout_seconds(self) -> int:
        try:
            from jarvis.config import get_settings
            return get_settings().tool_timeout_seconds
        except Exception:
            return 60

    def _record_tool_metric(self, name: str, duration_s: float) -> None:
        try:
            from jarvis.api.metrics import TOOL_DURATION
            TOOL_DURATION.labels(tool_name=name).observe(duration_s)
        except Exception:
            pass

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
                approved=-1,
                approver="auto",
                result_ok=result_ok,
                duration_ms=duration_ms,
                user_id=self._user_id,
            )
        except Exception:
            pass
