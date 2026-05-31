from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from collections.abc import Callable

import ollama
import structlog

from jarvis.models.router import ModelRouter
from jarvis.telemetry.tracing import get_tracer

log = structlog.get_logger()

_COMPRESS_KEEP_RECENT = 10  # always preserve this many recent messages
_CHARS_PER_TOKEN = 4        # rough estimate for token budgeting

_HEDGE_PHRASES = frozenset([
    "i'm not sure", "i'm uncertain", "i think", "might be", "could be",
    "i believe", "not certain", "i'm unsure", "possibly", "unclear",
    "i cannot be certain", "it's possible", "i'm not confident",
])


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
        self._turn_tool_calls: list[str] = []
        self._turn_count = 0
        self._last_messages: list[dict] = []
        try:
            from jarvis.config import get_settings
            s = get_settings()
            self._router = ModelRouter(model, s.fast_model, s.routing_strategy, s.agent_model_map)
        except Exception:
            self._router = ModelRouter(model, model, "always_primary")

    @abstractmethod
    def get_system_prompt(self) -> str: ...

    def _agent_type_key(self) -> str:
        """Return lowercase agent type name for model routing lookup (e.g. 'coder', 'researcher')."""
        name = type(self).__name__.lower()
        return name[:-5] if name.endswith("agent") else name

    def _to_ollama_tools(self) -> list[dict]:
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

    def _settings_flag(self, attr: str, default: bool) -> bool:
        try:
            from jarvis.config import get_settings
            return getattr(get_settings(), attr, default)
        except Exception:
            return default

    def _coaching_prefix(self) -> str:
        """Return tool-failure warnings to prepend to the system prompt, or empty string."""
        try:
            from jarvis.config import get_settings
            from jarvis.agents.failure_coach import get_failure_warnings
            db_path = get_settings().reports_dir / "jarvis.db"
            return get_failure_warnings(db_path)
        except Exception:
            return ""

    @staticmethod
    def _detect_hedges(text: str) -> bool:
        lower = text.lower()
        return any(phrase in lower for phrase in _HEDGE_PHRASES)

    def _reflect(self, response: str) -> str:
        """Silently review the draft response; return a revision if the model suggests one."""
        if len(response) < 100:
            return response
        prompt = (
            "Review this AI response. If it is complete, accurate, and well-structured, "
            "reply with exactly: LGTM\n\n"
            "If it needs improvement, reply with a revised version only — no explanation.\n\n"
            f"Response to review:\n{response[:2000]}"
        )
        try:
            resp = ollama.chat(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                options={"num_predict": min(self._max_tokens, 1024)},
            )
            revised = resp.message.content.strip()
            if revised and not revised.upper().startswith("LGTM"):
                log.info("reflection_revised", agent=type(self).__name__,
                         original_len=len(response), revised_len=len(revised))
                return revised
        except Exception:
            pass
        return response

    def _context_budget_tokens(self) -> int:
        try:
            from jarvis.config import get_settings
            return get_settings().context_budget_tokens
        except Exception:
            return 4096

    @staticmethod
    def _estimate_tokens(messages: list[dict]) -> int:
        total_chars = sum(len(str(m.get("content", ""))) for m in messages)
        return total_chars // _CHARS_PER_TOKEN

    def _compress_history(self, messages: list[dict]) -> list[dict]:
        """Summarize oldest messages when estimated token count exceeds the budget."""
        if self._estimate_tokens(messages) <= self._context_budget_tokens():
            return messages
        to_compress = messages[:-_COMPRESS_KEEP_RECENT]
        recent = messages[-_COMPRESS_KEEP_RECENT:]
        lines = [
            f"[{m['role'].upper()}]: {str(m.get('content', ''))[:400]}"
            for m in to_compress
            if m.get("role") in ("user", "assistant") and m.get("content")
        ]
        if not lines:
            return recent
        prompt = "Summarize this conversation in 3-5 sentences, preserving key facts, decisions, and open questions:\n\n" + "\n".join(lines)
        try:
            resp = ollama.chat(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                options={"num_predict": 300},
            )
            summary = resp.message.content.strip()
            log.info("history_compressed", agent=type(self).__name__,
                     summarized=len(to_compress), kept=len(recent))
            return [{"role": "system", "content": f"[Prior conversation summary]: {summary}"}] + recent
        except Exception:
            return recent  # fallback: truncate without summary

    def run_turn_structured(
        self,
        messages: list[dict],
        json_schema: dict,
    ) -> tuple[dict, list[dict]]:
        """Run a single turn and return a parsed JSON object matching json_schema.

        Uses Ollama's native format parameter when available; falls back to a
        prompt-level instruction asking for JSON.  Returns (parsed_dict, updated_messages).
        Raises ValueError if the model output cannot be parsed as JSON.
        """
        import json as _json

        messages = self._compress_history(messages)
        messages = self._surface_memory_context(messages)
        system = self.get_system_prompt()
        schema_hint = _json.dumps(json_schema, indent=2)
        json_instruction = (
            f"\n\nRespond ONLY with a valid JSON object matching this schema:\n{schema_hint}\n"
            "Do not include any text outside the JSON."
        )
        all_messages = [{"role": "system", "content": system + json_instruction}] + messages
        model = self._router.select(messages, agent_type=self._agent_type_key())
        try:
            response = ollama.chat(
                model=model,
                messages=all_messages,
                format="json",
                options={"num_predict": self._max_tokens},
            )
        except Exception:
            response = ollama.chat(
                model=model,
                messages=all_messages,
                options={"num_predict": self._max_tokens},
            )
        self._prompt_tokens += getattr(response, "prompt_eval_count", 0) or 0
        self._completion_tokens += getattr(response, "eval_count", 0) or 0
        raw = (response.message.content or "").strip()
        try:
            parsed = _json.loads(raw)
        except _json.JSONDecodeError as exc:
            raise ValueError(f"Model did not return valid JSON: {exc}\n\nRaw output: {raw[:500]}")
        updated = messages + [{"role": "assistant", "content": raw}]
        return parsed, updated

    def run_turn(
        self,
        messages: list[dict],
        on_chunk: Callable[[str], None] | None = None,
    ) -> tuple[str, list[dict]]:
        t0 = time.perf_counter()
        self._turn_tool_calls = []
        tracer = get_tracer()
        with tracer.start_as_current_span("agent.run_turn") as span:
            span.set_attribute("model", self._model)
            span.set_attribute("messages_count", len(messages))
            result = self._run_turn_inner(messages, on_chunk)
        self._turn_count += 1
        self._last_messages = result[1]
        self._log_turn((time.perf_counter() - t0) * 1000)
        self._maybe_checkpoint(result[1])
        return result

    def _log_turn(self, latency_ms: float) -> None:
        try:
            from jarvis.memory.turns import log_turn
            from jarvis.config import get_settings
            db_path = get_settings().reports_dir / "jarvis.db"
            log_turn(
                db_path=db_path,
                session_id=self._session_id,
                agent_type=type(self).__name__,
                model=self._model,
                input_tokens=self._prompt_tokens,
                output_tokens=self._completion_tokens,
                tool_calls=self._turn_tool_calls,
                latency_ms=latency_ms,
            )
        except Exception:
            pass

    def _maybe_checkpoint(self, messages: list[dict]) -> None:
        """Save a checkpoint every N turns when checkpointing is configured."""
        if not self._session_id:
            return
        try:
            from jarvis.config import get_settings
            interval = get_settings().checkpoint_interval
            if not isinstance(interval, int):
                interval = 0
        except Exception:
            interval = 0
        if interval <= 0 or self._turn_count % interval != 0:
            return
        try:
            from jarvis.agents.checkpoint import save_checkpoint
            from jarvis.config import get_settings
            db_path = get_settings().reports_dir / "jarvis.db"
            save_checkpoint(db_path, self._session_id, self._turn_count, type(self).__name__, messages)
        except Exception:
            pass

    def _surface_memory_context(self, messages: list[dict]) -> list[dict]:
        """Prepend relevant prior context to the last user message when proactive_memory is on."""
        if not self._settings_flag("proactive_memory_enabled", False):
            return messages
        last_user = next(
            (m for m in reversed(messages) if m.get("role") == "user"),
            None,
        )
        if not last_user:
            return messages
        query = str(last_user.get("content", ""))
        try:
            from jarvis.memory.surfacing import surface_memory
            from jarvis.config import get_settings
            db_path = get_settings().reports_dir / "jarvis.db"
            ctx = surface_memory(query, db_path, user_id=self._user_id)
        except Exception:
            return messages
        if not ctx:
            return messages
        patched = list(messages)
        idx = next(
            i for i in range(len(patched) - 1, -1, -1)
            if patched[i].get("role") == "user"
        )
        patched[idx] = dict(patched[idx], content=f"[Prior context]\n{ctx}\n\n{query}")
        return patched

    def _run_turn_inner(
        self,
        messages: list[dict],
        on_chunk: Callable[[str], None] | None = None,
    ) -> tuple[str, list[dict]]:
        messages = self._compress_history(messages)
        messages = self._surface_memory_context(messages)
        system = self.get_system_prompt()
        coaching = self._coaching_prefix()
        if coaching:
            system = coaching + system
        all_messages = [{"role": "system", "content": system}] + messages
        tools = self._to_ollama_tools()
        model = self._router.select(messages, agent_type=self._agent_type_key())

        fast_model_used = (model != self._router._primary and model == self._router._fast)

        if on_chunk is not None:
            full_text = ""
            tool_calls = []
            for chunk in ollama.chat(
                model=model,
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

            # Confidence gate: re-run with primary if fast model produced hedging language
            if fast_model_used and self._settings_flag("confidence_gate_enabled", True) and self._detect_hedges(full_text):
                log.info("confidence_gate_escalating", agent=type(self).__name__)
                return self._run_turn_inner_with_model(self._router._primary, all_messages, tools, messages)

            if self._settings_flag("reflection_enabled", False):
                full_text = self._reflect(full_text)

            return full_text, messages + [{"role": "assistant", "content": full_text}]

        else:
            response = ollama.chat(
                model=model,
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

            # Confidence gate: re-run with primary if fast model produced hedging language
            if fast_model_used and self._settings_flag("confidence_gate_enabled", True) and self._detect_hedges(text):
                log.info("confidence_gate_escalating", agent=type(self).__name__)
                return self._run_turn_inner_with_model(self._router._primary, all_messages, tools, messages)

            if self._settings_flag("reflection_enabled", False):
                text = self._reflect(text)

            return text, messages + [{"role": "assistant", "content": text}]

    def _run_turn_inner_with_model(
        self,
        model: str,
        all_messages: list[dict],
        tools: list[dict],
        original_messages: list[dict],
    ) -> tuple[str, list[dict]]:
        """Re-run a single non-streaming call with a specific model (used by confidence gate)."""
        try:
            response = ollama.chat(
                model=model,
                messages=all_messages,
                tools=tools or None,
                options={"num_predict": self._max_tokens},
            )
            self._prompt_tokens += getattr(response, "prompt_eval_count", 0) or 0
            self._completion_tokens += getattr(response, "eval_count", 0) or 0
            text = response.message.content or ""
            if self._settings_flag("reflection_enabled", False):
                text = self._reflect(text)
            return text, original_messages + [{"role": "assistant", "content": text}]
        except Exception as exc:
            log.error("confidence_gate_fallback_failed", error=str(exc))
            return "", original_messages

    def _execute_tool_calls(
        self,
        messages: list[dict],
        assistant_text: str,
        tool_calls: list,
    ) -> tuple[list[dict], list[dict]]:
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
            self._turn_tool_calls.append(name)
            tool_msgs.append({"role": "tool", "content": result})

        return updated, tool_msgs

    def _dispatch(self, name: str, tool_input: dict) -> str:
        handler = self._tool_registry.get(name)
        if handler is None:
            return f"ERROR: unknown tool '{name}'"

        # Quota check — enforce per-user tool call limits
        try:
            from jarvis.config import get_settings as _gs
            from jarvis.tools.quotas import check_quota, QuotaExceededError
            _qdb = _gs().reports_dir / "jarvis.db"
            check_quota(_qdb, self._user_id or "anonymous", name)
        except Exception as _qe:
            from jarvis.tools.quotas import QuotaExceededError
            if isinstance(_qe, QuotaExceededError):
                return f"ERROR: {_qe}"

        # Circuit breaker — skip call if service is known-failing
        breaker = None
        try:
            from jarvis.tools.circuit_breaker import get_breaker
            breaker = get_breaker(name)
            if breaker.is_open(name):
                return f"ERROR: tool '{name}' circuit open — service temporarily unavailable"
        except Exception:
            breaker = None

        # Cache lookup — avoid redundant external calls
        db_path = None
        try:
            from jarvis.config import get_settings
            db_path = get_settings().reports_dir / "jarvis.db"
            from jarvis.tools.cache import get_cached
            cached = get_cached(db_path, name, tool_input)
            if cached is not None:
                if breaker:
                    breaker.record_success(name)
                return cached
        except Exception:
            pass

        tracer = get_tracer()
        t0 = time.perf_counter()
        retry_count = 0
        with tracer.start_as_current_span(f"tool.{name}") as span:
            span.set_attribute("tool.name", name)
            try:
                from concurrent.futures import ThreadPoolExecutor, TimeoutError as _Timeout
                from jarvis.tools.retry import call_with_retry
                # Per-tool config overrides global settings
                _global_timeout = self._tool_timeout_seconds()
                _global_retries = self._tool_retry_max()
                try:
                    from jarvis.tools.tool_config import get_tool_timeout, get_tool_max_retries
                    from jarvis.config import get_settings as _gcfg
                    _cfg_db = _gcfg().reports_dir / "jarvis.db"
                    tool_timeout = get_tool_timeout(_cfg_db, name, _global_timeout)
                    max_retries = get_tool_max_retries(_cfg_db, name, _global_retries)
                except Exception:
                    tool_timeout = _global_timeout
                    max_retries = _global_retries
                base_delay = self._tool_retry_base_delay()

                def _timed_call(tool_input: dict) -> str:
                    with ThreadPoolExecutor(max_workers=1) as pool:
                        future = pool.submit(handler, tool_input)
                        try:
                            return future.result(timeout=tool_timeout)
                        except _Timeout:
                            return f"ERROR: tool '{name}' timed out after {tool_timeout}s"

                result, attempts = call_with_retry(
                    _timed_call, tool_input,
                    max_retries=max_retries, base_delay=base_delay,
                )
                retry_count = attempts - 1
            except Exception as exc:
                result = f"ERROR: tool '{name}' raised — {exc}"

            duration_ms = (time.perf_counter() - t0) * 1000
            result_ok = not str(result).startswith("ERROR")
            if retry_count > 0:
                span.set_attribute("tool.retries", retry_count)

            if breaker:
                if result_ok:
                    breaker.record_success(name)
                else:
                    breaker.record_failure(name)

            if not result_ok:
                span.set_attribute("tool.error", result)
                self._record_failure(name, tool_input, result)
            else:
                # Cache successful results for eligible tools
                try:
                    if db_path:
                        from jarvis.tools.cache import set_cached
                        set_cached(db_path, name, tool_input, result)
                except Exception:
                    pass
                # Record call for quota tracking
                try:
                    from jarvis.tools.quotas import record_call
                    if db_path:
                        record_call(db_path, self._user_id or "anonymous", name)
                except Exception:
                    pass

            self._record_tool_metric(name, duration_ms / 1000)
            self._write_audit(name, tool_input, int(result_ok), duration_ms)
            return result

    def _tool_timeout_seconds(self) -> int:
        try:
            from jarvis.config import get_settings
            return get_settings().tool_timeout_seconds
        except Exception:
            return 60

    def _tool_retry_max(self) -> int:
        try:
            from jarvis.config import get_settings
            return get_settings().tool_max_retries
        except Exception:
            return 2

    def _tool_retry_base_delay(self) -> float:
        try:
            from jarvis.config import get_settings
            return get_settings().tool_retry_base_delay
        except Exception:
            return 1.0

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
