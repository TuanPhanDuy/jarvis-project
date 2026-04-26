"""Multi-model router — drop-in replacement for anthropic.Anthropic.

Exposes the same .messages.create() / .messages.stream() interface so agents
require zero changes. Routing strategy is set via JARVIS_ROUTING_STRATEGY:

  always_primary (default) — always use the primary Claude model
  smart                    — use fast model for tool-dispatch turns,
                             primary model for synthesis/first turns

The fast model is configured via JARVIS_FAST_MODEL (default: claude-haiku-4-5-20251001).
"""
from __future__ import annotations


class _MessagesAPI:
    def __init__(self, router: "ModelRouter") -> None:
        self._router = router

    def create(self, **kwargs):
        params = dict(kwargs)
        if self._router.strategy == "smart":
            params["model"] = self._select_model(params)
        return self._router.primary.messages.create(**params)

    def stream(self, **kwargs):
        # Streaming always uses primary model for simplicity
        return self._router.primary.messages.stream(**kwargs)

    def _select_model(self, params: dict) -> str:
        """Use fast model for tool-dispatch turns, primary for all other turns.

        Tool-dispatch turns are identified by the presence of a user message
        whose content is a list (tool_result blocks), not a plain string.
        """
        messages = params.get("messages", [])
        in_tool_loop = any(
            m.get("role") == "user" and isinstance(m.get("content"), list)
            for m in messages
        )
        return self._router.fast_model if in_tool_loop else self._router.primary_model


class ModelRouter:
    """Route requests across Claude models based on a configurable strategy.

    Usage (replaces bare anthropic.Anthropic in agent construction):

        from jarvis.models.router import ModelRouter
        import anthropic

        client = ModelRouter(
            primary=anthropic.Anthropic(api_key=key),
            primary_model="claude-sonnet-4-6",
            fast_model="claude-haiku-4-5-20251001",
            strategy="smart",
        )
        agent = PlannerAgent(client=client, model=primary_model, ...)
    """

    def __init__(
        self,
        primary,
        primary_model: str,
        fast_model: str | None = None,
        strategy: str = "always_primary",
    ) -> None:
        self.primary = primary
        self.primary_model = primary_model
        self.fast_model = fast_model or primary_model
        self.strategy = strategy
        self.messages = _MessagesAPI(self)
