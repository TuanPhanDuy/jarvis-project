"""CriticAgent — evaluates sub-agent outputs and flags low-quality results.

Uses the fast model (Haiku) to score task results and optionally suggest
a revised task description for retry. Called by ExecutorAgent after each step.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Callable

import anthropic

from jarvis.agents.base_agent import BaseAgent
from jarvis.prompts.loader import load_prompt


@dataclass
class CritiqueResult:
    score: int               # 1-5
    issues: list[str]
    should_retry: bool
    revised_task: str | None = None


class CriticAgent(BaseAgent):
    """Fast quality evaluator using the haiku model."""

    def get_system_prompt(self) -> str:
        return load_prompt("critic")

    def critique(self, task: str, result: str) -> CritiqueResult:
        """Score a task result. Returns CritiqueResult with retry guidance."""
        prompt = f"TASK:\n{task}\n\nRESULT:\n{result[:3000]}"
        messages = [{"role": "user", "content": prompt}]
        try:
            response_text, _ = self.run_turn(messages)
            return _parse_critique(response_text)
        except Exception:
            return CritiqueResult(score=3, issues=[], should_retry=False)


def _parse_critique(text: str) -> CritiqueResult:
    lines = {
        k.strip(): v.strip()
        for line in text.strip().splitlines()
        if ":" in line
        for k, v in [line.split(":", 1)]
    }
    try:
        score = int(lines.get("SCORE", "3"))
    except ValueError:
        score = 3

    issues_raw = lines.get("ISSUES", "none")
    issues = [] if issues_raw.lower() == "none" else [i.strip() for i in issues_raw.split(",")]

    retry_raw = lines.get("RETRY", "no").lower()
    should_retry = retry_raw == "yes"

    revised = lines.get("REVISED_TASK", "none")
    revised_task = None if revised.lower() == "none" else revised

    return CritiqueResult(score=score, issues=issues, should_retry=should_retry, revised_task=revised_task)


def build_critic(client: anthropic.Anthropic, fast_model: str, max_tokens: int) -> CriticAgent:
    """Create a CriticAgent with no tools — it only reads and reasons."""
    return CriticAgent(
        client=client,
        model=fast_model,
        max_tokens=min(max_tokens, 512),
        tool_schemas=[],
        tool_registry={},
    )
