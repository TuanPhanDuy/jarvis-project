"""CriticAgent — evaluates sub-agent outputs and flags low-quality results."""
from __future__ import annotations

import json
from dataclasses import dataclass

import structlog

from jarvis.agents.base_agent import BaseAgent
from jarvis.prompts.loader import load_prompt

log = structlog.get_logger()


@dataclass
class CritiqueResult:
    score: int
    issues: list[str]
    should_retry: bool
    revised_task: str | None = None


class CriticAgent(BaseAgent):
    def get_system_prompt(self) -> str:
        return load_prompt("critic")

    def critique(self, task: str, result: str) -> CritiqueResult:
        prompt = f"TASK:\n{task}\n\nRESULT:\n{result[:3000]}"
        messages = [{"role": "user", "content": prompt}]
        try:
            response_text, _ = self.run_turn(messages)
            return _parse_critique(response_text)
        except Exception as exc:
            log.warning("critic_failed", task_preview=task[:100], error=str(exc))
            return CritiqueResult(score=3, issues=[], should_retry=False)


def _parse_critique(text: str) -> CritiqueResult:
    # Try JSON first (primary path — matches the updated critic prompt)
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            data = json.loads(text[start:end])
            score = int(data.get("score", 3))
            issues_raw = data.get("issues", [])
            issues = (
                issues_raw if isinstance(issues_raw, list)
                else ([] if str(issues_raw).lower() == "none" else [str(issues_raw)])
            )
            should_retry = bool(data.get("retry", False))
            revised = data.get("revised_task")
            revised_task = None if not revised or str(revised).lower() == "none" else str(revised)
            return CritiqueResult(score=score, issues=issues, should_retry=should_retry, revised_task=revised_task)
    except (ValueError, KeyError):
        pass

    # Fall back to legacy key-value format
    lines = {}
    for line in text.strip().splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            lines[k.strip()] = v.strip()

    try:
        score = int(lines.get("SCORE", "3"))
    except ValueError:
        score = 3

    issues_raw = lines.get("ISSUES", "none")
    issues = [] if issues_raw.lower() == "none" else [i.strip() for i in issues_raw.split(",")]
    should_retry = lines.get("RETRY", "no").lower() == "yes"
    revised = lines.get("REVISED_TASK", "none")
    revised_task = None if revised.lower() == "none" else revised

    return CritiqueResult(score=score, issues=issues, should_retry=should_retry, revised_task=revised_task)


def build_critic(model: str, max_tokens: int) -> CriticAgent:
    return CriticAgent(
        model=model,
        max_tokens=min(max_tokens, 512),
        tool_schemas=[],
        tool_registry={},
    )
