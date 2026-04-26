"""Tool: create_plan — decompose a multi-step task and execute it via ExecutorAgent.

Only available to PlannerAgent (not sub-agents, matching the delegate_task pattern).
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import anthropic

from jarvis.agents.executor import ExecutorAgent, PlanStep


def build_plan_handler(
    client: anthropic.Anthropic,
    model: str,
    fast_model: str,
    max_tokens: int,
    sub_tool_schemas: list[dict],
    sub_tool_registry: dict[str, Callable[[dict], str]],
    db_path: Path | None = None,
    session_id: str = "",
    user_id: str | None = None,
) -> Callable[[dict], str]:
    """Return a handler for the create_plan tool."""
    executor = ExecutorAgent(
        client=client,
        model=model,
        fast_model=fast_model,
        max_tokens=max_tokens,
        sub_tool_schemas=sub_tool_schemas,
        sub_tool_registry=sub_tool_registry,
        db_path=db_path,
        session_id=session_id,
        user_id=user_id,
    )

    def handle_create_plan(tool_input: dict) -> str:
        goal = tool_input.get("goal", "").strip()
        raw_steps = tool_input.get("steps", [])

        if not goal:
            return "ERROR: 'goal' is required"
        if not raw_steps:
            return "ERROR: 'steps' is required and must be non-empty"

        try:
            steps = [
                PlanStep(
                    id=s.get("id", str(i)),
                    description=s.get("description", ""),
                    agent_type=s.get("agent_type", "researcher"),
                    depends_on=s.get("depends_on", []),
                )
                for i, s in enumerate(raw_steps)
            ]
        except Exception as exc:
            return f"ERROR: invalid steps format — {exc}"

        try:
            return executor.execute_plan(goal=goal, steps=steps)
        except Exception as exc:
            return f"ERROR: plan execution failed — {exc}"

    return handle_create_plan


SCHEMA: dict = {
    "name": "create_plan",
    "description": (
        "Decompose a complex multi-step task into an ordered plan and execute it. "
        "Each step is delegated to the appropriate specialist agent. "
        "Use this instead of delegate_task when the task requires multiple steps with dependencies "
        "(e.g., research a topic, then write code based on it, then review the code)."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "goal": {
                "type": "string",
                "description": "The overall goal of the plan.",
            },
            "steps": {
                "type": "array",
                "description": "Ordered list of steps to execute.",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {
                            "type": "string",
                            "description": "Unique step identifier (e.g., 'step1', 'research', 'code').",
                        },
                        "description": {
                            "type": "string",
                            "description": "Clear, self-contained task description for this step.",
                        },
                        "agent_type": {
                            "type": "string",
                            "enum": ["researcher", "coder", "qa"],
                            "description": "Which specialist to use for this step.",
                        },
                        "depends_on": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "IDs of steps that must complete before this one.",
                            "default": [],
                        },
                    },
                    "required": ["id", "description", "agent_type"],
                },
            },
        },
        "required": ["goal", "steps"],
    },
}
