"""Tool: create_plan — decompose a multi-step task and execute it via ExecutorAgent."""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from jarvis.agents.executor import ExecutorAgent, PlanStep


def build_plan_handler(
    model: str,
    max_tokens: int,
    sub_tool_schemas: list[dict],
    sub_tool_registry: dict[str, Callable[[dict], str]],
    db_path: Path | None = None,
    session_id: str = "",
    user_id: str | None = None,
) -> Callable[[dict], str]:
    executor = ExecutorAgent(
        model=model,
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
            return "ERROR: 'steps' must be non-empty"

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
            return f"ERROR: invalid steps — {exc}"

        try:
            return executor.execute_plan(goal=goal, steps=steps)
        except Exception as exc:
            return f"ERROR: plan execution failed — {exc}"

    return handle_create_plan


SCHEMA: dict = {
    "name": "create_plan",
    "description": (
        "Decompose a complex multi-step task into an ordered plan and execute it using specialist agents. "
        "Each step runs a researcher, coder, or qa agent. Results from earlier steps flow as context "
        "into dependent steps. Use this for tasks like: research X, then implement it, then test it."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "goal": {"type": "string", "description": "The overall objective of this plan."},
            "steps": {
                "type": "array",
                "description": "Ordered list of steps.",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "description": "Unique step id (e.g. 'research', 'code', 'review')."},
                        "description": {"type": "string", "description": "Self-contained task for this step."},
                        "agent_type": {"type": "string", "enum": ["researcher", "coder", "qa", "analyst", "devops"]},
                        "depends_on": {"type": "array", "items": {"type": "string"}, "default": []},
                    },
                    "required": ["id", "description", "agent_type"],
                },
            },
        },
        "required": ["goal", "steps"],
    },
}
