"""Tool: delegate_to_team_member — spawn a role-specialised TeamAgent and return its result."""
from __future__ import annotations

from collections.abc import Callable


def build_team_delegation_handler(
    model: str,
    max_tokens: int,
    role_configs: dict[str, tuple[list[dict], dict[str, Callable[[dict], str]]]],
    allowed_roles: list[str],
) -> Callable[[dict], str]:
    def handle(tool_input: dict) -> str:
        role = tool_input.get("role", "").strip()
        task = tool_input.get("task", "").strip()

        if not task:
            return "ERROR: 'task' is required"
        if role not in allowed_roles:
            return (
                f"ERROR: invalid role '{role}'. "
                f"This agent can delegate to: {', '.join(allowed_roles)}"
            )

        schemas, registry = role_configs[role]
        try:
            from jarvis.agents.team_agent import TeamAgent
            agent = TeamAgent(
                model=model,
                max_tokens=max_tokens,
                tool_schemas=schemas,
                tool_registry=registry,
                role=role,
            )
            result, _ = agent.run_turn([{"role": "user", "content": task}])
            return f"[{role.upper().replace('_', ' ')}]\n{result}"
        except Exception as exc:
            return f"ERROR: delegation to '{role}' failed — {exc}"

    return handle


SCHEMA: dict = {
    "name": "delegate_to_team_member",
    "description": (
        "Delegate a task to a specialised team member and receive their result. "
        "'team_lead' handles architecture, 'frontend' handles UI/React/CSS, "
        "'backend' handles Python/API/database work."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "role": {
                "type": "string",
                "enum": ["team_lead", "frontend", "backend"],
                "description": "Which team member to delegate to.",
            },
            "task": {
                "type": "string",
                "description": "A complete, self-contained task with all context needed.",
            },
        },
        "required": ["role", "task"],
    },
}
