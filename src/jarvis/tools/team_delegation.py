"""Tool: delegate_to_team_member — spawn a role-specialised TeamAgent and return its result.

Used by Manager (→ team_lead, frontend, backend) and Team Lead (→ frontend, backend).
Each role receives its own tool set; leaf roles (frontend, backend) have no delegation tool.
"""
from __future__ import annotations

from collections.abc import Callable

import anthropic


def build_team_delegation_handler(
    client: anthropic.Anthropic,
    model: str,
    max_tokens: int,
    role_configs: dict[str, tuple[list[dict], dict[str, Callable[[dict], str]]]],
    allowed_roles: list[str],
) -> Callable[[dict], str]:
    """Return a handler for the delegate_to_team_member tool.

    Args:
        client: Shared Anthropic client.
        model: Model name passed to spawned agents.
        max_tokens: Token budget for sub-agent responses.
        role_configs: Maps role name → (tool_schemas, tool_registry) for that role.
        allowed_roles: Roles this handler is permitted to spawn.
    """

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
                client=client,
                model=model,
                max_tokens=max_tokens,
                tool_schemas=schemas,
                tool_registry=registry,
                role=role,
            )
            messages = [{"role": "user", "content": task}]
            result, _ = agent.run_turn(messages)
            label = role.upper().replace("_", " ")
            return f"[{label}]\n{result}"
        except Exception as exc:
            return f"ERROR: delegation to '{role}' failed — {exc}"

    return handle


SCHEMA: dict = {
    "name": "delegate_to_team_member",
    "description": (
        "Delegate a task to a specialised team member and receive their result. "
        "Each member has distinct expertise: 'team_lead' for architecture and cross-cutting concerns, "
        "'frontend' for UI/React/CSS, 'backend' for Python/API/database work."
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
                "description": (
                    "A complete, self-contained task description including all context "
                    "the team member needs — they have no memory of this conversation."
                ),
            },
        },
        "required": ["role", "task"],
    },
}
