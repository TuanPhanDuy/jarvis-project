"""Tool: delegate_task — spawn a specialized sub-agent and return its result.

Used by PlannerAgent to delegate work to ResearcherAgent, CoderAgent, or QAAgent.
Sub-agents receive the base tool registry (no recursive delegation) to prevent
infinite loops.

The handler is created via build_delegation_handler() which injects the shared
Anthropic client and sub-agent tool registry.
"""
from __future__ import annotations

from collections.abc import Callable

import anthropic


def build_delegation_handler(
    client: anthropic.Anthropic,
    model: str,
    max_tokens: int,
    sub_tool_schemas: list[dict],
    sub_tool_registry: dict[str, Callable[[dict], str]],
) -> Callable[[dict], str]:
    """Return a handler for the delegate_task tool.

    Sub-agents are created with sub_tool_schemas / sub_tool_registry — they do NOT
    have the delegate_task tool themselves, preventing recursive delegation.

    Args:
        client: Shared Anthropic client.
        model: Model name to use for sub-agents.
        max_tokens: Token budget for sub-agent responses.
        sub_tool_schemas: Tool schemas available to sub-agents (no delegate_task).
        sub_tool_registry: Tool handlers available to sub-agents.
    """

    def handle_delegate_task(tool_input: dict) -> str:
        agent_type = tool_input.get("agent_type", "")
        task = tool_input.get("task", "").strip()

        if not task:
            return "ERROR: 'task' is required"

        # Lazy imports avoid circular dependencies at module load time
        from jarvis.agents.coder import CoderAgent
        from jarvis.agents.qa import QAAgent
        from jarvis.agents.researcher import ResearcherAgent

        agent_classes = {
            "researcher": ResearcherAgent,
            "coder": CoderAgent,
            "qa": QAAgent,
        }

        AgentClass = agent_classes.get(agent_type)
        if AgentClass is None:
            return (
                f"ERROR: unknown agent_type '{agent_type}'. "
                f"Valid options: {', '.join(agent_classes)}"
            )

        try:
            agent = AgentClass(
                client=client,
                model=model,
                max_tokens=max_tokens,
                tool_schemas=sub_tool_schemas,
                tool_registry=sub_tool_registry,
            )
            messages = [{"role": "user", "content": task}]
            result, _ = agent.run_turn(messages)
            return f"[{agent_type.upper()} RESULT]\n{result}"
        except Exception as e:
            return f"ERROR: delegation to '{agent_type}' failed — {e}"

    return handle_delegate_task


SCHEMA: dict = {
    "name": "delegate_task",
    "description": (
        "Delegate a task to a specialized sub-agent and receive its result. "
        "Use this when the user's request matches a specialist: "
        "'researcher' for AI/ML research, 'coder' for writing/running code, "
        "'qa' for reviewing or testing code."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "agent_type": {
                "type": "string",
                "enum": ["researcher", "coder", "qa"],
                "description": "Which specialist to delegate to.",
            },
            "task": {
                "type": "string",
                "description": (
                    "A clear, self-contained task description. Include all context the "
                    "sub-agent needs — it has no memory of the current conversation."
                ),
            },
        },
        "required": ["agent_type", "task"],
    },
}
