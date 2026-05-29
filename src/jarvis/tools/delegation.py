"""Tool: delegate_task — spawn a specialist sub-agent and return its result."""
from __future__ import annotations

from collections.abc import Callable


def build_delegation_handler(
    model: str,
    max_tokens: int,
    sub_tool_schemas: list[dict],
    sub_tool_registry: dict[str, Callable[[dict], str]],
) -> Callable[[dict], str]:
    """Return a handler that delegates tasks to specialist sub-agents.

    Sub-agents receive base tools only (no delegate_task) to prevent
    recursive delegation loops.
    """

    def handle_delegate_task(tool_input: dict) -> str:
        agent_type = tool_input.get("agent_type", "")
        task = tool_input.get("task", "").strip()

        if not task:
            return "ERROR: 'task' is required"

        from jarvis.agents.coder import CoderAgent
        from jarvis.agents.consensus import ConsensusAgent
        from jarvis.agents.data_analyst import DataAnalystAgent
        from jarvis.agents.devops import DevOpsAgent
        from jarvis.agents.qa import QAAgent
        from jarvis.agents.researcher import ResearcherAgent

        if agent_type == "consensus":
            n = int(tool_input.get("n_agents", 3))
            try:
                agent = ConsensusAgent(
                    model=model,
                    max_tokens=max_tokens,
                    tool_schemas=sub_tool_schemas,
                    tool_registry=sub_tool_registry,
                    n_agents=n,
                )
                result = agent.run(task)
                return f"[CONSENSUS RESULT]\n{result}"
            except Exception as exc:
                return f"ERROR: consensus delegation failed — {exc}"

        agent_classes = {
            "researcher": ResearcherAgent,
            "coder": CoderAgent,
            "qa": QAAgent,
            "analyst": DataAnalystAgent,
            "devops": DevOpsAgent,
        }

        AgentClass = agent_classes.get(agent_type)
        if AgentClass is None:
            return (
                f"ERROR: unknown agent_type '{agent_type}'. "
                f"Valid: {', '.join(list(agent_classes) + ['consensus'])}"
            )

        try:
            agent = AgentClass(
                model=model,
                max_tokens=max_tokens,
                tool_schemas=sub_tool_schemas,
                tool_registry=sub_tool_registry,
            )
            result, _ = agent.run_turn([{"role": "user", "content": task}])
            return f"[{agent_type.upper()} RESULT]\n{result}"
        except Exception as e:
            return f"ERROR: delegation to '{agent_type}' failed — {e}"

    return handle_delegate_task


SCHEMA: dict = {
    "name": "delegate_task",
    "description": (
        "Delegate a task to a specialist sub-agent and receive its result. "
        "- 'researcher': web research, information gathering, topic synthesis\n"
        "- 'coder': write, run, and explain code\n"
        "- 'qa': review code for bugs, edge cases, and quality\n"
        "- 'analyst': query databases/CSV, data analysis, statistics, charts\n"
        "- 'devops': system diagnostics, shell automation, git, infrastructure\n"
        "- 'consensus': run N researchers in parallel and return the best-scored answer"
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "agent_type": {
                "type": "string",
                "enum": ["researcher", "coder", "qa", "analyst", "devops", "consensus"],
                "description": "Which specialist to delegate to.",
            },
            "task": {
                "type": "string",
                "description": (
                    "A clear, self-contained task description with all the context needed. "
                    "The sub-agent has no memory of the current conversation."
                ),
            },
            "n_agents": {
                "type": "integer",
                "description": "For 'consensus' only: number of parallel researchers (default 3).",
                "default": 3,
            },
        },
        "required": ["agent_type", "task"],
    },
}
