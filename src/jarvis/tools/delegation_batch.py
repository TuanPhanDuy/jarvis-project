"""Tool: delegate_batch — run multiple specialist sub-agents in parallel."""
from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed


def build_batch_delegation_handler(
    model: str,
    max_tokens: int,
    sub_tool_schemas: list[dict],
    sub_tool_registry: dict[str, Callable[[dict], str]],
) -> Callable[[dict], str]:
    """Return a handler that runs multiple agent tasks in parallel."""

    def handle_delegate_batch(tool_input: dict) -> str:
        tasks = tool_input.get("tasks", [])
        if not tasks:
            return "ERROR: 'tasks' must be non-empty"

        from jarvis.agents.coder import CoderAgent
        from jarvis.agents.data_analyst import DataAnalystAgent
        from jarvis.agents.devops import DevOpsAgent
        from jarvis.agents.qa import QAAgent
        from jarvis.agents.researcher import ResearcherAgent

        agent_classes = {
            "researcher": ResearcherAgent,
            "coder": CoderAgent,
            "qa": QAAgent,
            "analyst": DataAnalystAgent,
            "devops": DevOpsAgent,
        }

        def run_one(item: dict) -> tuple[str, str, str]:
            task_id = item.get("id", "task")
            agent_type = item.get("agent_type", "researcher")
            task = item.get("task", "").strip()
            if not task:
                return task_id, agent_type, "ERROR: task is empty"
            AgentClass = agent_classes.get(agent_type)
            if AgentClass is None:
                return task_id, agent_type, f"ERROR: unknown agent_type '{agent_type}'"
            try:
                agent = AgentClass(
                    model=model,
                    max_tokens=max_tokens,
                    tool_schemas=sub_tool_schemas,
                    tool_registry=sub_tool_registry,
                )
                result, _ = agent.run_turn([{"role": "user", "content": task}])
                return task_id, agent_type, result
            except Exception as e:
                return task_id, agent_type, f"ERROR: {e}"

        # Preserve input order in output
        ordered: list[tuple[str, str, str]] = [("", "", "")] * len(tasks)
        max_workers = min(len(tasks), 5)

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_to_idx = {pool.submit(run_one, item): i for i, item in enumerate(tasks)}
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                ordered[idx] = future.result()

        parts = [
            f"[{tid.upper()} — {atype.upper()}]\n{result}"
            for tid, atype, result in ordered
        ]
        return "\n\n".join(parts)

    return handle_delegate_batch


SCHEMA: dict = {
    "name": "delegate_batch",
    "description": (
        "Delegate multiple independent tasks to specialist sub-agents running in parallel. "
        "All tasks start simultaneously — use this when tasks are independent of each other. "
        "Agents: 'researcher' (research, synthesis), 'coder' (write/run code), "
        "'qa' (review, test), 'analyst' (data/SQL), 'devops' (system/infra). "
        "Results are returned together in the original task order."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "tasks": {
                "type": "array",
                "description": "Independent tasks to run in parallel (2–5 tasks).",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {
                            "type": "string",
                            "description": "Short identifier for this task (e.g. 'research_rlhf', 'code_attention').",
                        },
                        "agent_type": {
                            "type": "string",
                            "enum": ["researcher", "coder", "qa", "analyst", "devops"],
                            "description": "Which specialist handles this task.",
                        },
                        "task": {
                            "type": "string",
                            "description": (
                                "Self-contained task description with all required context. "
                                "The sub-agent has no memory of the current conversation."
                            ),
                        },
                    },
                    "required": ["id", "agent_type", "task"],
                },
                "minItems": 2,
                "maxItems": 5,
            }
        },
        "required": ["tasks"],
    },
}
