"""Tool: delegate_batch — run multiple specialist sub-agents in parallel."""
from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import TimeoutError as _FutureTimeout

import structlog

from jarvis.config import get_settings

log = structlog.get_logger()

_MAX_TASKS = 10
_MAX_WORKERS = 8


def build_batch_delegation_handler(
    model: str,
    max_tokens: int,
    sub_tool_schemas: list[dict],
    sub_tool_registry: dict[str, Callable[[dict], str]],
) -> Callable[[dict], str]:
    """Return a handler that runs multiple agent tasks in parallel."""

    def handle_delegate_batch(tool_input: dict) -> str:
        tasks = tool_input.get("tasks", [])
        timeout_seconds = tool_input.get("timeout_seconds") or None

        if not tasks:
            return "ERROR: 'tasks' must be non-empty"

        from jarvis.agents.coder import CoderAgent
        from jarvis.agents.consensus import ConsensusAgent
        from jarvis.agents.data_analyst import DataAnalystAgent
        from jarvis.agents.devops import DevOpsAgent
        from jarvis.agents.qa import QAAgent
        from jarvis.agents.researcher import ResearcherAgent

        agent_classes: dict[str, type] = {
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

            # consensus: run N researchers in parallel and pick the best-scored result
            if agent_type == "consensus":
                n = int(item.get("n_agents", get_settings().consensus_n_agents))
                try:
                    agent = ConsensusAgent(
                        model=model,
                        max_tokens=max_tokens,
                        tool_schemas=sub_tool_schemas,
                        tool_registry=sub_tool_registry,
                        n_agents=n,
                    )
                    result = agent.run(task)
                    return task_id, agent_type, result
                except Exception as exc:
                    return task_id, agent_type, f"ERROR: {exc}"

            AgentClass = agent_classes.get(agent_type)
            if AgentClass is None:
                return task_id, agent_type, (
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
                log.info("batch_task_done", task_id=task_id, agent=agent_type)
                return task_id, agent_type, result
            except Exception as exc:
                log.warning("batch_task_failed", task_id=task_id, agent=agent_type, error=str(exc))
                return task_id, agent_type, f"ERROR: {exc}"

        # Preserve input order in output
        ordered: list[tuple[str, str, str]] = [("", "", "ERROR: not started")] * len(tasks)
        n_workers = min(len(tasks), _MAX_WORKERS)

        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            future_to_idx = {pool.submit(run_one, item): i for i, item in enumerate(tasks)}
            try:
                for future in as_completed(future_to_idx, timeout=timeout_seconds):
                    idx = future_to_idx[future]
                    ordered[idx] = future.result()
            except _FutureTimeout:
                for future, idx in future_to_idx.items():
                    if ordered[idx][2] == "ERROR: not started":
                        task_id = tasks[idx].get("id", f"task_{idx}")
                        agent_type = tasks[idx].get("agent_type", "?")
                        ordered[idx] = (task_id, agent_type, "ERROR: timed out")

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
        "'qa' (review, test), 'analyst' (data/SQL), 'devops' (system/infra), "
        "'consensus' (run N researchers in parallel and return the best-scored answer). "
        "Results are returned together in the original task order."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "tasks": {
                "type": "array",
                "description": "Independent tasks to run in parallel (2–10 tasks).",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {
                            "type": "string",
                            "description": "Short identifier for this task (e.g. 'research_rlhf', 'code_attention').",
                        },
                        "agent_type": {
                            "type": "string",
                            "enum": ["researcher", "coder", "qa", "analyst", "devops", "consensus"],
                            "description": "Which specialist handles this task.",
                        },
                        "task": {
                            "type": "string",
                            "description": (
                                "Self-contained task description with all required context. "
                                "The sub-agent has no memory of the current conversation."
                            ),
                        },
                        "n_agents": {
                            "type": "integer",
                            "description": "For consensus agent_type only: how many researchers to run (default 3).",
                            "default": 3,
                        },
                    },
                    "required": ["id", "agent_type", "task"],
                },
                "minItems": 2,
                "maxItems": 10,
            },
            "timeout_seconds": {
                "type": "number",
                "description": "Optional wall-clock timeout in seconds for the entire batch.",
            },
        },
        "required": ["tasks"],
    },
}
