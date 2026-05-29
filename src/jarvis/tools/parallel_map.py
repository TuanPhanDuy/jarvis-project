"""Tool: parallel_map — scatter a task template over N topics using parallel agents.

Pattern: supply a task template with a ``{topic}`` placeholder and a list of
topics.  One specialist agent is spawned per topic and all run concurrently.
Results are collected in topic order; an optional synthesis step merges them.

Example tool call::

    parallel_map(
        task_template="Research the latest developments in {topic} and summarise key findings.",
        topics=["RLHF", "constitutional AI", "LoRA fine-tuning"],
        agent_type="researcher",
        synthesize=True,
    )
"""
from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import TimeoutError as _FutureTimeout

import structlog

log = structlog.get_logger()

_MAX_TOPICS = 10
_MAX_WORKERS = 8


def build_parallel_map_handler(
    model: str,
    max_tokens: int,
    sub_tool_schemas: list[dict],
    sub_tool_registry: dict[str, Callable[[dict], str]],
) -> Callable[[dict], str]:
    """Return a handler that maps a task template across topics in parallel."""

    def handle_parallel_map(tool_input: dict) -> str:
        task_template: str = tool_input.get("task_template", "").strip()
        topics: list[str] = tool_input.get("topics", [])
        agent_type: str = tool_input.get("agent_type", "researcher")
        synthesize: bool = bool(tool_input.get("synthesize", True))
        timeout_seconds = tool_input.get("timeout_seconds") or None

        if not task_template:
            return "ERROR: 'task_template' is required"
        if "{topic}" not in task_template:
            return "ERROR: 'task_template' must contain the {topic} placeholder"
        if not topics:
            return "ERROR: 'topics' must be non-empty"
        if len(topics) > _MAX_TOPICS:
            return f"ERROR: too many topics — max {_MAX_TOPICS}, got {len(topics)}"

        from jarvis.agents.coder import CoderAgent
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
        AgentClass = agent_classes.get(agent_type)
        if AgentClass is None:
            return (
                f"ERROR: unknown agent_type '{agent_type}'. "
                f"Valid: {', '.join(agent_classes)}"
            )

        def run_one(topic: str) -> tuple[str, str]:
            task = task_template.replace("{topic}", topic)
            try:
                agent = AgentClass(
                    model=model,
                    max_tokens=max_tokens,
                    tool_schemas=sub_tool_schemas,
                    tool_registry=sub_tool_registry,
                )
                result, _ = agent.run_turn([{"role": "user", "content": task}])
                log.info("parallel_map_topic_done", topic=topic[:40], agent=agent_type)
                return topic, result
            except Exception as exc:
                log.warning("parallel_map_topic_failed", topic=topic[:40], error=str(exc))
                return topic, f"ERROR: {exc}"

        # Run all topics in parallel; preserve order in output
        results: dict[str, str] = {}
        n_workers = min(len(topics), _MAX_WORKERS)

        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            future_to_topic = {pool.submit(run_one, t): t for t in topics}
            try:
                for future in as_completed(future_to_topic, timeout=timeout_seconds):
                    topic, result = future.result()
                    results[topic] = result
            except _FutureTimeout:
                for future, topic in future_to_topic.items():
                    if topic not in results:
                        results[topic] = "ERROR: timed out waiting for result"

        topic_sections = [
            f"### {topic}\n{results.get(topic, 'ERROR: no result')}"
            for topic in topics
        ]

        if not synthesize or len(topics) == 1:
            return "\n\n".join(topic_sections)

        # Synthesis pass: merge all per-topic results into a unified summary
        combined = "\n\n".join(
            f"[{topic}]:\n{results.get(topic, '')[:1500]}"
            for topic in topics
        )
        synthesis_prompt = (
            f"You have just completed research on {len(topics)} related topics. "
            "Synthesise the findings below into a single coherent summary. "
            "Highlight common themes, important differences, and cross-cutting insights. "
            "Be concise and well-structured.\n\n"
            + combined
        )
        try:
            synth_agent = ResearcherAgent(
                model=model,
                max_tokens=max_tokens,
                tool_schemas=sub_tool_schemas,
                tool_registry=sub_tool_registry,
            )
            synthesis, _ = synth_agent.run_turn(
                [{"role": "user", "content": synthesis_prompt}]
            )
            log.info("parallel_map_synthesis_done", topics=len(topics))
        except Exception as exc:
            synthesis = f"(synthesis failed: {exc})"

        topic_sections.append(f"## Cross-Topic Synthesis\n{synthesis}")
        return "\n\n".join(topic_sections)

    return handle_parallel_map


SCHEMA: dict = {
    "name": "parallel_map",
    "description": (
        "Scatter a single task template over multiple topics, running one specialist "
        "agent per topic in parallel. Results are collected in topic order and optionally "
        "synthesised into a unified summary. Ideal for comparative research, multi-domain "
        "analysis, or generating parallel reports on related subjects. "
        "The task_template must contain the literal placeholder {topic}."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "task_template": {
                "type": "string",
                "description": (
                    "Task description with a {topic} placeholder that is replaced for each topic. "
                    "Example: 'Research the latest breakthroughs in {topic} and summarise them.'"
                ),
            },
            "topics": {
                "type": "array",
                "description": "List of topics to substitute into the task template (2–10 items).",
                "items": {"type": "string"},
                "minItems": 2,
                "maxItems": 10,
            },
            "agent_type": {
                "type": "string",
                "enum": ["researcher", "coder", "qa", "analyst", "devops"],
                "description": "Specialist agent to use for each topic. Defaults to 'researcher'.",
                "default": "researcher",
            },
            "synthesize": {
                "type": "boolean",
                "description": (
                    "If true (default), a final synthesis step merges all topic results "
                    "into a unified cross-topic summary."
                ),
                "default": True,
            },
            "timeout_seconds": {
                "type": "number",
                "description": "Optional wall-clock timeout in seconds for the entire parallel phase.",
            },
        },
        "required": ["task_template", "topics"],
    },
}
