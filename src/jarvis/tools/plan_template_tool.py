"""Tool: plan_from_template — run a named plan template with parameter substitution."""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path


def build_plan_template_handler(
    model: str,
    max_tokens: int,
    sub_tool_schemas: list[dict],
    sub_tool_registry: dict[str, Callable[[dict], str]],
    db_path: Path | None = None,
    session_id: str = "",
    user_id: str | None = None,
) -> Callable[[dict], str]:
    from jarvis.agents.executor import ExecutorAgent

    executor = ExecutorAgent(
        model=model,
        max_tokens=max_tokens,
        sub_tool_schemas=sub_tool_schemas,
        sub_tool_registry=sub_tool_registry,
        db_path=db_path,
        session_id=session_id,
        user_id=user_id,
    )

    def handle(tool_input: dict) -> str:
        template_name = tool_input.get("template", "").strip()
        params = {k: str(v) for k, v in tool_input.items() if k != "template"}

        from jarvis.agents.plan_templates import get_template, list_templates

        result = get_template(template_name, **params)
        if result is None:
            available = [t["name"] for t in list_templates()]
            return f"ERROR: unknown template '{template_name}'. Available: {', '.join(available)}"

        goal, steps = result
        try:
            return executor.execute_plan(goal=goal, steps=steps)
        except Exception as exc:
            return f"ERROR: template '{template_name}' execution failed — {exc}"

    return handle


SCHEMA: dict = {
    "name": "plan_from_template",
    "description": (
        "Run a named multi-step plan template with parameter substitution. "
        "Each template expands into 5-6 specialist steps with maximum parallelism. "
        "Templates:\n"
        "- 'research-report' (params: topic) — 4 parallel research tracks + synthesis + QA review\n"
        "- 'code-review' (params: repo_path) — git + quality + security + coverage → summary\n"
        "- 'data-analysis' (params: data_source, question) — explore + stats + trends + interpret\n"
        "- 'implement-and-test' (params: feature, language) — research → implement → test → review → deploy check"
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "template": {
                "type": "string",
                "enum": ["research-report", "code-review", "data-analysis", "implement-and-test"],
                "description": "Which plan template to run.",
            },
            "topic": {
                "type": "string",
                "description": "Research subject (required for research-report).",
            },
            "repo_path": {
                "type": "string",
                "description": "Filesystem path to the repository (required for code-review).",
            },
            "data_source": {
                "type": "string",
                "description": "Path or table name for the data (required for data-analysis).",
            },
            "question": {
                "type": "string",
                "description": "The analysis question to answer (required for data-analysis).",
            },
            "feature": {
                "type": "string",
                "description": "Name of the feature to implement (required for implement-and-test).",
            },
            "language": {
                "type": "string",
                "description": "Programming language (required for implement-and-test).",
            },
        },
        "required": ["template"],
    },
}
