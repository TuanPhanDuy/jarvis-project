"""Named plan templates with parameter substitution.

Templates define reusable multi-step workflows. Parameters are substituted
using {param_name} placeholders in step descriptions.

Usage:
    goal, steps = get_template("research-report", topic="RLHF")
    executor.execute_plan(goal, steps)
"""
from __future__ import annotations

from copy import deepcopy

from jarvis.agents.executor import PlanStep


_TEMPLATES: dict[str, dict] = {
    "research-report": {
        "description": "Research {topic} from multiple angles and save a structured report.",
        "params": ["topic"],
        "steps": [
            {
                "id": "r_background",
                "agent_type": "researcher",
                "description": "Research {topic}: background, history, and key concepts. Include foundational papers and definitions.",
            },
            {
                "id": "r_sota",
                "agent_type": "researcher",
                "description": "Research {topic}: recent developments, state of the art, and open problems as of 2024-2025.",
            },
            {
                "id": "r_applications",
                "agent_type": "researcher",
                "description": "Research {topic}: practical applications, case studies, and real-world deployments.",
            },
            {
                "id": "r_criticism",
                "agent_type": "researcher",
                "description": "Research criticism and limitations of {topic}: known weaknesses, controversies, and alternative approaches.",
            },
            {
                "id": "synthesize",
                "agent_type": "researcher",
                "depends_on": ["r_background", "r_sota", "r_applications", "r_criticism"],
                "description": (
                    "Synthesize all research into a structured report on {topic}. "
                    "Sections: overview, key concepts, state of the art, applications, limitations, references. "
                    "Save the report."
                ),
            },
            {
                "id": "qa_review",
                "agent_type": "qa",
                "depends_on": ["synthesize"],
                "description": "Review the synthesized report on {topic} for accuracy, completeness, and clarity. List any gaps or factual errors.",
            },
        ],
    },
    "code-review": {
        "description": "Full code review of {repo_path}: git inspection, quality, security, and test coverage.",
        "params": ["repo_path"],
        "steps": [
            {
                "id": "git_history",
                "agent_type": "devops",
                "description": "Inspect git log and recent commits in {repo_path}. Summarize what changed recently and identify risky or large changes.",
            },
            {
                "id": "code_quality",
                "agent_type": "qa",
                "description": "Review the codebase at {repo_path} for code quality: naming conventions, structure, duplication, cyclomatic complexity.",
            },
            {
                "id": "security_audit",
                "agent_type": "qa",
                "description": "Audit {repo_path} for security issues: injection risks, auth gaps, exposed secrets, insecure dependencies.",
            },
            {
                "id": "test_coverage",
                "agent_type": "analyst",
                "description": "Analyze test coverage for {repo_path}: which modules lack tests, which tests are weak or redundant, coverage percentage.",
            },
            {
                "id": "summary",
                "agent_type": "qa",
                "depends_on": ["git_history", "code_quality", "security_audit", "test_coverage"],
                "description": (
                    "Combine all review findings for {repo_path} into a prioritized code review report. "
                    "Rank issues by severity (critical / high / medium / low). Include specific line references where possible."
                ),
            },
        ],
    },
    "data-analysis": {
        "description": "Structured data analysis of {data_source} to answer: {question}",
        "params": ["data_source", "question"],
        "steps": [
            {
                "id": "explore",
                "agent_type": "analyst",
                "description": "Explore '{data_source}': schema, row count, missing values, data types, and sample rows.",
            },
            {
                "id": "stats",
                "agent_type": "analyst",
                "description": "Compute descriptive statistics for '{data_source}': mean, median, std, quartiles, correlations for all numeric columns.",
            },
            {
                "id": "trends",
                "agent_type": "analyst",
                "description": "Identify trends, patterns, and anomalies in '{data_source}' relevant to: {question}",
            },
            {
                "id": "benchmarks",
                "agent_type": "researcher",
                "description": "Research domain benchmarks and literature relevant to: {question}. What do typical values look like in this domain?",
            },
            {
                "id": "interpret",
                "agent_type": "analyst",
                "depends_on": ["explore", "stats", "trends", "benchmarks"],
                "description": (
                    "Synthesize data exploration, statistics, trends, and benchmarks to answer: {question}. "
                    "Write a clear data analysis report with conclusions and confidence levels."
                ),
            },
        ],
    },
    "implement-and-test": {
        "description": "Full software implementation of {feature} in {language}: research → design → implement → test → review → deploy check.",
        "params": ["feature", "language"],
        "steps": [
            {
                "id": "research_api",
                "agent_type": "researcher",
                "description": "Research best practices, existing libraries, and APIs for implementing {feature} in {language}.",
            },
            {
                "id": "research_patterns",
                "agent_type": "researcher",
                "description": "Research design patterns and architecture approaches for {feature}. Focus on testability, extensibility, and performance.",
            },
            {
                "id": "implement",
                "agent_type": "coder",
                "depends_on": ["research_api", "research_patterns"],
                "description": "Implement {feature} in {language}. Apply research findings. Include type hints, docstrings, and error handling.",
            },
            {
                "id": "write_tests",
                "agent_type": "coder",
                "depends_on": ["implement"],
                "description": "Write comprehensive tests for the {feature} implementation: unit tests, edge cases, and integration tests.",
            },
            {
                "id": "code_review",
                "agent_type": "qa",
                "depends_on": ["implement", "write_tests"],
                "description": "Review {feature} implementation and tests: correctness, edge cases, error handling, code quality, test completeness.",
            },
            {
                "id": "deploy_check",
                "agent_type": "devops",
                "depends_on": ["code_review"],
                "description": "Verify {feature} is deployment-ready: check dependencies, environment requirements, build steps, and potential integration issues.",
            },
        ],
    },
}


def get_template(name: str, **params: str) -> tuple[str, list[PlanStep]] | None:
    """Return (goal, steps) for the named template with params substituted.

    Returns None if the template name is unknown.
    """
    tpl = _TEMPLATES.get(name)
    if tpl is None:
        return None

    goal = tpl["description"]
    for k, v in params.items():
        goal = goal.replace(f"{{{k}}}", v)

    steps: list[PlanStep] = []
    for raw in deepcopy(tpl["steps"]):
        desc = raw["description"]
        for k, v in params.items():
            desc = desc.replace(f"{{{k}}}", v)
        steps.append(PlanStep(
            id=raw["id"],
            description=desc,
            agent_type=raw["agent_type"],
            depends_on=raw.get("depends_on", []),
        ))
    return goal, steps


def list_templates() -> list[dict]:
    """Return all template metadata: name, description, params, step_count."""
    return [
        {
            "name": name,
            "description": tpl["description"],
            "params": tpl["params"],
            "step_count": len(tpl["steps"]),
        }
        for name, tpl in _TEMPLATES.items()
    ]
