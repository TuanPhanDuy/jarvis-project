"""Agent capability registry — centralises metadata about each agent type.

This module is the authoritative source for:
  - which tools each agent is allowed to use
  - a brief description of each agent's purpose
  - which prompt template file it loads

It is deliberately decoupled from the actual agent classes so it can be read
without instantiating agents (no Ollama / DB dependencies at import time).
"""
from __future__ import annotations


# ── Per-agent tool allowlists (mirrors the _*_TOOLS sets in each agent module)

_TOOL_ALLOWLISTS: dict[str, set[str] | None] = {
    "planner": None,   # PlannerAgent — receives all tools via delegate_task routing
    "researcher": {
        "web_search", "read_url", "browse", "ingest_document", "research_crawler",
        "save_report", "update_report", "search_memory", "search_episodic_memory",
        "query_knowledge_graph", "update_knowledge_graph", "summarize_youtube",
        "analyze_text", "get_weather", "ask_local_model", "record_feedback",
        "recall_user_preferences", "analyze_failures",
    },
    "coder": {
        "execute_python", "run_command", "filesystem_search", "git_context",
        "save_report", "update_report", "read_url", "analyze_text",
        "search_memory", "search_episodic_memory", "analyze_failures",
    },
    "qa": {
        "execute_python", "filesystem_search", "git_context", "analyze_text",
    },
    "data_analyst": {
        "query_database", "database_query", "execute_python", "filesystem_search",
        "save_report", "update_report", "analyze_text", "system_info",
    },
    "devops": {
        "run_command", "filesystem_search", "git_context", "system_info",
        "save_report", "update_report", "analyze_text",
    },
    "critic": set(),       # no tools — pure reasoning
    "team_agent": None,    # dynamic — tool set is configured per team role
}

_DESCRIPTIONS: dict[str, str] = {
    "planner":      "Orchestrates multi-step plans, delegates tasks to specialist agents.",
    "researcher":   "Web search, document ingestion, and knowledge synthesis.",
    "coder":        "Code generation, execution, and repository navigation.",
    "qa":           "Test writing, code quality analysis, and git review.",
    "data_analyst": "Database queries, data analysis, and statistical reporting.",
    "devops":       "Shell commands, system monitoring, and infrastructure tasks.",
    "critic":       "Evaluates agent outputs and flags low-quality or incorrect results.",
    "team_agent":   "Collaborative multi-role team; tool set depends on assigned role.",
}

_PROMPT_FILES: dict[str, str] = {
    "planner":      "planner",
    "researcher":   "researcher",
    "coder":        "coder",
    "qa":           "qa",
    "data_analyst": "data_analyst",
    "devops":       "devops",
    "critic":       "critic",
    "team_agent":   "team_lead",
}


def list_agents() -> list[dict]:
    """Return capability info for all known agent types."""
    from jarvis.prompts.overrides import get_override

    result = []
    for name in sorted(_TOOL_ALLOWLISTS):
        tools = _TOOL_ALLOWLISTS[name]
        prompt_file = _PROMPT_FILES.get(name, name)
        override = get_override(prompt_file)
        result.append({
            "name": name,
            "description": _DESCRIPTIONS.get(name, ""),
            "allowed_tools": sorted(tools) if tools is not None else None,
            "tool_count": len(tools) if tools is not None else "all",
            "prompt_file": prompt_file,
            "prompt_source": "override" if override else "file",
        })
    return result


def get_agent_info(name: str) -> dict | None:
    """Return capability info for a single agent type, or None if unknown."""
    name = name.lower()
    if name not in _TOOL_ALLOWLISTS:
        return None
    agents = list_agents()
    return next((a for a in agents if a["name"] == name), None)
