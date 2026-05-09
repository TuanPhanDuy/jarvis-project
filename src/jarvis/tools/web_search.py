from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SearchInput:
    query: str
    max_results: int = 5


def handle_web_search(tool_input: dict) -> str:
    try:
        from ddgs import DDGS

        inp = SearchInput(
            query=tool_input["query"],
            max_results=int(tool_input.get("max_results", 5)),
        )
        results = list(DDGS().text(inp.query, max_results=inp.max_results))
        if not results:
            return "No results found."
        lines: list[str] = []
        for i, r in enumerate(results, 1):
            lines.append(f"[{i}] {r.get('title', 'No title')}")
            lines.append(f"    URL: {r.get('href', '')}")
            lines.append(f"    {r.get('body', '')[:500]}")
            lines.append("")
        return "\n".join(lines).strip()
    except Exception as e:
        return f"ERROR: web_search failed — {e}"


SCHEMA: dict = {
    "name": "web_search",
    "description": (
        "Search the web for current information on any topic. "
        "Returns a list of relevant results with titles, URLs, and excerpts. "
        "Use specific queries for best results."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query. Be specific for best results.",
            },
            "max_results": {
                "type": "integer",
                "description": "Number of results to return (1–10). Defaults to 5.",
            },
        },
        "required": ["query"],
    },
}
