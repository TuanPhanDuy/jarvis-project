from __future__ import annotations

from dataclasses import dataclass

from tavily import TavilyClient

_client: TavilyClient | None = None


def _get_client(api_key: str) -> TavilyClient:
    global _client
    if _client is None:
        _client = TavilyClient(api_key=api_key)
    return _client


@dataclass
class SearchInput:
    query: str
    max_results: int = 5


def handle_web_search(tool_input: dict, tavily_api_key: str) -> str:
    try:
        inp = SearchInput(
            query=tool_input["query"],
            max_results=int(tool_input.get("max_results", 5)),
        )
        client = _get_client(tavily_api_key)
        response = client.search(inp.query, max_results=inp.max_results)
        results = response.get("results", [])
        if not results:
            return "No results found."
        lines: list[str] = []
        for i, r in enumerate(results, 1):
            lines.append(f"[{i}] {r.get('title', 'No title')}")
            lines.append(f"    URL: {r.get('url', '')}")
            lines.append(f"    {r.get('content', '')[:500]}")
            lines.append("")
        return "\n".join(lines).strip()
    except Exception as e:
        return f"ERROR: web_search failed — {e}"


SCHEMA: dict = {
    "name": "web_search",
    "description": (
        "Search the web for current information on AI research topics, papers, and techniques. "
        "Returns a list of relevant excerpts with source URLs. "
        "Use specific queries — include paper names, author names, or precise technical terms."
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
