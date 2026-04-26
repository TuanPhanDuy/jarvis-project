"""Tool: read_url — fetch a web page or arXiv paper and return its text content."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class URLInput:
    url: str
    max_chars: int = 8000


def handle_read_url(tool_input: dict) -> str:
    try:
        import httpx
        from bs4 import BeautifulSoup

        inp = URLInput(
            url=tool_input["url"],
            max_chars=int(tool_input.get("max_chars", 8000)),
        )

        url = inp.url
        # Redirect arXiv abstract pages to the HTML full-text version
        if "arxiv.org/abs/" in url:
            url = url.replace("arxiv.org/abs/", "arxiv.org/html/")

        headers = {"User-Agent": "JARVIS-Research-Agent/1.0 (educational use)"}
        response = httpx.get(url, headers=headers, follow_redirects=True, timeout=15)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove navigation, ads, scripts, styles
        for tag in soup(["nav", "header", "footer", "script", "style", "aside", "advertisement"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)
        # Collapse excessive blank lines
        lines = [ln for ln in text.splitlines() if ln.strip()]
        cleaned = "\n".join(lines)

        if len(cleaned) > inp.max_chars:
            cleaned = cleaned[: inp.max_chars] + f"\n\n[...truncated at {inp.max_chars} chars]"

        return cleaned or "ERROR: page returned no readable text"
    except Exception as e:
        return f"ERROR: read_url failed — {e}"


SCHEMA: dict = {
    "name": "read_url",
    "description": (
        "Fetch and read the text content of a web page or arXiv paper. "
        "Useful for reading primary sources directly — research papers, blog posts, documentation. "
        "For arXiv, pass the abstract URL (arxiv.org/abs/XXXX) and it will fetch the HTML version."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "Full URL of the page to read.",
            },
            "max_chars": {
                "type": "integer",
                "description": "Maximum characters to return (default 8000).",
                "default": 8000,
            },
        },
        "required": ["url"],
    },
}
