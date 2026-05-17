"""Plugin: crawl_research — fetch AI research papers and ingest into JARVIS memory.

Searches ArXiv, HuggingFace blog, Anthropic research, and Papers With Code,
fetches full text, and indexes each document into ChromaDB so it becomes
immediately searchable via search_memory.
"""
from __future__ import annotations

from jarvis.config import get_settings
from jarvis.training.crawler import ResearchCrawler

SCHEMA: dict = {
    "name": "crawl_research",
    "description": (
        "Crawl AI research papers and blog posts from the internet and ingest them into "
        "JARVIS memory. Sources: ArXiv, HuggingFace blog, Anthropic research, Papers With Code. "
        "Documents are immediately searchable via search_memory after ingestion."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "topic": {
                "type": "string",
                "description": "Research topic to search for (e.g. 'RLHF', 'vision transformers').",
            },
            "sources": {
                "type": "array",
                "items": {"type": "string", "enum": ["arxiv", "hf", "anthropic", "pwc"]},
                "description": "Sources to crawl. Default: all four.",
                "default": ["arxiv", "hf", "anthropic", "pwc"],
            },
            "max_per_source": {
                "type": "integer",
                "description": "Maximum documents to fetch per source (default 5).",
                "default": 5,
            },
        },
        "required": ["topic"],
    },
}


def handle(tool_input: dict) -> str:
    try:
        settings = get_settings()
        crawler = ResearchCrawler(settings.reports_dir)

        topic = tool_input["topic"]
        sources = tool_input.get("sources") or ["arxiv", "hf", "anthropic", "pwc"]
        max_n = int(tool_input.get("max_per_source", 5))

        summary_lines: list[str] = []
        total = 0

        if "arxiv" in sources:
            docs = crawler.crawl_arxiv(topic, max_results=max_n)
            names = crawler.ingest_all(docs)
            ok = len([n for n in names if not n.startswith("ERROR")])
            total += ok
            summary_lines.append(f"ArXiv: {ok}/{len(docs)} documents ingested")

        if "hf" in sources:
            docs = crawler.crawl_hf_blog(max_posts=max_n)
            names = crawler.ingest_all(docs)
            ok = len([n for n in names if not n.startswith("ERROR")])
            total += ok
            summary_lines.append(f"HuggingFace blog: {ok}/{len(docs)} documents ingested")

        if "anthropic" in sources:
            docs = crawler.crawl_anthropic(max_posts=max_n)
            names = crawler.ingest_all(docs)
            ok = len([n for n in names if not n.startswith("ERROR")])
            total += ok
            summary_lines.append(f"Anthropic research: {ok}/{len(docs)} documents ingested")

        if "pwc" in sources:
            docs = crawler.crawl_papers_with_code(topic, max_results=max_n)
            names = crawler.ingest_all(docs)
            ok = len([n for n in names if not n.startswith("ERROR")])
            total += ok
            summary_lines.append(f"Papers With Code: {ok}/{len(docs)} documents ingested")

        detail = "\n".join(f"  - {line}" for line in summary_lines)
        return (
            f"Research crawl complete for topic '{topic}'. "
            f"Total ingested: {total} documents.\n{detail}\n"
            "All documents are now searchable via search_memory."
        )
    except Exception as exc:
        return f"ERROR: crawl_research failed — {exc}"
