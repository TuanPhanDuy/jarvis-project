"""Research crawler — fetches AI papers from ArXiv, HuggingFace blog, Anthropic,
and Papers With Code, then ingests each into ChromaDB via the report pipeline.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import structlog
from jarvis.tools.memory import index_new_report
from jarvis.tools.url_reader import handle_read_url

log = structlog.get_logger()

_REQUEST_DELAY = 1.5  # seconds between HTTP requests (polite crawling)


@dataclass
class CrawledDoc:
    title: str
    url: str
    text: str
    source: str
    tags: list[str] = field(default_factory=list)


class ResearchCrawler:
    """Fetch full-text documents from AI research sources and ingest into ChromaDB."""

    def __init__(self, reports_dir: Path) -> None:
        self._reports_dir = reports_dir
        self._reports_dir.mkdir(parents=True, exist_ok=True)

    # ── Source-specific fetchers ─────────────────────────────────────────────

    def crawl_arxiv(self, topic: str, max_results: int = 10) -> list[CrawledDoc]:
        """Search ArXiv for recent papers matching `topic`."""
        docs: list[CrawledDoc] = []
        try:
            import arxiv

            client = arxiv.Client(page_size=max_results, delay_seconds=1.0)
            search = arxiv.Search(
                query=topic,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.SubmittedDate,
            )
            for result in client.results(search):
                text = self._fetch_url(result.entry_id)
                if text and not text.startswith("ERROR"):
                    docs.append(CrawledDoc(
                        title=result.title,
                        url=result.entry_id,
                        text=text,
                        source="arxiv",
                        tags=["arxiv"] + (result.categories or []),
                    ))
                    log.info("arxiv_fetched", title=result.title[:60])
                time.sleep(_REQUEST_DELAY)
        except Exception as exc:
            log.error("arxiv_crawl_failed", topic=topic, error=str(exc))
        return docs

    def crawl_hf_blog(self, max_posts: int = 10) -> list[CrawledDoc]:
        """Scrape recent HuggingFace blog posts."""
        docs: list[CrawledDoc] = []
        try:
            index_text = self._fetch_url("https://huggingface.co/blog", max_chars=20000)
            urls = re.findall(r'href="(/blog/[a-zA-Z0-9_-]+)"', index_text)
            seen: set[str] = set()
            for path in urls:
                if path in seen or len(seen) >= max_posts:
                    break
                seen.add(path)
                full_url = f"https://huggingface.co{path}"
                text = self._fetch_url(full_url)
                if text and not text.startswith("ERROR"):
                    title = path.split("/")[-1].replace("-", " ").title()
                    docs.append(CrawledDoc(
                        title=title,
                        url=full_url,
                        text=text,
                        source="hf_blog",
                        tags=["huggingface", "blog"],
                    ))
                    log.info("hf_blog_fetched", url=full_url)
                time.sleep(_REQUEST_DELAY)
        except Exception as exc:
            log.error("hf_blog_crawl_failed", error=str(exc))
        return docs

    def crawl_anthropic(self, max_posts: int = 10) -> list[CrawledDoc]:
        """Scrape Anthropic research page."""
        docs: list[CrawledDoc] = []
        try:
            index_text = self._fetch_url("https://www.anthropic.com/research", max_chars=20000)
            urls = re.findall(r'href="(/research/[a-zA-Z0-9_-]+)"', index_text)
            seen: set[str] = set()
            for path in urls:
                if path in seen or len(seen) >= max_posts:
                    break
                seen.add(path)
                full_url = f"https://www.anthropic.com{path}"
                text = self._fetch_url(full_url)
                if text and not text.startswith("ERROR"):
                    title = path.split("/")[-1].replace("-", " ").title()
                    docs.append(CrawledDoc(
                        title=title,
                        url=full_url,
                        text=text,
                        source="anthropic",
                        tags=["anthropic", "research"],
                    ))
                    log.info("anthropic_fetched", url=full_url)
                time.sleep(_REQUEST_DELAY)
        except Exception as exc:
            log.error("anthropic_crawl_failed", error=str(exc))
        return docs

    def crawl_papers_with_code(self, topic: str, max_results: int = 10) -> list[CrawledDoc]:
        """Fetch paper summaries from Papers With Code."""
        docs: list[CrawledDoc] = []
        try:
            import urllib.parse
            q = urllib.parse.quote(topic)
            search_url = f"https://paperswithcode.com/search?q_meta=&q_type=&q={q}"
            index_text = self._fetch_url(search_url, max_chars=20000)
            urls = re.findall(r'href="(/paper/[a-zA-Z0-9_-]+)"', index_text)
            seen: set[str] = set()
            for path in urls:
                if path in seen or len(seen) >= max_results:
                    break
                seen.add(path)
                full_url = f"https://paperswithcode.com{path}"
                text = self._fetch_url(full_url)
                if text and not text.startswith("ERROR"):
                    title = path.split("/")[-1].replace("-", " ").title()
                    docs.append(CrawledDoc(
                        title=title,
                        url=full_url,
                        text=text,
                        source="pwc",
                        tags=["papers-with-code", topic],
                    ))
                    log.info("pwc_fetched", url=full_url)
                time.sleep(_REQUEST_DELAY)
        except Exception as exc:
            log.error("pwc_crawl_failed", topic=topic, error=str(exc))
        return docs

    # ── Ingestion ────────────────────────────────────────────────────────────

    def ingest_doc(self, doc: CrawledDoc) -> str:
        """Save a CrawledDoc as a markdown report and index it into ChromaDB."""
        try:
            safe = re.sub(r"[^a-zA-Z0-9_-]", "_", doc.title[:60])
            report_name = f"research_{doc.source}_{safe}.md"
            report_path = self._reports_dir / report_name

            tags_line = " ".join(f"`{t}`" for t in doc.tags)
            content = (
                f"# {doc.title}\n\n"
                f"*Source: [{doc.source}]({doc.url})*  \n"
                f"*Tags: {tags_line}*\n\n"
                f"{doc.text}"
            )
            report_path.write_text(content, encoding="utf-8")
            index_new_report(self._reports_dir, report_name)
            return report_name
        except Exception as exc:
            return f"ERROR: ingest_doc failed — {exc}"

    def ingest_all(self, docs: list[CrawledDoc]) -> list[str]:
        """Ingest a list of docs and return their report filenames."""
        results = []
        for doc in docs:
            result = self.ingest_doc(doc)
            results.append(result)
        return results

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _fetch_url(self, url: str, max_chars: int = 12000) -> str:
        return handle_read_url({"url": url, "max_chars": max_chars})
