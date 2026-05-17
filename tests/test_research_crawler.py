"""Tests for ResearchCrawler and the crawl_research plugin."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch


# ── ResearchCrawler ──────────────────────────────────────────────────────────

class TestResearchCrawler:
    def _make_crawler(self, tmp_path: Path):
        from jarvis.training.crawler import ResearchCrawler
        return ResearchCrawler(tmp_path)

    def test_ingest_doc_writes_markdown_and_returns_filename(self, tmp_path):
        from jarvis.training.crawler import CrawledDoc, ResearchCrawler

        crawler = ResearchCrawler(tmp_path)
        doc = CrawledDoc(title="Test Paper", url="http://x.com", text="hello world", source="arxiv")

        with patch("jarvis.training.crawler.index_new_report") as mock_idx:
            result = crawler.ingest_doc(doc)

        assert result.endswith(".md")
        assert not result.startswith("ERROR")
        report_path = tmp_path / result
        assert report_path.exists()
        content = report_path.read_text()
        assert "Test Paper" in content
        assert "hello world" in content
        mock_idx.assert_called_once()

    def test_ingest_doc_sanitises_title(self, tmp_path):
        from jarvis.training.crawler import CrawledDoc, ResearchCrawler

        crawler = ResearchCrawler(tmp_path)
        doc = CrawledDoc(title="Title: with / special chars!", url="http://x.com",
                         text="body", source="hf_blog")
        with patch("jarvis.training.crawler.index_new_report"):
            result = crawler.ingest_doc(doc)
        assert "/" not in result
        assert ":" not in result

    def test_ingest_all_returns_list_of_filenames(self, tmp_path):
        from jarvis.training.crawler import CrawledDoc, ResearchCrawler

        crawler = ResearchCrawler(tmp_path)
        docs = [
            CrawledDoc(title=f"Paper {i}", url=f"http://x.com/{i}", text=f"text {i}", source="pwc")
            for i in range(3)
        ]
        with patch("jarvis.training.crawler.index_new_report"):
            results = crawler.ingest_all(docs)
        assert len(results) == 3
        assert all(r.endswith(".md") for r in results)

    def test_crawl_arxiv_returns_crawled_docs(self, tmp_path):
        from jarvis.training.crawler import ResearchCrawler

        mock_result = MagicMock()
        mock_result.title = "Attention Is All You Need"
        mock_result.entry_id = "https://arxiv.org/abs/1706.03762"
        mock_result.categories = ["cs.AI"]

        crawler = ResearchCrawler(tmp_path)

        with (
            patch("arxiv.Client") as MockClient,
            patch.object(crawler, "_fetch_url", return_value="some paper text"),
        ):
            mock_client_instance = MagicMock()
            MockClient.return_value = mock_client_instance
            mock_client_instance.results.return_value = iter([mock_result])
            docs = crawler.crawl_arxiv("attention mechanism", max_results=1)

        assert len(docs) == 1
        assert docs[0].title == "Attention Is All You Need"
        assert docs[0].source == "arxiv"
        assert docs[0].text == "some paper text"

    def test_crawl_arxiv_skips_error_results(self, tmp_path):
        from jarvis.training.crawler import ResearchCrawler

        mock_result = MagicMock()
        mock_result.title = "Bad Paper"
        mock_result.entry_id = "https://arxiv.org/abs/0000.0000"
        mock_result.categories = []

        crawler = ResearchCrawler(tmp_path)

        with (
            patch("arxiv.Client") as MockClient,
            patch.object(crawler, "_fetch_url", return_value="ERROR: timeout"),
        ):
            MockClient.return_value.results.return_value = iter([mock_result])
            docs = crawler.crawl_arxiv("test", max_results=1)

        assert len(docs) == 0

    def test_crawl_hf_blog_parses_urls(self, tmp_path):
        from jarvis.training.crawler import ResearchCrawler

        index_html = 'some text href="/blog/llm-post" other href="/blog/rlhf-guide" end'
        crawler = ResearchCrawler(tmp_path)

        with patch.object(crawler, "_fetch_url") as mock_fetch:
            mock_fetch.side_effect = [index_html, "post body 1", "post body 2"]
            docs = crawler.crawl_hf_blog(max_posts=2)

        assert len(docs) == 2
        assert all(d.source == "hf_blog" for d in docs)

    def test_fetch_url_delegates_to_url_reader(self, tmp_path):
        from jarvis.training.crawler import ResearchCrawler

        crawler = ResearchCrawler(tmp_path)
        with patch("jarvis.training.crawler.handle_read_url", return_value="page text") as mock_reader:
            result = crawler._fetch_url("https://example.com")

        mock_reader.assert_called_once_with({"url": "https://example.com", "max_chars": 12000})
        assert result == "page text"


# ── research_crawler plugin ──────────────────────────────────────────────────

class TestResearchCrawlerPlugin:
    def test_schema_has_required_fields(self):
        from jarvis.tools.plugins.research_crawler import SCHEMA
        assert SCHEMA["name"] == "crawl_research"
        assert "topic" in SCHEMA["input_schema"]["properties"]
        assert "topic" in SCHEMA["input_schema"]["required"]

    def test_handle_returns_summary_string(self, tmp_path):
        from jarvis.tools.plugins.research_crawler import handle

        with (
            patch("jarvis.tools.plugins.research_crawler.get_settings") as mock_settings,
            patch("jarvis.tools.plugins.research_crawler.ResearchCrawler") as MockCrawler,
        ):
            mock_settings.return_value.reports_dir = tmp_path
            mock_instance = MagicMock()
            MockCrawler.return_value = mock_instance
            mock_instance.crawl_arxiv.return_value = [MagicMock()]
            mock_instance.ingest_all.return_value = ["doc_paper.md"]
            mock_instance.crawl_hf_blog.return_value = []
            mock_instance.crawl_anthropic.return_value = []
            mock_instance.crawl_papers_with_code.return_value = []

            result = handle({"topic": "RLHF", "sources": ["arxiv"], "max_per_source": 2})

        assert "RLHF" in result
        assert "ingested" in result

    def test_handle_returns_error_string_on_exception(self):
        from jarvis.tools.plugins.research_crawler import handle

        with patch("jarvis.tools.plugins.research_crawler.get_settings", side_effect=RuntimeError("boom")):
            result = handle({"topic": "test"})

        assert result.startswith("ERROR")
