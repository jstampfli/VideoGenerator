"""
Unit tests for research_utils.py.
Mocks requests to avoid hitting Wikipedia API.
"""

import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

import research_utils


def _make_opensearch_response(titles: list[str]) -> list:
    """OpenSearch returns [query, [titles], [descriptions], [urls]]."""
    return ["query", titles, ["desc"] * len(titles), ["url"] * len(titles)]


def _make_parse_response(page_title: str, html_content: str) -> dict:
    """Wikipedia action=parse response structure. HTML is stripped to text in the real code."""
    return {
        "parse": {
            "title": page_title,
            "text": {"*": html_content},
        }
    }


class TestResearchContext(unittest.TestCase):
    """Test ResearchContext dataclass."""

    def test_is_empty_true_when_summary_empty(self):
        ctx = research_utils.ResearchContext(summary="")
        self.assertTrue(ctx.is_empty())

    def test_is_empty_true_when_summary_whitespace_only(self):
        ctx = research_utils.ResearchContext(summary="   \n  ")
        self.assertTrue(ctx.is_empty())

    def test_is_empty_false_when_summary_has_content(self):
        ctx = research_utils.ResearchContext(summary="Abraham Lincoln was the 16th president.")
        self.assertFalse(ctx.is_empty())


class TestSearchWikipedia(unittest.TestCase):
    """Test _search_wikipedia (OpenSearch)."""

    @patch("research_utils.requests.get")
    def test_returns_first_title(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = _make_opensearch_response(["Abraham Lincoln", "Abraham Lincoln (film)"])
        mock_get.return_value = mock_resp

        result = research_utils._search_wikipedia("Abraham Lincoln")
        self.assertEqual(result, "Abraham Lincoln")

    @patch("research_utils.requests.get")
    def test_returns_none_when_no_results(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = _make_opensearch_response([])
        mock_get.return_value = mock_resp

        result = research_utils._search_wikipedia("XyzNonexistentPerson123")
        self.assertIsNone(result)

    @patch("research_utils.requests.get")
    def test_returns_none_on_non_200(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "Internal Server Error"
        mock_get.return_value = mock_resp

        result = research_utils._search_wikipedia("Lincoln")
        self.assertIsNone(result)


class TestFetchWikipediaPageContent(unittest.TestCase):
    """Test _fetch_wikipedia_page_content (parse API)."""

    @patch("research_utils.requests.get")
    def test_returns_stripped_text(self, mock_get):
        html = "<p>Abraham Lincoln was the 16th president of the United States.</p>"
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = _make_parse_response("Abraham Lincoln", html)
        mock_get.return_value = mock_resp

        result = research_utils._fetch_wikipedia_page_content("Abraham Lincoln")
        self.assertIn("Abraham Lincoln was the 16th president", result)
        self.assertNotIn("<p>", result)

    @patch("research_utils.requests.get")
    def test_returns_none_on_api_error(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"error": {"info": "The page you specified doesn't exist."}}
        mock_get.return_value = mock_resp

        result = research_utils._fetch_wikipedia_page_content("Nonexistent")
        self.assertIsNone(result)

    @patch("research_utils.requests.get")
    def test_returns_none_on_non_200(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 503
        mock_resp.text = "Service Unavailable"
        mock_get.return_value = mock_resp

        result = research_utils._fetch_wikipedia_page_content("Lincoln")
        self.assertIsNone(result)


class TestTruncateAtParagraphs(unittest.TestCase):
    """Test _truncate_at_paragraphs."""

    def test_returns_unchanged_when_under_limit(self):
        text = "Short text."
        result = research_utils._truncate_at_paragraphs(text, 100)
        self.assertEqual(result, text)

    def test_truncates_at_paragraph_boundary(self):
        text = "First para.\n\nSecond para.\n\nThird para."
        result = research_utils._truncate_at_paragraphs(text, 25)
        self.assertLessEqual(len(result), 25)
        self.assertIn("First para", result)
        # Should end at a paragraph boundary
        self.assertTrue(result.endswith("para.") or result.endswith("\n\n"))

    def test_truncates_when_no_paragraph_break(self):
        text = "A" * 200
        result = research_utils._truncate_at_paragraphs(text, 50)
        self.assertEqual(len(result), 50)


class TestFetchResearch(unittest.TestCase):
    """Test fetch_research (main entry point)."""

    @patch("research_utils.time.sleep")  # avoid slowing tests
    @patch("research_utils._fetch_wikipedia_page_content")
    @patch("research_utils._search_wikipedia")
    def test_returns_research_context_on_success(self, mock_search, mock_fetch, mock_sleep):
        mock_search.return_value = "Abraham Lincoln"
        mock_fetch.return_value = "Abraham Lincoln was the 16th president."

        result = research_utils.fetch_research("Abraham Lincoln", use_cache=False)

        self.assertFalse(result.is_empty())
        self.assertIn("16th president", result.summary)
        self.assertEqual(result.source_page_title, "Abraham Lincoln")

    @patch("research_utils._search_wikipedia")
    def test_returns_empty_when_no_page_found(self, mock_search):
        mock_search.return_value = None

        result = research_utils.fetch_research("XyzNonexistent999", use_cache=False)

        self.assertTrue(result.is_empty())

    @patch("research_utils._fetch_wikipedia_page_content")
    @patch("research_utils._search_wikipedia")
    def test_returns_empty_when_content_empty(self, mock_search, mock_fetch):
        mock_search.return_value = "Some Page"
        mock_fetch.return_value = None

        result = research_utils.fetch_research("Some Person", use_cache=False)

        self.assertTrue(result.is_empty())

    @patch("research_utils._load_from_cache")
    def test_cache_hit_skips_fetch(self, mock_load):
        mock_load.return_value = research_utils.ResearchContext(
            summary="Cached content",
            source_page_title="Abraham Lincoln",
        )

        result = research_utils.fetch_research("Abraham Lincoln", use_cache=True)
        self.assertFalse(result.is_empty())
        self.assertEqual(result.summary, "Cached content")
        mock_load.assert_called_once()


class TestGetResearchContextBlock(unittest.TestCase):
    """Test get_research_context_block for prompt injection."""

    def test_returns_empty_for_empty_context(self):
        ctx = research_utils.ResearchContext(summary="")
        result = research_utils.get_research_context_block(ctx)
        self.assertEqual(result, "")

    def test_returns_formatted_block_with_content(self):
        ctx = research_utils.ResearchContext(summary="Lincoln was president.")
        result = research_utils.get_research_context_block(ctx)
        self.assertIn("RESEARCH CONTEXT", result)
        self.assertIn("Lincoln was president", result)
        self.assertIn("prefer them over your general knowledge", result)
        self.assertIn("When facts conflict, prioritize the research above", result)


def _get_printed_messages(mock_print):
    """Extract the first arg (message) from each print call."""
    return [c[0][0] if c[0] else "" for c in mock_print.call_args_list]


class TestResearchLogging(unittest.TestCase):
    """Test that [RESEARCH] log messages are emitted (via cap capturing print)."""

    @patch("research_utils.print")
    @patch("research_utils.time.sleep")
    @patch("research_utils.requests.get")
    def test_logs_cache_miss_and_fetch(self, mock_get, mock_sleep, mock_print):
        # Mock OpenSearch response
        opensearch_resp = MagicMock()
        opensearch_resp.status_code = 200
        opensearch_resp.json.return_value = ["Lincoln", ["Abraham Lincoln"], ["desc"], ["url"]]
        # Mock parse response
        parse_resp = MagicMock()
        parse_resp.status_code = 200
        parse_resp.json.return_value = _make_parse_response("Abraham Lincoln", "<p>Lincoln was president.</p>")
        mock_get.side_effect = [opensearch_resp, parse_resp]

        research_utils.fetch_research("Lincoln", use_cache=False)

        printed = _get_printed_messages(mock_print)
        self.assertTrue(any("Cache MISS" in p for p in printed))
        self.assertTrue(any("Fetching Wikipedia" in p or "Wikipedia" in p for p in printed))

    @patch("research_utils.print")
    @patch("research_utils._get_cache_path")
    def test_logs_cache_hit(self, mock_cache_path, mock_print):
        cache_file = Path(__file__).parent / "tmp_research_cache_lincoln.json"
        mock_cache_path.return_value = cache_file
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump({
                "summary": "Cached",
                "key_facts": [],
                "related_extracts": {},
                "source_page_title": "Lincoln",
            }, f)
        try:
            research_utils.fetch_research("Lincoln", use_cache=True)
            printed = _get_printed_messages(mock_print)
            self.assertTrue(any("Cache HIT" in p for p in printed))
        finally:
            if cache_file.exists():
                cache_file.unlink()

    @patch("research_utils.print")
    @patch("research_utils.time.sleep")
    @patch("research_utils.requests.get")
    def test_logs_response_status(self, mock_get, mock_sleep, mock_print):
        opensearch_resp = MagicMock()
        opensearch_resp.status_code = 200
        opensearch_resp.json.return_value = ["Lincoln", ["Abraham Lincoln"], ["desc"], ["url"]]
        parse_resp = MagicMock()
        parse_resp.status_code = 200
        parse_resp.json.return_value = _make_parse_response("Abraham Lincoln", "<p>Content</p>")
        mock_get.side_effect = [opensearch_resp, parse_resp]

        research_utils.fetch_research("Lincoln", use_cache=False)

        printed = _get_printed_messages(mock_print)
        self.assertTrue(any("status" in p.lower() or "Wikipedia" in p for p in printed))


if __name__ == "__main__":
    unittest.main()
