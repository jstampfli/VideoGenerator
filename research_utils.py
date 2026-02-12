"""
Research utilities for biopic script generation.
Fetches biographical content from Wikipedia (and optionally other sources)
and provides a ResearchContext for injection into prompts.

Note: The Wikipedia prop=extracts API has a hard 1200-character limit.
We use action=parse with prop=text to get full page content, then strip HTML.
"""
import html
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests

RESEARCH_CACHE_DIR = Path("research_cache")
WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "VideoGenerator/1.0 (biopic script research; https://github.com)"
MAX_SUMMARY_CHARS = 1_000_000  # Full article; Gemini 2.0 Flash has 1M token context
RESEARCH_DEBUG = os.getenv("DEBUG", "").lower() in ("1", "true", "yes")


def _log(msg: str, verbose_only: bool = False) -> None:
    """Log with [RESEARCH] prefix. Use verbose_only for extra debug output."""
    if verbose_only and not RESEARCH_DEBUG:
        return
    print(f"[RESEARCH] {msg}")


@dataclass
class ResearchContext:
    """Structured research context for prompt injection."""

    summary: str
    key_facts: list[str] = field(default_factory=list)
    related_extracts: dict[str, str] = field(default_factory=dict)
    source_page_title: str | None = None

    def is_empty(self) -> bool:
        return not bool(self.summary.strip())


def _search_wikipedia(query: str) -> str | None:
    """
    Use OpenSearch to find the best matching Wikipedia page title.
    Returns the first result (main article) or None if no results.
    """
    params = {
        "action": "opensearch",
        "search": query,
        "limit": 5,
        "format": "json",
    }
    url = f"{WIKIPEDIA_API}?action=opensearch&search={requests.utils.quote(query)}&limit=5&format=json"
    _log(f"OpenSearch: query='{query}'", verbose_only=True)
    _log(f"OpenSearch URL: {url}", verbose_only=True)

    try:
        resp = requests.get(
            WIKIPEDIA_API,
            params=params,
            headers={"User-Agent": USER_AGENT},
            timeout=15,
        )
        _log(f"OpenSearch response status: {resp.status_code}")

        if resp.status_code != 200:
            _log(f"OpenSearch non-200: {resp.text[:500]}")
            return None

        data = resp.json()
        # OpenSearch returns [query, [titles], [descriptions], [urls]]
        if not isinstance(data, list) or len(data) < 2:
            _log("OpenSearch: unexpected response structure")
            return None

        titles = data[1]
        if not titles:
            _log("OpenSearch: no results found")
            return None

        # Prefer the first result (most relevant)
        first_title = titles[0]
        _log(f"OpenSearch: resolved to page '{first_title}'")
        return first_title
    except Exception as e:
        _log(f"ERROR: OpenSearch failed: {e}")
        raise


def _strip_html_to_text(html_content: str) -> str:
    """Convert HTML to plain text by stripping tags and normalizing whitespace."""
    # Remove script and style blocks (and their content)
    text = re.sub(r"<script[^>]*>.*?</script>", "", html_content, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
    # Replace block elements with newlines before stripping
    text = re.sub(r"</(p|div|br|li|tr|h[1-6])>", "\n", text, flags=re.IGNORECASE)
    # Strip all remaining HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    # Decode HTML entities (&#32; -> space, &nbsp; -> nbsp, etc.)
    text = html.unescape(text)
    # Remove CSS-like content (e.g. ".mw-parser-output .hatnote{...}")
    text = re.sub(r"\.[a-zA-Z0-9_-]+\s*\{[^}]*\}", "", text)
    # Collapse multiple newlines and spaces
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def _fetch_wikipedia_page_content(page_title: str) -> str | None:
    """
    Fetch full page content from Wikipedia using action=parse.
    The prop=extracts API has a hard 1200-char limit; parse returns full HTML.
    Returns plain text stripped of HTML, or None if page is missing.
    """
    params = {
        "action": "parse",
        "page": page_title,
        "prop": "text",
        "format": "json",
        "redirects": 1,
    }
    _log(f"Fetching Wikipedia for: {page_title}")
    _log(f"Parse API: page={page_title!r}", verbose_only=True)

    try:
        resp = requests.get(
            WIKIPEDIA_API,
            params=params,
            headers={"User-Agent": USER_AGENT},
            timeout=30,
        )
        _log(f"Wikipedia response status: {resp.status_code}")

        if resp.status_code != 200:
            _log(f"Wikipedia non-200 response: {resp.text[:500]}")
            return None

        data = resp.json()
        if "parse" not in data or "text" not in data["parse"]:
            if "error" in data:
                _log(f"Wikipedia API error: {data['error'].get('info', 'Unknown')}")
            else:
                _log("Wikipedia: parse.text missing")
            return None

        html = data["parse"]["text"].get("*", "")
        if not html or not html.strip():
            _log("Wikipedia: empty page content")
            return None

        text = _strip_html_to_text(html)
        _log(f"Wikipedia: found page \"{page_title}\", content length {len(text)} chars")
        if RESEARCH_DEBUG:
            _log(f"Content preview (first 300 chars): {text[:300]}...")
        return text
    except Exception as e:
        _log(f"ERROR: {e}")
        raise


def _truncate_at_paragraphs(text: str, max_chars: int) -> str:
    """Truncate at paragraph boundary, preferring early sections."""
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    last_para = truncated.rfind("\n\n")
    if last_para > max_chars // 2:
        return truncated[: last_para + 1].strip()
    return truncated.strip()


def _get_cache_path(person: str) -> Path:
    """Path to cached research for a person."""
    safe_name = "".join(c if c.isalnum() or c in (" ", "-", "_") else "" for c in person)
    safe_name = safe_name.replace(" ", "_").lower()
    return RESEARCH_CACHE_DIR / f"{safe_name}.json"


def _load_from_cache(person: str) -> ResearchContext | None:
    """Load cached research if available and valid."""
    path = _get_cache_path(person)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        ctx = ResearchContext(
            summary=data.get("summary", ""),
            key_facts=data.get("key_facts", []),
            related_extracts=data.get("related_extracts", {}),
            source_page_title=data.get("source_page_title"),
        )
        _log(f"Cache HIT for {person}")
        return ctx
    except Exception as e:
        _log(f"Cache read failed: {e}")
        return None


def _save_to_cache(person: str, ctx: ResearchContext) -> None:
    """Save research to cache."""
    if ctx.is_empty():
        return
    RESEARCH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _get_cache_path(person)
    data = {
        "summary": ctx.summary,
        "key_facts": ctx.key_facts,
        "related_extracts": ctx.related_extracts,
        "source_page_title": ctx.source_page_title,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    _log(f"Cache saved for {person}", verbose_only=True)


def fetch_research(person_of_interest: str, use_cache: bool = True) -> ResearchContext:
    """
    Fetch research for a person from Wikipedia.
    Returns a ResearchContext with summary (truncated to token budget) and optional key_facts.
    """
    if use_cache:
        cached = _load_from_cache(person_of_interest)
        if cached is not None:
            return cached

    _log(f"Cache MISS for {person_of_interest}, fetching...")

    try:
        # Resolve page title via OpenSearch
        page_title = _search_wikipedia(person_of_interest)
        if not page_title:
            _log("No Wikipedia page found, returning empty research context")
            return ResearchContext(summary="")

        # Fetch full page content (parse API - no 1200 char limit)
        extract = _fetch_wikipedia_page_content(page_title)
        if not extract:
            return ResearchContext(summary="")

        # Truncate for prompt budget
        summary = _truncate_at_paragraphs(extract, MAX_SUMMARY_CHARS)
        if len(summary) < len(extract):
            _log(f"Truncated extract from {len(extract)} to {len(summary)} chars")

        ctx = ResearchContext(
            summary=summary,
            key_facts=[],
            related_extracts={},
            source_page_title=page_title,
        )

        # Rate limit: be nice to Wikipedia
        time.sleep(0.5)

        if use_cache:
            _save_to_cache(person_of_interest, ctx)

        return ctx
    except Exception as e:
        _log(f"ERROR: {e}")
        return ResearchContext(summary="")


def get_research_context_block(ctx: ResearchContext) -> str:
    """
    Format ResearchContext for injection into prompts.
    Returns empty string if context is empty.
    """
    if ctx.is_empty():
        return ""
    return f"""
RESEARCH CONTEXT (use these facts—prefer them over your general knowledge):
---
{ctx.summary}
---
When facts conflict, prioritize the research above.
"""
