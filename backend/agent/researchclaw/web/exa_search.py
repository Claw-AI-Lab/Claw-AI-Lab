"""Web search powered by Exa AI Search API.

Exa provides neural, keyword-free web search with built-in content
retrieval (text, highlights, summaries).  It complements Tavily as an
alternative high-quality search backend.

Usage::

    client = ExaSearchClient(api_key="exa-...")
    results = client.search("knowledge distillation survey 2024")
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

from researchclaw.web.search import SearchResult, WebSearchResponse

logger = logging.getLogger(__name__)


@dataclass
class ExaSearchConfig:
    """Configuration knobs for the Exa search client."""

    search_type: str = "auto"  # auto | neural | fast
    num_results: int = 10
    highlights: bool = True
    text_max_characters: int = 1000
    summary: bool = False
    category: str = ""  # company | research paper | news | etc.
    include_domains: list[str] = field(default_factory=list)
    exclude_domains: list[str] = field(default_factory=list)
    include_text: list[str] = field(default_factory=list)
    exclude_text: list[str] = field(default_factory=list)
    start_published_date: str = ""  # ISO 8601
    end_published_date: str = ""  # ISO 8601


class ExaSearchClient:
    """Web search client backed by the Exa AI Search API.

    Uses the ``exa-py`` SDK.  When no ``EXA_API_KEY`` is set the client
    is inert and ``search()`` returns an empty response.

    Parameters
    ----------
    api_key:
        Exa API key.  Falls back to ``EXA_API_KEY`` env var.
    config:
        Optional :class:`ExaSearchConfig` to override defaults.
    """

    def __init__(
        self,
        *,
        api_key: str = "",
        config: ExaSearchConfig | None = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("EXA_API_KEY", "")
        self.config = config or ExaSearchConfig()

    @property
    def available(self) -> bool:
        """True when an API key is configured."""
        return bool(self.api_key)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        *,
        num_results: int | None = None,
        search_type: str | None = None,
        category: str | None = None,
        include_domains: list[str] | None = None,
        exclude_domains: list[str] | None = None,
        include_text: list[str] | None = None,
        exclude_text: list[str] | None = None,
        start_published_date: str | None = None,
        end_published_date: str | None = None,
    ) -> WebSearchResponse:
        """Search the web via Exa and return results.

        Falls back to an empty response when no API key is set.
        """
        if not self.available:
            logger.debug("Exa search skipped — no API key")
            return WebSearchResponse(query=query, source="exa")

        t0 = time.monotonic()
        try:
            return self._search_exa(
                query,
                num_results=num_results,
                search_type=search_type,
                category=category,
                include_domains=include_domains,
                exclude_domains=exclude_domains,
                include_text=include_text,
                exclude_text=exclude_text,
                start_published_date=start_published_date,
                end_published_date=end_published_date,
                t0=t0,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Exa search failed: %s", exc)
            return WebSearchResponse(
                query=query,
                elapsed_seconds=time.monotonic() - t0,
                source="exa",
            )

    def search_multi(
        self,
        queries: list[str],
        *,
        num_results: int | None = None,
    ) -> list[WebSearchResponse]:
        """Run multiple search queries with cross-query URL deduplication."""
        responses: list[WebSearchResponse] = []
        seen_urls: set[str] = set()

        for query in queries:
            resp = self.search(query, num_results=num_results)
            unique = [r for r in resp.results if r.url not in seen_urls]
            seen_urls.update(r.url for r in unique)
            resp.results = unique
            responses.append(resp)

        return responses

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _search_exa(
        self,
        query: str,
        *,
        num_results: int | None,
        search_type: str | None,
        category: str | None,
        include_domains: list[str] | None,
        exclude_domains: list[str] | None,
        include_text: list[str] | None,
        exclude_text: list[str] | None,
        start_published_date: str | None,
        end_published_date: str | None,
        t0: float,
    ) -> WebSearchResponse:
        from exa_py import Exa

        cfg = self.config
        client = Exa(self.api_key)
        client.headers["x-exa-integration"] = "claw-ai-lab"

        # Build contents spec
        contents: dict[str, Any] = {}
        if cfg.highlights:
            contents["highlights"] = True
        if cfg.text_max_characters > 0:
            contents["text"] = {"maxCharacters": cfg.text_max_characters}
        if cfg.summary:
            contents["summary"] = True

        kwargs: dict[str, Any] = {
            "query": query,
            "type": search_type or cfg.search_type,
            "num_results": num_results or cfg.num_results,
        }

        # Domain filtering
        domains_inc = include_domains or cfg.include_domains
        domains_exc = exclude_domains or cfg.exclude_domains
        if domains_inc:
            kwargs["include_domains"] = domains_inc
        if domains_exc:
            kwargs["exclude_domains"] = domains_exc

        # Text filtering
        texts_inc = include_text or cfg.include_text
        texts_exc = exclude_text or cfg.exclude_text
        if texts_inc:
            kwargs["include_text"] = texts_inc
        if texts_exc:
            kwargs["exclude_text"] = texts_exc

        # Category
        cat = category or cfg.category
        if cat:
            kwargs["category"] = cat

        # Date filtering
        start_date = start_published_date or cfg.start_published_date
        end_date = end_published_date or cfg.end_published_date
        if start_date:
            kwargs["start_published_date"] = start_date
        if end_date:
            kwargs["end_published_date"] = end_date

        response = client.search_and_contents(**kwargs, **contents)

        results = _parse_exa_results(response)
        elapsed = time.monotonic() - t0

        return WebSearchResponse(
            query=query,
            results=results,
            elapsed_seconds=elapsed,
            source="exa",
        )


def _build_snippet(result: Any) -> str:
    """Extract the best available snippet from an Exa result.

    Cascades through highlights -> summary -> text, returning the first
    non-empty value found.
    """
    # Highlights (list of strings)
    highlights = getattr(result, "highlights", None)
    if highlights:
        return " ... ".join(highlights)

    # Summary (string)
    summary = getattr(result, "summary", None)
    if summary:
        return summary

    # Text (string, truncated)
    text = getattr(result, "text", None)
    if text:
        return text[:500]

    return ""


def _parse_exa_results(response: Any) -> list[SearchResult]:
    """Convert an Exa SDK response into a list of ``SearchResult``."""
    results: list[SearchResult] = []
    for item in getattr(response, "results", []):
        snippet = _build_snippet(item)
        text = getattr(item, "text", "") or ""

        results.append(SearchResult(
            title=getattr(item, "title", "") or "",
            url=getattr(item, "url", "") or "",
            snippet=snippet,
            content=text,
            score=0.0,
            source="exa",
        ))
    return results
