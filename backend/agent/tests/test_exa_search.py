"""Tests for researchclaw.web.exa_search — ExaSearchClient."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from researchclaw.web.exa_search import (
    ExaSearchClient,
    ExaSearchConfig,
    _build_snippet,
    _parse_exa_results,
)
from researchclaw.web.search import SearchResult, WebSearchResponse


# ---------------------------------------------------------------------------
# Fixtures — hardcoded Exa-like response objects
# ---------------------------------------------------------------------------

def _make_exa_result(
    *,
    title: str = "Example Result",
    url: str = "https://example.com",
    text: str = "Full text content of the page.",
    highlights: list[str] | None = None,
    summary: str | None = None,
    published_date: str | None = None,
) -> SimpleNamespace:
    """Build a fake Exa SDK result object."""
    obj = SimpleNamespace(
        title=title,
        url=url,
        text=text,
        published_date=published_date,
    )
    if highlights is not None:
        obj.highlights = highlights
    if summary is not None:
        obj.summary = summary
    return obj


def _make_exa_response(results: list[SimpleNamespace] | None = None) -> SimpleNamespace:
    return SimpleNamespace(results=results or [])


# ---------------------------------------------------------------------------
# Snippet / fallback logic
# ---------------------------------------------------------------------------


class TestBuildSnippet:
    def test_highlights_preferred(self):
        r = _make_exa_result(
            highlights=["First highlight", "Second highlight"],
            summary="A summary",
            text="Full text",
        )
        snippet = _build_snippet(r)
        assert "First highlight" in snippet
        assert "Second highlight" in snippet

    def test_summary_fallback_when_no_highlights(self):
        r = _make_exa_result(summary="A summary", text="Full text")
        assert _build_snippet(r) == "A summary"

    def test_text_fallback_when_no_highlights_or_summary(self):
        r = _make_exa_result(text="Some text content")
        snippet = _build_snippet(r)
        assert snippet == "Some text content"

    def test_empty_when_nothing_available(self):
        r = SimpleNamespace()
        assert _build_snippet(r) == ""

    def test_text_truncated_at_500(self):
        r = _make_exa_result(text="x" * 1000)
        snippet = _build_snippet(r)
        assert len(snippet) == 500

    def test_empty_highlights_falls_through(self):
        r = _make_exa_result(text="Fallback text")
        r.highlights = []  # empty list should fall through
        assert _build_snippet(r) == "Fallback text"


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


class TestParseExaResults:
    def test_basic_parsing(self):
        response = _make_exa_response([
            _make_exa_result(
                title="Paper A",
                url="https://arxiv.org/abs/1234",
                text="Content A",
                highlights=["Key finding"],
            ),
            _make_exa_result(
                title="Paper B",
                url="https://arxiv.org/abs/5678",
                text="Content B",
            ),
        ])
        results = _parse_exa_results(response)
        assert len(results) == 2
        assert results[0].title == "Paper A"
        assert results[0].url == "https://arxiv.org/abs/1234"
        assert results[0].source == "exa"
        assert "Key finding" in results[0].snippet
        assert results[1].content == "Content B"

    def test_empty_response(self):
        response = _make_exa_response([])
        assert _parse_exa_results(response) == []

    def test_missing_fields_handled(self):
        response = _make_exa_response([SimpleNamespace()])
        results = _parse_exa_results(response)
        assert len(results) == 1
        assert results[0].title == ""
        assert results[0].url == ""


# ---------------------------------------------------------------------------
# ExaSearchClient
# ---------------------------------------------------------------------------


class TestExaSearchClient:
    def test_available_with_key(self):
        client = ExaSearchClient(api_key="test-key")
        assert client.available is True

    def test_not_available_without_key(self):
        client = ExaSearchClient(api_key="")
        assert client.available is False

    def test_search_returns_empty_when_no_key(self):
        client = ExaSearchClient(api_key="")
        resp = client.search("test query")
        assert resp.source == "exa"
        assert len(resp.results) == 0

    @patch.dict("os.environ", {"EXA_API_KEY": "env-key"})
    def test_picks_up_env_var(self):
        client = ExaSearchClient()
        assert client.api_key == "env-key"
        assert client.available is True

    @patch.dict("os.environ", {}, clear=True)
    def test_disabled_without_env_var(self):
        client = ExaSearchClient()
        assert client.available is False

    def test_search_with_mocked_sdk(self):
        """Full search path with mocked Exa SDK."""
        mock_exa_instance = MagicMock()
        mock_exa_instance.headers = {}
        mock_exa_instance.search_and_contents.return_value = _make_exa_response([
            _make_exa_result(
                title="Neural Search Paper",
                url="https://arxiv.org/abs/2401.00001",
                text="We present a neural search method...",
                highlights=["neural search outperforms BM25"],
            ),
        ])

        mock_exa_module = MagicMock()
        mock_exa_module.Exa.return_value = mock_exa_instance

        with patch.dict("sys.modules", {"exa_py": mock_exa_module}):
            client = ExaSearchClient(api_key="test-key")
            response = client.search("neural search methods")

            assert response.source == "exa"
            assert len(response.results) == 1
            assert response.results[0].title == "Neural Search Paper"
            assert "neural search outperforms BM25" in response.results[0].snippet
            assert response.elapsed_seconds > 0

            # Verify integration header was set
            assert mock_exa_instance.headers["x-exa-integration"] == "claw-ai-lab"

    def test_search_handles_sdk_error(self):
        """Graceful failure when SDK raises."""
        mock_exa_module = MagicMock()
        mock_exa_instance = MagicMock()
        mock_exa_instance.headers = {}
        mock_exa_instance.search_and_contents.side_effect = RuntimeError("API error")
        mock_exa_module.Exa.return_value = mock_exa_instance

        with patch.dict("sys.modules", {"exa_py": mock_exa_module}):
            client = ExaSearchClient(api_key="test-key")
            response = client.search("test query")
            assert response.source == "exa"
            assert len(response.results) == 0

    def test_search_multi_deduplication(self):
        """Multiple queries deduplicate by URL."""
        mock_exa_instance = MagicMock()
        mock_exa_instance.headers = {}
        # Both calls return the same URL
        mock_exa_instance.search_and_contents.return_value = _make_exa_response([
            _make_exa_result(
                title="Same Paper",
                url="https://arxiv.org/abs/same",
            ),
        ])

        mock_exa_module = MagicMock()
        mock_exa_module.Exa.return_value = mock_exa_instance

        with patch.dict("sys.modules", {"exa_py": mock_exa_module}):
            client = ExaSearchClient(api_key="test-key")
            responses = client.search_multi(["query1", "query2"])
            assert len(responses) == 2
            # First query has the result
            assert len(responses[0].results) == 1
            # Second query has it deduplicated
            assert len(responses[1].results) == 0


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------


class TestExaSearchConfig:
    def test_defaults(self):
        cfg = ExaSearchConfig()
        assert cfg.search_type == "auto"
        assert cfg.num_results == 10
        assert cfg.highlights is True
        assert cfg.summary is False

    def test_custom_values(self):
        cfg = ExaSearchConfig(
            search_type="neural",
            category="research paper",
            include_domains=["arxiv.org"],
        )
        assert cfg.search_type == "neural"
        assert cfg.category == "research paper"
        assert cfg.include_domains == ["arxiv.org"]


# ---------------------------------------------------------------------------
# WebSearchConfig — Exa fields present
# ---------------------------------------------------------------------------


class TestWebSearchConfigExa:
    def test_exa_fields_exist(self):
        from researchclaw.config import WebSearchConfig
        cfg = WebSearchConfig()
        assert cfg.exa_api_key == ""
        assert cfg.exa_api_key_env == "EXA_API_KEY"
        assert cfg.exa_search_type == "auto"
