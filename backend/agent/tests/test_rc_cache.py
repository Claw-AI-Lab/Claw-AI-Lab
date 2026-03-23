"""Tests for literature query cache and degradation fallback."""

from __future__ import annotations

import importlib
from unittest.mock import patch

from researchclaw.literature.models import Author, Paper
from researchclaw.literature.search import search_papers

cache_mod = importlib.import_module("researchclaw.literature.cache")
cache_key = cache_mod.cache_key
cache_stats = cache_mod.cache_stats
clear_cache = cache_mod.clear_cache
get_cached = cache_mod.get_cached
put_cache = cache_mod.put_cache


class TestCacheKey:
    def test_deterministic(self, tmp_path):
        _ = tmp_path
        k1 = cache_key("transformer", "s2", 20)
        k2 = cache_key("transformer", "s2", 20)
        assert k1 == k2

    def test_different_query(self):
        k1 = cache_key("transformer", "s2", 20)
        k2 = cache_key("attention", "s2", 20)
        assert k1 != k2

    def test_case_insensitive(self):
        k1 = cache_key("Transformer", "S2", 20)
        k2 = cache_key("transformer", "s2", 20)
        assert k1 == k2

    def test_length_16(self):
        k = cache_key("test", "s2", 10)
        assert len(k) == 16


class TestGetPut:
    def test_put_and_get(self, tmp_path):
        papers = [{"paper_id": "1", "title": "Test Paper"}]
        put_cache("q1", "s2", 20, papers, cache_base=tmp_path)
        result = get_cached("q1", "s2", 20, cache_base=tmp_path)
        assert result is not None
        assert len(result) == 1
        assert result[0]["title"] == "Test Paper"

    def test_cache_miss(self, tmp_path):
        result = get_cached("nonexistent", "s2", 20, cache_base=tmp_path)
        assert result is None

    def test_cache_expired(self, tmp_path):
        papers = [{"paper_id": "1", "title": "Old"}]
        put_cache("q1", "s2", 20, papers, cache_base=tmp_path)
        result = get_cached("q1", "s2", 20, cache_base=tmp_path, ttl=0)
        assert result is None

    def test_cache_not_expired(self, tmp_path):
        papers = [{"paper_id": "1", "title": "Fresh"}]
        put_cache("q1", "s2", 20, papers, cache_base=tmp_path)
        result = get_cached("q1", "s2", 20, cache_base=tmp_path, ttl=9999)
        assert result is not None

    def test_corrupted_cache_returns_none(self, tmp_path):
        key = cache_key("q1", "s2", 20)
        (tmp_path / f"{key}.json").write_text("not json", encoding="utf-8")
        result = get_cached("q1", "s2", 20, cache_base=tmp_path)
        assert result is None


class TestClear:
    def test_clear_removes_all(self, tmp_path):
        put_cache("q1", "s2", 20, [{"id": "1"}], cache_base=tmp_path)
        put_cache("q2", "arxiv", 10, [{"id": "2"}], cache_base=tmp_path)
        count = clear_cache(cache_base=tmp_path)
        assert count == 2
        assert get_cached("q1", "s2", 20, cache_base=tmp_path) is None

    def test_clear_empty(self, tmp_path):
        count = clear_cache(cache_base=tmp_path)
        assert count == 0


class TestStats:
    def test_stats_empty(self, tmp_path):
        stats = cache_stats(cache_base=tmp_path)
        assert stats["entries"] == 0
        assert stats["total_bytes"] == 0

    def test_stats_with_entries(self, tmp_path):
        put_cache("q1", "s2", 20, [{"id": "1"}], cache_base=tmp_path)
        stats = cache_stats(cache_base=tmp_path)
        assert stats["entries"] == 1
        assert stats["total_bytes"] > 0


class TestSearchDegradation:
    def test_search_uses_cache_on_failure(self, tmp_path):
        cached_papers = [
            {
                "paper_id": "s2-123",
                "title": "Cached Paper",
                "authors": [],
                "year": 2024,
                "abstract": "",
                "venue": "",
                "citation_count": 10,
                "doi": "",
                "arxiv_id": "",
                "url": "",
                "source": "semantic_scholar",
            }
        ]
        put_cache(
            "test query",
            "semantic_scholar",
            20,
            cached_papers,
            cache_base=tmp_path,
        )

        with patch(
            "researchclaw.literature.search.search_openalex",
            side_effect=RuntimeError("API down"),
        ):
            with patch(
                "researchclaw.literature.search.search_semantic_scholar",
                side_effect=RuntimeError("API down"),
            ):
                with patch(
                    "researchclaw.literature.search.search_arxiv",
                    side_effect=RuntimeError("API down"),
                ):
                    with patch(
                        "researchclaw.literature.cache._DEFAULT_CACHE_DIR", tmp_path
                    ):
                        with patch(
                            "researchclaw.literature.search.time.sleep", lambda _: None
                        ):
                            results = search_papers("test query", limit=20)

        assert len(results) >= 1
        assert results[0].title == "Cached Paper"

    def test_search_caches_successful_results(self, tmp_path):
        mock_paper = Paper(
            paper_id="s2-test",
            title="Test",
            authors=(Author(name="Smith"),),
            year=2024,
            abstract="abs",
            source="semantic_scholar",
        )

        with patch(
            "researchclaw.literature.search.search_semantic_scholar",
            return_value=[mock_paper],
        ):
            with patch("researchclaw.literature.search.search_arxiv", return_value=[]):
                with patch(
                    "researchclaw.literature.cache._DEFAULT_CACHE_DIR", tmp_path
                ):
                    with patch(
                        "researchclaw.literature.search.time.sleep", lambda _: None
                    ):
                        _ = search_papers("test", limit=20)

        cached = get_cached("test", "semantic_scholar", 20, cache_base=tmp_path)
        assert cached is not None
        assert cached[0]["paper_id"] == "s2-test"
