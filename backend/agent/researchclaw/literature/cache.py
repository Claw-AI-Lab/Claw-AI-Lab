"""Local query cache for literature search results.

Caches search results by (query, source, limit) hash to avoid
redundant API calls. Cache entries expire after TTL_SEC seconds.
Cache directory: .researchclaw_cache/literature/
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_CACHE_DIR = Path(".researchclaw_cache") / "literature"
_TTL_SEC = 86400 * 7  # 7 days (default for S2, OpenAlex)

# Per-source TTLs: arXiv updates daily at midnight, so 24h cache is optimal.
# Citation verification results are permanent (verified papers don't change).
_SOURCE_TTL: dict[str, float] = {
    "arxiv": 86400,         # 24 hours — arXiv metadata updates once/day
    "semantic_scholar": 86400 * 3,  # 3 days
    "openalex": 86400 * 3,  # 3 days
    "citation_verify": 86400 * 365,  # ~permanent
}


def _cache_dir(base: Path | None = None) -> Path:
    d = base or _DEFAULT_CACHE_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def cache_key(query: str, source: str, limit: int) -> str:
    """Deterministic cache key from query parameters."""
    raw = f"{query.strip().lower()}|{source.strip().lower()}|{limit}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def get_cached(
    query: str,
    source: str,
    limit: int,
    *,
    cache_base: Path | None = None,
    ttl: float | None = None,
) -> list[dict[str, Any]] | None:
    """Return cached results or None if miss/expired.

    If *ttl* is not provided, uses source-specific TTL from
    ``_SOURCE_TTL``, falling back to the global ``_TTL_SEC``.
    """
    d = _cache_dir(cache_base)
    key = cache_key(query, source, limit)
    path = d / f"{key}.json"

    if not path.exists():
        return None

    effective_ttl = ttl if ttl is not None else _SOURCE_TTL.get(source, _TTL_SEC)

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        ts = data.get("timestamp", 0)
        age_sec = time.time() - ts
        if age_sec > effective_ttl:
            logger.debug("Cache expired for key %s (age=%.0fs > ttl=%.0fs)",
                         key, age_sec, effective_ttl)
            return None
        papers = data.get("papers", [])
        if not isinstance(papers, list):
            return None
        age_str = _format_age(age_sec)
        logger.info(
            "[cache] HIT query=%r source=%s age=%s (%d papers)",
            query[:50], source, age_str, len(papers),
        )
        return papers
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def _format_age(seconds: float) -> str:
    """Human-readable age string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.0f}m"
    if seconds < 86400:
        return f"{seconds / 3600:.1f}h"
    return f"{seconds / 86400:.1f}d"


def put_cache(
    query: str,
    source: str,
    limit: int,
    papers: list[dict[str, Any]],
    *,
    cache_base: Path | None = None,
) -> None:
    """Write search results to cache."""
    d = _cache_dir(cache_base)
    key = cache_key(query, source, limit)
    path = d / f"{key}.json"

    payload = {
        "query": query,
        "source": source,
        "limit": limit,
        "timestamp": time.time(),
        "papers": papers,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.debug("Cached %d papers for key %s", len(papers), key)


def clear_cache(*, cache_base: Path | None = None) -> int:
    """Remove all cache files. Return count of files deleted."""
    d = _cache_dir(cache_base)
    count = 0
    for f in d.glob("*.json"):
        f.unlink()
        count += 1
    return count


def cache_stats(*, cache_base: Path | None = None) -> dict[str, Any]:
    """Return cache statistics."""
    d = _cache_dir(cache_base)
    files = list(d.glob("*.json"))
    total_bytes = sum(f.stat().st_size for f in files)
    return {
        "entries": len(files),
        "total_bytes": total_bytes,
        "cache_dir": str(d),
    }
