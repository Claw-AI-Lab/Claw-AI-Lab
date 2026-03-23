"""Network connectivity pre-check for external services.

Run once at the start of literature collection to probe all external
endpoints and disable unreachable ones, avoiding minutes of wasted
timeout retries during the actual search.
"""

from __future__ import annotations

import logging
import socket
import time
from dataclasses import dataclass, field
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

# Services to probe — (name, url, timeout_sec)
_ENDPOINTS = [
    ("openalex", "https://api.openalex.org/works?filter=title.search:test&per_page=1", 8),
    ("semantic_scholar", "https://api.semanticscholar.org/graph/v1/paper/search?query=test&limit=1", 8),
    ("arxiv", "https://export.arxiv.org/api/query?search_query=test&max_results=1", 8),
    ("duckduckgo", "https://html.duckduckgo.com/", 3),
    ("google_scholar", "https://scholar.google.com/", 3),
    ("tavily", "https://api.tavily.com/", 5),
]

_USER_AGENT = "Mozilla/5.0 (ResearchClaw connectivity probe)"


@dataclass
class ConnectivityReport:
    """Results of a network connectivity check."""
    reachable: dict[str, bool] = field(default_factory=dict)
    latency_ms: dict[str, float] = field(default_factory=dict)
    elapsed_sec: float = 0.0

    def is_up(self, name: str) -> bool:
        return self.reachable.get(name, False)

    def summary(self) -> str:
        parts = []
        for name, ok in self.reachable.items():
            status = "OK" if ok else "UNREACHABLE"
            ms = self.latency_ms.get(name, 0)
            parts.append(f"{name}: {status}" + (f" ({ms:.0f}ms)" if ok else ""))
        return " | ".join(parts)


def _probe_one(name: str, url: str, timeout: int) -> tuple[str, bool, float]:
    """Probe a single endpoint. Returns (name, reachable, latency_ms)."""
    import urllib.error

    req = Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        start = time.monotonic()
        resp = urlopen(req, timeout=timeout)  # noqa: S310
        resp.read(1024)
        resp.close()
        return name, True, (time.monotonic() - start) * 1000
    except urllib.error.HTTPError:
        # HTTP error (4xx/5xx) means host IS reachable
        return name, True, 0.0
    except Exception:
        return name, False, 0.0


def probe_all(endpoints: list[tuple[str, str, int]] | None = None) -> ConnectivityReport:
    """Probe all endpoints concurrently and return a ConnectivityReport.

    Uses a thread pool so that unreachable endpoints (3s timeout each)
    are probed in parallel rather than sequentially.
    """
    from concurrent.futures import ThreadPoolExecutor

    report = ConnectivityReport()
    t0 = time.monotonic()
    targets = endpoints or _ENDPOINTS

    with ThreadPoolExecutor(max_workers=len(targets)) as pool:
        futures = [pool.submit(_probe_one, name, url, timeout) for name, url, timeout in targets]
        for f in futures:
            name, ok, ms = f.result()
            report.reachable[name] = ok
            if ok:
                report.latency_ms[name] = ms

    report.elapsed_sec = time.monotonic() - t0
    logger.info("Connectivity probe: %s (%.1fs)", report.summary(), report.elapsed_sec)
    return report
