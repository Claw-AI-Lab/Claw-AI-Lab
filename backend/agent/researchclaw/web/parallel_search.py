"""Anonymous Parallel Search MCP adapter for the synchronous research pipeline."""

from __future__ import annotations

import json
import math
import time
import uuid
from urllib.parse import urlsplit

from researchclaw.web.search import SearchResult, WebSearchResponse

MCP_URL = "https://search.parallel.ai/mcp"
USER_AGENT = "Claw-AI-Lab/0.3.1 (+https://github.com/Claw-AI-Lab/Claw-AI-Lab)"


class ParallelSearchClient:
    """Search over the official MCP Streamable HTTP transport without credentials.

    One client keeps a stable session identifier across related research queries.
    A deadline covers connection, initialization, tool execution and cleanup.
    Failures propagate to the existing agent error handler; no provider fallback
    is attempted. Page extraction continues to use the existing crawler.
    """

    def __init__(self, *, timeout: float = 20.0) -> None:
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Parallel timeout must be finite and positive")
        self.timeout = timeout
        self.session_id = str(uuid.uuid4())

    def search(self, query: str, *, max_results: int = 10) -> WebSearchResponse:
        """Bridge the sync caller to MCP, including callers with a running loop."""
        try:
            from anyio.from_thread import start_blocking_portal
            import mcp  # noqa: F401
        except ImportError as exc:
            raise ImportError("Install Parallel support with pip install '.[parallel]'") from exc

        started = time.monotonic()
        with start_blocking_portal() as portal:
            future = portal.start_task_soon(self._search, query, max_results, started)
            try:
                return future.result()
            finally:
                # Interrupting a synchronous caller also cancels the MCP task.
                if not future.done():
                    future.cancel()

    async def _search(
        self, query: str, max_results: int, started: float
    ) -> WebSearchResponse:
        import anyio
        import httpx
        from mcp import ClientSession
        from mcp.client.streamable_http import streamable_http_client

        with anyio.fail_after(self.timeout):
            async with httpx.AsyncClient(
                headers={"User-Agent": USER_AGENT}, timeout=self.timeout,
                follow_redirects=False,
            ) as http:
                async with streamable_http_client(MCP_URL, http_client=http) as (read, write, _):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        response = await session.call_tool("web_search", {
                            "objective": query,
                            "search_queries": [query],
                            "session_id": self.session_id,
                        })

        if response.isError:
            message = " ".join(
                block.text for block in response.content if block.type == "text"
            )
            raise RuntimeError(f"Parallel search failed: {message[:500]}")
        payload = response.structuredContent
        if payload is None:
            text = "\n".join(
                block.text for block in response.content if block.type == "text"
            )
            payload = json.loads(text)
        if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
            raise ValueError("Invalid Parallel search response: expected results list")

        results = []
        for row in payload["results"]:
            if not isinstance(row, dict):
                raise ValueError("Invalid Parallel search result")
            url = row.get("url")
            if not isinstance(url, str):
                continue
            parsed_url = urlsplit(url)
            if parsed_url.scheme not in ("http", "https") or not parsed_url.netloc:
                continue
            excerpts = row.get("excerpts", [])
            if not isinstance(excerpts, list) or any(not isinstance(e, str) for e in excerpts):
                raise ValueError("Invalid Parallel search excerpts")
            content = "\n\n".join(excerpts)
            title = row.get("title") or url
            if not isinstance(title, str):
                raise ValueError("Invalid Parallel search title")
            results.append(SearchResult(
                title=title, url=url, snippet=content[:500], content=content,
                source="parallel",
            ))
        return WebSearchResponse(
            query=query, results=results[:max(0, max_results)], source="parallel",
            elapsed_seconds=time.monotonic() - started,
        )
