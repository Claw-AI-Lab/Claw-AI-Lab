"""Parallel provider tests through config, research stages and real MCP transport.

HTTP fixtures replace only the network boundary; the MCP SDK runs normally.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import Mock

import anyio
import httpx
import pytest

from researchclaw.config import RCConfig
from researchclaw.web.agent import WebSearchAgent
from researchclaw.web.parallel_search import MCP_URL, USER_AGENT, ParallelSearchClient
from researchclaw.web.search import WebSearchClient, WebSearchResponse


@pytest.fixture(autouse=True)
def no_search_credentials(monkeypatch):
    for key in ("TAVILY_API_KEY", "EXA_API_KEY", "PARALLEL_API_KEY"):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def mcp_http(monkeypatch):
    requests = []
    payload = {"results": [
        {"title": "Source paper", "url": "https://example.org/paper",
         "excerpts": ["A useful research finding.", "Supporting evidence."]},
        {"title": None, "url": "https://example.org/other", "excerpts": []},
        {"title": "Unsafe", "url": "file:///etc/passwd", "excerpts": []},
    ]}
    state = {"payload": payload, "error": False, "mode": "structured", "closed": False}

    async def handle(request):
        requests.append(request)
        body = json.loads(request.content)
        if body["method"].startswith("notifications/"):
            return httpx.Response(202)
        if body["method"] == "initialize":
            result = {"protocolVersion": "2025-11-25", "capabilities": {},
                      "serverInfo": {"name": "fixture", "version": "1"}}
        elif body["method"] == "tools/list":
            result = {"tools": [{"name": "web_search", "inputSchema": {"type": "object"}}]}
        else:
            assert body["method"] == "tools/call"
            assert body["params"]["name"] == "web_search"
            if state["mode"] == "wait":
                try:
                    await anyio.sleep_forever()
                finally:
                    state["closed"] = True
            if state["mode"] == "http_error":
                return httpx.Response(503, text="unavailable")
            result = {"isError": state["error"], "content": [
                {"type": "text", "text": json.dumps(state["payload"])}]}
            if state["mode"] != "text":
                result["structuredContent"] = state["payload"]
        envelope = {"jsonrpc": "2.0", "id": body["id"], "result": result}
        if state["mode"] == "sse":
            return httpx.Response(200, headers={"content-type": "text/event-stream"},
                                  text="event: message\ndata: " + json.dumps(envelope) + "\n\n")
        return httpx.Response(200, json=envelope)

    real_client = httpx.AsyncClient

    class Client(real_client):
        def __init__(self, **kwargs):
            super().__init__(transport=httpx.MockTransport(handle), **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", Client)
    return state, requests


def agent():
    return WebSearchAgent(search_provider="parallel", enable_scholar=False,
                          enable_crawling=False, enable_pdf=False)


def test_dispatch_context_urls_headers_and_session(mcp_http):
    _, requests = mcp_http
    selected = agent()
    result = selected.search_and_extract("research", search_queries=["first query", "second query"])
    assert [r.url for r in result.web_results] == [
        "https://example.org/paper", "https://example.org/other"]
    assert all(r.source == "parallel" for r in result.web_results)
    context = result.to_context_string()
    assert "Source paper" in context and "A useful research finding." in context
    assert "https://example.org/other" in context
    calls = [json.loads(r.content) for r in requests]
    assert {c["method"] for c in calls} == {"initialize", "notifications/initialized", "tools/list", "tools/call"}
    assert all(str(r.url) == MCP_URL and r.headers["user-agent"] == USER_AGENT for r in requests)
    assert all("authorization" not in r.headers for r in requests)
    arguments = [c["params"]["arguments"] for c in calls if c["method"] == "tools/call"]
    assert arguments[0]["search_queries"] == ["first query"]
    assert arguments[0]["session_id"] == arguments[1]["session_id"]
    assert "model_name" not in arguments[0]


def test_streamable_http_sse_response(mcp_http):
    state, _ = mcp_http
    state["mode"] = "sse"
    assert WebSearchClient(provider="parallel").search("query").results[0].title == "Source paper"


def test_text_only_result_limit_and_empty(mcp_http):
    state, _ = mcp_http
    state["mode"] = "text"
    selected = WebSearchClient(provider="parallel")
    result = selected.search("query", max_results=1)
    assert len(result.results) == 1
    assert result.results[0].content == "A useful research finding.\n\nSupporting evidence."
    state["payload"] = {"results": []}
    assert not selected.search("empty").has_results


@pytest.mark.parametrize("payload,error", [({"bad": []}, False), ({"message": "quota exceeded"}, True)])
def test_protocol_errors_never_switch_provider(monkeypatch, mcp_http, payload, error):
    state, _ = mcp_http
    state.update(payload=payload, error=error)
    fallback = Mock(side_effect=AssertionError("unexpected provider fallback"))
    monkeypatch.setattr(WebSearchClient, "_search_duckduckgo", fallback)
    monkeypatch.setattr(WebSearchClient, "_search_tavily", fallback)
    with pytest.raises((ValueError, RuntimeError)):
        WebSearchClient(provider="parallel", api_key="saved-tavily").search("query")
    assert not fallback.called
    # The normal research agent handles provider errors without making up results.
    assert agent().search_and_extract("query", search_queries=["query"]).web_results == []


def test_http_error_does_not_fallback(mcp_http):
    state, _ = mcp_http
    state["mode"] = "http_error"
    with pytest.raises(Exception):
        WebSearchClient(provider="parallel").search("query")


def test_timeout_cancels_request(mcp_http):
    state, _ = mcp_http
    state["mode"] = "wait"
    with pytest.raises(TimeoutError):
        ParallelSearchClient(timeout=0.05).search("query")
    assert state["closed"]


def test_async_cancellation_closes_request(mcp_http):
    state, requests = mcp_http
    state["mode"] = "wait"

    async def run():
        task = asyncio.create_task(ParallelSearchClient()._search("query", 10, 0))
        with anyio.fail_after(2):
            while not any(json.loads(r.content)["method"] == "tools/call" for r in requests):
                if task.done():
                    await task
                await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    asyncio.run(run())
    assert state["closed"]


def test_sync_search_inside_running_loop(mcp_http):
    async def run():
        assert WebSearchClient(provider="parallel").search("query").has_results
    asyncio.run(run())


def test_default_and_saved_credentials_preserved(monkeypatch):
    parallel = Mock(side_effect=AssertionError("Parallel must be opt-in"))
    monkeypatch.setattr(ParallelSearchClient, "search", parallel)
    ddg = Mock(return_value=WebSearchResponse(query="query", source="duckduckgo"))
    tavily = Mock(return_value=WebSearchResponse(query="query", source="tavily"))
    monkeypatch.setattr(WebSearchClient, "_search_duckduckgo", ddg)
    monkeypatch.setattr(WebSearchClient, "_search_tavily", tavily)
    assert WebSearchClient().search("query").source == "duckduckgo"
    assert WebSearchClient(api_key="saved-key").search("query").source == "tavily"
    monkeypatch.setenv("TAVILY_API_KEY", "environment-key")
    assert WebSearchClient().search("query").source == "tavily"
    assert not parallel.called


def test_filters_unknown_provider_and_missing_extra(monkeypatch):
    with pytest.raises(ValueError, match="Unknown web search provider"):
        WebSearchClient(provider="typo")
    with pytest.raises(ValueError, match="domain filters"):
        WebSearchClient(provider="parallel").search("query", include_domains=["example.org"])
    import builtins
    real_import = builtins.__import__

    def without_mcp(name, *args, **kwargs):
        if name == "mcp":
            raise ImportError("no mcp")
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", without_mcp)
    with pytest.raises(ImportError, match=r"\.\[parallel\]"):
        WebSearchClient(provider="parallel").search("query")


def config(tmp_path, *, enabled=True, provider="parallel"):
    return RCConfig.from_dict({
        "project": {"name": "test"}, "research": {"topic": "research"},
        "runtime": {"timezone": "UTC"}, "notifications": {"channel": "console"},
        "knowledge_base": {"root": str(tmp_path / "kb")},
        "llm": {"api_key_env": "UNSET_TEST_KEY"},
        "web_search": {"enabled": enabled, "provider": provider, "enable_scholar": False,
                       "enable_crawling": False, "enable_pdf_extraction": False},
    }, project_root=tmp_path, check_paths=False)


@pytest.mark.parametrize("enabled", [True, False])
def test_research_stage_config_to_model_context(tmp_path, monkeypatch, mcp_http, enabled):
    from researchclaw.adapters import AdapterBundle
    from researchclaw.pipeline import executor
    from researchclaw.llm.client import LLMResponse

    _, requests = mcp_http
    cfg = config(tmp_path, enabled=enabled)
    assert cfg.web_search.provider == "parallel"
    run = tmp_path / "run"
    collect = run / "stage-04"
    collect.mkdir(parents=True)
    monkeypatch.setattr(executor, "_check_literature_sites_reachable", lambda **kw: {"arxiv": True})
    monkeypatch.setattr("researchclaw.literature.search.search_papers_multi_query", lambda *a, **kw: [])
    monkeypatch.setattr("researchclaw.literature.arxiv_client.search_arxiv", lambda *a, **kw: [])
    monkeypatch.setattr("researchclaw.data.load_seminal_papers", lambda *a: [])
    executor._execute_literature_collect(collect, run, cfg, AdapterBundle())
    if not enabled:
        assert not requests
        assert not (collect / "web_context.md").exists()
        return
    assert "A useful research finding." in (collect / "web_context.md").read_text()
    extract = run / "stage-06"
    extract.mkdir()
    llm = Mock()
    llm.chat.return_value = LLMResponse(content='{"cards": []}', model="scripted-test")
    executor._execute_knowledge_extract(extract, run, cfg, AdapterBundle(), llm=llm)
    assert "A useful research finding." in str(llm.chat.call_args)
    assert "https://example.org/paper" in str(llm.chat.call_args)


def test_saved_provider_round_trip_and_disabled_config(tmp_path):
    cfg = config(tmp_path, enabled=False)
    restored = RCConfig.from_dict(json.loads(json.dumps(cfg.to_dict())), project_root=tmp_path, check_paths=False)
    assert restored.web_search.provider == "parallel"
    assert restored.web_search.enabled is False
    assert config(tmp_path, provider="auto").web_search.provider == "auto"
