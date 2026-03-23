from __future__ import annotations

import urllib.error
from email.message import Message
from unittest.mock import patch

from researchclaw.llm.client import LLMClient, LLMConfig, LLMResponse


def _make_client(
    *,
    base_url: str = "https://api.example.com/v1",
    api_key: str = "test-key",
    primary_model: str = "gpt-test",
    fallback_models: list[str] | None = None,
    max_retries: int = 1,
) -> LLMClient:
    return LLMClient(
        LLMConfig(
            base_url=base_url,
            api_key=api_key,
            primary_model=primary_model,
            fallback_models=fallback_models or [],
            max_retries=max_retries,
        )
    )


class TestPreflight:
    def test_preflight_success(self):
        client = _make_client()
        mock_resp = LLMResponse(content="pong", model="gpt-test")
        with patch.object(client, "chat", return_value=mock_resp):
            ok, msg = client.preflight()
        assert ok is True
        assert "OK" in msg
        assert "gpt-test" in msg

    def test_preflight_401_invalid_key(self):
        client = _make_client()
        err = urllib.error.HTTPError("url", 401, "Unauthorized", Message(), None)
        with patch.object(client, "chat", side_effect=err):
            ok, msg = client.preflight()
        assert ok is False
        assert "Invalid API key" in msg

    def test_preflight_403_model_forbidden(self):
        client = _make_client()
        err = urllib.error.HTTPError("url", 403, "Forbidden", Message(), None)
        with patch.object(client, "chat", side_effect=err):
            ok, msg = client.preflight()
        assert ok is False
        assert "not allowed" in msg

    def test_preflight_404_bad_endpoint(self):
        client = _make_client()
        err = urllib.error.HTTPError("url", 404, "Not Found", Message(), None)
        with patch.object(client, "chat", side_effect=err):
            ok, msg = client.preflight()
        assert ok is False
        assert "Endpoint not found" in msg

    def test_preflight_429_rate_limited(self):
        client = _make_client()
        err = urllib.error.HTTPError("url", 429, "Too Many Requests", Message(), None)
        with patch.object(client, "chat", side_effect=err):
            ok, msg = client.preflight()
        assert ok is False
        assert "Rate limited" in msg

    def test_preflight_timeout(self):
        client = _make_client()
        err = urllib.error.URLError("timeout")
        with patch.object(client, "chat", side_effect=err):
            ok, msg = client.preflight()
        assert ok is False
        assert "Connection failed" in msg

    def test_preflight_all_models_failed(self):
        client = _make_client()
        err = RuntimeError("All models failed. Last error: ...")
        with patch.object(client, "chat", side_effect=err):
            ok, msg = client.preflight()
        assert ok is False
        assert "All models failed" in msg

    def test_preflight_unknown_http_error(self):
        client = _make_client()
        err = urllib.error.HTTPError("url", 500, "Server Error", Message(), None)
        with patch.object(client, "chat", side_effect=err):
            ok, msg = client.preflight()
        assert ok is False
        assert "HTTP 500" in msg
