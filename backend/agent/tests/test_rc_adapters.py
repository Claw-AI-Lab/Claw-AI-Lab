from __future__ import annotations

from researchclaw.adapters import (
    AdapterBundle,
    BrowserPage,
    FetchResponse,
    RecordingBrowserAdapter,
    RecordingCronAdapter,
    RecordingMemoryAdapter,
    RecordingMessageAdapter,
    RecordingSessionsAdapter,
    RecordingWebFetchAdapter,
)


def test_adapter_bundle_defaults_are_recording_types():
    bundle = AdapterBundle()
    assert isinstance(bundle.cron, RecordingCronAdapter)
    assert isinstance(bundle.message, RecordingMessageAdapter)
    assert isinstance(bundle.memory, RecordingMemoryAdapter)
    assert isinstance(bundle.sessions, RecordingSessionsAdapter)
    assert isinstance(bundle.web_fetch, RecordingWebFetchAdapter)
    assert isinstance(bundle.browser, RecordingBrowserAdapter)


def test_recording_cron_adapter_records_call_and_returns_id():
    adapter = RecordingCronAdapter()
    result = adapter.schedule_resume("run-1", 7, "gate opened")
    assert result == "cron-1"
    assert adapter.calls == [("run-1", 7, "gate opened")]


def test_recording_message_adapter_notify_records_call():
    adapter = RecordingMessageAdapter()
    result = adapter.notify("ops", "stage update", "stage 3 done")
    assert result == "message-1"
    assert adapter.calls == [("ops", "stage update", "stage 3 done")]


def test_recording_memory_adapter_append_records_entries():
    adapter = RecordingMemoryAdapter()
    result = adapter.append("runs", "run-1 started")
    assert result == "memory-1"
    assert adapter.entries == [("runs", "run-1 started")]


def test_recording_sessions_adapter_spawn_records_calls():
    adapter = RecordingSessionsAdapter()
    result = adapter.spawn("worker", ("python", "train.py"))
    assert result == "session-1"
    assert adapter.calls == [("worker", ("python", "train.py"))]


def test_recording_webfetch_fetch_returns_success_response():
    adapter = RecordingWebFetchAdapter()
    response = adapter.fetch("https://example.com")
    assert isinstance(response, FetchResponse)
    assert response.url == "https://example.com"
    assert response.status_code == 200
    assert "stub fetch" in response.text


def test_recording_browser_open_returns_browser_page():
    adapter = RecordingBrowserAdapter()
    page = adapter.open("https://example.com")
    assert isinstance(page, BrowserPage)
    assert page.url == "https://example.com"
    assert "Stub browser page" in page.title


def test_fetch_response_dataclass_fields():
    response = FetchResponse(url="u", status_code=201, text="ok")
    assert response.url == "u"
    assert response.status_code == 201
    assert response.text == "ok"


def test_browser_page_dataclass_fields():
    page = BrowserPage(url="https://a", title="A")
    assert page.url == "https://a"
    assert page.title == "A"


def test_all_adapters_start_with_empty_call_lists():
    cron = RecordingCronAdapter()
    message = RecordingMessageAdapter()
    memory = RecordingMemoryAdapter()
    sessions = RecordingSessionsAdapter()
    web_fetch = RecordingWebFetchAdapter()
    browser = RecordingBrowserAdapter()
    assert cron.calls == []
    assert message.calls == []
    assert memory.entries == []
    assert sessions.calls == []
    assert web_fetch.calls == []
    assert browser.calls == []
