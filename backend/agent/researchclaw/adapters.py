"""Typed adapter interfaces and deterministic recording stubs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class FetchResponse:
    url: str
    status_code: int
    text: str


@dataclass(frozen=True)
class BrowserPage:
    url: str
    title: str


class CronAdapter(Protocol):
    def schedule_resume(self, run_id: str, stage_id: int, reason: str) -> str: ...


class MessageAdapter(Protocol):
    def notify(self, channel: str, subject: str, body: str) -> str: ...


class MemoryAdapter(Protocol):
    def append(self, namespace: str, content: str) -> str: ...


class SessionsAdapter(Protocol):
    def spawn(self, name: str, command: tuple[str, ...]) -> str: ...


class WebFetchAdapter(Protocol):
    def fetch(self, url: str) -> FetchResponse: ...


class BrowserAdapter(Protocol):
    def open(self, url: str) -> BrowserPage: ...


@dataclass
class RecordingCronAdapter:
    calls: list[tuple[str, int, str]] = field(default_factory=list)

    def schedule_resume(self, run_id: str, stage_id: int, reason: str) -> str:
        self.calls.append((run_id, stage_id, reason))
        return f"cron-{len(self.calls)}"


@dataclass
class RecordingMessageAdapter:
    calls: list[tuple[str, str, str]] = field(default_factory=list)

    def notify(self, channel: str, subject: str, body: str) -> str:
        self.calls.append((channel, subject, body))
        return f"message-{len(self.calls)}"


@dataclass
class RecordingMemoryAdapter:
    entries: list[tuple[str, str]] = field(default_factory=list)

    def append(self, namespace: str, content: str) -> str:
        self.entries.append((namespace, content))
        return f"memory-{len(self.entries)}"


@dataclass
class RecordingSessionsAdapter:
    calls: list[tuple[str, tuple[str, ...]]] = field(default_factory=list)

    def spawn(self, name: str, command: tuple[str, ...]) -> str:
        self.calls.append((name, command))
        return f"session-{len(self.calls)}"


@dataclass
class RecordingWebFetchAdapter:
    calls: list[str] = field(default_factory=list)

    def fetch(self, url: str) -> FetchResponse:
        self.calls.append(url)
        return FetchResponse(url=url, status_code=200, text=f"stub fetch for {url}")


@dataclass
class RecordingBrowserAdapter:
    calls: list[str] = field(default_factory=list)

    def open(self, url: str) -> BrowserPage:
        self.calls.append(url)
        return BrowserPage(url=url, title=f"Stub browser page for {url}")


@dataclass
class AdapterBundle:
    cron: CronAdapter = field(default_factory=RecordingCronAdapter)
    message: MessageAdapter = field(default_factory=RecordingMessageAdapter)
    memory: MemoryAdapter = field(default_factory=RecordingMemoryAdapter)
    sessions: SessionsAdapter = field(default_factory=RecordingSessionsAdapter)
    web_fetch: WebFetchAdapter = field(default_factory=RecordingWebFetchAdapter)
    browser: BrowserAdapter = field(default_factory=RecordingBrowserAdapter)
