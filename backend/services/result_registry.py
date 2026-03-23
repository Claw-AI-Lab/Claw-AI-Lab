"""
Shared baseline experiment result registry.

Stores and queries baseline metrics across projects so that
identical baselines (e.g. ResNet-50 on CIFAR-100) are not re-run.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ResultEntry:
    id: str
    project_id: str
    description: str
    model: str
    dataset: str
    task: str
    metrics: dict[str, float]
    tags: list[str]
    source_stage: int
    timestamp: int

    def to_dict(self) -> dict:
        return self.__dict__

    @staticmethod
    def from_dict(d: dict) -> "ResultEntry":
        return ResultEntry(
            id=d["id"],
            project_id=d.get("project_id", ""),
            description=d.get("description", ""),
            model=d.get("model", ""),
            dataset=d.get("dataset", ""),
            task=d.get("task", ""),
            metrics=d.get("metrics", {}),
            tags=d.get("tags", []),
            source_stage=d.get("source_stage", 0),
            timestamp=d.get("timestamp", 0),
        )


class ResultRegistry:
    """File-backed registry for shared baseline experiment results."""

    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir)
        self.index_path = self.base_dir / "index.json"
        self.entries_dir = self.base_dir / "entries"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.entries_dir.mkdir(exist_ok=True)
        self._entries: list[ResultEntry] = []
        self._load()

    def _load(self):
        if self.index_path.exists():
            try:
                data = json.loads(self.index_path.read_text(encoding="utf-8"))
                self._entries = [ResultEntry.from_dict(d) for d in data]
            except (json.JSONDecodeError, KeyError):
                self._entries = []

    def _save(self):
        self.index_path.write_text(
            json.dumps([e.to_dict() for e in self._entries], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def register(
        self,
        project_id: str,
        description: str,
        model: str = "",
        dataset: str = "",
        task: str = "",
        metrics: dict[str, float] | None = None,
        tags: list[str] | None = None,
        source_stage: int = 16,
    ) -> ResultEntry:
        entry = ResultEntry(
            id=f"res-{uuid.uuid4().hex[:8]}",
            project_id=project_id,
            description=description,
            model=model,
            dataset=dataset,
            task=task,
            metrics=metrics or {},
            tags=tags or [],
            source_stage=source_stage,
            timestamp=int(time.time() * 1000),
        )
        self._entries.append(entry)
        self._save()

        detail_path = self.entries_dir / f"{entry.project_id}_{entry.id}.json"
        detail_path.write_text(
            json.dumps(entry.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return entry

    def all_entries(self) -> list[ResultEntry]:
        return list(self._entries)

    def count(self) -> int:
        return len(self._entries)

    def query_by_llm(self, llm_chat_fn, exp_plan: str, max_entries: int = 30) -> list[dict]:
        """Use LLM to semantically match cached results against a new experiment plan.

        Args:
            llm_chat_fn: callable(messages, system, json_mode) -> str
            exp_plan: the new project's experiment plan (YAML text)
            max_entries: max cached entries to include in prompt

        Returns:
            list of {"cached_id": str, "reason": str} for reusable baselines
        """
        if not self._entries:
            return []

        entries_text = ""
        for e in self._entries[:max_entries]:
            entries_text += (
                f"- ID: {e.id}\n"
                f"  Project: {e.project_id}\n"
                f"  Description: {e.description}\n"
                f"  Model: {e.model}, Dataset: {e.dataset}, Task: {e.task}\n"
                f"  Metrics: {json.dumps(e.metrics)}\n"
                f"  Tags: {', '.join(e.tags)}\n\n"
            )

        prompt = (
            "You have the following cached experiment results from previous projects:\n\n"
            f"{entries_text}\n"
            "A new project needs to run these experiments:\n\n"
            f"{exp_plan[:4000]}\n\n"
            "Which cached results can be directly reused as baselines for the new project?\n"
            "Only match if the model architecture, dataset, and evaluation protocol are substantially identical.\n"
            "Return a JSON array of objects: [{\"cached_id\": \"...\", \"reason\": \"...\"}]\n"
            "Return [] if no match."
        )

        try:
            response = llm_chat_fn(
                [{"role": "user", "content": prompt}],
                system="You are a research experiment analyst. Return valid JSON only.",
                json_mode=True,
            )
            result = json.loads(response)
            if isinstance(result, list):
                return result
            return []
        except (json.JSONDecodeError, Exception):
            return []

    def get_entry(self, entry_id: str) -> ResultEntry | None:
        for e in self._entries:
            if e.id == entry_id:
                return e
        return None

    def summary(self) -> dict:
        return {
            "total_entries": len(self._entries),
            "projects": list(set(e.project_id for e in self._entries)),
            "models": list(set(e.model for e in self._entries if e.model)),
            "datasets": list(set(e.dataset for e in self._entries if e.dataset)),
        }
