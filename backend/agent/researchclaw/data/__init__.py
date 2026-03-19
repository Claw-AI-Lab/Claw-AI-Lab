"""Static data assets for the ResearchClaw pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Framework documentation
# ---------------------------------------------------------------------------

_FRAMEWORK_DOCS_DIR = Path(__file__).parent / "framework_docs"

# Map of framework identifier -> (doc filename, keyword patterns for detection)
_FRAMEWORK_REGISTRY: dict[str, dict[str, Any]] = {
    "trl": {
        "file": "trl.md",
        "keywords": ["trl", "sft", "dpo", "grpo", "ppo trainer", "rlhf",
                      "sfttrainer", "dpotrainer", "grpotrainer"],
    },
    "peft": {
        "file": "peft.md",
        "keywords": ["peft", "lora", "qlora", "adapter", "low-rank",
                      "parameter-efficient", "dora"],
    },
    "transformers_training": {
        "file": "transformers_training.md",
        "keywords": ["transformers", "huggingface", "trainer", "trainingarguments",
                      "automodel", "fine-tun"],
    },
    "llamafactory": {
        "file": "llamafactory.md",
        "keywords": ["llamafactory", "llama-factory", "llama factory"],
    },
    "axolotl": {
        "file": "axolotl.md",
        "keywords": ["axolotl"],
    },
}


def detect_frameworks(topic: str, hypothesis: str = "", plan: str = "") -> list[str]:
    """Detect which ML training frameworks are relevant based on topic/hypothesis/plan.

    Returns a list of framework identifiers (e.g., ["trl", "peft"]).
    """
    combined = (topic + " " + hypothesis + " " + plan).lower()
    matched: list[str] = []
    for fw_id, info in _FRAMEWORK_REGISTRY.items():
        for kw in info["keywords"]:
            if kw in combined:
                matched.append(fw_id)
                break
    return matched


def load_framework_docs(framework_ids: list[str], max_chars: int = 8000) -> str:
    """Load and concatenate framework API documentation for the given IDs.

    Returns a single string with all relevant docs, truncated to max_chars
    to avoid overwhelming the prompt context.
    """
    parts: list[str] = []
    total = 0
    for fw_id in framework_ids:
        info = _FRAMEWORK_REGISTRY.get(fw_id)
        if not info:
            continue
        doc_path = _FRAMEWORK_DOCS_DIR / info["file"]
        if not doc_path.exists():
            logger.warning("Framework doc not found: %s", doc_path)
            continue
        content = doc_path.read_text(encoding="utf-8")
        if total + len(content) > max_chars:
            remaining = max_chars - total
            if remaining > 500:
                content = content[:remaining] + "\n... (truncated)\n"
            else:
                break
        parts.append(content)
        total += len(content)

    if not parts:
        return ""

    header = (
        "\n## Framework API Documentation (auto-detected)\n"
        "The following API references are relevant to your experiment. "
        "Use these exact APIs and patterns — do NOT guess the API.\n\n"
    )
    return header + "\n---\n\n".join(parts)

_SEMINAL_PAPERS_PATH = Path(__file__).parent / "seminal_papers.yaml"
_CACHE: list[dict[str, Any]] | None = None


def _load_all() -> list[dict[str, Any]]:
    """Load and cache the seminal papers list."""
    global _CACHE  # noqa: PLW0603
    if _CACHE is not None:
        return _CACHE
    try:
        data = yaml.safe_load(_SEMINAL_PAPERS_PATH.read_text(encoding="utf-8"))
        _CACHE = data.get("papers", []) if isinstance(data, dict) else []
    except Exception:  # noqa: BLE001
        logger.warning("Failed to load seminal_papers.yaml", exc_info=True)
        _CACHE = []
    return _CACHE


def load_seminal_papers(topic: str) -> list[dict[str, Any]]:
    """Return seminal papers whose keywords overlap with *topic*.

    Each returned dict has: title, authors, year, venue, cite_key, keywords.
    Matching is case-insensitive substring on the topic string.
    """
    all_papers = _load_all()
    topic_lower = topic.lower()
    matched: list[dict[str, Any]] = []
    seen_keys: set[str] = set()

    for paper in all_papers:
        keywords = paper.get("keywords", [])
        if not isinstance(keywords, list):
            continue
        for kw in keywords:
            if isinstance(kw, str) and kw.lower() in topic_lower:
                ck = paper.get("cite_key", "")
                if ck not in seen_keys:
                    seen_keys.add(ck)
                    matched.append(paper)
                break

    logger.debug(
        "load_seminal_papers(%r): matched %d papers", topic, len(matched)
    )
    return matched
