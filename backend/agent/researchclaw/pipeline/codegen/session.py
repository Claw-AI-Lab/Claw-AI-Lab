"""Session state tracker for code generation.

Inspired by claw-code's Session model which tracks messages, version,
and TokenUsage across a conversation. CodegenSession tracks files,
phase log entries, artifacts, and cumulative LLM/sandbox usage across
the entire code generation turn loop.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from researchclaw.pipeline.codegen.types import CodegenPhase, GeneratedFiles

logger = logging.getLogger(__name__)


@dataclass
class CodegenSession:
    """Mutable state accumulated across codegen phases.

    Analogous to claw-code's ``Session { messages, version }`` which
    persists conversation state, plus ``TurnSummary`` for per-turn stats.
    """

    stage_dir: Path
    files: GeneratedFiles = field(default_factory=dict)
    phase_log: list[str] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)
    llm_calls: int = 0
    sandbox_runs: int = 0
    strategy_used: str = ""
    best_score: float = 0.0
    tree_nodes_explored: int = 0
    review_rounds: int = 0
    _start_time: float = field(default_factory=time.monotonic)

    def log(self, phase: CodegenPhase | str, message: str) -> None:
        """Append a timestamped log entry."""
        elapsed = time.monotonic() - self._start_time
        phase_name = phase.name if isinstance(phase, CodegenPhase) else phase
        entry = f"[{elapsed:7.1f}s] [{phase_name}] {message}"
        self.phase_log.append(entry)
        logger.info("[CodegenSession] %s", entry)

    def add_artifact(self, name: str) -> None:
        if name not in self.artifacts:
            self.artifacts.append(name)

    def elapsed_sec(self) -> float:
        return time.monotonic() - self._start_time

    def persist(self) -> Path:
        """Write session log to stage directory for diagnostics."""
        log_path = self.stage_dir / "codegen_session.json"
        payload: dict[str, Any] = {
            "strategy_used": self.strategy_used,
            "llm_calls": self.llm_calls,
            "sandbox_runs": self.sandbox_runs,
            "best_score": self.best_score,
            "tree_nodes_explored": self.tree_nodes_explored,
            "review_rounds": self.review_rounds,
            "elapsed_sec": round(self.elapsed_sec(), 1),
            "files_generated": sorted(self.files.keys()),
            "artifacts": self.artifacts,
            "phase_log": self.phase_log,
        }
        log_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return log_path
