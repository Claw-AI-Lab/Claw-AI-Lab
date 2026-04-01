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

    Key debug feature: every ``log()`` call auto-persists both a JSON
    snapshot and a streaming text log for live ``tail -f`` debugging.
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
    current_phase: str = ""
    errors: list[str] = field(default_factory=list)
    _start_time: float = field(default_factory=time.monotonic)
    _auto_persist: bool = True

    def log(self, phase: CodegenPhase | str, message: str) -> None:
        """Append a timestamped log entry and auto-persist for live debugging."""
        elapsed = time.monotonic() - self._start_time
        phase_name = phase.name if isinstance(phase, CodegenPhase) else phase
        self.current_phase = phase_name
        entry = f"[{elapsed:7.1f}s] [{phase_name}] {message}"
        self.phase_log.append(entry)
        logger.info("[CodegenSession] %s", entry)
        # Stream to text log for `tail -f` debugging
        self._append_to_live_log(entry)
        if self._auto_persist:
            self._try_persist()

    def log_error(self, phase: CodegenPhase | str, message: str, exc: Exception | None = None) -> None:
        """Log an error with optional exception details."""
        error_msg = message
        if exc is not None:
            error_msg = f"{message}: {type(exc).__name__}: {exc}"
        self.errors.append(error_msg)
        self.log(phase, f"ERROR: {error_msg}")

    def log_data(self, phase: CodegenPhase | str, label: str, data: Any) -> None:
        """Log structured data (dict/list) as a separate JSON artifact."""
        self.log(phase, f"{label}: see {label}.json")
        try:
            path = self.stage_dir / f"debug_{label}.json"
            payload = data if isinstance(data, (dict, list)) else {"value": str(data)}
            path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        except Exception:
            pass

    def add_artifact(self, name: str) -> None:
        if name not in self.artifacts:
            self.artifacts.append(name)

    def elapsed_sec(self) -> float:
        return time.monotonic() - self._start_time

    def persist(self) -> Path:
        """Write full session snapshot to stage directory."""
        log_path = self.stage_dir / "codegen_session.json"
        payload: dict[str, Any] = {
            "strategy_used": self.strategy_used,
            "current_phase": self.current_phase,
            "llm_calls": self.llm_calls,
            "sandbox_runs": self.sandbox_runs,
            "best_score": self.best_score,
            "tree_nodes_explored": self.tree_nodes_explored,
            "review_rounds": self.review_rounds,
            "elapsed_sec": round(self.elapsed_sec(), 1),
            "files_generated": sorted(self.files.keys()),
            "artifacts": self.artifacts,
            "errors": self.errors,
            "phase_log": self.phase_log,
        }
        log_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return log_path

    def _try_persist(self) -> None:
        """Best-effort persist — never raises."""
        try:
            self.persist()
        except Exception:
            pass

    def _append_to_live_log(self, entry: str) -> None:
        """Append one line to the streaming text log (tail -f friendly)."""
        try:
            log_path = self.stage_dir / "codegen_live.log"
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(entry + "\n")
        except Exception:
            pass
