"""Core types for the codegen package.

Inspired by claw-code's typed data structures: Session (state model),
BootstrapPhase (phase enumeration), and ToolSpec (tool definitions).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any


class CodegenPhase(Enum):
    """Explicit phase enumeration for the code generation turn loop.

    Analogous to claw-code's BootstrapPhase enum which enumerates
    CLI entry, fast paths, and MainRuntime in a fixed sequence.
    """

    CONTEXT = auto()
    LLM_SETUP = auto()
    ROUTING = auto()
    GENERATE = auto()
    FALLBACK = auto()
    VALIDATE = auto()
    REVIEW = auto()
    FINALIZE = auto()


GeneratedFiles = dict[str, str]
"""Mapping of filename -> source code content."""


@dataclass
class HardwareProfile:
    """Parsed hardware profile from S1."""

    has_gpu: bool = False
    gpu_type: str = "cuda"
    gpu_name: str = ""
    tier: str = "limited"
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> HardwareProfile | None:
        if data is None:
            return None
        return cls(
            has_gpu=bool(data.get("has_gpu")),
            gpu_type=data.get("gpu_type", "cuda"),
            gpu_name=data.get("gpu_name", ""),
            tier=data.get("tier", "limited"),
            raw=data,
        )


@dataclass
class DiscoveredData:
    """Pre-discovered filesystem context, gathered before prompt building.

    Analogous to claw-code's ``ProjectContext`` fields ``git_status``
    and ``instruction_files`` — real data read from the filesystem and
    injected into the system prompt so the LLM has ground truth before
    writing any code.
    """

    checkpoint_model_index: dict[str, Any] = field(default_factory=dict)
    checkpoint_model_index_raw: str = ""
    checkpoint_class_name: str = ""
    checkpoint_files: list[str] = field(default_factory=list)
    dataset_files: list[str] = field(default_factory=list)
    dataset_sample: str = ""
    codebase_files: list[str] = field(default_factory=list)
    codebase_readme: str = ""


@dataclass
class CodegenContext:
    """All context needed for code generation, assembled once.

    Analogous to claw-code's ProjectContext which gathers cwd, git_status,
    and instruction_files before the SystemPromptBuilder consumes them.
    """

    topic: str
    exp_plan: str
    metric: str
    metric_direction: str
    time_budget_sec: int
    mode: str

    hw_profile: HardwareProfile | None = None
    codebase_info: str = "[]"
    datasets_dir: str = ""
    checkpoints_dir: str = ""
    codebases_dir: str = ""

    pkg_hint: str = ""
    compute_budget: str = ""
    extra_guidance: str = ""

    run_dir: Path | None = None
    stage_dir: Path | None = None

    discovered: DiscoveredData = field(default_factory=DiscoveredData)


@dataclass
class CodegenResult:
    """Result returned by a strategy's generate() call.

    Analogous to claw-code's TurnSummary returned by
    ConversationRuntime.run_turn().
    """

    files: GeneratedFiles = field(default_factory=dict)
    strategy_name: str = ""
    skip_review: bool = False
    elapsed_sec: float = 0.0
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
