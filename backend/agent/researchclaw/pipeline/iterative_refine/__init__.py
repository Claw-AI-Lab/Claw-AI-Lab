"""S15 ITERATIVE_REFINE — claw-code agentic experiment optimization.

The agent analyzes experiment results, identifies improvement
opportunities (hyperparameter tuning, architectural changes, bug fixes),
modifies the code, and re-runs the experiment — iterating until
metrics improve or the iteration budget is exhausted.

Public API: ``execute_iterative_refine()`` is the single entry point.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from researchclaw.config import RCConfig
from researchclaw.adapters import AdapterBundle
from researchclaw.pipeline.executor import StageResult


def execute_iterative_refine(
    stage_dir: Path,
    run_dir: Path,
    config: RCConfig,
    adapters: AdapterBundle,
    *,
    llm: Any | None = None,
    prompts: Any | None = None,
) -> StageResult:
    """Execute S15 ITERATIVE_REFINE via the claw-code agentic turn loop."""
    from researchclaw.pipeline.iterative_refine.runtime import IterativeRefineRuntime
    runtime = IterativeRefineRuntime()
    return runtime.execute(
        stage_dir=stage_dir,
        run_dir=run_dir,
        config=config,
        adapters=adapters,
        llm=llm,
    )
