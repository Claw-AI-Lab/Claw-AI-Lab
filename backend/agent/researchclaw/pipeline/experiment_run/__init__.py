"""S14 EXPERIMENT_RUN — claw-code agentic experiment execution.

The agent runs the full experiment, monitors for errors (OOM, NaN,
timeout), and can apply runtime fixes (reduce batch size, adjust lr,
handle missing data) without compromising scientific validity.

Public API: ``execute_experiment_run()`` is the single entry point.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from researchclaw.config import RCConfig
from researchclaw.adapters import AdapterBundle
from researchclaw.pipeline.executor import StageResult


def execute_experiment_run(
    stage_dir: Path,
    run_dir: Path,
    config: RCConfig,
    adapters: AdapterBundle,
    *,
    llm: Any | None = None,
    prompts: Any | None = None,
) -> StageResult:
    """Execute S14 EXPERIMENT_RUN via the claw-code agentic turn loop."""
    from researchclaw.pipeline.experiment_run.runtime import ExperimentRunRuntime
    runtime = ExperimentRunRuntime()
    return runtime.execute(
        stage_dir=stage_dir,
        run_dir=run_dir,
        config=config,
        adapters=adapters,
        llm=llm,
    )
