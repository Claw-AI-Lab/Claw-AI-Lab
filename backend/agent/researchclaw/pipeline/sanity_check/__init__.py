"""S12 SANITY_CHECK — claw-code agentic smoke test + auto-fix.

The agent runs the experiment code in SMOKE_TEST mode, diagnoses
any errors (import failures, shape mismatches, OOM, missing files),
and iteratively fixes them using the same tool loop as S11.

Public API: ``execute_sanity_check()`` is the single entry point.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from researchclaw.config import RCConfig
from researchclaw.adapters import AdapterBundle
from researchclaw.pipeline.executor import StageResult


def execute_sanity_check(
    stage_dir: Path,
    run_dir: Path,
    config: RCConfig,
    adapters: AdapterBundle,
    *,
    llm: Any | None = None,
    prompts: Any | None = None,
) -> StageResult:
    """Execute S12 SANITY_CHECK via the claw-code agentic turn loop."""
    from researchclaw.pipeline.sanity_check.runtime import SanityCheckRuntime
    runtime = SanityCheckRuntime()
    return runtime.execute(
        stage_dir=stage_dir,
        run_dir=run_dir,
        config=config,
        adapters=adapters,
        llm=llm,
    )
