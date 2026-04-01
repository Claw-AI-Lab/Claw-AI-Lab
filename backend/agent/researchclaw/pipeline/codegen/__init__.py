"""S11 CODE_GENERATION — refactored codegen package.

Public API: ``execute_code_generation()`` is the single entry point,
called by the thin wrapper in ``executor.py``.

Architecture inspired by claw-code's harness engineering patterns:
- StrategyRegistry (like ExecutionRegistry)
- CodegenRouter (like PortRuntime.route_prompt)
- CodegenRuntime (like ConversationRuntime.run_turn)
- CodegenPromptBuilder (like SystemPromptBuilder)
- CodegenSession (like Session)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from researchclaw.adapters import AdapterBundle
from researchclaw.config import RCConfig
from researchclaw.pipeline.codegen.runtime import CodegenRuntime
from researchclaw.pipeline.executor import StageResult


def execute_code_generation(
    stage_dir: Path,
    run_dir: Path,
    config: RCConfig,
    adapters: AdapterBundle,
    *,
    llm: Any | None = None,
    prompts: Any | None = None,
) -> StageResult:
    """Execute S11 CODE_GENERATION via the refactored codegen runtime.

    Drop-in replacement for the old monolithic ``_execute_code_generation()``
    in executor.py. Same signature, same return type, same side effects.
    """
    runtime = CodegenRuntime()
    return runtime.execute(
        stage_dir=stage_dir,
        run_dir=run_dir,
        config=config,
        adapters=adapters,
        llm=llm,
        prompts=prompts,
    )
