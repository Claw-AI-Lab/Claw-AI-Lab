"""Validation pipeline: chains gates, auto-repair, and LLM repair.

Inspired by claw-code's turn loop where each tool execution is followed
by permission checks before proceeding. Here each code generation result
passes through AST gates, programmatic fixes, and LLM-driven repair.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from researchclaw.pipeline.codegen.session import CodegenSession
from researchclaw.pipeline.codegen.types import (
    CodegenContext,
    CodegenPhase,
    GeneratedFiles,
)
from researchclaw.pipeline.codegen.validation.gates import (
    auto_fix_unbound,
    classify_critical_issues,
    run_hard_validation,
)
from researchclaw.pipeline.codegen.validation.repair import repair_critical_issues

logger = logging.getLogger(__name__)


class ValidationPipeline:
    """Run the full validation + repair chain on generated code.

    Steps:
    1. Programmatic auto-fix (UnboundLocalError patterns)
    2. Hard validation (AST gates, complexity, API correctness)
    3. LLM repair for critical issues
    4. Write complexity report
    """

    def __init__(
        self,
        llm: Any | None,
        prompts: Any | None,
        stage_dir: Path | None,
    ) -> None:
        self._llm = llm
        self._prompts = prompts
        self._stage_dir = stage_dir

    def run(
        self,
        files: GeneratedFiles,
        ctx: CodegenContext,
        session: CodegenSession,
    ) -> GeneratedFiles:
        session.log(CodegenPhase.VALIDATE, "Validation pipeline started")

        # Step 1: Programmatic auto-fix
        files, n_fixes = auto_fix_unbound(files)
        if n_fixes:
            session.log(CodegenPhase.VALIDATE, f"Auto-fixed {n_fixes} UnboundLocalError risks")

        # Step 2: Hard validation
        complexity_warnings, deep_warnings = run_hard_validation(files)
        if complexity_warnings:
            session.log(
                CodegenPhase.VALIDATE,
                f"{len(complexity_warnings)} complexity/quality warnings",
            )

        # Step 3: LLM repair for critical issues
        critical = classify_critical_issues(deep_warnings)
        if critical and self._llm is not None:
            session.log(
                CodegenPhase.VALIDATE,
                f"{len(critical)} critical issues — triggering LLM repair",
            )
            files = repair_critical_issues(
                files, critical, self._llm, prompts=self._prompts,
            )
            session.llm_calls += 1

        # Step 4: Write complexity report
        if complexity_warnings and self._stage_dir is not None:
            (self._stage_dir / "code_complexity.json").write_text(
                json.dumps(
                    {"code_complexity_warnings": complexity_warnings},
                    indent=2,
                ),
                encoding="utf-8",
            )
            session.add_artifact("code_complexity.json")

        # Write updated files to experiment directory
        exp_dir = self._stage_dir / "experiment" if self._stage_dir else None
        if exp_dir is not None:
            exp_dir.mkdir(parents=True, exist_ok=True)
            for fname, code in files.items():
                (exp_dir / fname).write_text(code, encoding="utf-8")

        session.log(CodegenPhase.VALIDATE, "Validation pipeline completed")
        return files
