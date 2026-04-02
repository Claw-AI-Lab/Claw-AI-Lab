"""S12 SANITY_CHECK runtime — orchestrates the agentic smoke test loop.

Follows the same pattern as S11's ClawAgentStrategy:
workspace preparation → system prompt → turn loop → result.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any

from researchclaw.config import RCConfig
from researchclaw.adapters import AdapterBundle
from researchclaw.pipeline.claw_engine import AgentTurnLoop, StageSession
from researchclaw.pipeline.sanity_check.system_prompt import (
    build_system_prompt,
    build_user_message,
)
from researchclaw.pipeline.stages import Stage, StageStatus

logger = logging.getLogger(__name__)


class SanityCheckRuntime:
    """Orchestration for S12 SANITY_CHECK using claw-code turn loop."""

    def execute(
        self,
        stage_dir: Path,
        run_dir: Path,
        config: RCConfig,
        adapters: AdapterBundle,
        *,
        llm: Any | None = None,
    ) -> Any:
        from researchclaw.pipeline.executor import StageResult

        stage_dir.mkdir(parents=True, exist_ok=True)
        session = StageSession(stage_dir=stage_dir, stage_name="sanity_check")
        session.log("INIT", "SanityCheckRuntime started")

        # Resolve coding LLM
        llm = self._resolve_coding_llm(llm, config)
        if llm is None:
            session.log_error("INIT", "No LLM client available")
            return StageResult(
                stage=Stage.SANITY_CHECK,
                status=StageStatus.FAILED,
                artifacts=(),
                error="No LLM client available",
            )

        llm_config = llm.config
        session.log("INIT", f"LLM: {llm_config.primary_model}")

        # Find experiment directory from prior stages
        experiment_dir = self._find_experiment_dir(run_dir)
        if not experiment_dir:
            session.log_error("INIT", "No experiment directory found")
            return StageResult(
                stage=Stage.SANITY_CHECK,
                status=StageStatus.FAILED,
                artifacts=(),
                error="No experiment/ directory found in prior stages",
            )
        session.log("INIT", f"Experiment dir: {experiment_dir}")

        # Prepare workspace (copy experiment into a fresh workspace)
        workspace = self._prepare_workspace(stage_dir, experiment_dir, run_dir, config)
        session.log("INIT", f"Workspace: {workspace}")

        # List experiment files
        exp_files = [
            str(f.relative_to(workspace))
            for f in sorted(workspace.rglob("*.py"))
            if not f.is_symlink()
            and not any(p.startswith(".") or p == "__pycache__" for p in f.relative_to(workspace).parts)
        ]
        session.log("INIT", f"Experiment files: {exp_files}")

        # Build prompts
        python_path = getattr(config.experiment.sandbox, "python_path", "") or ""
        system_prompt = build_system_prompt(
            python_path=python_path,
            workspace_path=str(workspace),
        )

        exp_plan_summary = self._load_plan_summary(run_dir)
        user_message = build_user_message(
            experiment_dir=str(workspace),
            experiment_files=exp_files,
            exp_plan_summary=exp_plan_summary,
        )

        # Save system prompt for debugging
        (stage_dir / "sanity_system_prompt.md").write_text(system_prompt, encoding="utf-8")
        session.add_artifact("sanity_system_prompt.md")

        # Build allowed read dirs
        allowed_reads = self._build_allowed_reads(config)

        max_iters = getattr(config.experiment, "sanity_check_max_iterations", 18)
        session.log("INIT", f"Max iterations: {max_iters}")

        # Create and run turn loop
        loop = AgentTurnLoop(
            llm_config=llm_config,
            workspace=workspace,
            system_prompt=system_prompt,
            session=session,
            allowed_read_dirs=allowed_reads,
            bash_timeout=180,
            max_iterations=min(max_iters, 30),
            python_path=python_path,
            trace_prefix="sanity",
        )

        session.log("EXECUTE", "Starting sanity check turn loop...")
        turn_result = loop.run_turn(user_message)

        # Copy fixed files back to experiment directory
        fixed_count = self._copy_fixes_back(workspace, experiment_dir, session)

        # Determine success: check if smoke test passed by examining final text
        success = self._check_success(turn_result, workspace)
        session.log(
            "RESULT",
            f"Sanity check {'PASSED' if success else 'FAILED'}: "
            f"{turn_result.iterations} iters, {turn_result.tool_calls} tool calls, "
            f"{fixed_count} files fixed, {turn_result.elapsed_sec:.1f}s",
        )

        # Save report
        report = {
            "passed": success,
            "iterations": turn_result.iterations,
            "tool_calls": turn_result.tool_calls,
            "files_fixed": fixed_count,
            "errors": turn_result.errors,
            "elapsed_sec": round(turn_result.elapsed_sec, 1),
        }
        (stage_dir / "sanity_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8",
        )
        session.add_artifact("sanity_report.json")

        return StageResult(
            stage=Stage.SANITY_CHECK,
            status=StageStatus.DONE if success else StageStatus.FAILED,
            artifacts=("sanity_report.json",),
            error=None if success else "Smoke test did not pass",
            evidence_refs=("experiment/",),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_coding_llm(llm: Any, config: RCConfig) -> Any:
        """Resolve to coding model if configured."""
        coding_model = getattr(config.llm, "coding_model", "") or ""
        if not coding_model or not llm:
            return llm

        if hasattr(llm, "config") and llm.config.primary_model == coding_model:
            return llm

        try:
            from researchclaw.llm import create_llm_client
            import dataclasses
            new_llm_cfg = dataclasses.replace(config.llm, primary_model=coding_model)
            new_config = dataclasses.replace(config, llm=new_llm_cfg)
            return create_llm_client(new_config)
        except Exception:
            return llm

    @staticmethod
    def _find_experiment_dir(run_dir: Path) -> Path | None:
        """Find the latest experiment/ directory from prior stages."""
        candidates = []
        for stage_d in sorted(run_dir.glob("stage-*"), reverse=True):
            exp_dir = stage_d / "experiment"
            if exp_dir.is_dir() and (exp_dir / "main.py").is_file():
                candidates.append(exp_dir)
        return candidates[0] if candidates else None

    @staticmethod
    def _prepare_workspace(
        stage_dir: Path,
        experiment_dir: Path,
        run_dir: Path,
        config: RCConfig,
    ) -> Path:
        """Create workspace with experiment code + symlinks to data."""
        ws = stage_dir / f"sanity_workspace_{int(time.time())}_{os.getpid()}"
        shutil.copytree(
            experiment_dir, ws,
            ignore=shutil.ignore_patterns("__pycache__", ".snapshots", "*.pyc"),
            dirs_exist_ok=True,
        )

        # Symlink datasets
        datasets_dir = getattr(config.experiment, "datasets_dir", "") or ""
        if datasets_dir and Path(datasets_dir).is_dir():
            link = ws / "datasets"
            if not link.exists():
                try:
                    link.symlink_to(Path(datasets_dir).resolve())
                except OSError:
                    pass

        # Symlink checkpoints
        checkpoints_dir = getattr(config.experiment, "checkpoints_dir", "") or ""
        if checkpoints_dir and Path(checkpoints_dir).is_dir():
            link = ws / "checkpoints"
            if not link.exists():
                try:
                    link.symlink_to(Path(checkpoints_dir).resolve())
                except OSError:
                    pass

        # Symlink codebases
        codebases_dir = getattr(config.experiment, "codebases_dir", "") or ""
        if codebases_dir and Path(codebases_dir).is_dir():
            link = ws / "codebases"
            if not link.exists():
                try:
                    link.symlink_to(Path(codebases_dir).resolve())
                except OSError:
                    pass

        # Ensure outputs dir
        (ws / "outputs").mkdir(exist_ok=True)
        return ws

    @staticmethod
    def _load_plan_summary(run_dir: Path) -> str:
        """Load experiment plan summary from prior stages."""
        for stage_d in sorted(run_dir.glob("stage-*"), reverse=True):
            plan_file = stage_d / "exp_plan.yaml"
            if plan_file.is_file():
                try:
                    text = plan_file.read_text(encoding="utf-8")
                    return text[:2000]
                except OSError:
                    pass
        return ""

    @staticmethod
    def _build_allowed_reads(config: RCConfig) -> list[Path]:
        dirs: list[Path] = []
        for attr in ("datasets_dir", "checkpoints_dir", "codebases_dir"):
            d = getattr(config.experiment, attr, "") or ""
            if d and Path(d).is_dir():
                dirs.append(Path(d))
        return dirs

    @staticmethod
    def _copy_fixes_back(workspace: Path, experiment_dir: Path, session: StageSession) -> int:
        """Copy fixed .py files back to the original experiment directory."""
        count = 0
        for py_file in workspace.rglob("*.py"):
            if py_file.is_symlink():
                continue
            rel = py_file.relative_to(workspace)
            if any(p.startswith(".") or p == "__pycache__" or p == "codebases" for p in rel.parts):
                continue
            dest = experiment_dir / rel
            try:
                new_content = py_file.read_text(encoding="utf-8")
                old_content = dest.read_text(encoding="utf-8") if dest.exists() else ""
                if new_content != old_content:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_text(new_content, encoding="utf-8")
                    count += 1
                    session.log("FIX", f"Updated {rel}")
            except OSError:
                pass
        return count

    @staticmethod
    def _check_success(turn_result: Any, workspace: Path) -> bool:
        """Determine if the smoke test passed."""
        final = turn_result.final_text.lower()
        if any(phrase in final for phrase in (
            "smoke test pass", "sanity check pass", "test passed",
            "completed successfully", "all checks pass",
            "exit code 0", "smoke_test passed",
        )):
            return True

        if turn_result.errors:
            return False

        if turn_result.iterations >= 2 and not turn_result.errors:
            return True

        return False
