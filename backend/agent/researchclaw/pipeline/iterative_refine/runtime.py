"""S15 ITERATIVE_REFINE runtime — orchestrates experiment optimization loop.

The agent iteratively improves the experiment by analyzing results,
modifying code, and re-running — up to max_iterations times.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from researchclaw.config import RCConfig
from researchclaw.adapters import AdapterBundle
from researchclaw.pipeline.claw_engine import AgentTurnLoop, StageSession
from researchclaw.pipeline.iterative_refine.system_prompt import (
    build_system_prompt,
    build_user_message,
)
from researchclaw.pipeline.stages import Stage, StageStatus

logger = logging.getLogger(__name__)


class IterativeRefineRuntime:
    """Orchestration for S15 ITERATIVE_REFINE using claw-code turn loop."""

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
        session = StageSession(stage_dir=stage_dir, stage_name="iterative_refine")
        session.log("INIT", "IterativeRefineRuntime started")

        # Resolve coding LLM
        llm = self._resolve_coding_llm(llm, config)
        if llm is None:
            session.log_error("INIT", "No LLM client available")
            return StageResult(
                stage=Stage.ITERATIVE_REFINE,
                status=StageStatus.FAILED,
                artifacts=(),
                error="No LLM client available",
            )

        llm_config = llm.config
        session.log("INIT", f"LLM: {llm_config.primary_model}")

        # Find experiment directory
        experiment_dir = self._find_experiment_dir(run_dir)
        if not experiment_dir:
            session.log_error("INIT", "No experiment directory found")
            return StageResult(
                stage=Stage.ITERATIVE_REFINE,
                status=StageStatus.FAILED,
                artifacts=(),
                error="No experiment/ directory found",
            )
        session.log("INIT", f"Experiment dir: {experiment_dir}")

        # Load baseline results from S14
        baseline_results = self._load_baseline_results(run_dir)
        session.log("INIT", f"Baseline results: {baseline_results[:200] if baseline_results else 'none'}")

        # Config values
        time_budget = getattr(config.experiment, "time_budget_sec", 3600)
        max_iterations = min(getattr(config.experiment, "max_iterations", 3), 10)
        metric_key = getattr(config.experiment, "metric_key", "primary_metric")
        metric_direction = getattr(config.experiment, "metric_direction", "minimize")
        python_path = getattr(config.experiment.sandbox, "python_path", "") or ""

        # Select GPU
        gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES") or self._find_free_gpu()
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        session.log("INIT", f"GPU: {gpu_id}")

        # Prepare workspace
        workspace = self._prepare_workspace(stage_dir, experiment_dir, run_dir, config)
        session.log("INIT", f"Workspace: {workspace}")

        # Copy baseline results into workspace
        if baseline_results:
            (workspace / "results.json").write_text(baseline_results, encoding="utf-8")

        # List files
        exp_files = self._list_experiment_files(workspace)

        # Load plan summary
        exp_plan_summary = self._load_plan_summary(run_dir)

        # Build prompts
        system_prompt = build_system_prompt(
            python_path=python_path,
            workspace_path=str(workspace),
            time_budget_sec=time_budget,
            metric_key=metric_key,
            metric_direction=metric_direction,
            max_refine_iterations=max_iterations,
        )

        user_message = build_user_message(
            experiment_dir=str(workspace),
            experiment_files=exp_files,
            baseline_results=baseline_results,
            metric_key=metric_key,
            metric_direction=metric_direction,
            max_iterations=max_iterations,
            exp_plan_summary=exp_plan_summary,
        )

        # Save system prompt
        (stage_dir / "refine_system_prompt.md").write_text(system_prompt, encoding="utf-8")
        session.add_artifact("refine_system_prompt.md")

        # Allowed read dirs
        allowed_reads = self._build_allowed_reads(config)

        # The refine loop gets generous iterations and bash timeout because
        # each refinement cycle involves a full experiment re-run
        loop = AgentTurnLoop(
            llm_config=llm_config,
            workspace=workspace,
            system_prompt=system_prompt,
            session=session,
            allowed_read_dirs=allowed_reads,
            bash_timeout=min(time_budget + 60, 7200),
            max_iterations=max_iterations * 8,  # ~8 LLM turns per refinement cycle
            python_path=python_path,
            trace_prefix="iterative_refine",
        )

        session.log("EXECUTE", "Starting iterative refinement turn loop...")
        turn_result = loop.run_turn(user_message)

        # Collect final results
        final_results = self._collect_final_results(workspace)
        improved = self._check_improvement(baseline_results, final_results, metric_key, metric_direction)

        # Copy back
        self._copy_results_back(workspace, experiment_dir, stage_dir, session)

        session.log(
            "RESULT",
            f"Iterative refine {'IMPROVED' if improved else 'no improvement'}: "
            f"{turn_result.iterations} iters, {turn_result.tool_calls} tool calls, "
            f"{turn_result.elapsed_sec:.1f}s",
        )

        # Save refinement log
        log_data = {
            "improved": improved,
            "baseline_metric": self._extract_metric(baseline_results, metric_key),
            "final_metric": self._extract_metric(final_results, metric_key),
            "metric_key": metric_key,
            "metric_direction": metric_direction,
            "iterations": turn_result.iterations,
            "tool_calls": turn_result.tool_calls,
            "errors": turn_result.errors,
            "elapsed_sec": round(turn_result.elapsed_sec, 1),
        }
        (stage_dir / "refinement_log.json").write_text(
            json.dumps(log_data, indent=2), encoding="utf-8",
        )
        session.add_artifact("refinement_log.json")

        # Write experiment_final/
        final_dir = stage_dir / "experiment_final"
        self._write_final_experiment(workspace, final_dir, session)
        session.add_artifact("experiment_final/")

        return StageResult(
            stage=Stage.ITERATIVE_REFINE,
            status=StageStatus.DONE,
            artifacts=("refinement_log.json", "experiment_final/"),
            evidence_refs=("experiment_final/",),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_coding_llm(llm: Any, config: RCConfig) -> Any:
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
        for stage_d in sorted(run_dir.glob("stage-*"), reverse=True):
            exp_dir = stage_d / "experiment"
            if exp_dir.is_dir() and (exp_dir / "main.py").is_file():
                return exp_dir
            final_dir = stage_d / "experiment_final"
            if final_dir.is_dir() and (final_dir / "main.py").is_file():
                return final_dir
        return None

    @staticmethod
    def _find_free_gpu() -> str:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode != 0:
                return "0"
            gpus: list[tuple[int, float]] = []
            for line in result.stdout.strip().split("\n"):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 4:
                    idx = int(parts[0])
                    mem_frac = float(parts[1]) / max(float(parts[2]), 1.0)
                    util_frac = float(parts[3]) / 100.0
                    gpus.append((idx, 0.5 * mem_frac + 0.5 * util_frac))
            if not gpus:
                return "0"
            gpus.sort(key=lambda x: x[1])
            return str(gpus[0][0])
        except Exception:
            return "0"

    @staticmethod
    def _load_baseline_results(run_dir: Path) -> str:
        """Load results.json from the most recent experiment run stage."""
        for stage_d in sorted(run_dir.glob("stage-*"), reverse=True):
            for candidate in (
                stage_d / "runs" / "results.json",
                stage_d / "runs" / "sandbox" / "_project" / "results.json",
            ):
                if candidate.is_file():
                    try:
                        return candidate.read_text(encoding="utf-8")
                    except OSError:
                        pass
        return ""

    @staticmethod
    def _load_plan_summary(run_dir: Path) -> str:
        for stage_d in sorted(run_dir.glob("stage-*"), reverse=True):
            plan_file = stage_d / "exp_plan.yaml"
            if plan_file.is_file():
                try:
                    return plan_file.read_text(encoding="utf-8")[:2000]
                except OSError:
                    pass
        return ""

    @staticmethod
    def _prepare_workspace(
        stage_dir: Path, experiment_dir: Path, run_dir: Path, config: RCConfig,
    ) -> Path:
        ws = stage_dir / f"refine_workspace_{int(time.time())}_{os.getpid()}"
        shutil.copytree(
            experiment_dir, ws,
            ignore=shutil.ignore_patterns("__pycache__", ".snapshots", "*.pyc"),
            dirs_exist_ok=True,
        )

        for attr, link_name in (
            ("datasets_dir", "datasets"),
            ("checkpoints_dir", "checkpoints"),
            ("codebases_dir", "codebases"),
        ):
            d = getattr(config.experiment, attr, "") or ""
            if d and Path(d).is_dir():
                link = ws / link_name
                if not link.exists():
                    try:
                        link.symlink_to(Path(d).resolve())
                    except OSError:
                        pass

        (ws / "outputs").mkdir(exist_ok=True)
        return ws

    @staticmethod
    def _list_experiment_files(workspace: Path) -> list[str]:
        return [
            str(f.relative_to(workspace))
            for f in sorted(workspace.rglob("*.py"))
            if not f.is_symlink()
            and not any(p.startswith(".") or p == "__pycache__" for p in f.relative_to(workspace).parts)
        ]

    @staticmethod
    def _build_allowed_reads(config: RCConfig) -> list[Path]:
        dirs: list[Path] = []
        for attr in ("datasets_dir", "checkpoints_dir", "codebases_dir"):
            d = getattr(config.experiment, attr, "") or ""
            if d and Path(d).is_dir():
                dirs.append(Path(d))
        return dirs

    @staticmethod
    def _collect_final_results(workspace: Path) -> str:
        results_file = workspace / "results.json"
        if results_file.is_file():
            try:
                return results_file.read_text(encoding="utf-8")
            except OSError:
                pass
        return ""

    @staticmethod
    def _extract_metric(results_str: str, metric_key: str) -> float | None:
        if not results_str:
            return None
        try:
            data = json.loads(results_str)
            if isinstance(data, dict):
                return data.get(metric_key)
        except (json.JSONDecodeError, TypeError):
            pass
        return None

    @staticmethod
    def _check_improvement(
        baseline_str: str, final_str: str,
        metric_key: str, metric_direction: str,
    ) -> bool:
        if not baseline_str or not final_str:
            return bool(final_str)
        try:
            baseline = json.loads(baseline_str)
            final = json.loads(final_str)
            base_val = baseline.get(metric_key)
            final_val = final.get(metric_key)
            if base_val is None or final_val is None:
                return False
            if metric_direction == "minimize":
                return float(final_val) < float(base_val)
            else:
                return float(final_val) > float(base_val)
        except (json.JSONDecodeError, TypeError, ValueError):
            return False

    @staticmethod
    def _copy_results_back(
        workspace: Path, experiment_dir: Path, stage_dir: Path, session: StageSession,
    ) -> None:
        # Copy results
        results_src = workspace / "results.json"
        if results_src.is_file():
            try:
                shutil.copy2(results_src, stage_dir / "results.json")
                session.log("RESULT", "Saved final results.json")
            except OSError:
                pass

        # Copy modified .py files back to experiment_dir
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
            except OSError:
                pass

    @staticmethod
    def _write_final_experiment(workspace: Path, final_dir: Path, session: StageSession) -> None:
        """Write the final experiment state to experiment_final/."""
        final_dir.mkdir(parents=True, exist_ok=True)
        for f in workspace.rglob("*"):
            if f.is_symlink() or not f.is_file():
                continue
            rel = f.relative_to(workspace)
            if any(p.startswith(".") or p == "__pycache__" or p == "codebases" for p in rel.parts):
                continue
            dest = final_dir / rel
            try:
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, dest)
            except OSError:
                pass
        session.log("FINALIZE", f"Wrote experiment_final/ ({len(list(final_dir.rglob('*')))} files)")
