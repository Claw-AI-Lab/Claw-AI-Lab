"""S14 EXPERIMENT_RUN runtime — orchestrates full experiment execution.

The agent runs the experiment, monitors for runtime errors, and
applies fixes while preserving scientific validity.
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
from researchclaw.pipeline.experiment_run.system_prompt import (
    build_system_prompt,
    build_user_message,
)
from researchclaw.pipeline.stages import Stage, StageStatus

logger = logging.getLogger(__name__)


class ExperimentRunRuntime:
    """Orchestration for S14 EXPERIMENT_RUN using claw-code turn loop."""

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
        session = StageSession(stage_dir=stage_dir, stage_name="experiment_run")
        session.log("INIT", "ExperimentRunRuntime started")

        # Resolve coding LLM
        llm = self._resolve_coding_llm(llm, config)
        if llm is None:
            session.log_error("INIT", "No LLM client available")
            return StageResult(
                stage=Stage.EXPERIMENT_RUN,
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
                stage=Stage.EXPERIMENT_RUN,
                status=StageStatus.FAILED,
                artifacts=(),
                error="No experiment/ directory found",
            )
        session.log("INIT", f"Experiment dir: {experiment_dir}")

        # Select GPU
        gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES") or self._find_free_gpu()
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        session.log("INIT", f"GPU: {gpu_id}")

        # Auto-install missing dependencies
        python_path = getattr(config.experiment.sandbox, "python_path", "") or ""
        self._ensure_deps(experiment_dir, python_path, session)

        # Prepare workspace
        workspace = self._prepare_workspace(stage_dir, experiment_dir, run_dir, config)
        session.log("INIT", f"Workspace: {workspace}")

        # List files
        exp_files = self._list_experiment_files(workspace)

        # Time budget
        time_budget = getattr(config.experiment, "time_budget_sec", 3600)

        # Build prompts
        system_prompt = build_system_prompt(
            python_path=python_path,
            workspace_path=str(workspace),
            time_budget_sec=time_budget,
            gpu_id=gpu_id,
        )

        prior_results = self._load_sanity_results(run_dir)
        user_message = build_user_message(
            experiment_dir=str(workspace),
            experiment_files=exp_files,
            time_budget_sec=time_budget,
            metric_key=getattr(config.experiment, "metric_key", "primary_metric"),
            metric_direction=getattr(config.experiment, "metric_direction", "minimize"),
            prior_results=prior_results,
        )

        # Save system prompt
        (stage_dir / "experiment_run_system_prompt.md").write_text(
            system_prompt, encoding="utf-8",
        )
        session.add_artifact("experiment_run_system_prompt.md")

        # Allowed read dirs
        allowed_reads = self._build_allowed_reads(config)

        # Create turn loop with generous bash timeout for full experiments
        loop = AgentTurnLoop(
            llm_config=llm_config,
            workspace=workspace,
            system_prompt=system_prompt,
            session=session,
            allowed_read_dirs=allowed_reads,
            bash_timeout=time_budget + 600,
            max_iterations=20,
            python_path=python_path,
            trace_prefix="experiment_run",
        )

        session.log("EXECUTE", "Starting experiment run turn loop...")
        turn_result = loop.run_turn(user_message)

        # Copy results back
        self._copy_results_back(workspace, experiment_dir, stage_dir, session)

        # Check for results.json
        results_json = workspace / "results.json"
        has_results = results_json.is_file()
        if has_results:
            try:
                results_data = json.loads(results_json.read_text(encoding="utf-8"))
                session.log("RESULT", f"results.json: {json.dumps(results_data)[:500]}")
                session.metadata["results"] = results_data
            except Exception as exc:
                session.log("RESULT", f"Failed to parse results.json: {exc}")
                has_results = False

        success = has_results and not turn_result.errors
        session.log(
            "RESULT",
            f"Experiment run {'SUCCEEDED' if success else 'FAILED'}: "
            f"{turn_result.iterations} iters, {turn_result.tool_calls} tool calls, "
            f"results.json={'yes' if has_results else 'no'}, "
            f"{turn_result.elapsed_sec:.1f}s",
        )

        # Save run report
        report = {
            "success": success,
            "has_results": has_results,
            "results": session.metadata.get("results", {}),
            "iterations": turn_result.iterations,
            "tool_calls": turn_result.tool_calls,
            "errors": turn_result.errors,
            "elapsed_sec": round(turn_result.elapsed_sec, 1),
            "gpu_id": gpu_id,
        }
        runs_dir = stage_dir / "runs"
        runs_dir.mkdir(exist_ok=True)
        (runs_dir / "run_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8",
        )
        session.add_artifact("runs/run_report.json")

        return StageResult(
            stage=Stage.EXPERIMENT_RUN,
            status=StageStatus.DONE if success else StageStatus.FAILED,
            artifacts=("runs/",),
            error=None if success else "Experiment did not produce valid results",
            evidence_refs=("runs/",),
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
    def _ensure_deps(experiment_dir: Path, python_path: str, session: StageSession) -> None:
        """Auto-install missing dependencies from experiment code."""
        import re
        code = ""
        for f in experiment_dir.rglob("*.py"):
            if f.is_file() and not f.is_symlink():
                try:
                    code += "\n" + f.read_text(encoding="utf-8")
                except OSError:
                    pass

        imports: set[str] = set()
        for line in code.splitlines():
            m = re.match(r"^(?:from|import)\s+(\w+)", line.strip())
            if m:
                imports.add(m.group(1))

        safe_packages = {
            "torch", "torchvision", "torchmetrics", "transformers", "diffusers",
            "accelerate", "peft", "safetensors", "einops", "PIL", "cv2",
            "numpy", "scipy", "pandas", "sklearn", "tqdm", "matplotlib",
        }
        to_check = imports & safe_packages
        py = python_path or "python3"

        for pkg in sorted(to_check):
            try:
                r = subprocess.run(
                    [py, "-c", f"import {pkg}"],
                    capture_output=True, timeout=10,
                )
                if r.returncode != 0:
                    pip_name = {"sklearn": "scikit-learn", "PIL": "Pillow", "cv2": "opencv-python"}.get(pkg, pkg)
                    session.log("DEPS", f"Installing {pip_name}")
                    subprocess.run(
                        [py, "-m", "pip", "install", pip_name, "--quiet"],
                        capture_output=True, timeout=120,
                    )
            except Exception:
                pass

    @staticmethod
    def _prepare_workspace(
        stage_dir: Path, experiment_dir: Path, run_dir: Path, config: RCConfig,
    ) -> Path:
        ws = stage_dir / f"run_workspace_{int(time.time())}_{os.getpid()}"
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
    def _load_sanity_results(run_dir: Path) -> str:
        for stage_d in sorted(run_dir.glob("stage-*"), reverse=True):
            report = stage_d / "sanity_report.json"
            if report.is_file():
                try:
                    return report.read_text(encoding="utf-8")[:1000]
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
    def _copy_results_back(
        workspace: Path, experiment_dir: Path, stage_dir: Path, session: StageSession,
    ) -> None:
        """Copy results and modified files back."""
        # Copy results.json
        results_src = workspace / "results.json"
        if results_src.is_file():
            try:
                runs_dir = stage_dir / "runs"
                runs_dir.mkdir(exist_ok=True)
                shutil.copy2(results_src, runs_dir / "results.json")
                session.log("RESULT", "Copied results.json to runs/")
            except OSError:
                pass

        # Copy any modified .py files back
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
                    session.log("FIX", f"Updated experiment/{rel}")
            except OSError:
                pass

        # Copy outputs/
        ws_outputs = workspace / "outputs"
        if ws_outputs.is_dir():
            stage_outputs = stage_dir / "runs" / "outputs"
            try:
                if stage_outputs.exists():
                    shutil.rmtree(stage_outputs)
                shutil.copytree(ws_outputs, stage_outputs, dirs_exist_ok=True)
            except OSError:
                pass
