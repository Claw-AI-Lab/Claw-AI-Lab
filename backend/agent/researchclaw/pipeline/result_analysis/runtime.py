"""S16 RESULT_ANALYSIS runtime — agentic analysis via claw-engine turn loop.

The agent reads raw experiment output files, writes analysis scripts,
runs them, and produces experiment_summary.json + analysis.md.
This adapts to any data format the experiment happened to produce.
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
from researchclaw.pipeline.result_analysis.system_prompt import (
    build_system_prompt,
    build_user_message,
)
from researchclaw.pipeline.stages import Stage, StageStatus

logger = logging.getLogger(__name__)

_COLLECT_EXTENSIONS = frozenset({
    ".json", ".csv", ".tsv", ".txt", ".yaml", ".yml", ".log", ".md",
})

_SKIP_DIRS = frozenset({
    "__pycache__", ".git", "codebases", "datasets", "checkpoints",
})


class ResultAnalysisRuntime:
    """Orchestration for S16 RESULT_ANALYSIS using claw-engine turn loop."""

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
        session = StageSession(stage_dir=stage_dir, stage_name="result_analysis")
        session.log("INIT", "ResultAnalysisRuntime started")

        llm = self._resolve_coding_llm(llm, config)
        if llm is None:
            session.log_error("INIT", "No LLM client available")
            return StageResult(
                stage=Stage.RESULT_ANALYSIS,
                status=StageStatus.FAILED,
                artifacts=(),
                error="No LLM client available for agentic result analysis",
            )

        llm_config = llm.config
        session.log("INIT", f"LLM: {llm_config.primary_model}")

        python_path = getattr(config.experiment.sandbox, "python_path", "") or ""

        workspace = self._prepare_workspace(stage_dir, run_dir, config)
        session.log("INIT", f"Workspace: {workspace}")

        data_files = self._list_data_files(workspace)
        session.log("INIT", f"Found {len(data_files)} data files")

        system_prompt = build_system_prompt(
            python_path=python_path,
            workspace_path=str(workspace),
        )

        topic = getattr(config.research, "topic", "") or ""
        metric_key = getattr(config.experiment, "metric_key", "primary_metric")
        metric_direction = getattr(config.experiment, "metric_direction", "minimize")

        user_message = build_user_message(
            workspace_path=str(workspace),
            data_files=data_files,
            metric_key=metric_key,
            metric_direction=metric_direction,
            topic=topic,
        )

        (stage_dir / "result_analysis_system_prompt.md").write_text(
            system_prompt, encoding="utf-8",
        )

        allowed_reads = self._build_allowed_reads(config, run_dir)

        loop = AgentTurnLoop(
            llm_config=llm_config,
            workspace=workspace,
            system_prompt=system_prompt,
            session=session,
            allowed_read_dirs=allowed_reads,
            bash_timeout=300,
            max_iterations=25,
            python_path=python_path,
            trace_prefix="result_analysis",
        )

        session.log("EXECUTE", "Starting result analysis turn loop...")
        turn_result = loop.run_turn(user_message)

        summary_path = workspace / "experiment_summary.json"
        analysis_path = workspace / "analysis.md"

        has_summary = summary_path.is_file()
        has_analysis = analysis_path.is_file()

        if has_summary:
            try:
                shutil.copy2(summary_path, stage_dir / "experiment_summary.json")
                session.log("RESULT", "Copied experiment_summary.json to stage_dir")
            except OSError as exc:
                session.log_error("RESULT", f"Failed to copy summary: {exc}")

        if has_analysis:
            try:
                shutil.copy2(analysis_path, stage_dir / "analysis.md")
                session.log("RESULT", "Copied analysis.md to stage_dir")
            except OSError as exc:
                session.log_error("RESULT", f"Failed to copy analysis: {exc}")

        success = has_summary and has_analysis and not turn_result.errors
        session.log(
            "RESULT",
            f"Result analysis {'SUCCEEDED' if success else 'FAILED'}: "
            f"{turn_result.iterations} iters, {turn_result.tool_calls} tool calls, "
            f"summary={'yes' if has_summary else 'no'}, "
            f"analysis={'yes' if has_analysis else 'no'}, "
            f"{turn_result.elapsed_sec:.1f}s",
        )

        artifacts = []
        if has_analysis:
            artifacts.append("analysis.md")
        if has_summary:
            artifacts.append("experiment_summary.json")

        # Copy charts if the agent generated any
        ws_charts = workspace / "charts"
        if ws_charts.is_dir() and any(ws_charts.iterdir()):
            stage_charts = stage_dir / "charts"
            try:
                if stage_charts.exists():
                    shutil.rmtree(stage_charts)
                shutil.copytree(ws_charts, stage_charts, dirs_exist_ok=True)
                artifacts.append("charts/")
                session.log("RESULT", "Copied charts/ to stage_dir")
            except OSError:
                pass

        if not success:
            return StageResult(
                stage=Stage.RESULT_ANALYSIS,
                status=StageStatus.FAILED,
                artifacts=tuple(artifacts),
                error="Agentic analysis did not produce required outputs",
                decision="retry",
            )

        return StageResult(
            stage=Stage.RESULT_ANALYSIS,
            status=StageStatus.DONE,
            artifacts=tuple(artifacts),
            evidence_refs=tuple(f"stage-16/{a}" for a in artifacts),
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
    def _prepare_workspace(
        stage_dir: Path, run_dir: Path, config: RCConfig,
    ) -> Path:
        ws = stage_dir / f"analysis_workspace_{int(time.time())}_{os.getpid()}"
        ws.mkdir(parents=True, exist_ok=True)

        # Copy result data from prior stages into the workspace
        for source_name in ("runs", "experiment_final"):
            for stage_d in sorted(run_dir.glob("stage-*"), reverse=True):
                src = stage_d / source_name
                if src.is_dir():
                    dst = ws / source_name
                    try:
                        shutil.copytree(
                            src, dst,
                            ignore=shutil.ignore_patterns(
                                "__pycache__", "*.pyc", ".git",
                            ),
                            dirs_exist_ok=True,
                        )
                    except OSError:
                        pass
                    break

        # Copy key standalone result files
        _RESULT_FILES = (
            "results.json", "results_v0.json", "refinement_log.json",
            "experiment_summary.json", "sanity_report.json",
        )
        for fname in _RESULT_FILES:
            for stage_d in sorted(run_dir.glob("stage-*"), reverse=True):
                src = stage_d / fname
                if src.is_file():
                    dst = ws / fname
                    if not dst.exists():
                        try:
                            shutil.copy2(src, dst)
                        except OSError:
                            pass
                    break
                # Also check one level deeper (e.g. runs/results.json, experiment_final/results_v0.json)
                for sub in stage_d.iterdir():
                    if sub.is_dir():
                        nested = sub / fname
                        if nested.is_file():
                            nested_dst = ws / sub.name / fname
                            nested_dst.parent.mkdir(parents=True, exist_ok=True)
                            if not nested_dst.exists():
                                try:
                                    shutil.copy2(nested, nested_dst)
                                except OSError:
                                    pass

        # Copy experiment plan for context
        for plan_name in ("exp_plan.yaml", "EXPERIMENT_PLAN.yaml"):
            for stage_d in sorted(run_dir.glob("stage-*"), reverse=True):
                src = stage_d / plan_name
                if src.is_file():
                    try:
                        shutil.copy2(src, ws / plan_name)
                    except OSError:
                        pass
                    break

        # Copy analysis.md if it already exists from prior S16 run
        for stage_d in sorted(run_dir.glob("stage-*"), reverse=True):
            src = stage_d / "analysis.md"
            if src.is_file():
                try:
                    shutil.copy2(src, ws / "prior_analysis.md")
                except OSError:
                    pass
                break

        (ws / "charts").mkdir(exist_ok=True)
        return ws

    @staticmethod
    def _list_data_files(workspace: Path) -> list[str]:
        files: list[str] = []
        for fpath in sorted(workspace.rglob("*")):
            if not fpath.is_file() or fpath.is_symlink():
                continue
            rel = fpath.relative_to(workspace)
            if any(p.startswith(".") or p in _SKIP_DIRS for p in rel.parts):
                continue
            if fpath.suffix.lower() not in _COLLECT_EXTENSIONS:
                continue
            if fpath.stat().st_size > 5 * 1024 * 1024:
                continue
            files.append(str(rel))
        return files

    @staticmethod
    def _build_allowed_reads(config: RCConfig, run_dir: Path) -> list[Path]:
        dirs: list[Path] = [run_dir]
        for attr in ("datasets_dir", "checkpoints_dir", "codebases_dir"):
            d = getattr(config.experiment, attr, "") or ""
            if d and Path(d).is_dir():
                dirs.append(Path(d))
        return dirs
