"""Claw-code agentic strategy — the sole code generation path.

Replaces Aider/Blueprint/SingleShot with a Python implementation of
claw-code's ``ConversationRuntime.run_turn()`` pattern: the LLM
iteratively calls 6 tools (bash, read_file, write_file, edit_file,
glob_search, grep_search) to generate experiment code.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any

from researchclaw.config import RCConfig
from researchclaw.pipeline.codegen.session import CodegenSession
from researchclaw.pipeline.codegen.system_prompt import build_system_prompt, build_user_message
from researchclaw.pipeline.codegen.turn_loop import ClawTurnLoop
from researchclaw.pipeline.codegen.types import CodegenContext, CodegenPhase, CodegenResult

logger = logging.getLogger(__name__)


class ClawAgentStrategy:
    """Agentic code generation via claw-code turn loop.

    The LLM uses tools to explore codebases, write experiment files,
    run them, and fix errors — iterating until the code works or
    max iterations are reached.
    """

    @property
    def name(self) -> str:
        return "claw_agent"

    def can_handle(self, ctx: CodegenContext, config: RCConfig) -> bool:
        return True

    def generate(
        self,
        ctx: CodegenContext,
        config: RCConfig,
        llm: Any,
        session: CodegenSession,
        prompts: Any | None = None,
    ) -> CodegenResult:
        if llm is None:
            return CodegenResult(
                strategy_name=self.name,
                error="No LLM client available",
            )

        session.log(CodegenPhase.GENERATE, "ClawAgentStrategy started")
        t0 = time.monotonic()

        # Resolve coding model config
        llm_config = self._resolve_llm_config(llm, config)
        session.log(
            CodegenPhase.GENERATE,
            f"LLM: model={llm_config.primary_model}, "
            f"base_url={llm_config.base_url[:60]}...",
        )

        # Prepare workspace
        workspace = self._prepare_workspace(ctx, session)
        session.log(CodegenPhase.GENERATE, f"Workspace: {workspace}")

        # Build system prompt (set workspace path so LLM knows the correct cwd)
        from researchclaw.pipeline.codegen.system_prompt import set_workspace_path
        set_workspace_path(str(workspace))
        system_prompt = build_system_prompt(ctx)
        session.log(
            CodegenPhase.GENERATE,
            f"System prompt: {len(system_prompt)} chars",
        )

        # Save system prompt for debugging
        if ctx.stage_dir:
            (ctx.stage_dir / "claw_system_prompt.md").write_text(
                system_prompt, encoding="utf-8",
            )
            session.add_artifact("claw_system_prompt.md")

        # Build allowed read directories
        allowed_reads = self._build_allowed_reads(ctx)
        session.log(
            CodegenPhase.GENERATE,
            f"Allowed read dirs: {[str(d) for d in allowed_reads]}",
        )

        # Build user message
        user_message = build_user_message(ctx)

        # Resolve python_path from config (experiment.sandbox.python_path)
        python_path = getattr(config.experiment.sandbox, "python_path", "") or ""
        if python_path:
            session.log(CodegenPhase.GENERATE, f"Python path from config: {python_path}")
        else:
            session.log(CodegenPhase.GENERATE, "No python_path in config — using system default")

        # Create and run turn loop
        loop = ClawTurnLoop(
            llm_config=llm_config,
            workspace=workspace,
            system_prompt=system_prompt,
            session=session,
            allowed_read_dirs=allowed_reads,
            bash_timeout=120,  # S11 verifies via import + smoke mode; full experiment remains the default code path
            max_iterations=40,
            python_path=python_path,
        )

        loop.set_exp_plan(ctx.exp_plan)
        session.log(CodegenPhase.GENERATE, "Starting turn loop...")
        turn_result = loop.run_turn(user_message)

        elapsed = time.monotonic() - t0

        # Save turn loop log
        if ctx.stage_dir:
            (ctx.stage_dir / "claw_agent_log.json").write_text(
                json.dumps({
                    "success": "main.py" in turn_result.files,
                    "iterations": turn_result.iterations,
                    "tool_calls": turn_result.tool_calls,
                    "files_produced": sorted(turn_result.files.keys()),
                    "file_sizes": {f: len(c) for f, c in turn_result.files.items()},
                    "errors": turn_result.errors,
                    "elapsed_sec": round(elapsed, 1),
                    "final_text_length": len(turn_result.final_text),
                }, indent=2),
                encoding="utf-8",
            )
            session.add_artifact("claw_agent_log.json")

        session.llm_calls += turn_result.tool_calls

        if turn_result.files and "main.py" in turn_result.files:
            session.log(
                CodegenPhase.GENERATE,
                f"ClawAgent SUCCESS — {len(turn_result.files)} files, "
                f"{turn_result.iterations} iterations, "
                f"{turn_result.tool_calls} tool calls, {elapsed:.1f}s",
            )
            return CodegenResult(
                files=turn_result.files,
                strategy_name=self.name,
                skip_review=False,
                elapsed_sec=elapsed,
            )

        session.log(
            CodegenPhase.GENERATE,
            f"ClawAgent produced no main.py — "
            f"{len(turn_result.files)} files: {sorted(turn_result.files.keys())}",
        )
        return CodegenResult(
            files=turn_result.files,
            strategy_name=self.name,
            elapsed_sec=elapsed,
            error="No main.py produced",
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_llm_config(llm: Any, config: RCConfig) -> Any:
        """Get the LLM config (may have coding_model override already applied)."""
        return llm.config

    def _prepare_workspace(self, ctx: CodegenContext, session: CodegenSession) -> Path:
        """Create a clean workspace with symlinks to data directories."""
        ws = ctx.stage_dir / f"claw_workspace_{int(time.time())}_{os.getpid()}"
        ws.mkdir(parents=True, exist_ok=True)

        # Save experiment plan
        if ctx.exp_plan:
            (ws / "EXPERIMENT_PLAN.yaml").write_text(
                ctx.exp_plan, encoding="utf-8",
            )

        # Generate CODEGEN.md directly in workspace (user-provided takes priority)
        codegen_dst = ws / "CODEGEN.md"
        user_codegen = ctx.run_dir / "CODEGEN.md" if ctx.run_dir else None
        if user_codegen and user_codegen.is_file():
            content = user_codegen.read_text(encoding="utf-8")
            codegen_dst.write_text(content, encoding="utf-8")
            session.log(CodegenPhase.GENERATE, f"Using user-provided CODEGEN.md ({len(content)} chars)")
        else:
            from researchclaw.pipeline.codegen.runtime import generate_codegen_md
            content = generate_codegen_md(ctx, codegen_dst)
            session.log(CodegenPhase.GENERATE, f"Auto-generated CODEGEN.md ({len(content)} chars)")

        # Symlink codebases
        if ctx.codebases_dir and Path(ctx.codebases_dir).is_dir():
            cb_link = ws / "codebases"
            try:
                cb_link.symlink_to(Path(ctx.codebases_dir).resolve())
                session.log(CodegenPhase.GENERATE, f"Linked codebases: {ctx.codebases_dir}")
            except OSError as e:
                session.log(CodegenPhase.GENERATE, f"Failed to link codebases: {e}")

        # Symlink datasets
        if ctx.datasets_dir and Path(ctx.datasets_dir).is_dir():
            ds_link = ws / "datasets"
            try:
                ds_link.symlink_to(Path(ctx.datasets_dir).resolve())
                session.log(CodegenPhase.GENERATE, f"Linked datasets: {ctx.datasets_dir}")
            except OSError:
                pass

        # Symlink checkpoints
        if ctx.checkpoints_dir and Path(ctx.checkpoints_dir).is_dir():
            ck_link = ws / "checkpoints"
            try:
                ck_link.symlink_to(Path(ctx.checkpoints_dir).resolve())
                session.log(CodegenPhase.GENERATE, f"Linked checkpoints: {ctx.checkpoints_dir}")
            except OSError:
                pass

        # Create outputs directory
        (ws / "outputs").mkdir(exist_ok=True)

        return ws

    @staticmethod
    def _build_allowed_reads(ctx: CodegenContext) -> list[Path]:
        """Build list of directories allowed for read operations."""
        dirs: list[Path] = []
        for d in (ctx.codebases_dir, ctx.datasets_dir, ctx.checkpoints_dir):
            if d and Path(d).is_dir():
                dirs.append(Path(d))
        return dirs
