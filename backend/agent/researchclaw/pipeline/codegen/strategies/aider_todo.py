"""Beast Mode strategy: Aider TODO-driven code generation.

Wraps ``OpenHandsBridge`` from ``openhands_bridge.py`` via composition.
The existing bridge handles workspace preparation, TODO loop, syntax
fixing, and file collection — this strategy just adapts the interface.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from researchclaw.config import RCConfig
from researchclaw.pipeline.codegen.prompt_builder import CodegenPromptBuilder
from researchclaw.pipeline.codegen.session import CodegenSession
from researchclaw.pipeline.codegen.types import CodegenContext, CodegenPhase, CodegenResult

logger = logging.getLogger(__name__)


class AiderTodoStrategy:
    """Beast Mode code generation via Aider CLI TODO loop.

    Analogous to claw-code's tool execution where ``execute_tool``
    dispatches to a typed handler — here we dispatch to ``OpenHandsBridge``.
    """

    @property
    def name(self) -> str:
        return "aider_todo"

    def can_handle(self, ctx: CodegenContext, config: RCConfig) -> bool:
        oc_cfg = config.experiment.opencode
        if not oc_cfg.enabled:
            return False
        from researchclaw.pipeline.openhands_bridge import score_complexity, count_historical_failures
        hist_failures = count_historical_failures(ctx.run_dir) if ctx.run_dir else 0
        cplx = score_complexity(
            exp_plan=ctx.exp_plan,
            topic=ctx.topic,
            historical_failures=hist_failures,
            threshold=oc_cfg.complexity_threshold,
        )
        return cplx.recommendation == "beast_mode"

    def generate(
        self,
        ctx: CodegenContext,
        config: RCConfig,
        llm: Any,
        session: CodegenSession,
        prompts: Any | None = None,
    ) -> CodegenResult:
        from researchclaw.llm import resolve_provider_base_url
        from researchclaw.pipeline.executor import _extract_selected_repos
        from researchclaw.pipeline.openhands_bridge import (
            OpenHandsBridge,
            count_historical_failures,
            score_complexity,
        )

        oc_cfg = config.experiment.opencode
        session.log(CodegenPhase.GENERATE, "Beast Mode (Aider TODO loop) ENGAGED")
        t0 = time.monotonic()

        hist_failures = count_historical_failures(ctx.run_dir) if ctx.run_dir else 0
        cplx = score_complexity(
            exp_plan=ctx.exp_plan,
            topic=ctx.topic,
            historical_failures=hist_failures,
            threshold=oc_cfg.complexity_threshold,
        )

        if ctx.stage_dir is not None:
            (ctx.stage_dir / "complexity_analysis.json").write_text(
                json.dumps({
                    "score": cplx.score,
                    "signals": cplx.signals,
                    "recommendation": cplx.recommendation,
                    "reason": cplx.reason,
                    "threshold": oc_cfg.complexity_threshold,
                    "historical_failures": hist_failures,
                }, indent=2),
                encoding="utf-8",
            )

        oc_model = oc_cfg.model or config.llm.primary_model
        oc_base_url = resolve_provider_base_url(
            getattr(config.llm, "provider", "openai-compatible"),
            getattr(config.llm, "base_url", ""),
        )

        bridge = OpenHandsBridge(
            model=f"openai/{oc_model}" if "/" not in oc_model else oc_model,
            llm_base_url=oc_base_url,
            api_key_env=config.llm.api_key_env,
            api_key=getattr(config.llm, "api_key", "") or "",
            timeout_sec=oc_cfg.timeout_sec,
            max_retries=oc_cfg.max_retries,
        )

        guidance = CodegenPromptBuilder(ctx, config, prompts).build_full_hint()
        selected_repos = _extract_selected_repos(ctx.codebase_info)

        oc_result = bridge.generate(
            stage_dir=ctx.stage_dir,
            topic=ctx.topic,
            exp_plan=ctx.exp_plan,
            metric=ctx.metric,
            pkg_hint=guidance,
            extra_guidance=ctx.extra_guidance,
            time_budget_sec=ctx.time_budget_sec,
            codebases_dir=ctx.codebases_dir,
            datasets_dir=ctx.datasets_dir,
            checkpoints_dir=ctx.checkpoints_dir,
            selected_repos=selected_repos,
        )

        elapsed = time.monotonic() - t0

        if ctx.stage_dir is not None:
            (ctx.stage_dir / "beast_mode_log.json").write_text(
                json.dumps({
                    "success": oc_result.success,
                    "elapsed_sec": oc_result.elapsed_sec,
                    "files": list(oc_result.files.keys()),
                    "error": oc_result.error,
                    "complexity_score": cplx.score,
                    "model": oc_model,
                }, indent=2),
                encoding="utf-8",
            )

        if oc_result.success and oc_result.files:
            session.log(
                CodegenPhase.GENERATE,
                f"Beast Mode SUCCESS — {len(oc_result.files)} files in {elapsed:.1f}s",
            )
            return CodegenResult(
                files=oc_result.files,
                strategy_name=self.name,
                skip_review=True,
                elapsed_sec=elapsed,
            )

        session.log(
            CodegenPhase.GENERATE,
            f"Beast Mode FAILED ({oc_result.error or 'unknown'}) — will fallback",
        )
        return CodegenResult(
            files={},
            strategy_name=self.name,
            elapsed_sec=elapsed,
            error=oc_result.error or "Beast mode produced no files",
        )
