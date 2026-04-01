"""Legacy single-shot code generation strategy.

Uses a single LLM call with the ``code_generation`` stage prompt to
produce experiment code. Falls back to a retry with higher token budget
if the first call returns empty.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from researchclaw.config import RCConfig
from researchclaw.pipeline.codegen.prompt_builder import CodegenPromptBuilder
from researchclaw.pipeline.codegen.session import CodegenSession
from researchclaw.pipeline.codegen.types import CodegenContext, CodegenPhase, CodegenResult

logger = logging.getLogger(__name__)


class SingleShotStrategy:
    """Single LLM call code generation — the legacy path."""

    @property
    def name(self) -> str:
        return "single_shot"

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
                error="No LLM client available for single-shot generation",
            )

        from researchclaw.pipeline.executor import (
            _chat_with_prompt,
            _extract_multi_file_blocks,
            _get_evolution_overlay,
        )
        from researchclaw.prompts import PromptManager

        session.log(CodegenPhase.GENERATE, "Single-shot LLM generation started")
        t0 = time.monotonic()

        pm = prompts or PromptManager()

        builder = CodegenPromptBuilder(ctx, config, prompts)
        full_hint = builder.build_full_hint()
        md = ctx.metric_direction
        md_hint = (
            f"`{md}` — use direction={'lower' if md == 'minimize' else 'higher'} "
            f"in METRIC_DEF. You MUST NOT use the opposite direction."
        )

        overlay = ""
        if ctx.run_dir is not None:
            try:
                overlay = _get_evolution_overlay(ctx.run_dir, "code_generation")
            except Exception:
                pass

        sp = pm.for_stage(
            "code_generation",
            evolution_overlay=overlay,
            topic=ctx.topic,
            metric=ctx.metric,
            pkg_hint=full_hint,
            exp_plan=ctx.exp_plan,
            metric_direction_hint=md_hint,
        )

        max_tokens = sp.max_tokens or 8192
        if any(config.llm.primary_model.startswith(p) for p in ("gpt-5", "o3", "o4")):
            max_tokens = max(max_tokens, 16384)

        resp = _chat_with_prompt(
            llm, sp.system, sp.user,
            json_mode=sp.json_mode,
            max_tokens=max_tokens,
        )
        session.llm_calls += 1

        files = _extract_multi_file_blocks(resp.content)

        if not files and not resp.content.strip():
            logger.warning(
                "R13-3: Empty LLM response for code_generation (len=%d, "
                "finish_reason=%s, tokens=%d). Retrying with 32768 tokens.",
                len(resp.content),
                resp.finish_reason,
                resp.total_tokens,
            )
            resp = _chat_with_prompt(
                llm, sp.system, sp.user,
                json_mode=sp.json_mode,
                max_tokens=32768,
            )
            session.llm_calls += 1
            files = _extract_multi_file_blocks(resp.content)

        if not files:
            logger.warning(
                "R13-2: _extract_multi_file_blocks returned empty. "
                "LLM response length=%d, first 300 chars: %s",
                len(resp.content),
                resp.content[:300],
            )

        elapsed = time.monotonic() - t0
        session.log(
            CodegenPhase.GENERATE,
            f"Single-shot done: {len(files)} files in {elapsed:.1f}s",
        )

        return CodegenResult(
            files=files,
            strategy_name=self.name,
            skip_review=False,
            elapsed_sec=elapsed,
        )
