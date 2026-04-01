"""Strategy router for code generation.

Inspired by claw-code's ``PortRuntime.route_prompt()`` which scores tokens
against command/tool modules and selects the best match. CodegenRouter
scores experiment complexity and config flags to select the right strategy.
"""

from __future__ import annotations

import logging
from typing import Any

from researchclaw.config import RCConfig
from researchclaw.pipeline.codegen.session import CodegenSession
from researchclaw.pipeline.codegen.strategies.base import CodegenStrategy
from researchclaw.pipeline.codegen.types import CodegenContext, CodegenPhase

logger = logging.getLogger(__name__)


class CodegenRouter:
    """Select the best code generation strategy for the given context.

    Strategy priority (analogous to claw-code's routing where commands
    are checked before tools, with the best-scoring match selected):

    1. **Beast Mode** (Aider) — if enabled and complexity exceeds threshold
    2. **Blueprint** (CodeAgent) — if enabled
    3. **Single-shot** — always available as fallback
    """

    def __init__(self, strategies: list[CodegenStrategy]) -> None:
        self._strategies = strategies

    def select(
        self,
        ctx: CodegenContext,
        config: RCConfig,
        adapters: Any,
        session: CodegenSession,
    ) -> CodegenStrategy:
        """Route to the best strategy.

        Tries strategies in priority order (Beast Mode > Blueprint > Single-shot).
        The first one whose ``can_handle()`` returns True is selected.

        For Beast Mode, respects the ``auto`` flag: if not auto, checks
        HITL adapter for confirmation.
        """
        for strategy in self._strategies:
            if strategy.can_handle(ctx, config):
                # Beast Mode needs HITL confirmation when not auto
                if strategy.name == "aider_todo":
                    oc_cfg = config.experiment.opencode
                    if not oc_cfg.auto:
                        if not self._confirm_beast_mode(adapters, ctx, config):
                            session.log(
                                CodegenPhase.ROUTING,
                                f"Beast Mode skipped (HITL declined or unavailable)",
                            )
                            continue

                session.log(
                    CodegenPhase.ROUTING,
                    f"Selected strategy: {strategy.name}",
                )
                return strategy

        # Should never reach here — single_shot always can_handle
        from researchclaw.pipeline.codegen.strategies.single_shot import SingleShotStrategy
        return SingleShotStrategy()

    @staticmethod
    def _confirm_beast_mode(adapters: Any, ctx: CodegenContext, config: RCConfig) -> bool:
        """Check HITL adapter for Beast Mode confirmation."""
        hitl = getattr(adapters, "hitl", None)
        if hitl is None:
            logger.info("Beast mode: no HITL adapter, skipping")
            return False
        try:
            from researchclaw.pipeline.openhands_bridge import score_complexity, count_historical_failures
            hist = count_historical_failures(ctx.run_dir) if ctx.run_dir else 0
            cplx = score_complexity(
                exp_plan=ctx.exp_plan,
                topic=ctx.topic,
                historical_failures=hist,
                threshold=config.experiment.opencode.complexity_threshold,
            )
            return hitl.confirm(
                f"Beast Mode: complexity={cplx.score:.2f} "
                f"(threshold={config.experiment.opencode.complexity_threshold}). "
                f"Route to Aider?"
            )
        except Exception:
            logger.info("Beast mode: HITL adapter unavailable")
            return False
