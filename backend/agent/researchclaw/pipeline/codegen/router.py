"""Strategy router for code generation.

Simplified from the previous multi-strategy routing: now there is only
one strategy (ClawAgentStrategy) which always handles all requests.
The router still exists for interface compatibility with CodegenRuntime.
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
    """Select the code generation strategy.

    With the claw-code agentic rewrite, there is only one strategy:
    ClawAgentStrategy. The router simply picks the first (and only)
    registered strategy that can handle the request.
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
        """Select the code generation strategy.

        With the claw-code agent as the sole strategy, this always
        returns it. The fallback (numpy) is handled by the runtime
        if the strategy produces no files.
        """
        for strategy in self._strategies:
            if strategy.can_handle(ctx, config):
                session.log(
                    CodegenPhase.ROUTING,
                    f"Selected strategy: {strategy.name}",
                )
                return strategy

        from researchclaw.pipeline.codegen.strategies.fallback import FallbackStrategy
        session.log(CodegenPhase.ROUTING, "No strategy available — using fallback")
        return FallbackStrategy()
