"""Strategy registry for code generation.

Inspired by claw-code's ``ExecutionRegistry`` which provides unified
lookup for commands and tools by name. StrategyRegistry registers
strategy implementations and builds the default set.
"""

from __future__ import annotations

import logging

from researchclaw.pipeline.codegen.strategies.base import CodegenStrategy

logger = logging.getLogger(__name__)


class StrategyRegistry:
    """Registry of available code generation strategies.

    Analogous to claw-code's ``build_execution_registry()`` which
    populates ``MirroredCommand`` and ``MirroredTool`` entries from
    snapshot data. Here we populate from concrete strategy classes.
    """

    def __init__(self) -> None:
        self._strategies: list[CodegenStrategy] = []

    def register(self, strategy: CodegenStrategy) -> None:
        self._strategies.append(strategy)
        logger.debug("Registered codegen strategy: %s", strategy.name)

    def all(self) -> list[CodegenStrategy]:
        return list(self._strategies)

    def get(self, name: str) -> CodegenStrategy | None:
        for s in self._strategies:
            if s.name == name:
                return s
        return None


def build_default_registry() -> StrategyRegistry:
    """Build the strategy registry with the claw-code agentic strategy.

    The ClawAgentStrategy is the sole code generation path — it uses
    claw-code's turn loop pattern where the LLM iteratively calls tools
    (bash, read_file, write_file, edit_file, glob_search, grep_search)
    to generate experiment code.

    FallbackStrategy is handled by the runtime if claw_agent produces
    no files — it does not need to be registered here.
    """
    from researchclaw.pipeline.codegen.strategies.claw_agent import ClawAgentStrategy

    registry = StrategyRegistry()
    registry.register(ClawAgentStrategy())
    return registry
