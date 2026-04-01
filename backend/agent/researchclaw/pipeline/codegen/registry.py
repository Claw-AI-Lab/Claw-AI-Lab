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
    """Build the standard strategy registry with all built-in strategies.

    Priority order matters — the router tries them in registration order.
    """
    from researchclaw.pipeline.codegen.strategies.aider_todo import AiderTodoStrategy
    from researchclaw.pipeline.codegen.strategies.blueprint import BlueprintStrategy
    from researchclaw.pipeline.codegen.strategies.single_shot import SingleShotStrategy

    registry = StrategyRegistry()
    registry.register(AiderTodoStrategy())
    registry.register(BlueprintStrategy())
    registry.register(SingleShotStrategy())
    return registry
