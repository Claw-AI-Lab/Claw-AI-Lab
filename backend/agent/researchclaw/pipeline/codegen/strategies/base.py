"""Strategy protocol for code generation.

Inspired by claw-code's tool dispatch pattern where each tool is
a ``ToolSpec`` with name/description/input_schema, and ``execute_tool``
dispatches to typed handlers via a central ``match`` on tool name.

Each CodegenStrategy has a ``name``, a ``can_handle`` predicate, and a
``generate`` method — analogous to a tool's schema check + execution.
"""

from __future__ import annotations

from typing import Any, Protocol

from researchclaw.config import RCConfig
from researchclaw.pipeline.codegen.session import CodegenSession
from researchclaw.pipeline.codegen.types import CodegenContext, CodegenResult


class CodegenStrategy(Protocol):
    """Protocol for a code generation strategy.

    Analogous to claw-code's tool protocol: each tool has a name,
    checks if it can handle the input, and then executes.
    """

    @property
    def name(self) -> str: ...

    def can_handle(self, ctx: CodegenContext, config: RCConfig) -> bool:
        """Return True if this strategy should be used for the given context."""
        ...

    def generate(
        self,
        ctx: CodegenContext,
        config: RCConfig,
        llm: Any,
        session: CodegenSession,
        prompts: Any | None = None,
    ) -> CodegenResult:
        """Generate experiment code files.

        Analogous to claw-code's ``execute_tool(name, input)`` — receives
        typed context, produces a typed result.
        """
        ...
