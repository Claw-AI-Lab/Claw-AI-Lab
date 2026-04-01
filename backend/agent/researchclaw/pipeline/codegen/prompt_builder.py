"""Fluent prompt builder for code generation.

Directly inspired by claw-code's ``SystemPromptBuilder`` which constructs
system prompts via chained ``.with_os().with_project_context()
.with_runtime_config().build()`` calls, with section ordering and a
dynamic boundary marker separating static rules from runtime context.

CodegenPromptBuilder follows the same pattern: each ``.with_*()`` method
appends a named section, and ``.build()`` joins them into the final
``pkg_hint`` string consumed by strategy implementations.
"""

from __future__ import annotations

import logging
from typing import Any

from researchclaw.pipeline.codegen.types import CodegenContext

logger = logging.getLogger(__name__)


class CodegenPromptBuilder:
    """Build the combined prompt guidance string for code generation.

    Usage::

        guidance = (CodegenPromptBuilder(ctx, config, prompts)
            .with_context_sections()
            .build())

    Each section is independently optional — if the relevant data is
    missing, the section is silently skipped.
    """

    def __init__(
        self,
        ctx: CodegenContext,
        config: Any,
        prompts: Any | None = None,
    ) -> None:
        self._ctx = ctx
        self._config = config
        self._pm = prompts
        self._sections: list[tuple[str, str]] = []

    def _append(self, name: str, content: str) -> CodegenPromptBuilder:
        if content.strip():
            self._sections.append((name, content))
        return self

    # ------------------------------------------------------------------
    # Section builders (claw-code pattern: each is a .with_*() method)
    # ------------------------------------------------------------------

    def with_packages(self) -> CodegenPromptBuilder:
        """Package availability hint (sandbox/docker)."""
        return self._append("packages", self._ctx.pkg_hint)

    def with_compute_budget(self) -> CodegenPromptBuilder:
        """Time budget and compute constraints."""
        return self._append("compute_budget", self._ctx.compute_budget)

    def with_extra_guidance(self) -> CodegenPromptBuilder:
        """All extra guidance sections assembled by ContextAssembler."""
        return self._append("extra_guidance", self._ctx.extra_guidance)

    def with_evolution_overlay(self) -> CodegenPromptBuilder:
        """Lessons learned from prior pipeline runs."""
        if self._ctx.run_dir is None:
            return self
        try:
            from researchclaw.pipeline.executor import _get_evolution_overlay
            overlay = _get_evolution_overlay(self._ctx.run_dir, "code_generation")
            return self._append("evolution", overlay)
        except Exception:
            return self

    def with_metric_direction(self) -> CodegenPromptBuilder:
        """Metric direction hint for the code generation prompt."""
        md = self._ctx.metric_direction
        direction = "lower" if md == "minimize" else "higher"
        hint = (
            f"`{md}` — use direction='{direction}' "
            f"in METRIC_DEF. You MUST NOT use the opposite direction."
        )
        return self._append("metric_direction", hint)

    def with_all(self) -> CodegenPromptBuilder:
        """Convenience: apply all standard sections in order.

        Analogous to claw-code's ``load_system_prompt()`` which chains
        all the ``.with_*()`` calls in the canonical order.
        """
        return (
            self
            .with_packages()
            .with_compute_budget()
            .with_extra_guidance()
            .with_evolution_overlay()
            .with_metric_direction()
        )

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self) -> str:
        """Join all sections into the final guidance string.

        Analogous to claw-code's ``SystemPromptBuilder.build()`` which
        joins intro, system, doing-tasks, dynamic-boundary, environment,
        project, instructions, and config sections with newlines.
        """
        parts: list[str] = []
        for _name, content in self._sections:
            parts.append(content)
        return "\n".join(parts)

    def build_full_hint(self) -> str:
        """Build the combined pkg_hint + compute_budget + extra_guidance.

        This is the single string passed to strategies as their guidance
        input — equivalent to the old ``pkg_hint + compute_budget + extra_guidance``
        concatenation in the monolithic executor function.
        """
        return (
            self._ctx.pkg_hint + "\n"
            + self._ctx.compute_budget + "\n"
            + self._ctx.extra_guidance
        )
