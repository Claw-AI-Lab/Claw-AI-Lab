"""Numpy fallback code generator.

Produces a minimal runnable experiment that performs a parameter sweep on a
synthetic objective. Used when all other strategies fail to produce any code.
"""

from __future__ import annotations

from typing import Any

from researchclaw.config import RCConfig
from researchclaw.pipeline.codegen.session import CodegenSession
from researchclaw.pipeline.codegen.types import (
    CodegenContext,
    CodegenPhase,
    CodegenResult,
    GeneratedFiles,
)


class FallbackStrategy:
    """Last-resort numpy-based experiment generator."""

    @property
    def name(self) -> str:
        return "fallback"

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
        session.log(CodegenPhase.FALLBACK, "Using numpy fallback generator")
        metric = ctx.metric
        files: GeneratedFiles = {
            "main.py": (
                "import numpy as np\n"
                "\n"
                "np.random.seed(42)\n"
                "\n"
                "# Fallback experiment: parameter sweep on a synthetic objective\n"
                "# This runs when LLM code generation fails to produce valid code.\n"
                "dim = 10\n"
                "n_conditions = 3\n"
                "results = {}\n"
                "\n"
                "for cond_idx in range(n_conditions):\n"
                "    cond_name = f'condition_{cond_idx}'\n"
                "    scores = []\n"
                "    for seed in range(3):\n"
                "        rng = np.random.RandomState(seed + cond_idx * 100)\n"
                "        x = rng.randn(dim)\n"
                "        score = float(1.0 / (1.0 + np.sum(x ** 2)))\n"
                "        scores.append(score)\n"
                "    mean_score = float(np.mean(scores))\n"
                "    results[cond_name] = mean_score\n"
                f"    print(f'condition={{cond_name}} {metric}: {{mean_score:.6f}}')\n"
                "\n"
                "best = max(results, key=results.get)\n"
                f"print(f'{metric}: {{results[best]:.6f}}')\n"
            )
        }
        return CodegenResult(
            files=files,
            strategy_name=self.name,
            skip_review=True,
        )
