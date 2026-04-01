"""Blueprint strategy: multi-phase CodeAgent code generation.

Wraps ``CodeAgent`` from ``code_agent.py`` via composition. The existing
agent handles blueprint planning, sequential generation, exec-fix,
tree search, and review — this strategy adapts the interface.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

from researchclaw.config import RCConfig
from researchclaw.pipeline.codegen.prompt_builder import CodegenPromptBuilder
from researchclaw.pipeline.codegen.session import CodegenSession
from researchclaw.pipeline.codegen.types import CodegenContext, CodegenPhase, CodegenResult

logger = logging.getLogger(__name__)


class BlueprintStrategy:
    """Multi-phase code generation: blueprint → sequential → exec-fix → review.

    Wraps the existing ``CodeAgent`` via composition.
    """

    @property
    def name(self) -> str:
        return "blueprint"

    def can_handle(self, ctx: CodegenContext, config: RCConfig) -> bool:
        return config.experiment.code_agent.enabled

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
                error="No LLM client available for CodeAgent",
            )

        from researchclaw.pipeline.code_agent import CodeAgent

        session.log(CodegenPhase.GENERATE, "Blueprint (CodeAgent) strategy started")
        t0 = time.monotonic()

        ca_cfg = config.experiment.code_agent

        sandbox_factory = None
        if config.experiment.mode in ("sandbox", "docker"):
            try:
                from researchclaw.experiment.factory import create_sandbox
                sandbox_factory = create_sandbox
            except ImportError:
                pass

        domain_profile = self._detect_domain(ctx)
        code_search_result = self._run_code_search(ctx, config, llm, domain_profile)

        agent = CodeAgent(
            llm=llm,
            prompts=prompts,
            config=ca_cfg,
            stage_dir=ctx.stage_dir,
            sandbox_factory=sandbox_factory,
            experiment_config=config.experiment,
            domain_profile=domain_profile,
            code_search_result=code_search_result,
        )

        guidance = CodegenPromptBuilder(ctx, config, prompts).build_full_hint()

        max_tokens = 8192
        if any(config.llm.primary_model.startswith(p) for p in ("gpt-5", "o3", "o4")):
            max_tokens = 16384

        agent_result = agent.generate(
            topic=ctx.topic,
            exp_plan=ctx.exp_plan,
            metric=ctx.metric,
            pkg_hint=guidance,
            max_tokens=max_tokens,
        )

        elapsed = time.monotonic() - t0

        if ctx.stage_dir is not None:
            (ctx.stage_dir / "code_agent_log.json").write_text(
                json.dumps({
                    "log": agent_result.validation_log,
                    "llm_calls": agent_result.total_llm_calls,
                    "sandbox_runs": agent_result.total_sandbox_runs,
                    "best_score": agent_result.best_score,
                    "tree_nodes_explored": agent_result.tree_nodes_explored,
                    "review_rounds": agent_result.review_rounds,
                }, indent=2),
                encoding="utf-8",
            )
            if agent_result.architecture_spec:
                (ctx.stage_dir / "architecture_spec.yaml").write_text(
                    agent_result.architecture_spec, encoding="utf-8",
                )

        session.llm_calls += agent_result.total_llm_calls
        session.sandbox_runs += agent_result.total_sandbox_runs
        session.best_score = agent_result.best_score
        session.tree_nodes_explored = agent_result.tree_nodes_explored
        session.review_rounds += agent_result.review_rounds

        session.log(
            CodegenPhase.GENERATE,
            f"CodeAgent done: {agent_result.total_llm_calls} LLM calls, "
            f"{agent_result.total_sandbox_runs} sandbox runs, "
            f"score={agent_result.best_score:.2f}",
        )

        return CodegenResult(
            files=agent_result.files,
            strategy_name=self.name,
            skip_review=True,
            elapsed_sec=elapsed,
        )

    @staticmethod
    def _detect_domain(ctx: CodegenContext) -> Any | None:
        try:
            from researchclaw.domains.detector import detect_domain
            return detect_domain(topic=ctx.topic)
        except Exception:
            return None

    @staticmethod
    def _run_code_search(
        ctx: CodegenContext,
        config: RCConfig,
        llm: Any,
        domain_profile: Any | None,
    ) -> Any | None:
        if domain_profile is None:
            return None
        try:
            from researchclaw.domains.detector import is_ml_domain
            has_local_codebases = (
                bool(ctx.codebases_dir)
                and os.path.isdir(ctx.codebases_dir)
                and any(
                    os.path.isdir(os.path.join(ctx.codebases_dir, d))
                    for d in os.listdir(ctx.codebases_dir)
                    if not d.startswith(".")
                )
            )
            if not is_ml_domain(domain_profile) or has_local_codebases:
                from researchclaw.agents.code_searcher import CodeSearchAgent
                cs_agent = CodeSearchAgent(llm=llm)
                result = cs_agent.search(topic=ctx.topic, domain=domain_profile)
                if result and result.patterns.has_content:
                    return result
        except Exception:
            logger.debug("Code search unavailable", exc_info=True)
        return None
