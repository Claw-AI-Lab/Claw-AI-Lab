"""Code generation runtime — the orchestration turn loop.

Inspired by claw-code's ``ConversationRuntime.run_turn()`` which executes:
  user → (assistant → tool →)* → final assistant

Our turn loop executes:
  context → routing → generate → (validate → repair →)* → review → finalize

Each phase is explicit, logged, and independently testable.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from researchclaw.adapters import AdapterBundle
from researchclaw.config import RCConfig
from researchclaw.pipeline.codegen.context import ContextAssembler
from researchclaw.pipeline.codegen.prompt_builder import CodegenPromptBuilder
from researchclaw.pipeline.codegen.registry import StrategyRegistry, build_default_registry
from researchclaw.pipeline.codegen.router import CodegenRouter
from researchclaw.pipeline.codegen.session import CodegenSession
from researchclaw.pipeline.codegen.strategies.fallback import FallbackStrategy
from researchclaw.pipeline.codegen.types import CodegenPhase, CodegenResult
from researchclaw.pipeline.codegen.validation.pipeline import ValidationPipeline
from researchclaw.pipeline.codegen.validation.review import run_code_review
from researchclaw.pipeline.executor import StageResult, _safe_json_loads, _utcnow_iso
from researchclaw.pipeline.stages import Stage, StageStatus

logger = logging.getLogger(__name__)


class CodegenRuntime:
    """Phase-based orchestration for S11 CODE_GENERATION.

    Analogous to claw-code's ``ConversationRuntime`` which:
    1. Builds an API request (system prompt + messages)
    2. Calls the API (streaming)
    3. Processes tool use in a loop
    4. Returns a TurnSummary

    Our phases:
    1. CONTEXT    — Build CodegenContext (like system prompt construction)
    2. LLM_SETUP  — Resolve coding model override
    3. ROUTING    — Score complexity, select strategy
    4. GENERATE   — Execute strategy (like tool execution)
    5. FALLBACK   — Numpy fallback if empty
    6. VALIDATE   — AST gates + auto-repair + LLM repair
    7. REVIEW     — LLM code review
    8. FINALIZE   — Write experiment/, experiment_spec.md, artifacts
    """

    def __init__(self, registry: StrategyRegistry | None = None) -> None:
        self._registry = registry or build_default_registry()

    def execute(
        self,
        stage_dir: Path,
        run_dir: Path,
        config: RCConfig,
        adapters: AdapterBundle,
        *,
        llm: Any | None = None,
        prompts: Any | None = None,
    ) -> StageResult:
        """Run the full code generation pipeline.

        This is the single entry point that replaces the monolithic
        ``_execute_code_generation()`` in executor.py.
        """
        session = CodegenSession(stage_dir=stage_dir)
        session.log(CodegenPhase.CONTEXT, "CodegenRuntime started")

        # ── Phase 1: CONTEXT ─────────────────────────────────────────────
        ctx = ContextAssembler(config, run_dir, stage_dir, prompts).build()
        session.log(
            CodegenPhase.CONTEXT,
            f"Context assembled: topic={ctx.topic[:60]!r}, "
            f"mode={ctx.mode}, metric={ctx.metric}",
        )

        # ── Phase 2: LLM_SETUP ──────────────────────────────────────────
        llm = self._resolve_coding_llm(llm, config)
        session.log(CodegenPhase.LLM_SETUP, "LLM client resolved")

        # ── Phase 3: ROUTING ─────────────────────────────────────────────
        router = CodegenRouter(self._registry.all())
        strategy = router.select(ctx, config, adapters, session)
        session.strategy_used = strategy.name

        # ── Phase 4: GENERATE ────────────────────────────────────────────
        result = strategy.generate(ctx, config, llm, session, prompts=prompts)
        files = result.files
        session.files = dict(files)

        # ── Phase 5: FALLBACK ────────────────────────────────────────────
        if not files:
            session.log(CodegenPhase.FALLBACK, "No files from strategy — using fallback")
            fallback = FallbackStrategy()
            fb_result = fallback.generate(ctx, config, llm, session, prompts=prompts)
            files = fb_result.files
            session.files = dict(files)
            session.strategy_used = f"{session.strategy_used}+fallback"

        # Write initial experiment directory
        exp_dir = stage_dir / "experiment"
        exp_dir.mkdir(parents=True, exist_ok=True)
        for fname, code in files.items():
            (exp_dir / fname).write_text(code, encoding="utf-8")

        # ── Phase 6: VALIDATE ────────────────────────────────────────────
        vp = ValidationPipeline(llm, prompts, stage_dir)
        files = vp.run(files, ctx, session)
        session.files = dict(files)

        # ── Phase 7: REVIEW ──────────────────────────────────────────────
        if llm is not None and not result.skip_review:
            files = run_code_review(files, ctx, config, llm, session, prompts=prompts)
            session.files = dict(files)
            # Write reviewed files
            for fname, code in files.items():
                (exp_dir / fname).write_text(code, encoding="utf-8")

        # Topic-experiment alignment check
        files = self._alignment_check(files, ctx, config, llm, session, prompts, exp_dir)
        session.files = dict(files)

        # Ablation distinctness check
        files = self._ablation_check(files, ctx, config, llm, session, prompts, exp_dir)
        session.files = dict(files)

        # ── Phase 8: FINALIZE ────────────────────────────────────────────
        artifacts = self._finalize(files, ctx, config, stage_dir, exp_dir, session)

        session.log(
            CodegenPhase.FINALIZE,
            f"Done: {len(files)} files, strategy={session.strategy_used}, "
            f"{session.llm_calls} LLM calls, {session.elapsed_sec():.1f}s",
        )
        session.persist()

        return StageResult(
            stage=Stage.CODE_GENERATION,
            status=StageStatus.DONE,
            artifacts=tuple(artifacts),
            evidence_refs=tuple(f"stage-10/{a}" for a in artifacts),
        )

    # ------------------------------------------------------------------
    # Phase helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_coding_llm(llm: Any | None, config: RCConfig) -> Any | None:
        """Swap to dedicated coding model if configured."""
        coding_model = getattr(config.llm, "coding_model", None) or ""
        if not coding_model or llm is None:
            return llm

        seen = {coding_model}
        full_fallbacks = []
        for m in [llm.config.primary_model] + list(llm.config.fallback_models):
            if m not in seen:
                seen.add(m)
                full_fallbacks.append(m)

        coding_cfg = dataclasses.replace(
            llm.config,
            primary_model=coding_model,
            fallback_models=full_fallbacks,
        )
        from researchclaw.llm.client import LLMClient
        new_llm = LLMClient(coding_cfg)
        fb_str = ", ".join(full_fallbacks)
        print(
            f"[CODE_GENERATION] Using coding model: {coding_model} "
            f"(fallbacks: {fb_str})",
            flush=True,
        )
        logger.info("S11 CODE_GENERATION: using coding model '%s'", coding_model)
        return new_llm

    def _alignment_check(
        self,
        files: dict[str, str],
        ctx: Any,
        config: RCConfig,
        llm: Any | None,
        session: CodegenSession,
        prompts: Any | None,
        exp_dir: Path,
    ) -> dict[str, str]:
        """Check topic-experiment alignment and regenerate if misaligned."""
        if llm is None:
            return files

        from researchclaw.pipeline.executor import (
            _chat_with_prompt,
            _extract_multi_file_blocks,
        )

        all_code = "\n\n".join(
            f"# --- {fname} ---\n{code}" for fname, code in files.items()
        )
        if len(all_code) > 8000:
            all_code = all_code[:8000] + "\n... [truncated]"

        align_prompt = (
            f"Research topic: {ctx.topic}\n\n"
            f"Experiment code:\n```python\n{all_code}\n```\n\n"
            "TASK: Evaluate whether this experiment code actually tests the "
            "stated research topic. Answer with JSON:\n"
            '{"aligned": true/false, "reason": "...", "suggestions": "..."}\n\n'
            "Check specifically:\n"
            "- Does the code implement models/methods relevant to the topic?\n"
            "- If the topic mentions LLMs/transformers/language models, does "
            "the code use or simulate them (not just small MLPs)?\n"
            "- If the topic mentions a specific technique (e.g. curriculum "
            "learning, RLHF), does the code actually implement it?\n"
            "- Are the experimental conditions meaningfully different from each other?\n"
        )

        try:
            resp = llm.chat(
                [{"role": "user", "content": align_prompt}],
                system="You are a scientific code reviewer checking topic-experiment alignment.",
                max_tokens=1024,
            )
            session.llm_calls += 1
            data = _safe_json_loads(resp.content, {})
            if isinstance(data, dict) and not data.get("aligned", True):
                reason = data.get("reason", "Misaligned")
                suggestions = data.get("suggestions", "")
                logger.warning("S11: Topic-experiment MISALIGNMENT: %s", reason)
                session.log(CodegenPhase.REVIEW, f"Alignment issue: {reason}")

                builder = CodegenPromptBuilder(ctx, config, prompts)
                hint = builder.build_full_hint()
                system_prompt = ""
                if prompts:
                    try:
                        system_prompt = prompts.prompts["code_generation"]["system"]
                    except (KeyError, AttributeError):
                        pass

                regen_prompt = (
                    f"The experiment code does NOT align with the research topic.\n\n"
                    f"TOPIC: {ctx.topic}\nMISALIGNMENT: {reason}\n"
                    f"SUGGESTIONS: {suggestions}\n\n"
                    f"REGENERATE the code to DIRECTLY test the stated topic.\n\n"
                    f"{hint}\nPLAN:\n{ctx.exp_plan}\n\n"
                    f"Return files using ```filename:xxx.py format.\n"
                    f"Do NOT use try/except blocks."
                )
                regen_resp = _chat_with_prompt(llm, system_prompt, regen_prompt, max_tokens=8192)
                session.llm_calls += 1
                regen_files = _extract_multi_file_blocks(regen_resp.content)
                if regen_files and "main.py" in regen_files:
                    for fname, code in regen_files.items():
                        (exp_dir / fname).write_text(code, encoding="utf-8")
                    session.log(CodegenPhase.REVIEW, "Code regenerated after alignment fix")
                    return regen_files
        except Exception as exc:
            logger.debug("Alignment check failed: %s", exc)

        return files

    def _ablation_check(
        self,
        files: dict[str, str],
        ctx: Any,
        config: RCConfig,
        llm: Any | None,
        session: CodegenSession,
        prompts: Any | None,
        exp_dir: Path,
    ) -> dict[str, str]:
        """Check ablation condition distinctness."""
        main_code = files.get("main.py", "")
        if llm is None or not main_code or "condition" not in main_code.lower():
            return files

        from researchclaw.pipeline.executor import (
            _chat_with_prompt,
            _extract_multi_file_blocks,
        )

        try:
            abl_prompt = (
                f"Examine this experiment code:\n```python\n{main_code[:6000]}\n```\n\n"
                "Check if any experimental conditions have IDENTICAL configurations. "
                "Answer JSON: "
                '{"has_duplicates": true/false, "details": "which conditions are identical"}'
            )
            resp = llm.chat(
                [{"role": "user", "content": abl_prompt}],
                system="You are a code reviewer checking experimental conditions.",
                max_tokens=512,
            )
            session.llm_calls += 1
            data = _safe_json_loads(resp.content, {})
            if isinstance(data, dict) and data.get("has_duplicates"):
                details = data.get("details", "")
                logger.warning("S11: Duplicate ablation conditions: %s", details)

                if ctx.stage_dir is not None:
                    (ctx.stage_dir / "ablation_warning.json").write_text(
                        json.dumps(data, indent=2), encoding="utf-8",
                    )

                all_code_ctx = "\n\n".join(
                    f"```filename:{f}\n{c}\n```" for f, c in files.items()
                )
                system_prompt = ""
                if prompts:
                    try:
                        system_prompt = prompts.prompts["code_generation"]["system"]
                    except (KeyError, AttributeError):
                        pass

                repair_prompt = (
                    f"ABLATION REPAIR REQUIRED — duplicate conditions:\n{details}\n\n"
                    "Rewrite so each condition is GENUINELY DIFFERENT.\n"
                    "Return ALL files using ```filename:xxx.py format.\n"
                    f"Do NOT use try/except blocks.\n\nCurrent code:\n{all_code_ctx}\n"
                )
                try:
                    repair_resp = _chat_with_prompt(llm, system_prompt, repair_prompt, max_tokens=8192)
                    session.llm_calls += 1
                    repaired = _extract_multi_file_blocks(repair_resp.content)
                    if repaired and "main.py" in repaired:
                        for fname, code in repaired.items():
                            (exp_dir / fname).write_text(code, encoding="utf-8")
                        session.log(CodegenPhase.REVIEW, "Ablation repair applied")
                        return repaired
                except Exception as exc:
                    logger.debug("Ablation repair failed: %s", exc)
        except Exception as exc:
            logger.debug("Ablation check skipped: %s", exc)

        return files

    @staticmethod
    def _finalize(
        files: dict[str, str],
        ctx: Any,
        config: RCConfig,
        stage_dir: Path,
        exp_dir: Path,
        session: CodegenSession,
    ) -> list[str]:
        """Write experiment_spec.md and collect artifact list."""
        from researchclaw.experiment.validator import validate_code

        session.log(CodegenPhase.FINALIZE, "Writing experiment spec and artifacts")

        file_list = ", ".join(f"`{f}`" for f in sorted(files.keys()))
        main_validation = validate_code(files.get("main.py", ""))

        spec = f"""# Experiment Specification

## Topic
{ctx.topic}

## Project Structure
Multi-file experiment project with {len(files)} file(s): {file_list}

## Entry Point
`main.py` — executed directly via sandbox

## Outputs
- `main.py` emits metric lines in `name: value` format
- Primary metric key: `{ctx.metric}`

## Constraints
- Time budget per run: {config.experiment.time_budget_sec}s
- Max iterations: {config.experiment.max_iterations}
- {"Uses local data/codebases from project config" if (ctx.datasets_dir or ctx.codebases_dir) else "Self-contained execution (no external data, no network)"}
- Validated: {main_validation.summary()}

## Strategy
{session.strategy_used} ({session.llm_calls} LLM calls, {session.sandbox_runs} sandbox runs)

## Generated
{_utcnow_iso()}
"""
        (stage_dir / "experiment_spec.md").write_text(spec, encoding="utf-8")

        artifacts = ["experiment/", "experiment_spec.md"]
        if (stage_dir / "validation_report.md").exists():
            artifacts.append("validation_report.md")
        if (stage_dir / "code_review.json").exists():
            artifacts.append("code_review.json")
        if (stage_dir / "codegen_session.json").exists():
            artifacts.append("codegen_session.json")

        session.add_artifact("experiment/")
        session.add_artifact("experiment_spec.md")

        return artifacts
