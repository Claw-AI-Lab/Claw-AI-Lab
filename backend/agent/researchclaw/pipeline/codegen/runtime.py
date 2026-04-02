"""Code generation runtime — claw-code agentic orchestration.

The LLM generates code by iteratively calling tools (bash, read_file,
write_file, edit_file, glob_search, grep_search) in a turn loop —
it writes code, runs it, reads errors, and fixes them autonomously.

No separate validation/repair/review phases are needed because the
agent handles all of that internally via tool calls.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path
from typing import Any

from researchclaw.adapters import AdapterBundle
from researchclaw.config import RCConfig
from researchclaw.pipeline.codegen.registry import StrategyRegistry, build_default_registry
from researchclaw.pipeline.codegen.router import CodegenRouter
from researchclaw.pipeline.codegen.session import CodegenSession
from researchclaw.pipeline.codegen.strategies.fallback import FallbackStrategy
from researchclaw.pipeline.codegen.types import CodegenContext, CodegenPhase, CodegenResult, DiscoveredData, HardwareProfile
from researchclaw.pipeline.executor import StageResult, _utcnow_iso
from researchclaw.pipeline.stages import Stage, StageStatus

logger = logging.getLogger(__name__)


def _discover_data(
    checkpoints_dir: str,
    datasets_dir: str,
    codebases_dir: str,
    session: Any,
) -> DiscoveredData:
    """Pre-discover filesystem context before prompt building.

    Analogous to claw-code's ``ProjectContext.discover_with_git()`` which
    reads git status and CLAUDE.md files BEFORE the SystemPromptBuilder
    consumes them. We read model_index.json, directory listings, and
    sample data files so the LLM has ground truth in its system prompt.
    """
    import json as _json
    from researchclaw.pipeline.codegen.types import DiscoveredData

    d = DiscoveredData()

    # ── Discover checkpoint model type ──
    if checkpoints_dir:
        ckpt_path = Path(checkpoints_dir)
        if ckpt_path.is_dir():
            # Search for model_index.json in checkpoint dir and subdirs
            for candidate in [
                ckpt_path / "model_index.json",
                *sorted(ckpt_path.glob("*/model_index.json")),
                *sorted(ckpt_path.glob("*/*/model_index.json")),
            ]:
                if candidate.is_file():
                    try:
                        raw = candidate.read_text(encoding="utf-8")
                        idx = _json.loads(raw)
                        d.checkpoint_model_index = idx
                        d.checkpoint_class_name = idx.get("_class_name", "")
                        # Store the raw JSON (truncated) so the LLM can
                        # see the full model structure and component names
                        d.checkpoint_model_index_raw = raw[:3000]
                        session.log(
                            CodegenPhase.CONTEXT,
                            f"Discovered model_index.json: _class_name={d.checkpoint_class_name!r} "
                            f"at {candidate}",
                        )
                        break
                    except Exception:
                        pass

            # List top-level checkpoint files
            try:
                d.checkpoint_files = sorted(
                    f.name for f in ckpt_path.iterdir()
                    if not f.name.startswith(".")
                )[:30]
            except OSError:
                pass

    # ── Discover dataset structure ──
    if datasets_dir:
        ds_path = Path(datasets_dir)
        if ds_path.is_dir():
            try:
                d.dataset_files = sorted(
                    f.name for f in ds_path.iterdir()
                    if not f.name.startswith(".")
                )[:30]
            except OSError:
                pass

            # Read first few lines of any .txt file as a sample
            for txt in sorted(ds_path.glob("*.txt"))[:1]:
                try:
                    lines = txt.read_text(encoding="utf-8").splitlines()[:5]
                    d.dataset_sample = f"Sample from {txt.name}:\n" + "\n".join(lines)
                except OSError:
                    pass

    # ── Discover codebase structure ──
    if codebases_dir:
        cb_path = Path(codebases_dir)
        if cb_path.is_dir():
            try:
                d.codebase_files = sorted(
                    str(f.relative_to(cb_path))
                    for f in cb_path.rglob("*.py")
                    if not any(p.startswith(".") or p == "__pycache__" for p in f.relative_to(cb_path).parts)
                )[:50]
            except OSError:
                pass

            # Read README if present
            for readme in [cb_path / "README.md", *cb_path.glob("*/README.md")]:
                if readme.is_file():
                    try:
                        text = readme.read_text(encoding="utf-8")
                        d.codebase_readme = text[:2000]
                        break
                    except OSError:
                        pass

    return d


def generate_codegen_md(ctx: CodegenContext, target_path: Path) -> str:
    """Auto-generate CODEGEN.md content from experiment plan and discovered data.

    Mirrors claw-code's CLAUDE.md pattern: project-specific instructions
    scoped to ONE project. Written directly into the agent workspace so
    the agent reads it on-demand via read_file (not injected into the
    system prompt which would bloat every LLM call).

    Returns the generated content (also written to target_path).
    """
    sections: list[str] = ["# Auto-generated project instructions for S11 code generation\n"]

    plan_dict: dict = {}
    if ctx.exp_plan:
        try:
            import yaml
            plan_dict = yaml.safe_load(ctx.exp_plan) or {}
        except Exception:
            pass

    # ── Model loading instructions ──
    model_info = plan_dict.get("model", {})
    if model_info:
        sections.append("## Model")
        name = model_info.get("name", "")
        path = model_info.get("path", "")
        pipeline_class = model_info.get("pipeline_class", "")
        load_code = model_info.get("load_code", "").strip()
        if name:
            sections.append(f"- Name: `{name}`")
        if path:
            sections.append(f"- Path: `{path}`")
        if pipeline_class:
            sections.append(f"- Pipeline class: `{pipeline_class}`")
            sections.append(
                f"- Use `from diffusers import {pipeline_class}` — do NOT use a generic/wrong class"
            )
        if load_code:
            sections.append(f"- Load code:\n```python\n{load_code}\n```")

        components = model_info.get("components", {})
        if components:
            sections.append("- Components:")
            for comp_name, comp_info in components.items():
                cls = comp_info.get("class", "") if isinstance(comp_info, dict) else ""
                if cls:
                    sections.append(f"  - `{comp_name}`: `{cls}`")
                    lora_targets = comp_info.get("lora_target_modules", [])
                    if lora_targets:
                        sections.append(f"    - LoRA target modules: `{lora_targets}`")
    elif ctx.discovered.checkpoint_class_name:
        sections.append("## Model (discovered from checkpoint)")
        sections.append(f"- Discovered `_class_name`: `{ctx.discovered.checkpoint_class_name}`")
        sections.append(
            "- Use `DiffusionPipeline.from_pretrained()` which auto-detects the correct class"
        )

    # ── Methods/conditions ──
    methods = plan_dict.get("methods", {})
    if methods:
        sections.append("\n## Experimental conditions")
        for method_name, method_info in methods.items():
            if not isinstance(method_info, dict):
                continue
            desc = method_info.get("description", "")
            sections.append(f"- **{method_name}**: {desc}")

    # ── Dataset ──
    dataset_info = plan_dict.get("dataset", {})
    if dataset_info:
        sections.append("\n## Dataset")
        ds_name = dataset_info.get("name", "")
        ds_source = dataset_info.get("source", "")
        ds_format = dataset_info.get("format", "").strip()
        if ds_name:
            sections.append(f"- Name: `{ds_name}`")
        if ds_source:
            sections.append(f"- Path: `{ds_source}`")
        subsets = dataset_info.get("subsets", [])
        for subset in subsets:
            if isinstance(subset, dict):
                sub_name = subset.get("name", "")
                sections.append(f"- Subset `{sub_name}`:")
                for key in ("train_file", "test_file", "first_frames_dir"):
                    val = subset.get(key)
                    if val:
                        sections.append(f"  - {key}: `{val}`")
        if ds_format:
            sections.append(f"- Format: {ds_format[:300]}")

    sections.append("\n## Absolute path usage (CRITICAL)")
    sections.append(
        "- When exploring datasets, checkpoints, or codebases, use the REAL absolute path via the tool `path` field."
    )
    sections.append(
        "- Do NOT assume workspace-relative symlinks like `datasets/` or `checkpoints/` are the source of truth."
    )
    sections.append(
        "- If a configured absolute path exists, prefer `glob_search(path=ABS_PATH, pattern=\"**/*\")` and `read_file(path=\"/abs/path/to/file\")`."
    )
    if ctx.datasets_dir:
        sections.append(f"- DATASETS_DIR: `{ctx.datasets_dir}`")
        sections.append(
            f"- Example exploration call: `glob_search(path=\"{ctx.datasets_dir}\", pattern=\"**/*\")`"
        )
    if ctx.checkpoints_dir:
        sections.append(f"- CHECKPOINTS_DIR: `{ctx.checkpoints_dir}`")
        sections.append(
            f"- Example exploration call: `glob_search(path=\"{ctx.checkpoints_dir}\", pattern=\"**/*\")`"
        )
    if ctx.codebases_dir:
        sections.append(f"- CODEBASES_DIR: `{ctx.codebases_dir}`")
        sections.append(
            f"- Example exploration call: `glob_search(path=\"{ctx.codebases_dir}\", pattern=\"**/*.py\")`"
        )

    # ── Evaluation ──
    eval_info = plan_dict.get("evaluation", {})
    if eval_info:
        sections.append("\n## Evaluation")
        primary = eval_info.get("primary_metric", "")
        direction = eval_info.get("direction", "")
        secondary = eval_info.get("secondary_metrics", [])
        test_protocol = eval_info.get("test_protocol", "").strip()
        if primary:
            sections.append(f"- Primary metric: `{primary}` (direction: {direction})")
        if secondary:
            sections.append(f"- Secondary metrics: {', '.join(f'`{m}`' for m in secondary)}")
        if test_protocol:
            sections.append(f"- Test protocol:\n{test_protocol[:500]}")

    sections.append("\n## Execution modes")
    sections.append("- Default run: `python main.py` must execute the FULL experiment plan.")
    sections.append("- Smoke run: `SMOKE_TEST=1 python main.py` must execute the SAME logic with smaller counts only.")
    sections.append("- Smoke mode may reduce steps / prompts / seeds / inference steps, but MUST NOT remove conditions or change algorithms.")
    sections.append("- If a metric is skipped in smoke/offline mode, report an explicit skipped reason instead of `NaN` or a fake number.")
    sections.append("\n## Epistemic honesty")
    sections.append("- Do NOT derive human labels, semantic ratings, or ground-truth classes from prompt text, file names, paths, clip IDs, or other heuristics unless the plan explicitly defines that as the official supervision source.")
    sections.append("- If the workspace does not contain the required annotations, trackers, judge outputs, or metadata for a planned metric/method, mark it as `not_implemented` or emit an explicit `skipped_reason` instead of inventing labels or surrogate scores.")
    sections.append("- Output summaries, reports, and artifact JSON files may only contain results that were actually computed during execution. Do NOT copy plan metadata into outputs just to satisfy coverage checks.")
    sections.append("- Declaring a method name in a summary/report does NOT count as implementing it. Only include methods that are actually executed in the experiment loop.")

    # ── Training ──
    training_info = plan_dict.get("training", {})
    if training_info:
        sections.append("\n## Training requirements")
        for key in (
            "max_steps",
            "batch_size",
            "gradient_accumulation_steps",
            "lr",
            "optimizer",
            "lr_scheduler",
            "warmup_steps",
            "mixed_precision",
            "gradient_checkpointing",
            "seed",
        ):
            val = training_info.get(key)
            if val is not None:
                sections.append(f"- {key}: `{val}`")
        notes = training_info.get("notes", "").strip()
        if notes:
            sections.append(f"- Notes:\n{notes[:800]}")

    # ── Compute ──
    compute_info = plan_dict.get("compute", {})
    if compute_info:
        sections.append("\n## Compute environment")
        gpu = compute_info.get("gpu", "")
        if gpu:
            sections.append(f"- GPU: `{gpu}`")
        pkgs = compute_info.get("key_packages", [])
        if pkgs:
            sections.append(f"- Key packages: {', '.join(f'`{p}`' for p in pkgs)}")

    # ── Technology-specific guidance (derived from plan analysis) ──
    from researchclaw.pipeline.codegen.system_prompt import _extract_plan_hints
    hints = _extract_plan_hints(ctx.exp_plan or "")
    if hints:
        sections.append("\n## Technical guidance (auto-derived from plan)")
        for hint in hints:
            sections.append(f"- {hint}")

    sections.append("\n## State isolation")
    sections.append(
        "- If a condition mutates model state (training / LoRA attach), isolate each condition and seed by reloading or deep-copying the base model."
    )
    sections.append(
        "- Do NOT share one mutable pipeline instance across multiple trained conditions."
    )

    # ── Reference paper algorithm details (for reproduce projects) ──
    if ctx.reference_paper_text:
        _paper_excerpt = ctx.reference_paper_text[:12000]
        sections.append("\n## Reference paper (algorithm details)")
        sections.append(
            "This is a REPRODUCE project. The following is the reference paper text. "
            "Your implementation MUST faithfully reproduce the algorithms, loss "
            "functions, and training procedures described here."
        )
        sections.append(f"\n```\n{_paper_excerpt}\n```")

    content = "\n".join(sections)
    try:
        target_path.write_text(content, encoding="utf-8")
    except OSError:
        pass
    return content


class CodegenRuntime:
    """Orchestration for S11 CODE_GENERATION using claw-code turn loop.

    Phases:
    1. CONTEXT    — Build CodegenContext from config + prior artifacts
    2. LLM_SETUP  — Resolve coding model override
    3. ROUTING    — Select strategy (claw_agent is the only one)
    4. GENERATE   — Execute claw-code turn loop (LLM + tools)
    5. FALLBACK   — Numpy fallback if agent produced nothing
    6. FINALIZE   — Write experiment/, experiment_spec.md
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
        session = CodegenSession(stage_dir=stage_dir)
        session.log(CodegenPhase.CONTEXT, "CodegenRuntime started")
        session.log(CodegenPhase.CONTEXT, f"stage_dir={stage_dir}")
        session.log(CodegenPhase.CONTEXT, f"run_dir={run_dir}")

        # ── Phase 1: CONTEXT ─────────────────────────────────────────────
        ctx = self._build_context(config, run_dir, stage_dir, session)

        # ── Phase 2: LLM_SETUP ──────────────────────────────────────────
        coding_model = getattr(config.llm, "coding_model", "") or ""
        session.log(
            CodegenPhase.LLM_SETUP,
            f"primary_model={config.llm.primary_model!r}, "
            f"coding_model={coding_model!r}, "
            f"llm_available={llm is not None}",
        )
        llm = self._resolve_coding_llm(llm, config)
        session.log(CodegenPhase.LLM_SETUP, "LLM client resolved")

        # ── Phase 3: ROUTING ─────────────────────────────────────────────
        available = [s.name for s in self._registry.all()]
        session.log(CodegenPhase.ROUTING, f"Available strategies: {available}")
        router = CodegenRouter(self._registry.all())
        strategy = router.select(ctx, config, adapters, session)
        session.strategy_used = strategy.name

        # ── Phase 4: GENERATE (claw-code turn loop) ──────────────────────
        session.log(CodegenPhase.GENERATE, f"Invoking strategy: {strategy.name}")
        try:
            result = strategy.generate(ctx, config, llm, session, prompts=prompts)
            files = result.files
            session.files = dict(files)
            session.log(
                CodegenPhase.GENERATE,
                f"Strategy returned: {len(files)} files={sorted(files.keys())}, "
                f"error={result.error!r}",
            )
        except Exception as exc:
            session.log_error(CodegenPhase.GENERATE, f"Strategy {strategy.name} raised", exc)
            files = {}
            result = CodegenResult(strategy_name=strategy.name, error=str(exc))

        # ── Phase 5: FALLBACK ────────────────────────────────────────────
        if not files or "main.py" not in files:
            session.log(CodegenPhase.FALLBACK, "No main.py from strategy — using numpy fallback")
            fallback = FallbackStrategy()
            fb_result = fallback.generate(ctx, config, llm, session, prompts=prompts)
            files = fb_result.files
            session.files = dict(files)
            session.strategy_used = f"{session.strategy_used}+fallback"

        # Write experiment directory
        exp_dir = stage_dir / "experiment"
        exp_dir.mkdir(parents=True, exist_ok=True)
        for fname, code in files.items():
            (exp_dir / fname).write_text(code, encoding="utf-8")
        session.log(CodegenPhase.GENERATE, f"Wrote {len(files)} files to {exp_dir}")

        # ── Phase 6: FINALIZE ────────────────────────────────────────────
        artifacts = self._finalize(files, ctx, config, stage_dir, exp_dir, session)

        session.log(
            CodegenPhase.FINALIZE,
            f"COMPLETE: {len(files)} files, strategy={session.strategy_used}, "
            f"{session.llm_calls} LLM calls, "
            f"{len(session.errors)} errors, {session.elapsed_sec():.1f}s total",
        )
        session.persist()

        return StageResult(
            stage=Stage.CODE_GENERATION,
            status=StageStatus.DONE,
            artifacts=tuple(artifacts),
            evidence_refs=tuple(f"stage-10/{a}" for a in artifacts),
        )

    # ------------------------------------------------------------------
    # Phase 1: Context assembly (replaces the deleted context.py)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_context(
        config: RCConfig, run_dir: Path, stage_dir: Path, session: CodegenSession,
    ) -> CodegenContext:
        from researchclaw.pipeline.codegen.types import DiscoveredData
        from researchclaw.pipeline.executor import _load_hardware_profile, _read_prior_artifact

        exp = config.experiment
        hw_raw = _load_hardware_profile(run_dir)
        hw = HardwareProfile.from_dict(hw_raw)

        exp_plan = _read_prior_artifact(run_dir, "exp_plan.yaml") or ""
        codebase_info = _read_prior_artifact(run_dir, "codebase_candidates.json") or "[]"
        reference_paper_text = _read_prior_artifact(run_dir, "reference_paper_text.md") or ""

        datasets_dir = getattr(exp, "datasets_dir", "") or ""
        checkpoints_dir = getattr(exp, "checkpoints_dir", "") or ""
        codebases_dir = getattr(exp, "codebases_dir", "") or ""

        # ── Discover filesystem context (claw-code pattern) ──
        # Like claw-code's ProjectContext.discover_with_git(), we read
        # real files BEFORE the prompt is built so the LLM has ground truth.
        discovered = _discover_data(
            checkpoints_dir, datasets_dir, codebases_dir, session,
        )

        ctx = CodegenContext(
            topic=config.research.topic,
            exp_plan=exp_plan,
            metric=exp.metric_key,
            metric_direction=exp.metric_direction,
            time_budget_sec=exp.time_budget_sec,
            mode=exp.mode,
            hw_profile=hw,
            codebase_info=codebase_info,
            datasets_dir=datasets_dir,
            checkpoints_dir=checkpoints_dir,
            codebases_dir=codebases_dir,
            reference_paper_text=reference_paper_text,
            run_dir=run_dir,
            stage_dir=stage_dir,
            discovered=discovered,
        )

        session.log(
            CodegenPhase.CONTEXT,
            f"Context: topic={ctx.topic[:60]!r}, mode={ctx.mode}, "
            f"metric={ctx.metric}, hw={'GPU:'+hw.gpu_name if hw and hw.has_gpu else 'no GPU'}, "
            f"exp_plan={len(exp_plan)} chars, "
            f"datasets={ctx.datasets_dir!r}, checkpoints={ctx.checkpoints_dir!r}, "
            f"codebases={ctx.codebases_dir!r}",
        )
        return ctx

    # ------------------------------------------------------------------
    # Phase 2: LLM setup
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_coding_llm(llm: Any | None, config: RCConfig) -> Any | None:
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
            llm.config, primary_model=coding_model, fallback_models=full_fallbacks,
        )
        from researchclaw.llm.client import LLMClient
        new_llm = LLMClient(coding_cfg)
        print(
            f"[CODE_GENERATION] Using coding model: {coding_model} "
            f"(fallbacks: {', '.join(full_fallbacks)})",
            flush=True,
        )
        logger.info("S11 CODE_GENERATION: using coding model '%s'", coding_model)
        return new_llm

    # ------------------------------------------------------------------
    # Phase 6: Finalize
    # ------------------------------------------------------------------

    @staticmethod
    def _finalize(
        files: dict[str, str],
        ctx: CodegenContext,
        config: RCConfig,
        stage_dir: Path,
        exp_dir: Path,
        session: CodegenSession,
    ) -> list[str]:
        session.log(CodegenPhase.FINALIZE, "Writing experiment spec")

        file_list = ", ".join(f"`{f}`" for f in sorted(files.keys()))
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
- {"Uses local data/codebases from project config" if (ctx.datasets_dir or ctx.codebases_dir) else "Self-contained execution"}

## Strategy
{session.strategy_used} (claw-code agentic turn loop)
LLM calls: {session.llm_calls}, Errors: {len(session.errors)}

## Generated
{_utcnow_iso()}
"""
        (stage_dir / "experiment_spec.md").write_text(spec, encoding="utf-8")

        artifacts = ["experiment/", "experiment_spec.md"]
        for name in (
            "generation_trace.md",
            "claw_agent_log.json",
            "claw_system_prompt.md",
            "codegen_session.json",
            "codegen_live.log",
            "turn_loop_conversation.json",
        ):
            if (stage_dir / name).exists():
                artifacts.append(name)

        session.add_artifact("experiment/")
        session.add_artifact("experiment_spec.md")
        return artifacts
