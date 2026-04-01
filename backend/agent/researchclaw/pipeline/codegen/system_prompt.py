"""System prompt builder for the claw-code agentic turn loop.

Ported from claw-code ``rust/crates/runtime/src/prompt.rs``
``SystemPromptBuilder``. Uses the same section ordering:

  Intro → System → Doing tasks → Executing actions
  ── DYNAMIC BOUNDARY ──
  Environment → Experiment → Data paths → Constraints

Tools are NOT embedded here — they go via the API ``tools`` field.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from researchclaw.pipeline.codegen.types import CodegenContext

DYNAMIC_BOUNDARY = "────────────────────────────────────────"

_workspace_path: str = ""


def set_workspace_path(path: str) -> None:
    """Set the actual workspace path so the system prompt can include it."""
    global _workspace_path
    _workspace_path = path


def build_system_prompt(ctx: CodegenContext) -> str:
    """Build the full system prompt from context.

    Follows claw-code's ``SystemPromptBuilder.build()`` section order.
    """
    sections: list[str] = [
        _intro_section(),
        _system_section(),
        _doing_tasks_section(),
        _actions_section(),
        _anti_simulation_section(),
        DYNAMIC_BOUNDARY,
        _environment_section(ctx),
        _experiment_section(ctx),
        _data_paths_section(ctx),
        _project_instructions_section(ctx),
        _constraints_section(ctx),
    ]
    return "\n\n".join(s for s in sections if s)


def build_user_message(ctx: CodegenContext) -> str:
    """Build the initial user message that starts the agent loop."""
    md = "lower" if ctx.metric_direction == "minimize" else "higher"
    return f"""Generate a complete experiment for the following research topic.

TOPIC: {ctx.topic}
PRIMARY METRIC: {ctx.metric} (direction: {ctx.metric_direction} — {md} is better)
TIME BUDGET: {ctx.time_budget_sec} seconds

Your workspace contains two critical files — read them FIRST before writing any code:
 - `CODEGEN.md` — project-specific instructions: model loading code, dataset format, evaluation protocol, technical guidance
 - `EXPERIMENT_PLAN.yaml` — full experiment plan with all conditions, training details, and evaluation protocol

STEP-BY-STEP WORKFLOW:

Step 1 — EXPLORE (mandatory before writing any code):
  - FIRST: Use read_file on CODEGEN.md — it contains project-specific model loading code, dataset format, evaluation protocol, and technical guidance
  - Use glob_search to list CHECKPOINTS_DIR contents — find which pretrained models are available
  - Use glob_search to list DATASETS_DIR contents — find data format and structure
  - Use read_file on any config files, README, or sample data files to understand the API
  - Use glob_search/read_file on CODEBASES_DIR to find reusable code

Step 2 — DESIGN based on what you found:
  - Read model_index.json in the checkpoint directory to find the correct pipeline class
  - Use the EXACT class specified in model_index.json `_class_name` field — NOT a generic fallback
  - Example: if model_index.json says `"_class_name": "WanImageToVideoPipeline"`, use that class
  - CRITICAL: Do NOT use `StableDiffusionPipeline` for non-SD models — it will fail silently or crash
  - If model_index.json is not found → use `diffusers.DiffusionPipeline.from_pretrained()` which auto-detects
  - If no checkpoint available → use `torchvision.models` with pretrained=True (small download)
  - NEVER substitute a real model with `torch.nn.Linear` — this makes the experiment meaningless

Step 3 — WRITE main.py using write_file:
  - Load the REAL pretrained model identified in Step 1
  - Implement genuinely different conditions (not just renamed copies)
  - Use REAL evaluation metrics (FID via torchmetrics, CLIP score, etc.)
  - Training loss must be task-appropriate (diffusion loss, cross-entropy, etc.) — NOT `output.mean()`
  - `python3 main.py` must run the FULL experiment specified in `EXPERIMENT_PLAN.yaml`
  - `SMOKE_TEST=1 python3 main.py` must run a LIGHTWEIGHT verification path using the SAME algorithms
  - Smoke mode may only reduce counts (steps / prompts / seeds / inference steps); it must NOT remove conditions, swap algorithms, or use fake metrics
  - Print "{ctx.metric}: <value>" for each condition and seed
  - Save visual artifacts to outputs/

Step 4 — VERIFY (quick smoke test only):
  - Run a QUICK syntax + import check: `python3 -c "import main; print('imports OK')"`
  - If that passes, run a SHORT smoke test: `timeout 30 env SMOKE_TEST=1 python3 main.py` (30 second limit)
  - The purpose is ONLY to catch import errors, syntax errors, and obvious crashes
  - Do NOT wait for the full experiment to finish during S11 verification
  - If the 30-second test shows the model loading, evaluation starting, and training loop starting, that's SUFFICIENT
  - Fix any import/syntax errors with edit_file, then move on

Step 5 — ANTI-SIMULATION SELF-CHECK (mandatory before finishing):
  Run these verification commands:
  ```
  grep -n "nn.Linear" main.py | grep -v "lora\\|adapter\\|projection\\|head\\|classifier\\|fc\\|linear_probe"
  grep -n "mock\\|_mock\\|random.uniform" main.py
  grep -n "from_pretrained\\|load_state_dict\\|create_model\\|timm\\.\\|torchvision\\.models" main.py
  ```
  - First grep must return EMPTY (no bare Linear as model substitute)
  - Second grep must return EMPTY (no mock functions)
  - Third grep must return AT LEAST ONE match (proof of real model loading)
  If any check fails, REWRITE the offending code with a real implementation.

Step 6 — VERIFY outputs:
  - Check that outputs/ contains at least one .png or .jpg file
  - Check that the printed primary metric is a real number (not 0.0 or NaN)
  - If an optional metric cannot run offline, print an explicit skipped status/reason instead of NaN or a fake number

CRITICAL RULES:
- Use REAL pretrained models — NEVER simulate with nn.Linear or random numbers
- Load models from LOCAL paths (CHECKPOINTS_DIR) when available
- Read model_index.json FIRST to determine the correct pipeline class
- NO try/except blocks around model loading or training — if it crashes, we need the traceback
- The ONLY place try/except is allowed is inside a save_outputs() function for file I/O
- NO hardcoded fallback metrics — if the model fails, the code must crash, not return fake numbers
- NO argparse — hardcode all parameters as constants
- NO mock functions — every metric must come from real model computation
- Default execution (`python3 main.py`) must run the FULL experiment plan
- Smoke execution (`SMOKE_TEST=1 python3 main.py`) must use the SAME code path with only smaller counts
- If a condition mutates model state (training / LoRA attach), isolate conditions and seeds by reloading or deep-copying the base model as needed
- Use 3 seeds (42, 123, 456) for statistical robustness
- Complete within the time budget ({ctx.time_budget_sec}s)"""


# ------------------------------------------------------------------
# Static sections (ported from claw-code prompt.rs)
# ------------------------------------------------------------------

def _intro_section() -> str:
    return (
        "You are a research coding agent that generates experiment code by using tools. "
        "You have access to bash, read_file, write_file, edit_file, glob_search, and grep_search. "
        "Use these tools to explore the workspace, write experiment code, run it, and fix errors "
        "iteratively until the experiment produces valid results.\n\n"
        "IMPORTANT: Do NOT describe what you plan to do — just DO it by calling tools. "
        "Every response should include at least one tool call until the experiment is complete."
    )


def _system_section() -> str:
    return (
        "# System\n"
        " - All text you output outside of tool use is logged but not shown to the user.\n"
        " - Tools execute in a sandboxed workspace with limited permissions.\n"
        " - Tool results may be truncated for large outputs — use offset/limit for big files.\n"
        " - bash commands run with a timeout; long-running processes will be killed.\n"
        " - File writes are restricted to the workspace directory.\n"
        " - File reads are allowed in the workspace and configured data directories."
    )


def _doing_tasks_section() -> str:
    return (
        "# Doing tasks\n"
        " - Read relevant code and data before writing — understand the codebase API first.\n"
        " - Keep changes tightly scoped: fix one thing at a time.\n"
        " - Do not add speculative abstractions or unrelated cleanup.\n"
        " - If an approach fails, diagnose the failure before switching tactics.\n"
        " - Use edit_file for targeted fixes instead of rewriting entire files.\n"
        " - After writing code, ALWAYS run it with bash to verify it works.\n"
        " - Report outcomes faithfully: if verification fails, fix it rather than ignoring."
    )


def _actions_section() -> str:
    return (
        "# Executing actions with care\n"
        "Write code incrementally: create the skeleton first, test it, then add complexity. "
        "If a bash command fails, read the error carefully and fix the root cause. "
        "Never mask errors with try/except — the experiment must crash cleanly so errors "
        "can be diagnosed.\n\n"
        "CRITICAL — S11 MUST PRODUCE DUAL-MODE EXPERIMENT CODE:\n"
        " - `python3 main.py` must execute the FULL experiment described by the plan.\n"
        " - `SMOKE_TEST=1 python3 main.py` must execute a LIGHTWEIGHT verification path.\n"
        " - Smoke mode may only shrink counts (steps, prompts, seeds, inference steps); it must NOT change algorithms.\n"
        " - Verify with a QUICK smoke test only: `timeout 30 env SMOKE_TEST=1 python3 main.py`\n"
        " - If the smoke test shows model loading + evaluation starting + first training step starting → DONE.\n"
        " - Do NOT remove training/evaluation code because the smoke test times out.\n"
        " - If OOM during smoke test → reduce only smoke-mode counts or batch size. Keep the full experiment path intact.\n"
        " - Once code is verified (imports work, model loads, training starts), STOP calling tools."
    )


def _anti_simulation_section() -> str:
    return """# ANTI-SIMULATION RULES (MANDATORY — VIOLATION = EXPERIMENT REJECTED)

You MUST use REAL pretrained models for experiments. The following patterns are
STRICTLY FORBIDDEN and will cause the experiment to be rejected:

## Forbidden Patterns (instant rejection)
 - `torch.nn.Linear` as a substitute for a real model (SD, ViT, ResNet, LLM, etc.)
 - `torch.nn.Sequential(Conv2d, ReLU, Conv2d)` as a "feature extractor" replacing a real model
 - `np.random.uniform` / `torch.rand` to generate fake metric values
 - `output.mean()` as a training loss (meaningless optimization target)
 - Functions named `compute_*_mock` or `*_mock` — no mock implementations allowed
 - Flattening images to 1D vectors and feeding them to Linear layers
 - Any function that returns a hardcoded or random number as a metric
 - `try/except` around model loading/training that returns hardcoded fallback metrics
 - `except: pass` or `except: return` with fake values — if the model fails to load, the code MUST crash
 - Loading a model with the WRONG class — always read config/metadata files first to determine the correct loader

## Required Patterns (must be present)
 - Load a REAL pretrained model via the appropriate library API (e.g. `from_pretrained`, `torch.hub.load`, etc.)
 - Use REAL evaluation metrics from a validated library — not hand-rolled approximations
 - Training must use a proper task-specific loss — NOT `output.mean()`
 - If CHECKPOINTS_DIR contains model weights, you MUST load them with the appropriate library
 - Every metric in the return dict must be computed from real data, NEVER hardcoded to 0.0 or any constant

## Self-Verification Protocol
After writing main.py, you MUST verify it is not a simulation by running:
```bash
grep -n "nn.Linear" main.py | grep -v "lora\\|adapter\\|projection\\|head\\|classifier"
```
If this finds any `nn.Linear` used as the primary model, the code is a simulation and MUST be rewritten.

Also verify:
```bash
grep -n "mock\\|_mock\\|random.uniform\\|np.random" main.py
```
If this finds mock functions or random metric generation, rewrite with real implementations."""


# ------------------------------------------------------------------
# Dynamic sections (experiment-specific context)
# ------------------------------------------------------------------

def _environment_section(ctx: CodegenContext) -> str:
    lines = ["# Environment"]
    ws = _workspace_path or str(ctx.stage_dir or "workspace")
    lines.append(f" - Working directory: `{ws}`")
    lines.append(f" - IMPORTANT: Use RELATIVE paths for write_file (e.g. `main.py`, NOT `/workspace/main.py`)")
    lines.append(f"   All write_file/edit_file calls use the working directory as base.")
    lines.append(f"   bash commands also run in this directory.")
    lines.append(f" - Mode: {ctx.mode}")

    if ctx.hw_profile and ctx.hw_profile.has_gpu:
        hw = ctx.hw_profile
        lines.append(f" - GPU: {hw.gpu_name} ({hw.gpu_type})")
        if hw.gpu_type == "npu":
            lines.append(" - CRITICAL: Huawei Ascend NPU — use `import torch_npu` and `device = torch.device('npu')`")
        else:
            lines.append(f" - Use `device = torch.device('{hw.gpu_type}')`")
    else:
        lines.append(" - No GPU detected — design CPU-friendly experiments")

    if ctx.pkg_hint:
        lines.append(f"\n{ctx.pkg_hint}")

    return "\n".join(lines)


def _experiment_section(ctx: CodegenContext) -> str:
    lines = ["# Experiment"]
    lines.append(f" - Topic: {ctx.topic}")
    lines.append(f" - Primary metric: {ctx.metric} (direction: {ctx.metric_direction})")
    lines.append(f" - Time budget: {ctx.time_budget_sec}s per run")
    if ctx.compute_budget:
        lines.append(ctx.compute_budget)
    return "\n".join(lines)


def _data_paths_section(ctx: CodegenContext) -> str:
    """Minimal data path pointers — detailed info is in CODEGEN.md."""
    lines = ["# Available data paths"]
    has_any = False

    if ctx.checkpoints_dir:
        lines.append(f" - Checkpoints: `{ctx.checkpoints_dir}`")
        has_any = True
    if ctx.datasets_dir:
        lines.append(f" - Datasets: `{ctx.datasets_dir}`")
        has_any = True
    if ctx.codebases_dir:
        lines.append(f" - Codebases: `{ctx.codebases_dir}`")
        has_any = True

    if not has_any:
        lines.append(" - No pre-configured data paths — generate synthetic data or download small datasets")

    lines.append("See CODEGEN.md for detailed model/dataset info. Use glob_search and read_file to explore.")
    return "\n".join(lines)


def _project_instructions_section(ctx: CodegenContext) -> str:
    """Lightweight pointer to workspace instruction files.

    CODEGEN.md is generated directly in the workspace by _prepare_workspace.
    We only add a brief pointer here — the agent reads the full file on-demand.
    """
    return (
        "# Project instructions\n"
        "Your workspace contains `CODEGEN.md` with project-specific instructions: "
        "model loading code, dataset format, evaluation protocol, and technical guidance.\n"
        "Read it with `read_file` BEFORE writing any code."
    )


def _extract_plan_hints(plan_text: str) -> list[str]:
    """Analyze the experiment plan and generate targeted technical hints.

    This replaces hardcoded advice: hints are only generated for technologies
    actually mentioned in the plan. Different plans get different hints.
    """
    if not plan_text:
        return []

    plan_lower = plan_text.lower()
    hints: list[str] = []

    if "lora" in plan_lower:
        hints.append(
            "LoRA detected: use `peft.LoraConfig` applied to a REAL pretrained model. "
            "NEVER apply LoRA to a bare nn.Linear — it must wrap a real model's layers."
        )
        if "adaptive" in plan_lower or "rank_pattern" in plan_lower or "per-layer" in plan_lower:
            hints.append(
                "Adaptive/per-layer LoRA ranks detected: peft does NOT support `rank_pattern` in LoraConfig. "
                "Instead, apply multiple LoraConfig objects with different `r` values and `adapter_name` parameters. "
                "Example: group layers by depth, then `model.add_adapter(LoraConfig(r=8, ...), adapter_name='early')` "
                "and `model.add_adapter(LoraConfig(r=32, ...), adapter_name='late')`. "
                "NEVER just set `module.r = 32` on an existing adapter — that does NOT change the matrix shape."
            )

    if "clip" in plan_lower and ("score" in plan_lower or "metric" in plan_lower):
        hints.append(
            "CLIP score metric detected: use `torchmetrics.multimodal.CLIPScore` with a real CLIP model. "
            "If network is unavailable, use a locally cached CLIP model or omit the metric entirely. "
            "NEVER return a hardcoded clip_score value or `NaN` without an explicit skipped reason."
        )

    if "fid" in plan_lower:
        hints.append(
            "FID metric detected: use `torchmetrics.image.fid.FrechetInceptionDistance`. "
            "FID requires reference (real) images and generated images — ensure both sets exist. "
            "If reference images are not available, skip FID rather than returning a fake value."
        )

    if "diffus" in plan_lower:
        hints.append(
            "Diffusion model detected: load using `DiffusionPipeline.from_pretrained()` which auto-detects "
            "the correct pipeline class from model_index.json. Do NOT hardcode StableDiffusionPipeline "
            "unless model_index.json specifically indicates that class."
        )

    if "video" in plan_lower or "i2v" in plan_lower or "t2v" in plan_lower:
        hints.append(
            "Video generation detected: video pipelines produce frame sequences, not single images. "
            "Ensure metrics handle video tensors (B, T, C, H, W) correctly."
        )

    if "training" in plan_lower or "evaluation" in plan_lower:
        hints.append(
            "Dual-mode execution required: `python main.py` must run the full planned experiment, while "
            "`SMOKE_TEST=1 python main.py` may only reduce steps/prompts/seeds/inference steps for quick verification. "
            "Smoke mode must never change the algorithm, remove conditions, or fake metrics."
        )

    if "train" in plan_lower and ("loss" in plan_lower or "fine-tun" in plan_lower or "finetun" in plan_lower):
        hints.append(
            "Training/fine-tuning detected: use a task-appropriate loss function (diffusion noise loss, "
            "cross-entropy, MSE, etc.) — NOT `output.mean()` which is meaningless."
        )

    if "gradient_checkpointing" in plan_lower or "memory" in plan_lower:
        hints.append(
            "Memory optimization detected: enable `model.enable_gradient_checkpointing()` and use "
            "`torch.cuda.amp.autocast()` to reduce VRAM usage."
        )

    return hints


def _constraints_section(ctx: CodegenContext) -> str:
    return (
        "# Constraints\n"
        " - Each experimental condition must implement a genuinely DIFFERENT algorithm\n"
        " - Metrics must be computed from actual model outputs — NEVER hardcode values\n"
        " - Save visual artifacts to `outputs/` directory (PNG, plots, samples)\n"
        " - Use 3 random seeds (42, 123, 456) for each condition\n"
        " - Print results in format: `{metric}: <value>` for pipeline parsing\n"
        " - Support both full mode (`python main.py`) and lightweight smoke mode (`SMOKE_TEST=1 python main.py`)\n"
        " - Smoke mode may only shrink counts; it must not change algorithms, conditions, or metric semantics\n"
        " - NO try/except blocks (except in save_outputs for file I/O)\n"
        " - Code must complete within the time budget"
    ).replace("{metric}", ctx.metric)
