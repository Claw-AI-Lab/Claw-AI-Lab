"""Context assembly for code generation.

Inspired by claw-code's ``ProjectContext.discover_with_git()`` which
gathers cwd, git_status, and instruction_files before the prompt builder
consumes them. ContextAssembler gathers hardware, domain, codebase, data
paths, benchmarks, and framework docs into a single CodegenContext.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from researchclaw.config import RCConfig
from researchclaw.pipeline.codegen.types import CodegenContext, HardwareProfile

logger = logging.getLogger(__name__)


class ContextAssembler:
    """Build a CodegenContext from config, run_dir, and prior stage artifacts.

    Analogous to claw-code's ``ProjectContext.discover()`` which walks
    directory ancestors to collect CLAUDE.md instruction files. Here we
    walk prior stage directories to collect experiment plans, codebase
    candidates, benchmark plans, and hardware profiles.
    """

    def __init__(
        self,
        config: RCConfig,
        run_dir: Path,
        stage_dir: Path,
        prompts: Any | None = None,
    ) -> None:
        self._config = config
        self._run_dir = run_dir
        self._stage_dir = stage_dir
        self._pm = prompts

    def build(self) -> CodegenContext:
        from researchclaw.pipeline.executor import (
            _load_hardware_profile,
            _read_prior_artifact,
        )

        cfg = self._config
        exp = cfg.experiment

        hw_raw = _load_hardware_profile(self._run_dir)
        hw = HardwareProfile.from_dict(hw_raw)

        exp_plan = _read_prior_artifact(self._run_dir, "exp_plan.yaml") or ""
        codebase_info = (
            _read_prior_artifact(self._run_dir, "codebase_candidates.json") or "[]"
        )

        ctx = CodegenContext(
            topic=cfg.research.topic,
            exp_plan=exp_plan,
            metric=exp.metric_key,
            metric_direction=exp.metric_direction,
            time_budget_sec=exp.time_budget_sec,
            mode=exp.mode,
            hw_profile=hw,
            codebase_info=codebase_info,
            datasets_dir=getattr(exp, "datasets_dir", "") or "",
            checkpoints_dir=getattr(exp, "checkpoints_dir", "") or "",
            codebases_dir=getattr(exp, "codebases_dir", "") or "",
            run_dir=self._run_dir,
            stage_dir=self._stage_dir,
        )

        ctx.pkg_hint = self._build_pkg_hint(ctx, hw)
        ctx.compute_budget = self._build_compute_budget(ctx)
        ctx.extra_guidance = self._build_extra_guidance(ctx)

        return ctx

    # ------------------------------------------------------------------
    # Section builders (each is a self-contained, testable unit)
    # ------------------------------------------------------------------

    def _build_pkg_hint(
        self, ctx: CodegenContext, hw: HardwareProfile | None
    ) -> str:
        """Build the available-packages hint for sandbox/docker mode."""
        pm = self._pm
        cfg = self._config

        if ctx.mode not in ("sandbox", "docker"):
            return ""

        if ctx.mode == "docker":
            pkg_prefix = "docker mode"
            net_policy = cfg.experiment.docker.network_policy
            base_pkgs = (
                ", torchvision, torchaudio, matplotlib, seaborn, scipy, "
                "tqdm, torchdiffeq, gymnasium, networkx, PyYAML, Pillow, "
                "transformers, datasets, accelerate, peft, bitsandbytes, "
                "timm, einops, torchmetrics, h5py"
            )
            if net_policy == "none":
                pkg_extras = base_pkgs + " (ONLY pre-installed packages — NO pip install available)"
            elif net_policy in ("setup_only", "pip_only"):
                pkg_extras = base_pkgs + ", and additional pip-installable packages via requirements.txt"
            else:
                pkg_extras = base_pkgs + ", and additional pip-installable packages (auto-detected from imports)"
        else:
            pkg_prefix = "sandbox mode (local subprocess — full filesystem access)"
            sandbox_net = getattr(cfg.experiment.sandbox, "network_policy", "full")
            base_pkgs = (
                ", torchvision, torchaudio, matplotlib, seaborn, scipy, "
                "tqdm, gymnasium, networkx, PyYAML, Pillow, "
                "transformers, datasets, accelerate, peft, bitsandbytes, "
                "timm, einops, torchmetrics, h5py, diffusers, safetensors, huggingface_hub"
            )
            if sandbox_net == "full":
                pkg_extras = base_pkgs + ", and any pip-installable packages"
            elif sandbox_net == "none":
                pkg_extras = base_pkgs + " (ONLY pre-installed packages — NO pip install available)"
            else:
                pkg_extras = base_pkgs

        if hw and hw.has_gpu:
            npu_hint = self._npu_hint(hw) if hw.gpu_type == "npu" else ""
            device_hint = f"torch.device('{hw.gpu_type}')"
            if hw.tier == "high":
                return (
                    f"\nAVAILABLE PACKAGES ({pkg_prefix}): Python stdlib, numpy, torch, sklearn, scipy, pandas{pkg_extras}.\n"
                    f"GPU: {hw.gpu_name} ({hw.gpu_type}). You MAY use PyTorch with GPU acceleration.\n"
                    f"Use `device = {device_hint}` for tensor operations.\n"
                    f"{npu_hint}"
                )
            else:
                return (
                    f"\nAVAILABLE PACKAGES ({pkg_prefix}): Python stdlib, numpy, torch, sklearn, scipy, pandas{pkg_extras}.\n"
                    f"GPU: {hw.gpu_name} ({hw.gpu_type}) — LIMITED performance.\n"
                    f"Use `device = {device_hint}` but design LIGHTWEIGHT experiments:\n"
                    f"- Small models (<1M parameters)\n"
                    f"- Few epochs (<=20)\n"
                    f"- Small datasets (<=10K samples)\n"
                    f"- Avoid large batch sizes\n"
                    f"{npu_hint}"
                )

        if pm is not None:
            try:
                return pm.block("pkg_hint_sandbox")
            except Exception:
                pass
        return ""

    @staticmethod
    def _npu_hint(hw: HardwareProfile) -> str:
        if hw.gpu_type != "npu":
            return ""
        return (
            "\n## CRITICAL: Huawei Ascend NPU Device\n"
            "This machine uses Ascend NPU, NOT NVIDIA CUDA.\n"
            "You MUST follow these rules:\n"
            "1. Add `import torch_npu` at the top of main.py (BEFORE any torch.npu calls)\n"
            "2. Use `device = torch.device('npu')` instead of 'cuda'\n"
            "3. Use `torch.npu.is_available()` instead of `torch.cuda.is_available()`\n"
            "4. NEVER use `torch.cuda.*` APIs — they will fail\n"
            "5. DataLoader MUST use `pin_memory=False` (pin_memory is CUDA-only)\n"
            "   but DO use `num_workers=8` for parallel data loading (critical for performance)\n"
            "6. Example device setup:\n"
            "   ```python\n"
            "   import torch\n"
            "   import torch_npu  # MUST import before using torch.npu\n"
            "   device = torch.device('npu' if torch.npu.is_available() else 'cpu')\n"
            "   loader = DataLoader(dataset, batch_size=128, shuffle=True,\n"
            "                       num_workers=8, pin_memory=False)\n"
            "   ```\n"
        )

    def _build_compute_budget(self, ctx: CodegenContext) -> str:
        if self._pm is not None:
            try:
                return self._pm.block("compute_budget").replace(
                    "{time_budget_sec}", str(ctx.time_budget_sec)
                )
            except Exception:
                pass
        return (
            f"\n## Compute Budget Constraint\n"
            f"- Total execution time limit: {ctx.time_budget_sec} seconds\n"
            f"- Design experiments that complete within this budget\n"
            f"- Implement a time guard using `time.time()` checks (NOT try/except) to stop at 80% of budget\n"
        )

    def _build_extra_guidance(self, ctx: CodegenContext) -> str:
        """Assemble all extra guidance sections."""
        parts: list[str] = []
        parts.append(self._codebase_candidates_guidance(ctx))
        parts.append(self._data_paths_guidance(ctx))
        parts.append(self._anti_simulation_guidance(ctx))
        parts.append(self._network_and_dataset_guidance(ctx))
        parts.append(self._benchmark_plan_guidance(ctx))
        parts.append(self._topic_specific_guidance(ctx))
        parts.append(self._framework_docs_guidance(ctx))
        parts.append(self._domain_specific_guidance(ctx))
        parts.append(self._local_data_constraint(ctx))
        parts.append(self._visual_outputs_guidance())
        return "\n".join(p for p in parts if p)

    def _codebase_candidates_guidance(self, ctx: CodegenContext) -> str:
        try:
            cb_list = json.loads(ctx.codebase_info)
            direct = [
                c for c in cb_list
                if isinstance(c, dict) and c.get("download_status") == "success"
            ]
            if not direct:
                return ""
            lines = [
                "\n## EXISTING CODEBASE (MUST USE AS BASE)",
                "The following codebases have been downloaded and are available locally.",
                "You MUST build upon this existing code rather than writing from scratch.\n",
            ]
            for cb in direct:
                lines.append(f"- **{cb.get('repo_url', '?')}** → `{cb.get('local_path', '?')}`")
                lines.append(f"  Description: {cb.get('description', '?')}")
                lines.append(f"  Key files: {', '.join(cb.get('key_files', []))}\n")
            return "\n".join(lines)
        except Exception:
            return ""

    def _data_paths_guidance(self, ctx: CodegenContext) -> str:
        """Build LOCAL DATA PATHS section from configured directories."""
        block = "\n## LOCAL DATA PATHS (MUST USE)\n"
        has_any = False

        if ctx.datasets_dir:
            os.makedirs(ctx.datasets_dir, exist_ok=True)
            existing = [d for d in os.listdir(ctx.datasets_dir) if not d.startswith(".")] if os.path.isdir(ctx.datasets_dir) else []
            block += f"### Datasets Directory: `{ctx.datasets_dir}`\n"
            if existing:
                block += f"Available datasets: {', '.join(existing)}\n"
                block += "Use these local datasets directly via `os.path.join(DATASETS_DIR, '<name>')`. Do NOT download datasets.\n"
            else:
                block += "Directory is empty. Generate synthetic data or download minimal test data programmatically.\n"
            block += f"In code: `DATASETS_DIR = '{ctx.datasets_dir}'`\n\n"
            has_any = True

        if ctx.checkpoints_dir:
            os.makedirs(ctx.checkpoints_dir, exist_ok=True)
            existing = [f for f in os.listdir(ctx.checkpoints_dir) if not f.startswith(".")] if os.path.isdir(ctx.checkpoints_dir) else []
            block += f"### Checkpoints Directory: `{ctx.checkpoints_dir}`\n"
            if existing:
                block += f"Available checkpoints: {', '.join(existing)}\n"
                block += "Load these checkpoints directly. Do NOT re-download if already present.\n"
            else:
                block += "No checkpoints yet. Download required model weights to this directory using `huggingface_hub` or `torch.hub`.\n"
                block += "Always check if file exists before downloading: `if not os.path.exists(path): download()`\n"
            block += f"In code: `CHECKPOINTS_DIR = '{ctx.checkpoints_dir}'`\n\n"
            has_any = True

        if ctx.codebases_dir:
            os.makedirs(ctx.codebases_dir, exist_ok=True)
            all_repos = [
                d for d in os.listdir(ctx.codebases_dir)
                if os.path.isdir(os.path.join(ctx.codebases_dir, d)) and not d.startswith(".")
            ] if os.path.isdir(ctx.codebases_dir) else []

            from researchclaw.pipeline.executor import _extract_selected_repos
            selected = _extract_selected_repos(ctx.codebase_info)
            repos = [r for r in all_repos if selected is None or r in selected] if all_repos else []

            if repos:
                from researchclaw.utils.codebase_manifest import generate_manifest, manifest_to_prompt
                block += f"### Codebases Directory: `{ctx.codebases_dir}`\n"
                block += (
                    "**CRITICAL**: You MUST build your experiment code ON TOP of these existing codebases. "
                    "Do NOT write everything from scratch. Import, extend, or wrap the existing code.\n\n"
                )
                for repo_name in repos:
                    repo_path = os.path.join(ctx.codebases_dir, repo_name)
                    try:
                        manifest = generate_manifest(repo_path)
                        block += manifest_to_prompt(manifest) + "\n\n"
                    except Exception as e:
                        block += f"#### Codebase: `{repo_name}` (manifest generation failed: {e})\n"
                        block += f"Path: `{repo_path}` — add to sys.path and explore manually.\n\n"
                block += f"In code: `CODEBASES_DIR = '{ctx.codebases_dir}'`\n\n"
                has_any = True

        return block if has_any else ""

    def _anti_simulation_guidance(self, ctx: CodegenContext) -> str:
        has_codebases = False
        if ctx.codebases_dir:
            has_codebases = any(
                os.path.isdir(os.path.join(ctx.codebases_dir, d))
                for d in (os.listdir(ctx.codebases_dir) if os.path.isdir(ctx.codebases_dir) else [])
                if not d.startswith(".")
            )
        if has_codebases or not (ctx.checkpoints_dir or ctx.datasets_dir):
            return ""
        block = (
            "\n## NO CODEBASE PROVIDED — USE REAL ML LIBRARIES\n"
            "No pre-existing codebase is available. You MUST write experiment code "
            "using standard ML libraries (torch, diffusers, transformers, peft, etc.).\n\n"
            "**CRITICAL ANTI-SIMULATION RULES:**\n"
            "1. NEVER simulate model behavior with numpy/PIL image manipulation.\n"
            "2. NEVER create fake training loops that blend/compose images instead of "
            "running real gradient-based optimization.\n"
            "3. Every experimental condition MUST run REAL model forward passes through "
            "actual neural network weights.\n"
            "4. If CHECKPOINTS_DIR contains model weights, load them with the appropriate "
            "library's `from_pretrained(CHECKPOINTS_DIR, local_files_only=True)`.\n"
            "5. Use the appropriate ML framework for the topic:\n"
            "   - Diffusion models / image generation → `diffusers`\n"
            "   - LoRA / fine-tuning → `peft` (LoraConfig, get_peft_model)\n"
            "   - LLM training → `transformers` + `peft` + `accelerate`\n"
            "   - General deep learning → `torch` + `torchvision`\n"
            "6. If the BenchmarkAgent suggests generic datasets (CIFAR, MNIST) but the "
            "topic requires a specific model (Stable Diffusion, GPT, etc.), PRIORITIZE "
            "the topic requirements and build the pipeline around the actual model.\n"
        )
        if ctx.checkpoints_dir:
            ck_lower = os.path.basename(ctx.checkpoints_dir).lower()
            if any(kw in ck_lower for kw in ("stable-diffusion", "sd-", "sdxl", "diffusion")):
                block += (
                    "\n**Detected: Stable Diffusion checkpoints.**\n"
                    "You MUST use `diffusers.StableDiffusionPipeline` to load the model:\n"
                    "```python\n"
                    "from diffusers import StableDiffusionPipeline\n"
                    f"pipe = StableDiffusionPipeline.from_pretrained('{ctx.checkpoints_dir}', "
                    "local_files_only=True)\n"
                    "pipe = pipe.to('cuda')\n"
                    "```\n"
                )
        return block

    def _network_and_dataset_guidance(self, ctx: CodegenContext) -> str:
        """Assemble network policy, dataset, HP reporting, and multi-seed guidance."""
        if ctx.mode not in ("sandbox", "docker"):
            return ""
        pm = self._pm
        cfg = self._config
        parts: list[str] = []
        has_data_paths = bool(ctx.datasets_dir or ctx.checkpoints_dir or ctx.codebases_dir)

        net_policy = (
            cfg.experiment.docker.network_policy
            if ctx.mode == "docker"
            else getattr(cfg.experiment.sandbox, "network_policy", "full")
        )

        if net_policy == "none":
            parts.append(self._try_block(pm, "network_disabled_guidance"))
        elif net_policy == "full":
            if ctx.mode == "sandbox":
                parts.append(self._try_block(pm, "sandbox_local_guidance"))
            if not has_data_paths:
                parts.append(self._try_block(pm, "dataset_guidance"))
            parts.append(self._try_block(pm, "network_full_guidance"))
        else:
            if not has_data_paths:
                parts.append(self._try_block(pm, "dataset_guidance"))
            if ctx.mode == "docker":
                parts.append(self._try_block(pm, "setup_script_guidance"))

        parts.append(self._try_block(pm, "hp_reporting"))
        parts.append(self._try_block(pm, "multi_seed_enforcement"))
        return "\n".join(p for p in parts if p)

    def _benchmark_plan_guidance(self, ctx: CodegenContext) -> str:
        """Inject BenchmarkAgent plan from Stage 9."""
        if ctx.run_dir is None:
            return ""
        bp_path = None
        for s9_dir in sorted(ctx.run_dir.glob("stage-09*"), reverse=True):
            candidate = s9_dir / "benchmark_plan.json"
            if candidate.exists():
                bp_path = candidate
                break
        if bp_path is None:
            return ""
        try:
            bp_data = json.loads(bp_path.read_text(encoding="utf-8"))
            from researchclaw.agents.benchmark_agent.orchestrator import BenchmarkPlan
            bp = BenchmarkPlan(
                selected_benchmarks=bp_data.get("selected_benchmarks", []),
                selected_baselines=bp_data.get("selected_baselines", []),
                data_loader_code=bp_data.get("data_loader_code", ""),
                baseline_code=bp_data.get("baseline_code", ""),
                experiment_notes=bp_data.get("experiment_notes", ""),
            )
            bp_block = bp.to_prompt_block()
            if bp_block:
                return (
                    "\n\n## BenchmarkAgent Selections (USE THESE)\n"
                    "The following datasets, baselines, and code snippets were "
                    "automatically selected and validated by the BenchmarkAgent. "
                    "You MUST use these selections in your experiment code.\n\n"
                    + bp_block
                )
        except Exception as exc:
            logger.debug("BA: Failed to load benchmark plan: %s", exc)
        return ""

    def _topic_specific_guidance(self, ctx: CodegenContext) -> str:
        """Detect LLM training and RL topics, inject relevant guidance."""
        parts: list[str] = []
        topic_lower = ctx.topic.lower()
        cfg = self._config

        _llm_keywords = (
            "language model", "llm", "fine-tun", "lora", "qlora", "peft",
            "instruction tun", "rlhf", "dpo", "sft", "alignment",
            "transformer train", "causal lm", "chat model", "qwen", "llama",
            "mistral", "phi-", "gemma", "pretraining", "tokeniz",
        )
        is_llm_topic = any(kw in topic_lower for kw in _llm_keywords)

        _rl_keywords = (
            "reinforcement learning", "policy gradient", "ppo", "sac", "td3",
            "ddpg", "dqn", "a2c", "a3c", "mujoco", "locomotion", "continuous control",
            "reward shaping", "exploration", "multi-agent rl", "marl", "curriculum rl",
            "imitation learning", "inverse rl", "offline rl", "model-based rl",
            "actor-critic", "reinforce", "gym", "gymnasium",
        )
        is_rl_topic = any(kw in topic_lower for kw in _rl_keywords)

        if is_rl_topic:
            parts.append(self._try_block(self._pm, "rl_step_guidance"))

        if is_llm_topic and ctx.mode == "docker":
            parts.append(self._try_block(self._pm, "llm_training_guidance"))
            parts.append(self._try_block(self._pm, "llm_eval_guidance"))
            if ctx.time_budget_sec < 3600:
                parts.append(
                    "\n## COMPUTE BUDGET WARNING\n"
                    f"Current time_budget_sec={ctx.time_budget_sec} is likely TOO SHORT "
                    "for LLM fine-tuning. Typical LoRA training needs 1-4 hours. "
                    "Design a LIGHTWEIGHT experiment:\n"
                    "- Use a small dataset (<=5000 samples)\n"
                    "- Train for 1-3 epochs only\n"
                    "- Use small batch size (1-2) with gradient accumulation\n"
                    "- Use 4-bit quantization (QLoRA) to minimize memory\n"
                    "- Limit max_seq_length to 512-1024\n"
                    "- If possible, use a smaller model (<=7B parameters)\n"
                )

        return "\n".join(p for p in parts if p)

    def _framework_docs_guidance(self, ctx: CodegenContext) -> str:
        try:
            from researchclaw.data import detect_frameworks, load_framework_docs
            from researchclaw.pipeline.executor import _read_prior_artifact
            hypothesis_text = _read_prior_artifact(ctx.run_dir, "hypotheses.md") or ""
            fw_ids = detect_frameworks(ctx.topic, hypothesis_text, ctx.exp_plan or "")
            if fw_ids:
                fw_docs = load_framework_docs(fw_ids, max_chars=8000)
                if fw_docs:
                    logger.info("F-01: Injected framework docs for: %s", fw_ids)
                    return fw_docs
        except Exception:
            logger.debug("F-01: Framework doc injection skipped", exc_info=True)
        return ""

    def _domain_specific_guidance(self, ctx: CodegenContext) -> str:
        try:
            from researchclaw.domains.detector import detect_domain, is_ml_domain
            dp = detect_domain(topic=ctx.topic)
            if not is_ml_domain(dp):
                from researchclaw.domains.prompt_adapter import get_adapter
                adapter = get_adapter(dp)
                blocks = adapter.get_code_generation_blocks({})
                parts: list[str] = []
                if blocks.compute_budget:
                    ctx.compute_budget = blocks.compute_budget
                if blocks.dataset_guidance:
                    parts.append(blocks.dataset_guidance)
                if blocks.code_generation_hints:
                    parts.append(blocks.code_generation_hints)
                if blocks.output_format_guidance:
                    parts.append(blocks.output_format_guidance)
                logger.info("Injected domain-specific guidance for %s", dp.domain_id)
                return "\n".join(parts)
        except Exception:
            logger.debug("Domain guidance injection skipped", exc_info=True)
        return ""

    def _local_data_constraint(self, ctx: CodegenContext) -> str:
        has_data_paths = bool(ctx.datasets_dir or ctx.checkpoints_dir or ctx.codebases_dir)
        if not has_data_paths:
            return ""
        return (
            "\n\n## LOCAL DATA CONSTRAINT\n"
            "Use ONLY the local datasets, checkpoints, and codebases listed in "
            "the LOCAL DATA PATHS section above. Do NOT download external datasets "
            "(CelebA, CIFAR, ImageNet, etc.) or model weights from the internet "
            "when local alternatives are available.\n"
        )

    @staticmethod
    def _visual_outputs_guidance() -> str:
        return (
            "\n\n## SAVE INTERMEDIATE VISUAL RESULTS (MANDATORY)\n"
            "You MUST create an `outputs/` directory and save intermediate visual results.\n"
            "This is critical for verifying experiment quality beyond numerical metrics.\n\n"
            "### Implementation:\n"
            "```python\n"
            "OUTPUT_DIR = 'outputs'\n"
            "os.makedirs(OUTPUT_DIR, exist_ok=True)\n"
            "```\n\n"
            "### What to save (choose based on task type):\n"
            "- **Image generation** (diffusion, GAN, etc.): generated images as PNG\n"
            "- **Video generation** (video diffusion, frame interpolation, etc.): save key frames "
            "as PNG grid or short clips as MP4/GIF to `outputs/`\n"
            "- **Image processing** (style transfer, SR, etc.): before/after comparison images\n"
            "- **Model training** (LoRA, fine-tuning, pretraining): loss/metric curves over "
            "steps using matplotlib\n"
            "- **Classification/regression**: confusion matrices, prediction scatter plots\n"
            "- **RL** (reinforcement learning): reward curves, episode return plots, "
            "agent trajectory visualizations\n"
            "- **NLP/text generation**: sample outputs saved to .txt, or token distribution plots\n"
            "- **Time series**: forecast vs actual overlay plots\n"
            "- **Attention/feature analysis**: attention heatmaps, feature maps\n"
            "- **General**: save whatever artifact best PROVES the model actually ran "
            "and produced meaningful results\n\n"
            "### Rules:\n"
            "- Name files descriptively: `outputs/{condition}_{seed}_{description}.png`\n"
            "- Inside the save_outputs() function ONLY, wrap file-saving logic in try/except so I/O failures do not crash the experiment. This is the ONLY place try/except is allowed.\n"
            "- NEVER add try/except in main() or run_condition(). All model errors MUST crash with full traceback so the fix system can diagnose them.\n"
            "- Save at least ONE visual artifact per condition to prove the model actually ran\n"
        )

    @staticmethod
    def _try_block(pm: Any | None, name: str) -> str:
        if pm is None:
            return ""
        try:
            return pm.block(name)
        except Exception:
            return ""
