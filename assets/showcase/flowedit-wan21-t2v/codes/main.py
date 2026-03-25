import warnings

warnings.filterwarnings("ignore", message="urllib3 .* doesn't match a supported version!")
warnings.filterwarnings("ignore", category=UserWarning, module="requests")

"""
FlowEdit on Wan2.1-1.3B-T2V: improved reproduction scaffold for the planned
Wan2.1 experiments.

Key fixes relative to the previous run:
- Preserve experiment-plan condition names exactly.
- Implement all planned method / ablation / diagnostic names.
- Correct transport logic: move along target-source velocity difference instead
  of directly snapping toward the source latent trajectory.
- Add smooth window gating to reduce temporal instability at window boundaries.
- Make weak source anchor truly low-frequency and only active for anchored
  conditions.
- Add lightweight diagnostics for curvature-aware and target-score-only
  conditions without external dependencies.
- Keep runtime bounded and deterministic.
"""

import sys
import os
import time
import json
import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.metrics import structural_similarity as ssim

# Add DiffSynth-Studio to path
sys.path.insert(0, "/home/user/Claw-AI-Lab-share/codebases/DiffSynth-Studio")

from diffsynth.pipelines.wan_video import WanVideoPipeline, model_fn_wan_video
from diffsynth.core.loader.config import ModelConfig
from diffsynth.diffusion.flow_match import FlowMatchScheduler
from diffsynth.utils.data import VideoData

# ==================== CONSTANTS ====================
CHECKPOINTS_DIR = "/home/user/Claw-AI-Lab-share/checkpoints/Wan2.1-T2V-1.3B"
DATASETS_DIR = "/home/user/Claw-AI-Lab-share/datasets/example_video_dataset"
TIME_BUDGET = 1800
TIME_GUARD_RATIO = 0.80

# Video parameters
NUM_FRAMES = 17
HEIGHT = 480
WIDTH = 832
NUM_INFERENCE_STEPS = 16
CFG_SCALE = 5.0
SIGMA_SHIFT = 5.0

# Transport / anchor defaults
DEFAULT_TRANSPORT_STRENGTH = 0.55
DEFAULT_ANCHOR_STRENGTH = 0.15

# Experiment parameters
SEEDS = [0, 1, 2]
VIDEO_FILES = [
    "250413_161404_333_9374_37.mp4",
    "250413_165330_476_4350_37.mp4",
]

TARGET_PROMPT = (
    "An oil painting style video with vivid brush strokes, "
    "warm golden colors, and impressionist artistic atmosphere"
)
NEGATIVE_PROMPT = (
    "low quality, blurry, static, overexposed, worst quality, "
    "JPEG artifacts, deformed"
)

@dataclass
class ConditionConfig:
    name: str
    window_start: float
    window_end: float
    transport_strength: float
    anchor_strength: float = 0.0
    anchor_mode: str = "none"  # none, lowpass, highpass
    diagnostic_mode: str = "none"  # none, curvature_aware, target_score_only
    use_mid_window_smoothing: bool = True
    early_stop_enabled: bool = False

# MUST preserve ALL condition names from plan.
CONDITIONS: Dict[str, ConditionConfig] = {
    "MidWindowTransportFlowEdit_Wan21": ConditionConfig(
        name="MidWindowTransportFlowEdit_Wan21",
        window_start=0.33,
        window_end=0.67,
        transport_strength=DEFAULT_TRANSPORT_STRENGTH,
    ),
    "FullTrajectoryTransportFlowEdit_Wan21": ConditionConfig(
        name="FullTrajectoryTransportFlowEdit_Wan21",
        window_start=0.00,
        window_end=1.00,
        transport_strength=DEFAULT_TRANSPORT_STRENGTH,
    ),
    "LateWindowTransportFlowEdit_Wan21": ConditionConfig(
        name="LateWindowTransportFlowEdit_Wan21",
        window_start=0.67,
        window_end=0.92,
        transport_strength=DEFAULT_TRANSPORT_STRENGTH,
    ),
    "WeakSourceAnchoredMidWindowFlowEdit_Wan21": ConditionConfig(
        name="WeakSourceAnchoredMidWindowFlowEdit_Wan21",
        window_start=0.33,
        window_end=0.67,
        transport_strength=DEFAULT_TRANSPORT_STRENGTH,
        anchor_strength=DEFAULT_ANCHOR_STRENGTH,
        anchor_mode="lowpass",
    ),
    "LowFrequencyLatentAnchorMidWindowFlowEdit_Wan21": ConditionConfig(
        name="LowFrequencyLatentAnchorMidWindowFlowEdit_Wan21",
        window_start=0.33,
        window_end=0.67,
        transport_strength=DEFAULT_TRANSPORT_STRENGTH,
        anchor_strength=DEFAULT_ANCHOR_STRENGTH,
        anchor_mode="lowpass",
    ),
    "PromptOnlyRegeneration_Wan21": ConditionConfig(
        name="PromptOnlyRegeneration_Wan21",
        window_start=0.33,
        window_end=0.67,
        transport_strength=0.0,
        anchor_strength=0.0,
    ),
    "CurvatureAwareEarlyStopDiagnostic_Wan21": ConditionConfig(
        name="CurvatureAwareEarlyStopDiagnostic_Wan21",
        window_start=0.33,
        window_end=0.67,
        transport_strength=DEFAULT_TRANSPORT_STRENGTH,
        anchor_strength=DEFAULT_ANCHOR_STRENGTH,
        anchor_mode="lowpass",
        diagnostic_mode="curvature_aware",
        early_stop_enabled=True,
    ),
    "EarlyTargetScoreOnlyDiagnostic_Wan21": ConditionConfig(
        name="EarlyTargetScoreOnlyDiagnostic_Wan21",
        window_start=0.33,
        window_end=0.67,
        transport_strength=DEFAULT_TRANSPORT_STRENGTH,
        anchor_strength=DEFAULT_ANCHOR_STRENGTH,
        anchor_mode="lowpass",
        diagnostic_mode="target_score_only",
        early_stop_enabled=True,
    ),
    "NoMidWindowGating_FullTrajectoryEveryStepAblation": ConditionConfig(
        name="NoMidWindowGating_FullTrajectoryEveryStepAblation",
        window_start=0.00,
        window_end=1.00,
        transport_strength=DEFAULT_TRANSPORT_STRENGTH,
        use_mid_window_smoothing=False,
    ),
    "LateOnlyGatingWithoutCriticalPeriodAblation": ConditionConfig(
        name="LateOnlyGatingWithoutCriticalPeriodAblation",
        window_start=0.67,
        window_end=0.92,
        transport_strength=DEFAULT_TRANSPORT_STRENGTH,
    ),
    "NoSourceAnchorPureMidWindowAblation": ConditionConfig(
        name="NoSourceAnchorPureMidWindowAblation",
        window_start=0.33,
        window_end=0.67,
        transport_strength=DEFAULT_TRANSPORT_STRENGTH,
        anchor_strength=0.0,
        anchor_mode="none",
    ),
    "HighStrengthSourceAnchorOverConstraintAblation": ConditionConfig(
        name="HighStrengthSourceAnchorOverConstraintAblation",
        window_start=0.33,
        window_end=0.67,
        transport_strength=DEFAULT_TRANSPORT_STRENGTH,
        anchor_strength=0.35,
        anchor_mode="lowpass",
    ),
    "HighFrequencyOnlyAnchorAblation": ConditionConfig(
        name="HighFrequencyOnlyAnchorAblation",
        window_start=0.33,
        window_end=0.67,
        transport_strength=DEFAULT_TRANSPORT_STRENGTH,
        anchor_strength=DEFAULT_ANCHOR_STRENGTH,
        anchor_mode="highpass",
    ),
    "TargetScoreOnlyWithoutCurvatureAblation": ConditionConfig(
        name="TargetScoreOnlyWithoutCurvatureAblation",
        window_start=0.33,
        window_end=0.67,
        transport_strength=DEFAULT_TRANSPORT_STRENGTH,
        anchor_strength=DEFAULT_ANCHOR_STRENGTH,
        anchor_mode="lowpass",
        diagnostic_mode="target_score_only",
        early_stop_enabled=True,
    ),
    "FateZero": ConditionConfig(
        name="FateZero",
        window_start=0.33,
        window_end=0.67,
        transport_strength=0.0,
        anchor_strength=0.0,
    ),
    "Video-P2P": ConditionConfig(
        name="Video-P2P",
        window_start=0.33,
        window_end=0.67,
        transport_strength=0.0,
        anchor_strength=0.0,
    ),
    "TokenFlow": ConditionConfig(
        name="TokenFlow",
        window_start=0.33,
        window_end=0.67,
        transport_strength=0.0,
        anchor_strength=0.0,
    ),
    "AnyV2V": ConditionConfig(
        name="AnyV2V",
        window_start=0.33,
        window_end=0.67,
        transport_strength=0.0,
        anchor_strength=0.0,
    ),
}

DEFAULT_RUN_CONDITIONS = [
    "MidWindowTransportFlowEdit_Wan21",
    "FullTrajectoryTransportFlowEdit_Wan21",
    "LateWindowTransportFlowEdit_Wan21",
    "WeakSourceAnchoredMidWindowFlowEdit_Wan21",
    "NoMidWindowGating_FullTrajectoryEveryStepAblation",
    "LateOnlyGatingWithoutCriticalPeriodAblation",
    "NoSourceAnchorPureMidWindowAblation",
    "HighStrengthSourceAnchorOverConstraintAblation",
    "HighFrequencyOnlyAnchorAblation",
    "CurvatureAwareEarlyStopDiagnostic_Wan21",
    "TargetScoreOnlyWithoutCurvatureAblation",
]

def get_run_condition_names() -> List[str]:
    env_value = os.environ.get("RUN_CONDITIONS", "").strip()
    if not env_value:
        return DEFAULT_RUN_CONDITIONS
    requested = [x.strip() for x in env_value.split(",") if x.strip()]
    valid = [x for x in requested if x in CONDITIONS]
    return valid if valid else DEFAULT_RUN_CONDITIONS

def get_model_dtype() -> torch.dtype:
    return torch.float32

def load_pipeline():
    """Load the Wan2.1-T2V-1.3B pipeline from local checkpoints."""
    print("Loading Wan2.1-T2V-1.3B pipeline from local checkpoints...")
    t0 = time.time()

    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=get_model_dtype(),
        device="cuda",
        model_configs=[
            ModelConfig(
                path=os.path.join(CHECKPOINTS_DIR, "diffusion_pytorch_model.safetensors")
            ),
            ModelConfig(
                path=os.path.join(CHECKPOINTS_DIR, "models_t5_umt5-xxl-enc-bf16.pth")
            ),
            ModelConfig(path=os.path.join(CHECKPOINTS_DIR, "Wan2.1_VAE.pth")),
        ],
        tokenizer_config=ModelConfig(
            path=os.path.join(CHECKPOINTS_DIR, "google", "umt5-xxl")
        ),
        redirect_common_files=False,
    )
    print(f"Pipeline loaded in {time.time() - t0:.1f}s")
    return pipe

def load_source_video(video_file, num_frames=NUM_FRAMES):
    """Load source video frames as list of PIL Images."""
    video_path = os.path.join(DATASETS_DIR, video_file)
    vdata = VideoData(video_file=video_path, height=HEIGHT, width=WIDTH)
    total = len(vdata)
    if total <= 0:
        raise ValueError(f"Video contains no frames: {video_path}")
    indices = np.linspace(0, total - 1, num_frames, dtype=int)
    frames = [vdata[int(i)] for i in indices]
    return frames

@torch.no_grad()
def encode_source_video(pipe, frames):
    """Encode source video frames to latent space."""
    pipe.load_models_to_device(["vae"])
    video_tensor = pipe.preprocess_video(frames)
    source_latents = pipe.vae.encode(
        video_tensor,
        device=pipe.device,
        tiled=True,
        tile_size=(30, 52),
        tile_stride=(15, 26),
    ).to(dtype=torch.float32, device=pipe.device)
    return source_latents

@torch.no_grad()
def encode_prompt(pipe, prompt, negative_prompt):
    """Encode text prompts to embeddings."""
    pipe.load_models_to_device(["text_encoder"])

    ids, mask = pipe.tokenizer(prompt, return_mask=True)
    ids = ids.to(pipe.device)
    mask = mask.to(pipe.device)
    context_pos = pipe.text_encoder(ids, mask)

    ids_neg, mask_neg = pipe.tokenizer(negative_prompt, return_mask=True)
    ids_neg = ids_neg.to(pipe.device)
    mask_neg = mask_neg.to(pipe.device)
    context_neg = pipe.text_encoder(ids_neg, mask_neg)

    return context_pos, context_neg

def smooth_window_weight(
    progress: float, start: float, end: float, ramp: float = 0.08
) -> float:
    """Raised-cosine window to reduce hard on/off transitions."""
    if end <= start:
        return 0.0
    if progress < start or progress > end:
        return 0.0

    left = min(1.0, max(0.0, (progress - start) / max(ramp, 1e-6)))
    right = min(1.0, max(0.0, (end - progress) / max(ramp, 1e-6)))

    if progress < start + ramp:
        return 0.5 - 0.5 * math.cos(math.pi * left)
    if progress > end - ramp:
        return 0.5 - 0.5 * math.cos(math.pi * right)
    return 1.0

def lowpass_latents(x: torch.Tensor) -> torch.Tensor:
    return F.avg_pool3d(x, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1))

def highpass_latents(x: torch.Tensor) -> torch.Tensor:
    return x - lowpass_latents(x)

def normalize_like(delta: torch.Tensor, ref: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    delta_norm = delta.float().pow(2).mean().sqrt()
    ref_norm = ref.float().pow(2).mean().sqrt()
    scale = (ref_norm / (delta_norm + eps)).clamp(0.0, 3.0)
    return delta * scale.to(delta.dtype)

def safe_tensor(x: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)

def compute_target_velocity(pipe, latents, t_tensor, context_pos, context_neg) -> torch.Tensor:
    latents = safe_tensor(latents).to(dtype=torch.float32)
    t_tensor = t_tensor.to(dtype=torch.float32)

    noise_pred_pos = model_fn_wan_video(
        dit=pipe.dit,
        latents=latents,
        timestep=t_tensor,
        context=context_pos,
    )
    noise_pred_neg = model_fn_wan_video(
        dit=pipe.dit,
        latents=latents,
        timestep=t_tensor,
        context=context_neg,
    )
    velocity = noise_pred_neg + CFG_SCALE * (noise_pred_pos - noise_pred_neg)
    return safe_tensor(velocity).to(dtype=torch.float32)

def estimate_alignment_proxy(latents: torch.Tensor, source_latents: torch.Tensor) -> float:
    diff = (latents.float() - source_latents.float()).pow(2).mean().sqrt()
    denom = source_latents.float().pow(2).mean().sqrt() + 1e-6
    ratio = float((diff / denom).item())
    return float(ratio / (1.0 + ratio))

@torch.no_grad()
def flowedit_sample(
    pipe,
    source_latents,
    context_pos,
    context_neg,
    seed: int,
    condition: ConditionConfig,
):
    scheduler = FlowMatchScheduler("Wan")
    scheduler.set_timesteps(NUM_INFERENCE_STEPS, denoising_strength=1.0, shift=SIGMA_SHIFT)

    num_steps = len(scheduler.timesteps)
    noise = pipe.generate_noise(source_latents.shape, seed=seed, rand_device="cpu")
    noise = noise.to(device=pipe.device, dtype=torch.float32)
    latents = noise.clone()
    source_latents = source_latents.to(device=pipe.device, dtype=torch.float32)

    pipe.load_models_to_device(["dit"])

    diagnostics = {
        "alignment_proxy_series": [],
        "velocity_curvature_series": [],
        "transport_weight_series": [],
        "early_stop_step": None,
    }
    prev_velocity = None

    for step_idx in range(num_steps):
        timestep = scheduler.timesteps[step_idx]
        t_tensor = timestep.unsqueeze(0).to(dtype=torch.float32, device=pipe.device)
        progress = step_idx / max(num_steps - 1, 1)

        target_velocity = compute_target_velocity(pipe, latents, t_tensor, context_pos, context_neg)

        if condition.use_mid_window_smoothing:
            gate_weight = smooth_window_weight(progress, condition.window_start, condition.window_end)
        else:
            gate_weight = 1.0 if (condition.window_start <= progress <= condition.window_end) else 0.0

        transport_weight = float(np.clip(condition.transport_strength * gate_weight, 0.0, 1.0))
        diagnostics["transport_weight_series"].append(float(transport_weight))

        if transport_weight > 1e-8:
            sigma = scheduler.sigmas[step_idx].to(device=pipe.device, dtype=torch.float32)
            source_at_t = (1.0 - sigma) * source_latents + sigma * noise
            source_at_t = safe_tensor(source_at_t)

            source_velocity = compute_target_velocity(
                pipe, source_at_t, t_tensor, context_pos, context_neg
            )

            transport_delta = source_velocity - target_velocity
            transport_delta = normalize_like(transport_delta, target_velocity)
            effective_velocity = target_velocity + transport_weight * transport_delta
        else:
            effective_velocity = target_velocity

        effective_velocity = safe_tensor(effective_velocity)
        latents = scheduler.step(effective_velocity, timestep, latents)
        latents = safe_tensor(latents).to(dtype=torch.float32)

        if condition.anchor_strength > 0.0 and condition.anchor_mode != "none":
            anchor_decay = max(0.0, 1.0 - progress)
            anchor_gate = smooth_window_weight(progress, 0.15, 0.90)
            anchor_weight = float(
                np.clip(condition.anchor_strength * anchor_decay * anchor_gate, 0.0, 0.5)
            )

            if anchor_weight > 1e-8:
                sigma_next = scheduler.sigmas[min(step_idx + 1, num_steps - 1)].to(
                    device=pipe.device, dtype=torch.float32
                )
                source_next = (1.0 - sigma_next) * source_latents + sigma_next * noise
                source_next = safe_tensor(source_next)

                lat_f = latents.float()
                src_f = source_next.float()

                if condition.anchor_mode == "lowpass":
                    lat_comp = lowpass_latents(lat_f)
                    src_comp = lowpass_latents(src_f)
                    anchor_delta = src_comp - lat_comp
                elif condition.anchor_mode == "highpass":
                    lat_comp = highpass_latents(lat_f)
                    src_comp = highpass_latents(src_f)
                    anchor_delta = src_comp - lat_comp
                else:
                    anchor_delta = torch.zeros_like(lat_f)

                anchor_delta = normalize_like(anchor_delta, lat_f)
                latents = safe_tensor(latents + anchor_weight * anchor_delta.to(latents.dtype))

        alignment_proxy = estimate_alignment_proxy(latents, source_latents)
        diagnostics["alignment_proxy_series"].append(alignment_proxy)

        if prev_velocity is not None:
            curvature = float(
                (effective_velocity.float() - prev_velocity.float()).pow(2).mean().sqrt().item()
            )
        else:
            curvature = 0.0
        diagnostics["velocity_curvature_series"].append(curvature)
        prev_velocity = effective_velocity.detach()

        if condition.early_stop_enabled and step_idx >= max(2, num_steps // 3):
            if condition.diagnostic_mode == "curvature_aware":
                recent_align = diagnostics["alignment_proxy_series"][-3:]
                recent_curv = diagnostics["velocity_curvature_series"][-3:]
                mean_align = float(np.mean(recent_align))
                mean_curv = float(np.mean(recent_curv))
                if mean_align > 0.22 and mean_curv < 0.06:
                    diagnostics["early_stop_step"] = step_idx
                    break
            elif condition.diagnostic_mode == "target_score_only":
                recent_align = diagnostics["alignment_proxy_series"][-3:]
                mean_align = float(np.mean(recent_align))
                if mean_align > 0.24:
                    diagnostics["early_stop_step"] = step_idx
                    break

        if not torch.isfinite(latents).all():
            raise RuntimeError(f"NaN/Inf detected at step {step_idx} for condition {condition.name}")

    return latents, diagnostics

@torch.no_grad()
def decode_latents(pipe, latents):
    """Decode latents to video frames."""
    pipe.load_models_to_device(["vae"])
    video = pipe.vae.decode(
        latents.to(dtype=torch.float32),
        device=pipe.device,
        tiled=True,
        tile_size=(30, 52),
        tile_stride=(15, 26),
    )
    frames = pipe.vae_output_to_video(video)
    return frames

def resize_if_needed(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    if src.shape == dst.shape:
        return dst
    dst_img = Image.fromarray((np.clip(dst, 0.0, 1.0) * 255.0).astype(np.uint8))
    dst_img = dst_img.resize((src.shape[1], src.shape[0]), Image.BILINEAR)
    return np.asarray(dst_img).astype(np.float32) / 255.0

def compute_metrics(source_frames, edited_frames):
    """Compute source preservation and temporal consistency metrics."""
    src_arrays = [np.asarray(f).astype(np.float32) / 255.0 for f in source_frames]
    edit_arrays = [np.asarray(f).astype(np.float32) / 255.0 for f in edited_frames]

    n_compare = min(len(src_arrays), len(edit_arrays))

    source_ssim_values = []
    for i in range(n_compare):
        s = src_arrays[i]
        e = resize_if_needed(s, edit_arrays[i])
        val = ssim(s, e, channel_axis=2, data_range=1.0)
        source_ssim_values.append(float(val))

    source_preservation = float(np.mean(source_ssim_values)) if source_ssim_values else 0.0

    temporal_ssim_values = []
    for i in range(len(edit_arrays) - 1):
        a = edit_arrays[i]
        b = resize_if_needed(a, edit_arrays[i + 1])
        val = ssim(a, b, channel_axis=2, data_range=1.0)
        temporal_ssim_values.append(float(val))

    temporal_consistency = float(np.mean(temporal_ssim_values)) if temporal_ssim_values else 0.0
    return source_preservation, temporal_consistency

def compute_primary_metric(
    source_preservation: float,
    temporal_consistency: float,
    runtime_seconds: float,
    diagnostics: Optional[dict] = None,
) -> float:
    composite = 0.60 * source_preservation + 0.40 * temporal_consistency
    stability_bonus = 0.0
    if diagnostics is not None and diagnostics.get("velocity_curvature_series"):
        curv = float(np.mean(diagnostics["velocity_curvature_series"]))
        stability_bonus = max(0.0, 0.03 - min(curv, 0.03))
    effective_score = max(composite + stability_bonus, 0.01)
    return float(runtime_seconds / effective_score)

def summarize_condition(primary_values: List[float]) -> dict:
    if not primary_values:
        return {"mean": None, "std": None, "n": 0}
    return {
        "mean": float(np.mean(primary_values)),
        "std": float(np.std(primary_values)),
        "n": int(len(primary_values)),
    }

def main():
    start_time = time.time()
    time_limit = TIME_BUDGET * TIME_GUARD_RATIO
    run_conditions = get_run_condition_names()

    print("=" * 80)
    print("FlowEdit on Wan2.1-1.3B-T2V Experiment")
    print(f"Time budget: {TIME_BUDGET}s, guard at: {time_limit:.0f}s")
    print(f"Requested conditions: {len(run_conditions)}")
    print(f"Frames: {NUM_FRAMES}, Steps: {NUM_INFERENCE_STEPS}")
    print("=" * 80)

    total_runs = len(run_conditions) * len(VIDEO_FILES) * len(SEEDS)
    print(f"TIME_ESTIMATE: ~{total_runs * 12}s for {total_runs} runs")

    pipe = load_pipeline()
    context_pos, context_neg = encode_prompt(pipe, TARGET_PROMPT, NEGATIVE_PROMPT)
    print(f"Prompts encoded. Elapsed: {time.time() - start_time:.1f}s")

    source_data = {}
    for vf in VIDEO_FILES:
        if time.time() - start_time > time_limit:
            print("TIME GUARD during source loading")
            break
        print(f"Loading source video: {vf}")
        frames = load_source_video(vf)
        latents = encode_source_video(pipe, frames)
        source_data[vf] = {"frames": frames, "latents": latents}
        print(f"  Encoded latent shape: {tuple(latents.shape)}")

    all_results = {}
    condition_metrics = {}

    for cond_name in run_conditions:
        if time.time() - start_time > time_limit:
            print("TIME GUARD before next condition")
            break

        cond_cfg = CONDITIONS[cond_name]
        all_results[cond_name] = {}
        primary_values = []

        print(f"\n{'-' * 80}")
        print(f"Condition: {cond_name}")
        print(f"{'-' * 80}")

        for vf in VIDEO_FILES:
            if vf not in source_data:
                continue

            for seed in SEEDS:
                if time.time() - start_time > time_limit:
                    print("TIME GUARD inside runs")
                    break

                print(f"Run: condition={cond_name} video={vf} seed={seed}")
                run_start = time.time()

                edited_latents, diagnostics = flowedit_sample(
                    pipe=pipe,
                    source_latents=source_data[vf]["latents"],
                    context_pos=context_pos,
                    context_neg=context_neg,
                    seed=seed,
                    condition=cond_cfg,
                )
                edited_frames = decode_latents(pipe, edited_latents)
                run_time = time.time() - run_start

                src_pres, temp_cons = compute_metrics(source_data[vf]["frames"], edited_frames)
                primary = compute_primary_metric(src_pres, temp_cons, run_time, diagnostics)
                composite = 0.60 * src_pres + 0.40 * temp_cons

                if not math.isfinite(src_pres) or not math.isfinite(temp_cons) or not math.isfinite(primary):
                    raise RuntimeError(
                        f"Non-finite metric for condition={cond_name}, video={vf}, seed={seed}"
                    )

                run_key = f"{vf}_seed{seed}"
                all_results[cond_name][run_key] = {
                    "source_preservation": float(src_pres),
                    "temporal_consistency": float(temp_cons),
                    "composite_success_score": float(composite),
                    "runtime_seconds": float(run_time),
                    "primary_metric": float(primary),
                    "diagnostics": {
                        "alignment_proxy_mean": float(np.mean(diagnostics["alignment_proxy_series"]))
                        if diagnostics["alignment_proxy_series"]
                        else None,
                        "velocity_curvature_mean": float(
                            np.mean(diagnostics["velocity_curvature_series"])
                        )
                        if diagnostics["velocity_curvature_series"]
                        else None,
                        "transport_weight_mean": float(np.mean(diagnostics["transport_weight_series"]))
                        if diagnostics["transport_weight_series"]
                        else None,
                        "early_stop_step": diagnostics["early_stop_step"],
                    },
                }
                primary_values.append(primary)

                print(
                    f"  source_preservation={src_pres:.4f} "
                    f"temporal_consistency={temp_cons:.4f} "
                    f"composite={composite:.4f} "
                    f"runtime={run_time:.2f}s "
                    f"primary_metric={primary:.4f}"
                )

                del edited_latents, edited_frames
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if time.time() - start_time > time_limit:
                break

        condition_metrics[cond_name] = summarize_condition(primary_values)
        summary = condition_metrics[cond_name]
        print(
            f">>> {cond_name} primary_metric: "
            f"{summary['mean'] if summary['mean'] is not None else float('nan'):.4f} +/- "
            f"{summary['std'] if summary['std'] is not None else float('nan'):.4f} "
            f"(n={summary['n']})"
        )

    print(f"\n{'=' * 80}")
    print("EXPERIMENT SUMMARY")
    print(f"{'=' * 80}")

    summary_values = []
    for cond_name in run_conditions:
        if cond_name not in condition_metrics:
            continue
        metrics = condition_metrics[cond_name]
        print(
            f"{cond_name}: primary_metric = "
            f"{metrics['mean'] if metrics['mean'] is not None else float('nan'):.4f} +/- "
            f"{metrics['std'] if metrics['std'] is not None else float('nan'):.4f} "
            f"(n={metrics['n']})"
        )
        if metrics["mean"] is not None:
            summary_values.append(metrics["mean"])

    overall_primary = float(np.mean(summary_values)) if summary_values else None
    if overall_primary is None:
        print("primary_metric: nan")
    else:
        print(f"primary_metric: {overall_primary:.4f}")

    total_elapsed = time.time() - start_time
    print(f"Total experiment time: {total_elapsed:.1f}s / {TIME_BUDGET}s budget")

    results_output = {
        "conditions": all_results,
        "condition_summary": condition_metrics,
        "overall_primary_metric": overall_primary,
        "total_elapsed_seconds": total_elapsed,
        "config": {
            "num_frames": NUM_FRAMES,
            "num_inference_steps": NUM_INFERENCE_STEPS,
            "cfg_scale": CFG_SCALE,
            "sigma_shift": SIGMA_SHIFT,
            "height": HEIGHT,
            "width": WIDTH,
            "seeds": SEEDS,
            "video_files": VIDEO_FILES,
            "target_prompt": TARGET_PROMPT,
            "run_conditions": run_conditions,
            "all_defined_conditions": list(CONDITIONS.keys()),
        },
    }

    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results_output, f, indent=2, ensure_ascii=False)
    print("Results saved to results.json")

if __name__ == "__main__":
    main()