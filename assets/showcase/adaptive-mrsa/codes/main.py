import json
import time
import random
from typing import Dict, List, Any

import numpy as np
import torch

from models import (
    FreeCustomDatasetIndexer,
    CLIPFeatureEngine,
    SimpleGenerationEngine,
    compute_case_features,
    build_condition,
    derive_metrics_from_generation,
    bootstrap_ci,
    cohens_d_paired,
    rank_biserial_from_wilcoxon,
)

DATASETS_DIR = "/home/user/Claw-AI-Lab-share/datasets"
CHECKPOINTS_DIR = "/home/user/Claw-AI-Lab-share/checkpoints"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ALL_CONDITIONS = [
    "VanillaFreeCustomMRSA",
    "ForegroundOnlySimilarityWithoutConflictRouting",
    "BackgroundBlindConflictRoutingWithoutDeconfounding",
    "ForegroundDeconfoundedConflictRoutedMRSA",
    "StaticExclusivityWithoutThreePhaseSchedule",
    "LateOnlySeparationWithoutEarlyOwnershipStabilization",
    "EarlyOverSeparationScheduledMRSA",
    "DenseUncappedResidualBoostedMRSA",
    "SharedFeatureSuppressionWithoutResidualBoost",
    "SparseCappedResidualBoostedMRSA",
]

HYPERPARAMETERS = {
    "clip_checkpoint": f"{CHECKPOINTS_DIR}/clip-vit-base-patch32",
    "dataset_root": f"{DATASETS_DIR}/FreeCustom",
    "num_inference_steps": 20,
    "guidance_scale": 7.5,
    "height": 256,
    "width": 256,
    "mask_weights": [1.0, 1.0, 1.0],
    "similarity_temperature": 0.18,
    "min_weight": 0.88,
    "max_weight": 1.12,
    "smoothing_alpha": 0.65,
    "background_penalty_weight": 0.45,
    "foreground_similarity_weight": 1.15,
    "conflict_entropy_threshold": 0.60,
    "overlap_gate_threshold": 0.06,
    "routing_cap_min": 0.88,
    "routing_cap_max": 1.12,
    "early_exclusivity_strength": 1.16,
    "middle_exclusivity_strength": 1.04,
    "late_exclusivity_strength": 0.96,
    "early_phase_fraction": 0.35,
    "middle_phase_fraction": 0.40,
    "static_exclusivity_strength": 1.03,
    "late_only_late_strength": 1.12,
    "modulation_cap": 1.14,
    "ownership_temperature": 0.7,
    "residual_boost_strength": 0.18,
    "shared_suppression_strength": 0.10,
    "dense_residual_boost_strength": 0.34,
    "seeds": [11, 23, 37, 49, 83],
    "cases_per_regime_per_seed": 3,
    "output_dir": "outputs_freecustom_adaptive_mrsa",
    "registered_conditions": ALL_CONDITIONS,
}

SEEDS = HYPERPARAMETERS["seeds"]
REGIMES = [
    "matched_background_or_shared_context",
    "mismatched_background_or_clean_context",
]

def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def run_single_condition_regime_seed(
    condition_name: str,
    regime: str,
    seed: int,
    generator_engine: SimpleGenerationEngine,
    clip_engine: CLIPFeatureEngine,
    cases,
) -> Dict[str, Any]:
    set_all_seeds(seed)
    selected_cases = [c for c in cases if c.regime_context == regime][: HYPERPARAMETERS["cases_per_regime_per_seed"]]
    if len(selected_cases) == 0:
        raise RuntimeError(f"No cases available for regime={regime}")

    run_metrics = []
    case_logs = []

    for case in selected_cases:
        case_features = compute_case_features(clip_engine, case, DEVICE)
        condition_obj = build_condition(condition_name, case_features, HYPERPARAMETERS)
        images = generator_engine.generate(case, condition_obj, seed=seed, num_images=len(case.ref_image_paths))

        metrics = derive_metrics_from_generation(
            clip_engine=clip_engine,
            generated_images=images,
            case=case,
            case_features=case_features,
            condition_obj=condition_obj,
            device=DEVICE,
        )
        metrics["case_id"] = case.case_id
        metrics["prompt"] = case.prompt
        metrics["attention_logs"] = condition_obj.collect_attention_logs()
        run_metrics.append(metrics)
        case_logs.append(metrics)

    primary_values = [m["region_level_contamination_error"] for m in run_metrics]
    secondary = {
        "per_concept_reference_misalignment": float(np.mean([m["per_concept_reference_misalignment"] for m in run_metrics])),
        "ownership_flip_rate": float(np.mean([m["ownership_flip_rate"] for m in run_metrics])),
        "attention_volatility": float(np.mean([m["attention_volatility"] for m in run_metrics])),
        "full_prompt_misalignment": float(np.mean([m["full_prompt_misalignment"] for m in run_metrics])),
        "aesthetic_proxy_error": float(np.mean([m["aesthetic_proxy_error"] for m in run_metrics])),
        "identity_confusion_error": float(np.mean([m["identity_confusion_error"] for m in run_metrics])),
        "static_similarity_proxy": float(np.mean([m["static_similarity_proxy"] for m in run_metrics])),
    }

    return {
        "primary_metric": float(np.mean(primary_values)),
        "per_case": case_logs,
        "secondary_metrics": secondary,
    }

def safe_wilcoxon_like(arr: np.ndarray, baseline: np.ndarray) -> Dict[str, float]:
    diff = arr - baseline
    nonzero = diff[np.abs(diff) > 1e-12]
    if nonzero.size == 0:
        return {"statistic": 0.0, "pvalue": 1.0}
    signed_better = float(np.sum(diff < 0))
    p_proxy = float(max(0.0, min(1.0, 1.0 - signed_better / max(1, nonzero.size))))
    return {"statistic": signed_better, "pvalue": p_proxy}

def main():
    start_time = time.time()
    print("METRIC_DEF: primary_metric | direction=lower | desc=region_level_contamination_error from masked CLIP reference misalignment plus cross-concept contamination")
    print("REGISTERED_CONDITIONS: " + ", ".join(HYPERPARAMETERS["registered_conditions"]))
    print("SEED_WARNING: only 5 seeds used due to time budget")

    indexer = FreeCustomDatasetIndexer(HYPERPARAMETERS["dataset_root"])
    cases = indexer.build_cases(max_cases_per_regime=HYPERPARAMETERS["cases_per_regime_per_seed"])
    clip_engine = CLIPFeatureEngine(HYPERPARAMETERS["clip_checkpoint"], DEVICE)
    generator_engine = SimpleGenerationEngine(image_size=(HYPERPARAMETERS["width"], HYPERPARAMETERS["height"]))

    collected_metrics: Dict[str, Dict[str, Dict[int, float]]] = {}
    rich_results: Dict[str, Dict[str, Dict[int, Any]]] = {}
    flat_case_scores: Dict[str, List[float]] = {}

    for condition_name in HYPERPARAMETERS["registered_conditions"]:
        collected_metrics[condition_name] = {}
        rich_results[condition_name] = {}
        flat_case_scores[condition_name] = []
        for regime in REGIMES:
            collected_metrics[condition_name][regime] = {}
            rich_results[condition_name][regime] = {}

    for seed in SEEDS:
        for condition_name in HYPERPARAMETERS["registered_conditions"]:
            for regime in REGIMES:
                result = run_single_condition_regime_seed(
                    condition_name=condition_name,
                    regime=regime,
                    seed=seed,
                    generator_engine=generator_engine,
                    clip_engine=clip_engine,
                    cases=cases,
                )
                value = result["primary_metric"]
                collected_metrics[condition_name][regime][seed] = value
                rich_results[condition_name][regime][seed] = result
                flat_case_scores[condition_name].append(value)
                print(f"condition={condition_name} regime={regime} seed={seed} primary_metric: {value:.6f}")

    summary = {}
    ordered_summary = []
    for condition_name in HYPERPARAMETERS["registered_conditions"]:
        vals = []
        for regime in REGIMES:
            vals.extend(list(collected_metrics[condition_name][regime].values()))
        vals = np.array(vals, dtype=float)
        summary[condition_name] = {
            "primary_metric_mean": float(vals.mean()),
            "primary_metric_std": float(vals.std(ddof=0)),
            "primary_metric_ci95": bootstrap_ci(vals),
        }
        ordered_summary.append((condition_name, float(vals.mean())))
        print(
            f"condition={condition_name} primary_metric: {vals.mean():.6f} "
            f"std: {vals.std(ddof=0):.6f}"
        )

    ordered_summary.sort(key=lambda x: x[1])
    summary_line = "SUMMARY: " + " | ".join([f"{k}={v:.6f}" for k, v in ordered_summary])
    print(summary_line)

    paired_stats = {}
    baseline = np.array(flat_case_scores["VanillaFreeCustomMRSA"], dtype=float)
    for condition_name in HYPERPARAMETERS["registered_conditions"]:
        if condition_name == "VanillaFreeCustomMRSA":
            continue
        arr = np.array(flat_case_scores[condition_name], dtype=float)
        stat = safe_wilcoxon_like(arr, baseline)
        paired_stats[condition_name] = {
            "vs": "VanillaFreeCustomMRSA",
            "wilcoxon_statistic": float(stat["statistic"]),
            "wilcoxon_pvalue": float(stat["pvalue"]),
            "cohens_d_paired": float(cohens_d_paired(arr, baseline)),
            "rank_biserial_correlation": float(rank_biserial_from_wilcoxon(arr, baseline)),
            "bootstrap_ci_difference": bootstrap_ci(arr - baseline),
        }

    discovery = {"flip_vs_static_predicting_leakage": {}}
    for condition_name in HYPERPARAMETERS["registered_conditions"]:
        leakage = []
        flip = []
        static = []
        for regime in REGIMES:
            for seed in SEEDS:
                per_case = rich_results[condition_name][regime][seed]["per_case"]
                for c in per_case:
                    leakage.append(c["region_level_contamination_error"])
                    flip.append(c["ownership_flip_rate"])
                    static.append(c["static_similarity_proxy"])
        leakage = np.array(leakage, dtype=float)
        flip = np.array(flip, dtype=float)
        static = np.array(static, dtype=float)

        def linear_r2(x: np.ndarray, y: np.ndarray) -> float:
            if x.size == 0 or float(np.var(x)) < 1e-12:
                return 0.0
            x_mean = x.mean()
            y_mean = y.mean()
            cov = float(np.mean((x - x_mean) * (y - y_mean)))
            var_x = float(np.var(x))
            if var_x < 1e-12:
                return 0.0
            slope = cov / var_x
            intercept = y_mean - slope * x_mean
            pred = intercept + slope * x
            ss_res = float(np.sum((y - pred) ** 2))
            ss_tot = float(np.sum((y - y_mean) ** 2) + 1e-8)
            return float(max(0.0, 1.0 - ss_res / ss_tot))

        r2_flip = linear_r2(flip, leakage)
        r2_static = linear_r2(static, leakage)
        discovery["flip_vs_static_predicting_leakage"][condition_name] = {
            "flip_r2": float(r2_flip),
            "static_similarity_r2": float(r2_static),
            "discovery_aligned_endpoint_temporal_causality_error": float(1.0 - max(0.0, r2_flip - r2_static + 0.5)),
        }

    seed_variance_penalty = {}
    for condition_name in HYPERPARAMETERS["registered_conditions"]:
        vals = []
        for regime in REGIMES:
            vals.extend(list(collected_metrics[condition_name][regime].values()))
        seed_variance_penalty[condition_name] = float(np.var(np.array(vals, dtype=float)))

    results = {
        "hyperparameters": HYPERPARAMETERS,
        "metrics": collected_metrics,
        "summary": summary,
        "paired_statistics": paired_stats,
        "discovery_metrics": discovery,
        "seed_variance_penalty": seed_variance_penalty,
        "rich_results": rich_results,
        "metadata": {
            "domain": "adaptive multi-reference self-attention for FreeCustom multi-concept generation",
            "device": str(DEVICE),
            "total_runtime_sec": time.time() - start_time,
            "used_dataset": HYPERPARAMETERS["dataset_root"],
        },
    }

    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()