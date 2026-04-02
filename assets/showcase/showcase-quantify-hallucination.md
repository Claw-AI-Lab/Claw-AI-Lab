# Quantifying Hallucination in Generated Video Models

**Project:** `i-want-to-quantify-the-hallucination-exh` · **Track:** Lab Explore

---

## 📄 Paper Title

> **Toward Quantifying Hallucination in Generated Video Model Outputs**

---

## 💡 Idea

Hallucination in generated videos is not just a matter of low visual quality. It also appears as **semantic state-change errors, object permanence failures, identity discontinuity, causal implausibility, and prompt-event completion failure**. This project frames video hallucination evaluation as a **multi-track benchmark problem**: instead of collapsing all failures into one generic realism score, it separates short-horizon faithfulness, long-horizon world consistency, intervention-based occlusion robustness, and denoising-time instability signals, then tests whether these tracks better align with human severity and downstream usability judgments.

---

## ⚙️ Pipeline Journey

| Field | Details |
| :--- | :--- |
| **Track** | Lab Explore — CV evaluation pipeline for generated-video hallucination |
| **Topic** | Quantify hallucination exhibited in generated results of video models |
| **Current Scope** | Showcase currently reflects the project **through code generation / sanity / resource planning only**; experiment results are not included yet |
| **Stages** | Available artifacts cover **S11 → S13**: experiment implementation → sanity check → resource planning |
| **Data** | Local `quant_hallu` dataset with generated videos and matched GT videos |
| **Model Anchor** | Local `Wan2.1-T2V-1.3B-Diffusers` checkpoint |
| **Compute Plan** | 1 x 24GB GPU, 16 vCPU, 64GB RAM recommended; 256x256 frames, 8 FPS, 16/32-frame windows |
| **Planned Conditions** | Registered evaluation plan with **19 total conditions**: 5 baselines + 6 proposed methods + 8 ablations |
| **Artifacts** | `EXPERIMENT_PLAN.yaml`, generated `main.py`, sanity traces, resource-planning outputs |

### Stage Breakdown

| Phase | Stages | Description |
| :--- | :--- | :--- |
| **L2 · Experiment Design** | S9 | Benchmark plan organized around 4 hypothesis families, 19 registered conditions, regime-wise reporting, and local-dataset-only execution |
| **L3 · Coding** | S11 → S13 | Generated experiment harness, implemented method classes and output writing, passed sanity check, then produced execution/resource plan |

---

## 🧪 Registered Evaluation Agenda

### Hypothesis Families

1. **Internal instability as early-warning signal**: denoising-time transition structure may predict future hallucination better than raw uncertainty or scene difficulty alone.
2. **Counterfactual occlusion debt as object-permanence metric**: matched original-vs-occluded clips may capture reappearance failure more directly than generic quality scores.
3. **Hallucination is multi-factor**: a factor-separated profile may be more faithful than a single global scalar.
4. **Application-dependent horizon relevance**: short-horizon faithfulness may matter more in creative T2V, while long-horizon consistency may add more value in embodied settings.

### Implemented Method Set

- **Baselines (5)**: CLIP-style temporal faithfulness, realism-only VBench-style proxy, scene-difficulty regressor, uncertainty-only latent variance, universal single-score aggregator
- **Proposed methods (6)**: denoising criticality forecaster, counterfactual occlusion debt evaluator, factor-separated hallucination profile, short-horizon predictor, long-horizon predictor, application-weighted multi-track composite
- **Ablations (8)**: identity-free occlusion debt, raw variance in place of transition structure, global weighting instead of application weighting, and other mechanism-removal variants registered in `EXPERIMENT_PLAN.yaml`

---

## 🔑 Key Code Snippets

### Multi-modal preprocessing scaffold

This benchmark does not rely on RGB frames alone. The preprocessing stage explicitly prepares motion, tracks, occlusion masks, and denoising-time hooks as reusable signals:

```python
def preprocessing(self, sample: VideoSample) -> Dict[str, object]:
    return {
        "rgb_frames": sample.long_frames,
        "optical_flow": self.extract_motion_magnitude(sample.long_frames),
        "object_tracks": self.object_tracks(sample.long_frames),
        "occlusion_masks": self.occlusion_masks(sample.long_frames),
        "denoising_intermediate_latents_when_generating_new_samples": "computed_via_hooks",
    }
```

### Counterfactual occlusion debt

Instead of only asking whether a clip looks realistic, the code constructs a synthetic occlusion intervention and measures whether entities reappear consistently in geometry, identity, and trajectory:

```python
def aggregate_occlusion_debt(self, iou: float, identity_similarity: float, trajectory_continuity: float, tracker_reliability: float) -> float:
    debt = 0.4 * (1 - iou) + 0.3 * (1 - identity_similarity) + 0.3 * (1 - trajectory_continuity)
    tracker_reliability_weighting_loss = self.lambda_tracker_reliability * (1 - tracker_reliability) * debt
    return float(debt + tracker_reliability_weighting_loss)
```

### Denoising criticality signal

One proposed track models hallucination as an **early-warning instability problem** by computing jump statistics, reversal behavior, sensitivity, and temporal concentration from intermediate latent states:

```python
def compute_critical_transition_score(self, sample: VideoSample) -> torch.Tensor:
    latents = self.extract_refinement_latents(sample)
    jumps = torch.linalg.vector_norm(latents[1:] - latents[:-1], dim=tuple(range(1, latents.ndim)))
    reversals = torch.sign(jumps[1:] - jumps[:-1]).lt(0).float().mean().unsqueeze(0)
    sensitivity = latents.var(dim=0).mean().unsqueeze(0)
    concentration = (jumps.max() / torch.clamp(jumps.sum(), min=1e-6)).unsqueeze(0)
    score = torch.cat([jumps.mean().unsqueeze(0), jumps.std().unsqueeze(0), reversals, sensitivity, concentration], dim=0)
    score = torch.nan_to_num(score, nan=0.0, posinf=1e3, neginf=-1e3)
    return score
```

---

## 💻 Generated Code

The generated experiment code already includes:

- Local dataset loading from generated-video and GT-video directories
- Windowed preprocessing for short-horizon and long-horizon analysis
- Proxy annotation construction for severity, permanence, rejection likelihood, and future-hallucination labels
- Condition classes for the full planned baseline / proposed-method set
- Output writing for `summary.json`, `benchmark_card.json`, representative GIFs, and frame grids

👉 Generated [`main.py`](quantifying-hallucination/main.py)

---


*Generated by Claw AI Lab pipeline · Lab Explore · showcase currently reflects code-generation stage only*
