Here are 4 unconventional, testable hypotheses that go beyond simple “similarity-aware weighting.”

---

## Hypothesis 1: **Anti-similarity inversion beats similarity amplification**
### Bold claim
When two concepts are highly similar, the best adaptive MRSA policy is often the opposite of the obvious one: **downweight the most similar reference features and amplify the residual, dissimilar features**, because identity-confusing overlap is concentrated in shared features while distinguishing evidence lives in the mismatch.

### Cross-domain inspiration
- **Immunology:** self/non-self discrimination works by reacting to difference, not sameness.
- **Siamese verification:** hard decisions often depend more on discriminative residuals than shared embeddings.
- **Style transfer disentanglement:** shared style/content components can be nuisance factors when preserving identity.

### Rationale grounded in literature gaps
The synthesis assumes similarity should guide scaling, but does not require that higher similarity implies higher weight. That is a hidden conventional assumption. In multi-concept composition, especially same-category identities, the shared subspace may be exactly where leakage occurs. Existing MRSA-like methods mostly fuse references or refine attention, but they do not explicitly test whether **difference subspaces are more useful than similarity subspaces** under overlap. This addresses:
- Gap 1: explicit similarity-conditioned policy
- Gap 2: similar/overlapping concepts
- Gap 4: feature-level grounding

A feasible implementation is training-free:
1. Compute per-reference visual embeddings from a frozen encoder.
2. For each pair of references, decompose token features into:
   - shared component
   - residual component
3. In contested attention regions, suppress shared components and boost residual ones.

This is more novel than scalar weighting because it changes **what part of a reference** is attended to, not just how much.

### Measurable prediction
On a small similarity-stress test set of same-class concept pairs, **residual-boosted MRSA** will improve mean per-concept CLIP-image alignment by **≥ 3% relative** over vanilla similarity-proportional MRSA, while reducing a leakage score (e.g. cross-concept attention contamination or identity confusion rate) by **≥ 10% relative**.

### Failure condition
Reject if, over at least 10 prompts × 3 seeds:
- CLIP-image alignment gain is **< 1% relative**, or
- leakage reduction is **< 5% relative**, or
- image quality drops by **> 0.02 LPIPS-equivalent fidelity proxy / > 5% aesthetic proxy** versus baseline.

### Estimated risk
**High**

---

## Hypothesis 2: **Attention should be routed by local surprise, not similarity**
### Bold claim
The most effective adaptive MRSA controller for multi-concept composition is not based on concept similarity directly, but on **local prediction surprise**: reference attention should increase where a reference explains a latent region poorly under the current denoising state, because confusion manifests as unresolved prediction error before it appears as visible leakage.

### Cross-domain inspiration
- **Predictive coding / neuroscience:** surprise drives resource allocation.
- **Active perception:** systems focus computation where current models fail.
- **Mixture-of-experts routing:** gating by residual error can outperform static similarity gating.

### Rationale grounded in literature gaps
The synthesis highlights that static one-shot similarity is likely noisy and that temporal dynamics matter. Prior project notes also warn that one-shot similarity may be too weak as the sole intervention signal. A more radical alternative is to discard similarity as the primary gate and instead treat it as prior context, while routing MRSA according to **which reference best reduces local denoising uncertainty/residual**.

Concretely:
1. For each denoising step, estimate per-region disagreement between base latent prediction and each reference-conditioned prediction.
2. Route more attention to the reference whose injection most reduces local prediction entropy/residual.
3. Only use similarity as a weak regularizer.

This is novel because it redefines adaptive MRSA from a similarity controller into an **online error-correcting controller**.

### Measurable prediction
A surprise-routed MRSA will outperform similarity-only MRSA on a 2–3 concept benchmark by **≥ 5 percentage points** on a region-concept assignment metric (e.g. segmentation-conditioned CLIP match or masked DINO similarity), especially in the highest-similarity subset.

### Failure condition
Reject if, over at least 8–12 similarity-stress prompts × 3 seeds:
- region-concept assignment improves by **< 2 percentage points**, or
- high-similarity subset shows **no advantage**, or
- runtime overhead exceeds **1.5×** baseline and total run no longer fits within ~30 minutes on one GPU.

### Estimated risk
**Medium-high**

---

## Hypothesis 3: **Intentional early-step over-separation creates better final blending**
### Bold claim
The best final multi-concept composition may come from a counter-intuitive denoising schedule: **force concepts to become more separated than necessary in early steps**, then relax barriers later. Temporary over-fragmentation can reduce irreversible identity fusion.

### Cross-domain inspiration
- **Embryonic morphogenesis:** coarse compartment boundaries are established before fine integration.
- **Curriculum learning:** exaggerated separation early can stabilize later joint optimization.
- **Annealing:** start with high exclusion, then soften constraints.

### Rationale grounded in literature gaps
The synthesis identifies weak understanding of temporal dynamics across denoising steps and limited integration of similarity with spatial reasoning. Existing approaches generally aim for stable balanced fusion throughout. But if early diffusion steps define coarse semantic ownership, then allowing overlap too early may create persistent mixed identities. A stronger approach is:
- early steps: aggressively enforce concept exclusivity in MRSA heads/regions
- middle steps: gradually allow negotiated sharing
- late steps: permit detail borrowing only in non-identity channels

This is not just step-aware weighting; it proposes **deliberate transient exaggeration of concept separation** as a mechanism.

Feasible test:
- Add a simple three-phase schedule to MRSA:
  - early: sharpen/sparsify and penalize cross-reference overlap
  - middle: normal adaptive routing
  - late: residual detail fusion only

### Measurable prediction
Compared with constant adaptive MRSA, scheduled over-separation will reduce temporal ownership flips by **≥ 15%** and improve final concept fidelity by **≥ 2% relative** on high-similarity pairs.

### Failure condition
Reject if:
- ownership-flip rate does not drop by at least **10%**, or
- fidelity improves by **< 1% relative**, or
- layout coherence drops by **> 5%** on a simple spatial consistency proxy.

### Estimated risk
**Medium**

---

## Hypothesis 4: **The right similarity signal is anti-background similarity**
### Bold claim
Adaptive MRSA should be driven less by concept similarity and more by **background-deconfounded similarity**; in one-shot references, removing shared background/context cues will improve composition more than adding sophisticated attention logic.

### Cross-domain inspiration
- **Causal inference:** control confounders before estimating effect.
- **Re-identification systems:** background suppression often matters more than backbone complexity.
- **Style/content disentanglement:** nuisance factors dominate naive similarity metrics.

### Rationale grounded in literature gaps
The synthesis explicitly warns that similarity computed on entangled features may amplify leakage instead of reducing it. FreeCustom-style references often contain contextual clutter. If two references share background color, pose, lighting, or scene structure, naive similarity estimates can mistake these for concept overlap and misroute attention.

A novel, lightweight idea:
1. Compute two similarity scores:
   - full-image similarity
   - foreground-only or saliency-masked similarity
2. Use their difference as a confound score.
3. If confound score is high, suppress similarity-based attention adaptation and revert to local routing.

This is unconventional because it says the main problem is not adaptive attention policy but **causally wrong similarity estimation**. It is feasible with off-the-shelf saliency/segmentation or simple thresholded attention masks.

### Measurable prediction
Foreground-deconfounded similarity gating will outperform naive global-similarity gating by **≥ 4% relative** on per-concept identity preservation for reference pairs with matched backgrounds, while giving **≤ 1%** change on mismatched-background controls.

### Failure condition
Reject if:
- matched-background subset improves by **< 2% relative**, or
- control subset improves equally, suggesting no confound-specific effect, or
- saliency extraction overhead pushes inference beyond the 30-minute budget.

### Estimated risk
**Medium**

---

# Fastest-to-test shortlist
If you want the most practical two to test first under the 30-minute constraint:

1. **Hypothesis 3: Intentional early-step over-separation**
   - easiest to implement
   - only needs a scheduling change
   - strong mechanistic link to denoising dynamics

2. **Hypothesis 4: Anti-background similarity**
   - cheap to evaluate
   - directly addresses a concrete literature gap
   - easy falsification with matched vs mismatched background subsets

If you want, I can next turn these into a **minimal ablation plan with exact metrics, prompts, and 30-minute single-GPU protocol**.