Here is a synthesis that keeps the strongest ideas, but does not pretend the perspectives fully agree.

## Final hypothesis set

### Hypothesis 1: **Local deconfounded routing will outperform global similarity gating on the hardest compositions**
**Claim**  
For multi-concept MRSA, the best practical control signal is not raw global concept similarity, but **region-level conflict routing using deconfounded similarity**: foreground/local features should drive attention adaptation, while shared background/context cues should be suppressed.

**Why this survives synthesis**
- **Strongest novel element from innovator:** anti-background similarity; shared context is a confound, not useful evidence.
- **Strongest practical element from pragmatist:** region-aware gating is feasible training-free using existing masks/attention maps.
- **Critical contrarian concern addressed:** global concept similarity may be the wrong abstraction because interference is local and nuisance-driven.

**Rationale**  
Global similarity is too coarse for MRSA failures that arise in specific contested regions. If references share background, pose, lighting, or texture, global similarity can misroute attention and amplify leakage. A feasible improvement is:
1. compute global similarity only as a weak prior,
2. compute foreground-only or saliency-masked similarity,
3. trigger stronger routing only in low-confidence, overlapping regions,
4. suppress background-driven similarity when full-image and foreground similarity disagree.

This combines causal deconfounding with practical local control, without requiring training.

**Measurable prediction**
Compared with:
- vanilla MRSA, and
- global similarity-only gating,

a **region-aware deconfounded gate** will achieve:
- **≥ 5% reduction** in region-level contamination/leakage on high-conflict prompts,
- **≥ 3% relative improvement** in per-concept identity/reference alignment on matched-background or shared-context subsets,
- with **≤ 1–2% loss** in full-prompt alignment or aesthetic proxy.

The key signature should be that gains are strongest when references share nuisance context, not uniformly everywhere.

**Failure condition**
Reject if:
- local deconfounded routing does **not** beat global gating on nuisance-overlap subsets,
- matched-background and mismatched-background subsets improve equally, implying no confound-specific effect,
- or local control introduces instability/artifacts that reduce total fidelity beyond the small allowed margin.

---

### Hypothesis 2: **Temporal ownership stability is a stronger causal predictor of leakage than static similarity, and early-step over-separation is a feasible way to improve it**
**Claim**  
The main driver of concept leakage is **temporal instability in which concept owns a region over denoising**, not static pairwise similarity alone. A simple three-phase MRSA schedule that enforces **early over-separation, then gradual relaxation** will reduce ownership flips and improve final concept fidelity.

**Why this survives synthesis**
- **Strongest novel element from innovator:** intentional early-step over-separation.
- **Critical contrarian concern addressed directly:** static similarity may be only a proxy; temporal routing instability may be the real mechanism.
- **Pragmatic feasibility:** scheduling is cheap, training-free, and easier to verify than a fully online surprise controller.

**Rationale**  
If regions switch reference ownership repeatedly during early and middle denoising, the model can irreversibly blend identities or attributes. Static similarity may identify hard cases, but does not explain the mechanism. A practical intervention is:
- **early steps:** enforce sharper exclusivity / stronger anti-overlap,
- **middle steps:** allow controlled negotiation,
- **late steps:** permit only limited detail borrowing.

This tests a causal mechanism while staying within a modest engineering budget.

**Measurable prediction**
Two linked predictions:
1. **Mechanism:** temporal ownership-flip rate / attention volatility will predict final leakage better than static similarity score alone.
2. **Intervention:** a scheduled over-separation MRSA will produce:
   - **≥ 15% reduction** in ownership flips,
   - **≥ 2% relative gain** in per-concept fidelity on high-similarity or high-ambiguity prompts,
   - with **≤ 5% drop** in spatial/layout coherence proxy.

**Failure condition**
Reject if:
- temporal instability metrics do not predict leakage better than similarity,
- early over-separation does not reduce flip rate meaningfully,
- or flip reduction does not translate to final fidelity gains.

This hypothesis also fails if stabilization merely locks in the wrong concept, improving consistency but not correctness.

---

### Hypothesis 3: **Adaptive MRSA should emphasize discriminative residuals, but only under sparse, capped activation**
**Claim**  
When concepts are highly similar, MRSA should often **downweight shared features and upweight discriminative residual features** in contested regions; however, this should be applied **sparsely and with capped strength** to avoid damaging the base model prior.

**Why this survives synthesis**
- **Strongest novel element from innovator:** anti-similarity inversion / residual boosting.
- **Critical contrarian concern addressed:** stronger adaptation can destabilize the generative prior; therefore the intervention must be weak and confidence-triggered, not global.
- **Feasibility preserved:** can be approximated training-free using frozen embeddings and token/feature decomposition only in flagged conflict zones.

**Rationale**  
For same-class identities or overlapping concepts, the shared feature subspace is often where confusion lives, while distinguishing evidence is in the mismatch. But applying this everywhere is risky: shared features are also useful for realism and coherence. So the most credible version is:
- detect contested regions,
- decompose reference evidence into shared vs residual components,
- suppress shared components and boost residual ones only where conflict is high,
- cap modulation magnitude.

This is more mechanistically specific than scalar gating and less brittle than unrestricted inversion.

**Measurable prediction**
Against vanilla MRSA and standard similarity-proportional gating, a **sparse residual-boosted controller** will yield on same-category/high-overlap pairs:
- **≥ 3% relative gain** in per-concept reference alignment,
- **≥ 10% relative reduction** in identity confusion / cross-concept contamination,
- without **> 5%** degradation in image quality/aesthetic proxy or increased seed variance beyond baseline tolerance.

A useful secondary prediction: benefits should concentrate in high-similarity pairs, not low-similarity controls.

**Failure condition**
Reject if:
- gains are absent or tiny on high-overlap pairs,
- quality/realism drops materially,
- or a simpler weak residual baseline matches results, implying residual inversion is unnecessary.

---

## Unresolved disagreements between perspectives

These are real disagreements worth preserving in the proposal:

1. **Is similarity causal or only a proxy?**
   - Innovator/pragmatist: similarity is useful if made local, deconfounded, or feature-specific.
   - Contrarian: temporal instability may be the true cause, and similarity only correlates with hard cases.
   - Resolution in proposal: treat similarity as a **testable prior**, not an assumed causal driver.

2. **Should control become smarter or weaker?**
   - Innovator: more targeted feature/routing logic can unlock gains.
   - Contrarian: aggressive adaptation may hurt the base diffusion prior.
   - Resolution: all adaptive hypotheses include **caps, sparse activation, and variance/robustness checks**.

3. **What granularity matters most: concept, region, or time?**
   - Pragmatist: region-aware control is the best next step.
   - Contrarian: time/ownership trajectories may matter more than concept-level signals.
   - Innovator: feature subspaces may matter even more than either.
   - Resolution: the final set intentionally spans **region**, **time**, and **feature-subspace** mechanisms.

4. **What should be tested first?**
   - Pragmatist would prioritize region-aware and simple global gating.
   - Innovator would push early-step scheduling and residual inversion as more novel.
   - Contrarian would insist every positive claim be checked against temporal and weak-routing alternatives.
   - Resolution: prioritize Hypothesis 1 and 2 first, with Hypothesis 3 as a higher-risk mechanistic extension.

## Recommended execution order
1. **Hypothesis 1** — best balance of novelty, feasibility, and causal sharpness.
2. **Hypothesis 2** — strongest response to the contrarian and still cheap to implement.
3. **Hypothesis 3** — most novel, but should only be pursued after the diagnostics from 1–2 are in place.

If you want, I can convert this into a **1-page proposal table** with columns: hypothesis, method, metrics, ablations, risks, and required compute.