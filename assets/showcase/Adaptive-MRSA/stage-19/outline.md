# Paper Outline for Adaptive MRSA in FreeCustom

## Proposed method name
**AMRA**

Rationale: short, pronounceable, and directly evokes **Adaptive Multi-Reference Attention**. It fits the conference style preference for compact system names.

---

## Candidate titles

1. **AMRA: Adaptive Multi-Reference Attention for FreeCustom Composition**
   - Memorability: **5/5**
   - Specificity: **5/5**
   - Novelty signal: **4/5**

2. **AMRA: Similarity-Aware MRSA for Multi-Concept Image Composition**
   - Memorability: **4/5**
   - Specificity: **5/5**
   - Novelty signal: **4/5**

3. **AMRA: Deconfounded Similarity Routing in FreeCustom Self-Attention**
   - Memorability: **4/5**
   - Specificity: **4/5**
   - Novelty signal: **5/5**

**Recommended title:** **AMRA: Adaptive Multi-Reference Attention for FreeCustom Composition**  
This is the strongest balance of clarity, scope, and branding.

---

# High-level paper positioning

## Central research question
Can **adaptive multi-reference self-attention (MRSA)** in **FreeCustom** improve **multi-concept composition quality** by dynamically scaling attention weights according to **concept similarity**, especially in hard cases where concepts share background or contextual features?

## Current evidence posture
The paper should be outlined as a **mechanism-driven refinement study**, not as a conclusively validated improvement paper yet. The current results support a paper plan centered on:
- a clear technical proposal,
- a principled evaluation design,
- a hard-case benchmark slice,
- and transparent reporting of unresolved evidence.

## Core claim the paper should aim to test
Adaptive, similarity-aware MRSA should reduce **cross-concept contamination** in compositional generation by downweighting misleading shared-context matches and preserving foreground-specific correspondences.

## Evidence links to keep central throughout
- **Primary metric:** `region_level_contamination_error` (lower is better)
- **Key stress test:** `matched_background_or_shared_context`
- **Vanilla baseline overall:** 0.063727
- **Vanilla hard-slice:** 0.103694
- **Proposed full variant overall:** 0.066970
- **Proposed full hard-slice:** 0.110118
- **Critical issue:** currently reported numbers do **not** support the intended hypothesis under the stated metric direction
- **Critical missing baseline:** `GlobalSimilarityWeightedMRSA`

These should be cited in the outline as the motivation for a careful paper structure, not as final claim material.

---

# Detailed paper outline

## 1. Introduction
**Target length:** **850-950 words**  
**Primary goal:** Establish the composition problem, explain why fixed MRSA fails under concept similarity and shared context, introduce AMRA, and frame the paper around a hard benchmark slice and mechanism-level evaluation.

### Paragraph 1 — Motivation and task importance
- Introduce **multi-concept customization/composition** in text-to-image generation.
- Explain why combining multiple personalized concepts is difficult when concepts share visual context, background, pose priors, or co-occurring attributes.
- Motivate the need for **region-faithful composition**, not just global image plausibility.
- Connect to FreeCustom and multi-reference self-attention as an appealing backbone because it naturally aligns multiple concept references during generation.

**Evidence links**
- Hard subset chosen in current experiments: `matched_background_or_shared_context`
- Metric focus: `region_level_contamination_error`

### Paragraph 2 — Gap in prior approaches
- Explain that standard or globally weighted MRSA can over-attend to features that are similar but semantically assigned to the wrong concept.
- Discuss the failure mode: shared backgrounds or context produce attention leakage across concepts.
- Position the gap as **insufficient disentanglement between foreground ownership and contextual similarity**.
- Cite prior strands: compositional generation, personalized generation, attention control, reference-guided diffusion/transformers, token routing.

**Must include**
- Why global similarity weighting alone is not enough
- Why local conflict-aware routing may matter

**Evidence links**
- Missing but necessary comparator: `GlobalSimilarityWeightedMRSA`
- Current ablation structure already suggests this decomposition:
  - similarity-only
  - routing without deconfounding
  - deconfounded routing
  - schedule variants
  - residual variants

### Paragraph 3 — Proposed approach
- Introduce **AMRA** by sentence 1 or 2 of this paragraph.
- Describe the intuition: dynamically scale MRSA weights using concept similarity, but modulate that similarity with **foreground deconfounding** and **conflict routing** so that shared context does not dominate ownership.
- Briefly explain three ingredients:
  1. concept-conditioned similarity estimation,
  2. conflict-aware routing between candidate references,
  3. stage-aware scheduling across denoising/generation layers.

**Evidence links**
- Relevant named variants in current experiments:
  - `ForegroundOnlySimilarityWithoutConflictRouting`
  - `BackgroundBlindConflictRoutingWithoutDeconfounding`
  - `ForegroundDeconfoundedConflictRoutedMRSA`
  - `StaticExclusivityWithoutThreePhaseSchedule`
  - `LateOnlySeparationWithoutEarlyOwnershipStabilization`
  - `EarlyOverSeparationScheduledMRSA`

### Paragraph 4 — Contributions
Write as flowing prose leading into a short bullet list.

**Contribution bullets**
- A similarity-aware adaptive MRSA mechanism for FreeCustom that explicitly targets **multi-concept contamination** under shared-context ambiguity.
- A mechanistic decomposition separating **global similarity weighting**, **foreground deconfounding**, **conflict routing**, **temporal scheduling**, and **residual amplification**.
- A benchmark protocol emphasizing the **matched-background/shared-context** stress test as the primary endpoint.
- A transparent empirical analysis that distinguishes pilot signals from validated gains and highlights metric-integrity requirements for composition research.

**Introduction evidence links**
- Vanilla hard-slice: **0.103694**
- Proposed full variant hard-slice: **0.110118**
- Current evidence status: **pilot / refine**, not confirmatory
- Reported run issue: no usable uncertainty due to **n=1 per cell**

---

## 2. Related Work
**Target length:** **650-800 words**  
**Primary goal:** Situate AMRA across three literature threads and repeatedly clarify how this paper differs.

### Subsection 2.1 — Multi-concept personalized generation
- Cover subject-driven generation and composition of multiple learned concepts.
- Discuss identity preservation vs concept entanglement.
- Explain that prior work often focuses on concept retention but not explicitly on **region-level contamination** under shared contexts.

**How AMRA differs**
- Targets attention-level conflict resolution during composition, not only concept embedding or adapter fusion.

### Subsection 2.2 — Attention control and reference-guided generation
- Review methods that manipulate cross-attention/self-attention, reference features, spatial masks, or token-level guidance.
- Discuss strengths and limitations of static gating and global reference weighting.

**How AMRA differs**
- Uses **dynamic**, concept-specific, conflict-aware weighting within MRSA rather than fixed or globally pooled reference fusion.

### Subsection 2.3 — Compositionality evaluation and disentanglement
- Cover metrics for compositional correctness, leakage, binding failures, and object-attribute consistency.
- Highlight why whole-image metrics are inadequate for the present question.
- Motivate the use of **region-level contamination** and hard-slice evaluation.

**How AMRA differs**
- Ties the mechanism directly to a failure-targeted evaluation slice: `matched_background_or_shared_context`.

**Evidence links**
- Primary metric: `region_level_contamination_error`
- Need to explain why lower values indicate less concept leakage
- Need to justify inclusion of complementary metrics in final paper because the current study is too narrow

---

## 3. Method
**Target length:** **1100-1400 words**  
**Primary goal:** Present AMRA as a precise technical method, with notation, objective, adaptive attention equations, routing logic, and schedule design.

---

### 3.1 Problem formulation
- Define FreeCustom composition with \(K\) concepts, concept references \(R_1, \dots, R_K\), prompt \(p\), and latent/image tokens \(X\).
- Define MRSA as attending from current latent tokens to a bank of reference tokens from all concepts.
- State the problem: standard MRSA produces contamination when reference tokens from concept \(j\) receive high weight for regions belonging to concept \(i\).

**Key notation**
- Query token \(q_t\)
- Reference keys/values \(K^{(i)}, V^{(i)}\)
- Similarity score \(s_t^{(i)}\)
- Ownership/conflict variable \(o_t^{(i)}\)
- Adaptive attention weight \(\alpha_t^{(i)}\)

### 3.2 Vanilla MRSA in FreeCustom
- Briefly formalize baseline MRSA.
- Show that vanilla attention aggregates references using standard similarity without concept-specific contamination control.
- This section is important because the baseline must be mathematically clear before the adaptive extension.

**Evidence links**
- VanillaFreeCustomMRSA overall: **0.063727**
- Vanilla hard slice: **0.103694**
- Vanilla clean slice: **0.023800**

### 3.3 Adaptive similarity weighting
- Introduce AMRA’s first component: concept-conditioned similarity scaling.
- Define a similarity gate \(g_t^{(i)} = f(q_t, K^{(i)}, c_i)\), where \(c_i\) is the concept identity or summary token.
- Explain that not all high-similarity matches should be trusted equally; similarity should be conditioned on concept ownership cues.

**Technical angle**
- Optionally split similarity into foreground and context channels:
  \[
  s_t^{(i)} = \lambda_f s_{t,\text{fg}}^{(i)} + \lambda_c s_{t,\text{ctx}}^{(i)}
  \]
  with adaptive suppression of context similarity when cross-concept conflict is detected.

### 3.4 Foreground deconfounding
- This is the conceptual core.
- Explain how AMRA estimates whether the similarity signal comes from concept-defining foreground evidence or from background/contextual co-occurrence.
- Introduce a deconfounded score:
  \[
  \tilde{s}_t^{(i)} = s_t^{(i)} - \beta b_t^{(i)}
  \]
  where \(b_t^{(i)}\) estimates background/shared-context affinity.

- Describe practical implementations:
  - segmentation-informed masking,
  - learned foregroundness predictor,
  - attention-map sharpening over subject regions,
  - or concept-specific token attribution.

**Evidence links**
- Current ablation explicitly tests no-deconfounding condition:
  `BackgroundBlindConflictRoutingWithoutDeconfounding`

### 3.5 Conflict-aware routing
- Formalize how ambiguous tokens choose among competing concept references.
- For each query token, compute competing deconfounded scores across concepts and apply a routing mechanism:
  \[
  r_t^{(i)} = \text{Softmax}_i(\tau \tilde{s}_t^{(i)})
  \]
- Explain alternatives:
  - soft routing,
  - capped sparse routing,
  - top-1/ top-k exclusivity,
  - entropy regularization.

- Clarify why routing matters beyond similarity weighting: it imposes ownership resolution when multiple references are plausible.

**Evidence links**
- Similarity-only ablation:
  `ForegroundOnlySimilarityWithoutConflictRouting`
- Full proposal:
  `ForegroundDeconfoundedConflictRoutedMRSA`

### 3.6 Three-phase schedule across denoising/generation
- Explain why early, middle, and late generation need different behavior:
  - early: stabilize coarse ownership,
  - middle: enforce separation,
  - late: preserve fine details without oversplitting.
- Present a three-phase coefficient schedule:
  \[
  \gamma(\ell) = 
  \begin{cases}
  \gamma_{\text{early}} & \ell \in \mathcal{L}_1 \\
  \gamma_{\text{mid}} & \ell \in \mathcal{L}_2 \\
  \gamma_{\text{late}} & \ell \in \mathcal{L}_3
  \end{cases}
  \]
- Relate this to the tested schedule ablations.

**Evidence links**
- `StaticExclusivityWithoutThreePhaseSchedule`
- `LateOnlySeparationWithoutEarlyOwnershipStabilization`
- `EarlyOverSeparationScheduledMRSA`

### 3.7 Residual preservation and detail recovery
- Discuss the risk that aggressive separation hurts realism or detail retention.
- Introduce residual boosting to preserve useful shared features while still suppressing leakage.
- Distinguish dense vs sparse and capped vs uncapped residuals.

**Evidence links**
- `DenseUncappedResidualBoostedMRSA`
- `SharedFeatureSuppressionWithoutResidualBoost`
- `SparseCappedResidualBoostedMRSA`

### 3.8 Final AMRA update rule
- Provide the unified equation for adaptive MRSA:
  \[
  \alpha_t^{(i)} \propto \exp\left(
  \frac{q_t^\top k^{(i)}}{\sqrt{d}}
  + \eta g_t^{(i)}
  + \rho r_t^{(i)}
  - \beta b_t^{(i)}
  \right)
  \]
- Then define the output aggregation:
  \[
  y_t = \sum_i \alpha_t^{(i)} V^{(i)} + \delta \, \text{Residual}(t)
  \]

### 3.9 Complexity and implementation details
- Report expected compute/memory overhead of adaptive scoring and routing.
- Clarify compatibility with FreeCustom backbone.
- Keep this technical, not infrastructural.

---

## 4. Experimental Setup
**Target length:** **900-1100 words**  
**Primary goal:** Describe the evaluation design so clearly that the reader understands what the method is supposed to improve and how the conclusions will be judged.

### 4.1 Research hypotheses
State explicit hypotheses:
- **H1:** AMRA reduces contamination on `matched_background_or_shared_context`.
- **H2:** Gains do not come at disproportionate cost on `mismatched_background_or_clean_context`.
- **H3:** Local deconfounded routing outperforms similarity-only and routing-only alternatives.
- **H4:** Stage-aware scheduling and residual control mediate the tradeoff between separation and fidelity.

### 4.2 Dataset/task setup
- Describe the multi-concept composition benchmark, prompt structure, concept pairs, and reference-image setup.
- Define the two primary slices:
  - `matched_background_or_shared_context`
  - `mismatched_background_or_clean_context`

### 4.3 Baselines and ablations
This subsection is crucial.

**Baselines to include in final paper**
- VanillaFreeCustomMRSA
- **GlobalSimilarityWeightedMRSA** — must be restored and reported
- Similarity-only local weighting baseline
- Conflict-routing-only baseline if distinct

**Ablations already scaffolded**
- `ForegroundOnlySimilarityWithoutConflictRouting`
- `BackgroundBlindConflictRoutingWithoutDeconfounding`
- `ForegroundDeconfoundedConflictRoutedMRSA`
- `StaticExclusivityWithoutThreePhaseSchedule`
- `LateOnlySeparationWithoutEarlyOwnershipStabilization`
- `EarlyOverSeparationScheduledMRSA`
- `DenseUncappedResidualBoostedMRSA`
- `SharedFeatureSuppressionWithoutResidualBoost`
- `SparseCappedResidualBoostedMRSA`

### 4.4 Metrics
- Primary: `region_level_contamination_error` (**lower is better**)
- Secondary metrics to add in the final paper:
  - per-concept identity retention
  - prompt alignment
  - realism/fidelity
  - human pairwise preference on contamination and composition correctness

### 4.5 Statistical protocol
Because the current evidence is weak, this must be explicit.
- Aggregate over prompts and seeds
- Use paired comparisons against vanilla
- Report mean, standard deviation, and bootstrap confidence intervals
- Predefine primary endpoint as hard slice

**Evidence links**
- Current reporting problem: **n=1 per cell**
- Current status: **failed**
- These must motivate a stronger evaluation protocol, not be framed as findings

### 4.6 Hardware and compute
- Briefly note GPU availability for reproducibility.
- Mention A800 80GB only as practical compute context, not as contribution.

**Evidence links**
- NVIDIA A800-SXM4-80GB

### 4.7 Tables and figures to plan
- **Table 1:** Hyperparameters
- **Table 2:** Main benchmark results
- **Table 3:** Mechanism ablations
- **Figure 1:** AMRA architecture diagram
- **Figure 2:** Attention routing visualization under shared context
- **Figure 3:** Tradeoff frontier between contamination and fidelity

---

## 5. Results
**Target length:** **650-800 words**  
**Primary goal:** Present the empirical outcomes with complete honesty: what is supported, what is not, and what the current numbers imply.

### 5.1 Main benchmark results
- Compare vanilla, global similarity baseline, and AMRA.
- Organize by overall, hard slice, and clean slice.
- The narrative must center on the hard slice as the primary endpoint.

**Evidence links from current pilot**
- Vanilla overall: **0.063727**
- Vanilla hard slice: **0.103694**
- Vanilla clean slice: **0.023800**
- Full deconfounded routed variant overall: **0.066970**
- Full deconfounded routed variant hard slice: **0.110118**
- Full deconfounded routed variant clean slice: **0.025301**

**Planned interpretation**
- Under current numbers and current metric direction, AMRA does **not** beat vanilla.
- Therefore this section should be written in two layers:
  1. pilot observation,
  2. confirmatory rerun plan.

### 5.2 Mechanism ablations
Use the ablations to reason about components rather than to overclaim wins.

Possible observations from current pilot:
- Similarity-only and routing-only variants also trail vanilla overall.
- Some schedule and residual variants improve one slice while hurting another.
- `DenseUncappedResidualBoostedMRSA` looks strongest overall among non-vanilla variants (**0.063039**) but worsens the clean subset relative to vanilla (**0.026115 vs 0.023800**).
- `LateOnlySeparationWithoutEarlyOwnershipStabilization` improves the hard slice in some seeds but substantially hurts the clean slice and overall.

### 5.3 Slice-wise tradeoffs
- Analyze hard-vs-clean subset tradeoffs.
- Explain that lower contamination in shared-context conditions is only useful if identity retention and non-hard-slice quality do not collapse.
- This section should motivate multi-metric evaluation.

### 5.4 Visual evidence
- Plan qualitative examples where vanilla leaks attributes/backgrounds and AMRA corrects them.
- Also include failure cases where aggressive separation fragments the image or suppresses useful shared detail.

---

## 6. Discussion
**Target length:** **450-600 words**  
**Primary goal:** Interpret the mechanism-level implications without overstating the evidence.

### Discussion themes
- Why shared-context composition is the right failure mode for testing adaptive MRSA.
- Why the pilot results may indicate that **naive similarity scaling is insufficient**.
- Why deconfounding and routing could still be right ideas but currently need better calibration or evaluation integrity.
- Why residual boosting and scheduling appear to expose a real separation-vs-fidelity tradeoff.
- How this connects to broader literature on compositional binding and attention competition.

**Evidence links**
- Most variants worsen hard-slice contamination relative to vanilla under current reporting
- `DenseUncappedResidualBoostedMRSA` is an interesting tradeoff point
- Missing `GlobalSimilarityWeightedMRSA` prevents direct testing of the paper’s strongest intended contrast

---

## 7. Limitations
**Target length:** **220-300 words**  
**Primary goal:** Move all caveats here, concretely and precisely.

### Limitations to state
- Current evidence is **pilot-only** due to incomplete/failed execution status.
- Reported cells are effectively **n=1**, preventing uncertainty quantification.
- The central baseline `GlobalSimilarityWeightedMRSA` is missing from quantitative reporting.
- Metric integrity must be audited because narrative and numeric direction appear inconsistent.
- The primary metric alone is too narrow to support broad claims about composition quality.
- Attention-based deconfounding may depend on mask/feature quality and may not generalize uniformly across concept types.

**Evidence links**
- failed run
- n=1 per cell
- sign inconsistency
- missing baseline

---

## 8. Conclusion
**Target length:** **120-180 words**  
**Primary goal:** End with a disciplined summary and forward-looking statement.

### Content plan
- Reiterate the core idea: adaptive, similarity-aware MRSA for reducing multi-concept contamination in FreeCustom.
- Emphasize the conceptual contribution: deconfounded, conflict-aware reference routing.
- State that the present evidence establishes a promising evaluation and method design, but not yet conclusive empirical superiority.
- Close with next-step agenda: complete baseline coverage, metric validation, and statistically sound reruns on the hard slice.

---

# Abstract planning guide

## Target length
**190-210 words**

## PMR+ structure

### Sentences 1-2: Problem
- Multi-concept customization methods struggle when concepts share context or background, causing cross-concept contamination in attention-mediated composition.
- Existing MRSA schemes in FreeCustom treat reference similarity too uniformly, allowing contextually similar but semantically incorrect references to dominate.

### Sentences 3-4: Method
- Introduce **AMRA**.
- Explain adaptive similarity weighting, foreground deconfounding, conflict-aware routing, and stage-wise scheduling.

### Sentences 5-6: Results
Because the current evidence is not confirmatory, do **not** invent strong improvement claims. Instead, draft two versions:
- **Target abstract for after rerun**
- **Truthful pilot abstract for workshop/internal draft**

## Safe result language for current evidence
“Across mechanism ablations, the matched-background/shared-context slice emerged as the most informative stress test for contamination, while the current pilot results highlight the importance of metric validation and baseline completeness before drawing confirmatory conclusions.”

---

# Main tables and figures plan

## Table 1 — Hyperparameters for AMRA and baselines
**Goal:** Reproducibility and fairness.

## Table 2 — Main benchmark results on overall and context-stratified slices
**Goal:** Show whether AMRA improves the primary endpoint.

**Columns**
- Method
- Overall contamination error
- Shared-context contamination error
- Clean-context contamination error
- Identity retention
- Human preference

## Table 3 — Mechanism ablations for AMRA
**Goal:** Isolate contribution of similarity weighting, deconfounding, routing, scheduling, residuals.

## Figure 1 — AMRA architecture
**Goal:** Visualize adaptive weighting and routing inside FreeCustom MRSA.

## Figure 2 — Token/reference routing under shared context
**Goal:** Show how ambiguity is resolved compared with vanilla MRSA.

## Figure 3 — Tradeoff curves
**Goal:** Show contamination vs fidelity tradeoffs across ablations.

## Figure 4 — Qualitative success and failure cases
**Goal:** Demonstrate actual composition behavior.

---

# Evidence map by claim

## Claim 1
**Shared-context cases are the right stress test.**  
**Evidence link:** all analyses agree that `matched_background_or_shared_context` is the intended failure mode.

## Claim 2
**Mechanism decomposition is scientifically useful.**  
**Evidence link:** current ablation suite spans similarity, routing, deconfounding, schedule, residuals.

## Claim 3
**Current evidence is insufficient for a strong win claim.**  
**Evidence link:** failed run, n=1 per cell, missing `GlobalSimilarityWeightedMRSA`, metric inconsistency.

## Claim 4
**As currently reported, the full proposed variant underperforms vanilla.**  
**Evidence link:**  
- overall: **0.066970 vs 0.063727**  
- hard slice: **0.110118 vs 0.103694**  
- clean slice: **0.025301 vs 0.023800**

## Claim 5
**The paper should be framed as a refined method-and-evaluation study unless rerun evidence improves.**  
**Evidence link:** decision = **REFINE**

---

# Writing guidance for the final manuscript

## Tone
- Confident on the **problem importance** and **method motivation**
- Precise and transparent on the **current empirical status**
- Avoid overclaiming from pilot evidence

## What to emphasize
- Hard-case benchmark design
- Technical novelty in adaptive attention and conflict routing
- Mechanism-oriented ablation structure
- Evaluation rigor as part of the scientific contribution

## What to avoid
- Do not treat run failure or system warnings as results
- Do not present numerical underperformance as improvement
- Do not omit the missing global similarity baseline in the final paper
- Do not generalize beyond the measured contamination setting

---

# Recommended final paper narrative

If writing **now**, frame this as:
> a technically motivated method paper with a strong evaluation redesign and preliminary ablations, emphasizing that confirmatory conclusions require completed reruns.

If writing **after rerun**, frame this as:
> a targeted advance in FreeCustom composition showing that adaptive, deconfounded MRSA reduces contamination on shared-context benchmarks while preserving clean-case fidelity.

---

If you want, I can next turn this outline into:
1. a **200-word NeurIPS-style abstract**,  
2. a **full Introduction draft**, or  
3. a **main results table skeleton with suggested captions**.