## Methodology Audit

## Bottom line

The methodology is **not yet strong enough to support the claimed hypothesis**.  
There are some good instincts here—especially the presence of multiple comparator variants and stratification by matched-background vs clean-context subsets—but the experiment has major weaknesses in:

- **baseline completeness and fairness**
- **metric validation**
- **evaluation robustness**
- **reproducibility**
- **statistical reliability**

Most critically, the reported results appear to come from **a failed run**, with **only 5 seeds**, **n=1 per cell**, and possible inconsistency between the registered conditions and the actually reported ones. That makes the current evidence **exploratory at best**, not confirmatory.

---

## 1. Baseline fairness and completeness

## What is good

You did include a reasonably thoughtful set of **internal baselines/ablations**, such as:

- `VanillaFreeCustomMRSA`
- `ForegroundOnlySimilarityWithoutConflictRouting`
- `BackgroundBlindConflictRoutingWithoutDeconfounding`
- `ForegroundDeconfoundedConflictRoutedMRSA`
- schedule variants
- residual boost variants
- shared-feature suppression variants

This is useful because it partially decomposes the proposed method into ingredients:
- foreground-only similarity
- deconfounding
- conflict routing
- scheduling
- residual boosting
- sparsity/capping

That is the right general direction for a mechanism paper.

## Main fairness problems

### 1.1 Missing strongest external baselines
For a claim about improving **multi-concept composition quality in personalization/generation**, comparing mainly against your own MRSA variants is insufficient.

You need at least:

- **Base FreeCustom without MRSA modification**
- **Uniform multi-reference attention fusion**
- **Static hand-tuned reference weights**
- **Global similarity weighted MRSA** as a real reported baseline
- If feasible: a representative **non-FreeCustom multi-concept personalization baseline** or plug-in attention-control baseline

The run log says:

> `REGISTERED_CONDITIONS: VanillaFreeCustomMRSA, GlobalSimilarityWeightedMRSA, ForegroundDeconfoundedConflictRoutedMRSA, EarlyOverSeparationScheduledMRSA`

But the quantitative results do **not** visibly include `GlobalSimilarityWeightedMRSA`; instead they include many other variants. That mismatch is a serious auditing concern:
- either the baseline was planned but not actually run,
- or the output is incomplete,
- or condition naming/reporting is inconsistent.

Since your central hypothesis explicitly contrasts **local deconfounded routing vs global similarity gating**, the absence of a clearly reported `GlobalSimilarityWeightedMRSA` result is a **major omission**.

### 1.2 Baselines may not be compute-fair
There is no evidence that all methods were matched on:
- denoising steps
- guidance scale
- number of references
- backbone weights
- prompt templates
- seed lists
- mask source and quality
- preprocessing pipeline
- inference-time overhead budget

If the proposed method uses extra signals such as:
- saliency/foreground masks
- conflict detection
- three-phase schedule
- residual boosting

then fairness requires either:
1. giving baselines access to equivalent side information where applicable, or
2. explicitly framing this as a **quality-vs-compute tradeoff** and reporting latency/FLOPs.

Right now, the comparison appears to be **quality-only**, while the proposed method may enjoy extra machinery.

### 1.3 Baseline naming suggests uneven optimization
Some baselines look like intentionally weakened variants:
- `WithoutConflictRouting`
- `WithoutDeconfounding`
- `WithoutResidualBoost`
- `WithoutThreePhaseSchedule`

These are useful ablations, but they are **not substitutes for strong tuned baselines**. An ablation is not automatically a fair competitor.

### 1.4 No evidence of hyperparameter parity
Did each baseline get its own tuning?
If not, then results may reflect:
- your method being tuned,
- baselines using inherited defaults.

For fairness, you need:
- a shared tuning budget per method, or
- frozen canonical settings from prior work, or
- explicit no-tuning policy applied equally.

---

## 2. Metric appropriateness for the research question

## What the metric seems to target well

The metric definition is:

> `region_level_contamination_error from masked CLIP reference misalignment plus cross-concept contamination`  
> direction = lower

This is directionally aligned with the hypothesis. The paper is about:
- concept leakage
- identity blending
- attribute collapse
- local conflict under shared context

A **region-level contamination/error metric** is much better than just whole-image CLIP similarity for this question.

## But there are major metric problems

### 2.1 The metric is composite and underexplained
The metric merges:
- masked CLIP reference misalignment
- cross-concept contamination

Without a formal definition, the reader cannot assess:
- weighting between terms
- sensitivity to mask quality
- whether lower is really better in all cases
- whether the metric penalizes desired shared attributes
- whether it correlates with human judgments

A composite metric can hide failure modes.

### 2.2 Metric validity is not established
For a claim about multi-concept composition quality, a single custom metric is not enough unless you validate it.

You need evidence that it correlates with:
- human preference on concept disentanglement
- identity preservation
- prompt adherence
- edit faithfulness, if editing is part of the task

Without that, the metric could over-reward over-separation or under-reward realistic blending where appropriate.

### 2.3 Missing complementary metrics
The hypothesis explicitly predicts tradeoffs:
- lower contamination
- improved per-concept identity/reference alignment
- minimal loss in full-prompt alignment or aesthetics

But only one primary metric is reported. That is not enough.

You should also report:

- **Per-concept identity similarity**  
  e.g. masked image-image similarity or identity embedding similarity per subject
- **Prompt/text alignment**
- **Aesthetic/realism proxy**
- **Composition success rate**
- **Localization accuracy** for concept-region assignment
- **Human pairwise preference** on leakage vs realism

Otherwise, a method might simply reduce contamination by making concepts weaker, blurrier, or spatially segregated in unnatural ways.

### 2.4 Scale and effect size interpretability are poor
Values range roughly:
- ~0.024 on clean-context subset
- ~0.10–0.11 on matched/shared-context subset
- ~0.063–0.067 overall

But there is no explanation of:
- what counts as a meaningful improvement,
- what the natural variance is,
- whether a 0.002 absolute change matters perceptually.

Given the tiny numerical gaps between methods, metric noise could dominate.

---

## 3. Evaluation protocol: leakage, contamination, and internal validity risks

## Positive aspect

You at least stratified by:
- `matched_background_or_shared_context`
- `mismatched_background_or_clean_context`

This is exactly the right stress test family for the stated hypothesis.

## Major concerns

### 3.1 Possible data leakage / confounding from mask generation
Because the method and metric both appear to depend on foreground/local masks or saliency, there is risk of **evaluation entanglement**:

- If the same mask generator or attention-derived localization is used in both method and evaluation, the metric may be biased toward your method.
- If masks are generated from reference images using the same features used for routing, then the evaluation is not independent.

You need to specify:
- how masks are obtained,
- whether they are frozen,
- whether evaluation masks are independent from routing masks,
- whether any model used for evaluation was used in the method itself.

### 3.2 CLIP contamination risk / benchmark overfitting
Using CLIP-derived similarity for both method design and evaluation creates a classic proxy-overfitting risk.
If similarity/conflict routing is designed around CLIP-like embeddings and the metric is also CLIP-based, gains may reflect better optimization of the evaluator, not actual image quality.

You need at least one non-CLIP evaluation channel:
- human study
- DINO/face-ID/object-ID embeddings
- task-specific recognition models
- segmentation consistency

### 3.3 No train/validation/test protocol is described
Even for a training-light or inference-only method, there are still methodological split questions:

- Were the concepts/prompts used to choose thresholds/schedules distinct from those used for final reporting?
- Was the matched-background subset hand-curated after observing failures?
- Were prompt templates fixed before seeing results?

Without a held-out stress set, there is high risk of iterative benchmark tuning.

### 3.4 Failed run status undermines trust
The run context says:

```json
"status": "failed"
```

This is a major red flag. Even though many numbers are reported, a failed run raises questions:

- Were all conditions executed?
- Were partial outputs aggregated?
- Were missing conditions silently dropped?
- Did post-processing complete consistently?

A failed run should not be used as primary evidence unless you clearly explain:
- failure cause,
- which outputs are valid,
- whether failure affected any condition asymmetrically.

### 3.5 Small seed count and effectively no replication
The log states:

> `SEED_WARNING: only 5 seeds used due to time budget`

And every reported cell has:
- `n=1`

This means the displayed statistics are not really statistics; they are essentially single observations per seed-condition-subset cell. There is no confidence interval, no variance estimate, no significance test, and no robustness claim.

For generative image evaluation, this is a severe weakness because stochasticity is substantial.

### 3.6 Potential cherry-picking risk in subset emphasis
The hypothesis says gains should be strongest on matched/shared-context subsets. That makes sense. But if the dataset construction, subset definitions, or reporting emphasis evolved after inspecting results, then this becomes hypothesis-fitting.

Need:
- predeclared subset criteria
- exact counts
- examples
- fixed prompt/reference lists

---

## 4. Ablation completeness

## What is good

The ablation set is actually one of the stronger parts. It tries to isolate:
- foreground-only similarity
- deconfounding
- conflict routing
- early/late scheduling
- residual boost design
- sparsity/capping

This is good mechanism-oriented experimentation.

## What is still incomplete

### 4.1 Missing full factorization
To support the central causal claim, the ideal ablation table should cleanly isolate:

| Component | Off | On |
|---|---:|---:|
| global similarity prior | ✓ | ✓ |
| foreground masking | ✓ | ✓ |
| deconfounding against background | ✓ | ✓ |
| local conflict routing | ✓ | ✓ |
| scheduling | static / dynamic |
| residual boost | none / dense / sparse capped |

Right now, the naming suggests several combinations, but it is not clear that the design is a **complete factorial or near-factorial study**.

### 4.2 Missing global similarity baseline in ablations
Again, this is crucial.
Your hypothesis is specifically about **local deconfounded routing outperforming global similarity gating**, but I do not see a proper quantitative row for that direct comparison.

### 4.3 Missing “oracle” and “sanity” ablations
You should add:
- **Oracle foreground masks** vs automatic masks
- **Random masks** or shuffled masks
- **Random reference weighting**
- **Shuffled concept-reference assignment**
- **Routing disabled after certain timesteps**
- **Apply deconfounding only to background-clean subset** as a negative control

These would show whether the gains are real and mechanism-specific.

### 4.4 Missing failure analysis ablations
The paper’s novelty is about conflict cases. You need ablations by:
- similarity level
- overlap severity
- number of concepts
- concept type pairings: same species, same color family, similar texture, etc.
- pose/view similarity
- background-sharing only vs foreground-sharing only

That would directly test whether the method addresses the stated failure source.

---

## 5. Reproducibility assessment

## Current assessment: weak

### Reasons

#### 5.1 Insufficient run stability
- only 5 seeds
- `n=1` at result cells
- failed run
- no intervals
- no variance summaries

This is not reproducible evidence.

#### 5.2 Incomplete experimental traceability
Critical missing items include:
- exact dataset/prompt/reference list
- subset construction rules
- preprocessing details
- mask generation method
- all hyperparameters
- denoising schedule
- which layers use MRSA
- conflict thresholding rule
- deconfounding formula
- residual boost cap values
- schedule timings

For an attention manipulation method, tiny implementation details can change behavior a lot.

#### 5.3 Inconsistent condition registry vs outputs
The registered conditions mention `GlobalSimilarityWeightedMRSA`, but outputs emphasize different variants. This inconsistency hurts reproducibility and interpretability.

#### 5.4 No compute reporting
For a plug-in/training-light method, reproducibility should include:
- runtime/image
- VRAM
- number of added forward passes
- mask extraction cost
- batch size constraints

Otherwise others cannot fairly reproduce deployment conditions.

---

## 6. Reading the actual results cautiously

Even taking the numbers at face value, the evidence is mixed.

### 6.1 On the hard subset, proposed method is not clearly dominant
For `matched_background_or_shared_context`, `ForegroundDeconfoundedConflictRoutedMRSA` is often good, but not consistently best.

Examples:
- seed 11: `0.110118`, while `SparseCappedResidualBoostedMRSA` is `0.111553`
- seed 23: `0.110516`, while `EarlyOverSeparationScheduledMRSA` is `0.110419` and `StaticExclusivityWithoutThreePhaseSchedule` is `0.108509`
- seed 37: `0.109637`, while `SparseCappedResidualBoostedMRSA` is `0.111567`
- seed 49: `0.107964`, while `StaticExclusivityWithoutThreePhaseSchedule` is `0.110312`
- seed 83: `0.109986`, while `StaticExclusivityWithoutThreePhaseSchedule` is `0.110319`

So even by the primary metric, the proposed “full” method does not clearly and consistently outperform all ablations.

### 6.2 Overall metric gains are tiny
Overall:
- Vanilla: `0.063727`
- Proposed: `0.06697`

Since lower is better, the proposed method actually appears **worse overall** than vanilla on the aggregate metric reported there.

That may not be a fair conclusion if:
- the overall numbers aggregate subsets in a strange way,
- or the reported overall values are from a different aggregation pass.

But as presented, this directly conflicts with the narrative.

### 6.3 Subset-specific gains may exist but are not enough
Compared with vanilla on matched/shared-context:
- Vanilla around `0.1006–0.1050`
- Proposed around `0.1080–0.1105`

If higher were better, that would look good. But your metric says **lower is better**. This creates a serious interpretation problem.

Either:
1. the subset metrics are actually a different direction,
2. the labels are inconsistent,
3. or the proposed method is worse on the hard subset.

This must be resolved before any claim can be evaluated.

### 6.4 Metric direction inconsistency is a critical methodological issue
The header says:
- primary metric direction = lower

But many “better-looking” methods have numerically larger values on subsets.
That suggests one of:
- reporting bug
- sign inversion
- metric mismatch between overall and subset outputs
- post-processing error

Any of those would invalidate the current experimental conclusion.

---

## 7. Specific methodology improvements needed

## Highest-priority fixes

### 7.1 Re-run the entire evaluation successfully
Do not base conclusions on a failed run.
Require:
- status = completed
- all registered conditions present
- same evaluation script for all methods
- explicit missingness check

### 7.2 Fix metric definition and directionality
Publish:
- exact formula
- term weights
- normalization
- mask source
- whether higher/lower is better
- examples showing metric behavior on success/failure cases

Then verify that all tables obey the same direction convention.

### 7.3 Add the missing key baseline
You must include and clearly report:
- `GlobalSimilarityWeightedMRSA`

Because that is the central alternative your hypothesis is supposed to beat.

Also add:
- base FreeCustom without MRSA adaptation
- uniform reference fusion
- static tuned weighting

### 7.4 Increase statistical rigor
Minimum:
- 10–20 seeds for stochastic generation, or enough to stabilize estimates
- mean ± std / bootstrap CI
- paired statistical testing across prompts/seeds
- per-prompt win rates

Given small effect sizes, this is essential.

### 7.5 Predefine and freeze evaluation splits
Create:
- dev set for threshold/schedule selection
- held-out test set for final reporting

And publish:
- prompt list
- concept pairs/triples
- subset rules
- counts in each subset

### 7.6 Use multiple metrics aligned to the claim
At minimum report:
- contamination/leakage metric
- per-concept identity alignment
- full prompt alignment
- realism/aesthetics
- human pairwise preference

And report tradeoffs, not just one score.

---

## 8. Recommended improved experiment design

A stronger methodology would look like this:

### Dataset / benchmark
- Construct a benchmark of multi-concept prompts with:
  - 2-concept and 3-concept cases
  - matched-background/shared-context subset
  - clean-context subset
  - varying similarity/conflict severity
- Freeze all prompts and references before experiments.

### Methods compared
- Base FreeCustom
- Vanilla FreeCustom MRSA
- Uniform multi-reference fusion
- Static tuned weighting
- GlobalSimilarityWeightedMRSA
- Proposed ForegroundDeconfoundedConflictRoutedMRSA
- Best simplified variant from ablations

### Tuning protocol
- Equal tuning budget on a dev set
- Freeze hyperparameters before test evaluation

### Metrics
- Primary: region-level contamination/leakage
- Secondary:
  - per-concept identity/reference alignment
  - prompt alignment
  - realism/aesthetic
  - human preference
  - runtime overhead

### Robustness checks
- 10–20 seeds
- report means, stds, CIs
- paired comparisons
- stratify by conflict severity and concept type

### Ablation suite
- global only
- foreground only
- deconfounding only
- routing only
- full model
- oracle masks
- random masks
- shuffled references
- schedule variants
- layer/timestep ablations

### Contamination safeguards
- Independent mask generator for evaluation
- Non-CLIP secondary evaluators
- Held-out prompt/reference sets
- No metric-guided threshold tuning on test data

---

## 9. Final verdict by category

### Baseline fairness and completeness
**Rating: Weak to moderate**
- Internal ablations are decent
- External baselines are insufficient
- Key comparison to global similarity gating is missing/inconsistent

### Metric appropriateness
**Rating: Moderate in concept, weak in execution**
- Region-level contamination is relevant
- But metric definition, validation, and directionality are unclear
- Single-metric evaluation is inadequate

### Evaluation protocol
**Rating: Weak**
- Possible evaluator-method entanglement
- failed run
- no clear held-out protocol
- low seed count
- contamination/leakage safeguards not documented

### Ablation completeness
**Rating: Moderate**
- Better than average internal decomposition
- But still missing critical controls and full causal isolation

### Reproducibility
**Rating: Weak**
- failed run
- inconsistent condition reporting
- insufficient experimental details
- no robust statistics

---

## Final assessment

The current methodology is **promising as a pilot study but not publishable as strong causal evidence** for the claimed contribution.

The biggest blockers are:

1. **missing/unclear key baseline (`GlobalSimilarityWeightedMRSA`)**
2. **failed run and inconsistent reporting**
3. **single custom metric with unclear directionality**
4. **too little replication**
5. **lack of external validity and independent evaluation channels**

If you fix those, the work could become much more convincing, because the experimental framing itself—especially the shared-context stress test and component-wise mechanism ablations—is actually well aligned with the research question.