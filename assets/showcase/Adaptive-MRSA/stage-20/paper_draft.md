## Title

AMRA: Adaptive Multi-Reference Attention for FreeCustom Composition

## Abstract

Multi-concept customization remains difficult when concepts share background or contextual cues, because multi-reference self-attention (MRSA) can assign attention to semantically incorrect but visually similar references. We study AMRA, an adaptive MRSA variant for FreeCustom that combines similarity-aware weighting, foreground deconfounding, conflict-aware routing, and stage-wise scheduling. The evaluation targets region-level contamination error, where lower is better, with particular emphasis on the matched-background/shared-context slice. In the current pilot run, the evidence is not confirmatory: execution status is failed, reported cells are effectively \(n=1\), and one critical baseline is missing. Moreover, the full proposed variant does not improve over vanilla FreeCustomMRSA under the stated metric direction. Nevertheless, the ablation design is informative and identifies shared-context composition as the appropriate stress test for future reruns.

## Introduction

Composing multiple customized concepts in text-to-image generation is challenging because concept references often share backgrounds, poses, or contextual features. In such cases, a model may retrieve visually similar evidence from the wrong reference and leak attributes across concepts. This failure mode is especially relevant for FreeCustom-style pipelines that rely on multi-reference self-attention (MRSA), since MRSA directly mixes information from several concept-specific reference banks during generation. Good overall image quality is therefore insufficient: the central requirement is region-faithful concept binding.

The main limitation of vanilla MRSA is that reference similarity is treated too uniformly. When two concepts have overlapping contextual statistics, high attention scores can reflect background similarity rather than foreground ownership. This suggests that global or unstructured similarity weighting alone is unlikely to resolve cross-concept contamination. Instead, the attention mechanism should distinguish concept-defining foreground evidence from shared context and should explicitly arbitrate conflicts among competing references.

We investigate this idea through AMRA, an adaptive MRSA mechanism for FreeCustom composition. AMRA dynamically modulates reference attention using concept-conditioned similarity, subtracts background-sensitive similarity components, routes ambiguous tokens toward competing concepts, and applies stage-wise scheduling to balance early ownership stabilization against later detail preservation. The design is accompanied by mechanism-oriented ablations that isolate similarity-only weighting, routing without deconfounding, schedule variants, and residual variants.

This paper makes three contributions. First, it formulates a concise adaptive-attention design for contamination-aware multi-concept composition. Second, it evaluates the method on a targeted stress test, matched-background/shared-context examples, using region-level contamination error as the primary endpoint. Third, it reports the current pilot evidence transparently. Under the stated metric direction, lower is better, yet the current numbers do not support a performance gain for the full proposed variant. We therefore position the study as a method-and-evaluation refinement rather than a confirmatory improvement claim.

## Method

Let \(K\) denote the number of concepts, with reference token banks \(\{K^{(i)}, V^{(i)}\}_{i=1}^K\). In vanilla FreeCustomMRSA, a query token \(q_t\) attends over all reference tokens using standard similarity, and the resulting mixture is used to guide generation. This mechanism is flexible but vulnerable to contamination: a token belonging to concept \(i\) may attend strongly to concept \(j\) when both references share similar context.

AMRA modifies this attention process with four components. First, it introduces concept-conditioned similarity weighting. For each query token and concept, AMRA computes a similarity score \(s_t^{(i)}\) that is intended to reflect not only raw feature affinity but also concept relevance. Second, it applies foreground deconfounding. We model the observed similarity as a mixture of foreground evidence and shared-context evidence, and suppress the latter via a deconfounded score
\[
\tilde{s}_t^{(i)} = s_t^{(i)} - \beta b_t^{(i)},
\]
where \(b_t^{(i)}\) estimates background or context affinity. Intuitively, this downweights references that are similar for the wrong reason.

Third, AMRA performs conflict-aware routing. Given deconfounded scores across concepts, the method computes ownership weights
\[
r_t^{(i)} = \mathrm{Softmax}_i(\tau \tilde{s}_t^{(i)}),
\]
so that ambiguous tokens are assigned more selectively when several references compete. This distinguishes the method from similarity-only gating, which may still distribute mass across conflicting concepts. Fourth, AMRA uses stage-wise scheduling across denoising or generation depth. Early stages prioritize coarse ownership stabilization, middle stages emphasize separation, and late stages relax exclusivity to preserve detail.

The final attention weight is a combination of raw similarity, deconfounded evidence, routing, and optional residual preservation:
\[
\alpha_t^{(i)} \propto \exp\!\left(
\frac{q_t^\top k^{(i)}}{\sqrt d}
+ \eta g_t^{(i)}
+ \rho r_t^{(i)}
- \beta b_t^{(i)}
\right).
\]
This formulation yields a family of ablations already instantiated in the experiments: similarity-only weighting, routing without deconfounding, schedule variants, and residual-boosting variants. The goal is not merely to improve one score, but to expose which ingredients matter under shared-context ambiguity.

## Experiments

We evaluate AMRA and related ablations using region-level contamination error, for which lower is better. The most important slice is matched_background_or_shared_context, designed to stress cross-concept leakage when concepts share contextual cues. We also report mismatched_background_or_clean_context as a cleaner control slice. Hardware was an NVIDIA A800-SXM4-80GB GPU. However, the run status is failed, and although five seeds are listed, reported cells are effectively \(n=1\), so no uncertainty estimates are available.

The vanilla baseline, VanillaFreeCustomMRSA, reports an overall contamination error of 0.063727. On the hard matched/shared-context slice, its values across seeds 11, 23, 37, 49, and 83 are 0.103694, 0.103994, 0.100578, 0.104976, and 0.104119. On the clean mismatched slice, the corresponding values are 0.023800, 0.023803, 0.024026, 0.023917, and 0.024364.

The full proposed variant, ForegroundDeconfoundedConflictRoutedMRSA, does not improve over vanilla under this metric direction. Its overall score is 0.066970, worse than 0.063727. On the matched/shared-context slice, its values are 0.110118, 0.110516, 0.109637, 0.107964, and 0.109986, all above the vanilla range. On the clean slice, it reports 0.025301, 0.024083, 0.023627, 0.023867, and 0.024601, again not showing a clear advantage.

The ablations are informative but similarly mixed. ForegroundOnlySimilarityWithoutConflictRouting scores 0.066168 overall, BackgroundBlindConflictRoutingWithoutDeconfounding 0.067201, and StaticExclusivityWithoutThreePhaseSchedule 0.067161. DenseUncappedResidualBoostedMRSA is the strongest non-vanilla overall result at 0.063039, slightly below vanilla’s 0.063727, and also reaches relatively low matched/shared-context values in some seeds, including 0.094044 and 0.097295. However, it worsens the clean slice, with values such as 0.026115 and 0.029874 versus vanilla near 0.0238–0.0244. EarlyOverSeparationScheduledMRSA also shows a tradeoff: matched/shared-context values of 0.095095 and 0.101947 in some seeds, but clean-slice degradation up to 0.031204.

Two conclusions follow. First, the matched/shared-context subset is indeed the right stress test, because it reveals strong separation–fidelity tradeoffs hidden by aggregate numbers. Second, the present pilot evidence does not validate the main hypothesis that adaptive deconfounded routing improves FreeCustom composition. This interpretation is further limited by the absence of the critical GlobalSimilarityWeightedMRSA comparator.

## Conclusion

We presented AMRA, an adaptive MRSA variant for FreeCustom that combines similarity-aware weighting, foreground deconfounding, conflict-aware routing, and stage-wise scheduling to reduce multi-concept contamination. The method is well motivated for shared-context composition, and the ablation suite provides a useful mechanism-level decomposition. However, the current empirical evidence is preliminary and not confirmatory. The run failed, reported cells are effectively \(n=1\), a key baseline is missing, and the full proposed variant underperforms vanilla FreeCustomMRSA on overall, hard-slice, and clean-slice contamination error under the stated lower-is-better metric. The main outcome of this workshop study is therefore methodological: it identifies the correct stress test, exposes tradeoffs among routing and scheduling choices, and motivates a rigorous rerun with complete baselines and proper statistical reporting.