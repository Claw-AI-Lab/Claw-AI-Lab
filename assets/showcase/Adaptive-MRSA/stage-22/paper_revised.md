# ReMRSA: Adaptive Reference Attention for FreeCustom Composition

# Abstract

Multi-concept customization remains difficult when concept references share backgrounds or contextual cues, because standard multi-reference self-attention can bind a generated region to the wrong reference even when overall image quality remains high. Prior FreeCustom-style approaches improve compositional generation by mixing concept-specific reference banks, yet they typically treat reference similarity too uniformly and therefore remain vulnerable to cross-concept contamination under shared-context ambiguity [freecustom2024, customdiffusion2023, dreambooth2022]. We study ReMRSA, an adaptive multi-reference self-attention variant for FreeCustom that rescales attention weights using concept similarity, subtracts background-sensitive affinity, routes ambiguous tokens through concept competition, and schedules these effects across denoising stages. Across the reported evaluation, the full deconfounded routing variant reaches a contamination error of 0.066970, compared with 0.063727 for vanilla FreeCustomMRSA, and therefore does not improve the primary lower-is-better metric. A residual-boosted ablation attains the strongest aggregate score among the non-vanilla variants at 0.063039, while the hardest matched-background/shared-context slice reveals sharper separation–fidelity tradeoffs than the clean control slice. These findings show that adaptive scaling in MRSA is useful primarily as an analysis tool for exposing when stronger concept separation helps and when it degrades clean-context fidelity.

# Introduction

Personalized text-to-image generation has advanced rapidly, but reliable multi-concept composition remains unresolved when several customized subjects must appear together in a single scene [dreambooth2022, customdiffusion2023, elite2023, subjectdiffusion2024]. The difficulty is not merely producing realistic images; rather, the model must preserve concept identity while binding each visual region to the correct customized reference. This requirement becomes especially demanding when references share a background, pose, lighting pattern, or scene context, because similarity in those attributes can overwhelm the weaker signals that distinguish foreground ownership. In FreeCustom-style systems, the problem is particularly acute because generation relies on multi-reference self-attention (MRSA), which explicitly mixes token banks extracted from multiple concept examples [freecustom2024]. As a result, a region meant to depict one concept can borrow texture, color, or structure from another concept’s reference if contextual similarity is spuriously rewarded. Composition quality therefore depends not only on global realism or prompt adherence, but on contamination-resistant concept binding under ambiguous visual evidence.

Recent work on diffusion personalization and compositional generation has improved identity preservation, text alignment, and training efficiency, yet these gains do not directly solve the attention-allocation problem that emerges when multiple customized references are used together [dreambooth2022, customdiffusion2023, perfusion2023, instantid2024, ipadapter2023]. Several methods strengthen conditioning through subject-specific embeddings, low-rank adaptation, or image prompt injection, but they often assume that better reference access will translate into better composition [textualinversion2022, lora2022, blipdiffusion2023]. That assumption breaks down when references are visually similar for the wrong reason. Even methods that introduce structured attention or region guidance usually focus on correspondence or controllability rather than selective rejection of shared context during concept competition [composer2023, attendandexcite2023, boxdiff2023, layoutguidance2023]. Consequently, vanilla MRSA can remain competitive on aggregate metrics while failing in exactly the regime that matters most for compositional reliability: matched-background or shared-context examples. This gap motivates a narrower question than broad image quality benchmarking: can adaptive MRSA modulate attention according to concept similarity in a way that reduces region-level contamination where contextual overlap is strongest?

To answer that question, we develop ReMRSA, an adaptive MRSA mechanism for FreeCustom that treats reference competition as a token-level ownership problem rather than a uniform similarity lookup. The central idea is simple. Raw similarity should not be used directly when it is partly driven by background or contextual overlap, because such overlap can attract attention away from the concept that actually owns the queried region. ReMRSA therefore computes concept-conditioned similarity, estimates a background-sensitive component, subtracts that component to obtain a deconfounded score, and converts the resulting concept scores into routing weights that sharpen competition among references. Building on this observation, the method further applies stage-wise scheduling across denoising steps so that ownership is emphasized early, separation is reinforced in the middle of generation, and detail recovery is relaxed later when fine structure matters more than exclusivity. This design preserves the topic focus of adaptive MRSA for FreeCustom while turning the paper toward a concrete empirical question: which kinds of adaptive scaling help under shared-context ambiguity, and which kinds simply introduce a different failure mode?

The study contributes the following.  
- It formulates ReMRSA, a dynamic multi-reference attention mechanism for FreeCustom that combines concept-conditioned similarity weighting, background deconfounding, conflict-aware routing, and denoising-stage scheduling within a single attention rule.  
- It introduces a contamination-centered evaluation protocol that separates matched-background/shared-context cases from mismatched-background/clean-context controls, allowing the analysis to focus on the regime where MRSA should be most vulnerable.  
- It shows empirically that the full adaptive deconfounded routing design does not outperform vanilla FreeCustomMRSA on the primary contamination metric, while several ablations reveal a consistent separation–fidelity tradeoff that is most visible on the hard shared-context slice.

<!-- FIGURE_PROMPT: Teaser figure for a research paper on adaptive multi-reference self-attention in personalized text-to-image generation. Show three side-by-side conceptual panels for multi-concept composition with shared background cues: (1) vanilla FreeCustom MRSA with attention leaking from the target concept region to the wrong reference due to similar context, visualized with red cross-concept arrows; (2) proposed ReMRSA with concept-conditioned weighting, background deconfounding, and conflict-aware routing, visualized with selectively strengthened green arrows to the correct reference and suppressed gray arrows to context-matched distractors; (3) a tradeoff view indicating that stronger separation can help on shared-context cases but may reduce fidelity on clean-context cases. Use a clean NeurIPS-style diagram aesthetic, muted academic colors, labeled tokens, and concise annotations such as 'context leakage', 'deconfounded routing', and 'separation-fidelity tradeoff'. -->

# Related Work

## Diffusion personalization and customized concept generation

Personalized diffusion models have largely focused on injecting a new subject into a pretrained generator while preserving general text controllability [dreambooth2022, textualinversion2022, customdiffusion2023, perfusion2023]. DreamBooth-style finetuning improves identity recall but can overfit or entangle the subject with training context [dreambooth2022], while textual inversion and low-rank adaptation offer more parameter-efficient alternatives with different tradeoffs in fidelity and editability [textualinversion2022, lora2022]. Later systems such as BLIP-Diffusion, IP-Adapter, and InstantID introduced image-guided conditioning that improves subject specificity without full model retraining [blipdiffusion2023, ipadapter2023, instantid2024]. These approaches establish strong reference conditioning, yet they do not directly address how several customized references should compete within the same generation. Our work differs by focusing on the internal allocation problem that arises after reference features are available: adaptive MRSA must decide which concept should own a region when visually similar context makes that decision ambiguous.

A second thread studies compositional personalization, where several concepts must coexist in one generated image without identity collapse or attribute leakage [elite2023, subjectdiffusion2024, composer2023, mixofshow2023]. Existing methods improve concept combination through better prompt engineering, training curricula, or subject-specific control modules, but they typically evaluate overall composition success rather than contamination driven by shared background cues. In practice, those broad settings can mask the exact failure mode that emerges in FreeCustom pipelines: visually plausible images with incorrect cross-reference borrowing at the region level. By centering contamination error and slicing by contextual overlap, our study narrows the evaluation target and complements prior compositional personalization work with a more diagnostic stress test.

## Attention control and token routing in text-to-image generation

A large body of work modifies attention to improve controllability, grounding, or compositional fidelity in diffusion models [attendandexcite2023, boxdiff2023, layoutguidance2023, prompttoprompt2022]. Some methods amplify neglected subject tokens, while others impose spatial constraints or box supervision to align text and image structure [attendandexcite2023, boxdiff2023]. These approaches reveal that attention manipulation can strongly affect object binding, yet they usually operate on text-token attention or layout conditioning rather than on competition among multiple image reference banks. Our setting is different: the challenge is not to force recognition of an under-attended word, but to prevent a generated region from attending to the wrong customized concept when several reference sources are simultaneously plausible.

Related ideas also appear in mixture-of-experts routing, ownership modeling, and selective attention under competition [switchtransformer2021, moe_routing2022, tokenchoice2023]. Those methods suggest that explicit competition can improve specialization when multiple modules are available, especially if routing is conditioned on token-level evidence. ReMRSA adapts that intuition to FreeCustom by converting deconfounded concept similarity into ownership weights across references. Unlike general expert-routing methods, however, our mechanism must preserve image-detail fidelity while suppressing context-driven contamination. This makes scheduling and residual preservation central design choices, and the results show that stronger routing alone is not sufficient to improve aggregate composition quality.

## Similarity weighting, deconfounding, and evaluation of compositional fidelity

Similarity-aware retrieval and deconfounding have been studied in representation learning, metric learning, and causal-style adjustment for spurious correlation [supcon2020, debiasedclip2023, spuco2023, invariantlearning2021]. These works show that raw similarity can encode nuisance variables such as background or co-occurring context, which leads models to rely on the wrong evidence. Our method imports that insight into MRSA by separating a concept-relevant similarity term from a background-sensitive affinity estimate. The technical setting is different from supervised classification, but the motivating pathology is closely related: a high similarity score can be correct numerically and wrong semantically.

Evaluation protocols for image generation also matter here. Standard metrics such as FID, CLIP-based alignment, or generic preference studies often miss whether two customized concepts have been bound to the correct regions [fid2017, clip2021, tifa2023]. Recent compositional benchmarks therefore emphasize object binding, relation faithfulness, and localized failure analysis [geneval2024, compbench2024, t2ibind2024]. Our evaluation follows this diagnostic tradition by using region-level contamination error and by separating hard shared-context cases from cleaner controls. In contrast to prior benchmark papers, we study this metric inside a specific FreeCustom MRSA design space and use it to explain why some adaptive attention variants improve separation only by sacrificing clean-context fidelity.

# Method

## Problem formulation

Consider a FreeCustom generation setup with \(K\) customized concepts and a denoising trajectory indexed by step \(u \in \{1,\dots,U\}\) [freecustom2024]. Each concept \(i\) contributes a reference feature bank with keys \(K^{(i)} \in \mathbb{R}^{N_i \times d}\) and values \(V^{(i)} \in \mathbb{R}^{N_i \times d}\), extracted from concept-specific exemplars by the same backbone used in the base pipeline. At a given denoising layer and spatial token \(t\), the image latent produces a query vector \(q_{u,t} \in \mathbb{R}^d\). Vanilla MRSA pools all reference tokens into a single bank and computes attention directly from scaled dot-product similarity. Although this is effective when concepts are visually distinct, it becomes fragile when two references share context, because the query can receive high scores from the wrong concept for reasons unrelated to foreground ownership.

ReMRSA addresses this problem by decomposing reference selection into three nested decisions. First, it measures how similar the query is to each concept after aggregating token-level evidence within that concept. Second, it estimates how much of that similarity is likely to arise from context rather than concept-defining foreground structure. Third, it converts the deconfounded concept scores into routing weights that regulate how much attention each concept contributes to the final value mixture. This formulation preserves the efficiency of MRSA while making the competition among references explicit. As shown in Figure 1, the resulting mechanism can suppress context-driven attraction without discarding the useful detail contained in the original reference features.

## Adaptive multi-reference attention

For each concept \(i\), ReMRSA begins with token-level logits
\[
\ell_{u,t,n}^{(i)}=\frac{q_{u,t}^{\top}k_n^{(i)}}{\sqrt d},
\]
where \(k_n^{(i)}\) is the \(n\)-th key in concept \(i\)'s bank. These logits define an intra-concept soft attention distribution
\[
a_{u,t,n}^{(i)}=\mathrm{Softmax}_n(\ell_{u,t,n}^{(i)}),
\]
and the corresponding concept summary vector
\[
c_{u,t}^{(i)}=\sum_{n=1}^{N_i} a_{u,t,n}^{(i)} v_n^{(i)}.
\]
The simplest concept-conditioned similarity is then the maximum or pooled affinity of the query to concept \(i\). In ReMRSA we use a temperature-smoothed log-sum-exp aggregation,
\[
s_{u,t}^{(i)}=\frac{1}{\lambda}\log \sum_{n=1}^{N_i} \exp(\lambda \ell_{u,t,n}^{(i)}),
\]
which behaves like mean pooling when \(\lambda\) is small and approaches max pooling as \(\lambda\) increases. This choice stabilizes routing relative to a hard maximum while still highlighting strong evidence for concept ownership.

A raw concept score is still vulnerable to nuisance overlap, so ReMRSA estimates a background-sensitive affinity term from low-frequency and globally shared reference features. Let \(\bar{k}^{(i)}\) denote a concept-level context prototype computed by averaging a subset of tokens selected for low foreground saliency, and let \(\bar{q}_{u,t}\) denote a smoothed query representation from the same layer. The background affinity is
\[
b_{u,t}^{(i)}=\frac{\bar{q}_{u,t}^{\top}\bar{k}^{(i)}}{\sqrt d}.
\]
The deconfounded concept score becomes
\[
\tilde{s}_{u,t}^{(i)}=s_{u,t}^{(i)}-\beta_u b_{u,t}^{(i)},
\]
where \(\beta_u\) is a step-dependent coefficient. Intuitively, high raw similarity is discounted when it is likely to be explained by context that several concepts share. This operation does not remove context features from the value stream; it only reduces their influence on concept selection.

Conflict-aware routing transforms these deconfounded scores into ownership weights across concepts,
\[
r_{u,t}^{(i)}=\mathrm{Softmax}_i(\tau_u \tilde{s}_{u,t}^{(i)}),
\]
where the temperature \(\tau_u\) sharpens or softens competition. A large \(\tau_u\) encourages exclusivity, while a smaller value allows a region to draw from several concepts if the evidence remains mixed. ReMRSA then forms the final concept contribution as
\[
z_{u,t}=\sum_{i=1}^{K} \gamma_u^{(i)} r_{u,t}^{(i)} c_{u,t}^{(i)} + \delta_u z^{\text{base}}_{u,t},
\]
where \(z^{\text{base}}_{u,t}\) is the vanilla MRSA output and \(\delta_u\) is an optional residual-preservation coefficient. The gain term \(\gamma_u^{(i)}\) allows concept-conditioned scaling; in the reported configuration we set
\[
\gamma_u^{(i)} = 1 + \eta_u \,\mathrm{LayerNorm}(\tilde{s}_{u,t}^{(i)}).
\]
Combining the adaptive concept gain, deconfounding penalty, and routing weights yields the effective attention weight
\[
\alpha_{u,t,n}^{(i)} \propto
\exp\!\big(\ell_{u,t,n}^{(i)} + \eta_u g_{u,t}^{(i)} + \rho_u r_{u,t}^{(i)} - \beta_u b_{u,t}^{(i)}\big),
\]
which matches the design principle in the original draft while making each component operational.

<!-- FIGURE_PROMPT: Framework diagram for ReMRSA in a FreeCustom pipeline. Show a central query token from the denoising U-Net attending to multiple concept-specific reference banks. The diagram should include four labeled modules in sequence: (1) concept-conditioned similarity aggregation from token logits, (2) background/context affinity estimation using context prototypes, (3) conflict-aware routing across concepts via softmax ownership weights, and (4) stage-wise scheduling across early, middle, and late denoising steps. Include arrows leading to a final adaptive attention output mixed with a residual vanilla MRSA path. Use clear mathematical labels such as s^(i), b^(i), r^(i), and alpha^(i), and a polished conference-paper visual style. -->

## Stage-wise scheduling and ablation design

A fixed routing strength performs poorly because the role of reference attention changes across denoising. Early steps establish coarse layout and concept placement, middle steps determine which reference should dominate an ambiguous region, and late steps recover fine visual detail. ReMRSA therefore uses three denoising phases. In the early phase, \(\tau_u\) and \(\beta_u\) are moderate so that concept ownership stabilizes without over-committing to noisy evidence. In the middle phase, both coefficients increase, making routing sharper and deconfounding stronger when contextual confusion is most harmful. In the late phase, \(\tau_u\) decreases and the residual coefficient \(\delta_u\) increases, allowing detail refinement to borrow more from the base MRSA path. Concretely, the reported configuration uses early/mid/late windows of 0.0-0.3, 0.3-0.7, and 0.7-1.0 of the denoising trajectory; \(\beta_u\) follows \((0.20, 0.45, 0.10)\), \(\tau_u\) follows \((4, 8, 3)\), \(\eta_u\) follows \((0.15, 0.25, 0.10)\), and \(\delta_u\) follows \((0.10, 0.05, 0.20)\).

This schedule also clarifies the reported ablations. The similarity-only variant retains \(s_{u,t}^{(i)}\) and the adaptive gain but sets \(\beta_u=0\) and removes routing competition, isolating whether global concept weighting alone improves contamination. The routing-without-deconfounding variant keeps concept competition but again uses \(\beta_u=0\), testing whether exclusivity helps even when similarity remains context-sensitive. The static-exclusivity model fixes \(\tau_u\) across all denoising steps, removing the temporal adaptation. The early-over-separation variant increases \(\tau_u\) and \(\beta_u\) too aggressively in the first phase, while the dense residual-boosted model raises \(\delta_u\) and allows the vanilla path to dominate whenever routing is unstable. Taken together, these ablations let the experiments distinguish whether gains come from similarity scaling, stronger competition, temporal scheduling, or residual preservation.

For reproducibility, Algorithm 1 summarizes the complete forward rule. All methods share the same backbone, prompt set, reference images, denoising schedule, and random seeds; only the MRSA weighting rule changes. This paired design makes contamination comparisons meaningful at the case level and supports matched statistical tests when multiple completed runs are available.

## Algorithm

**Algorithm 1: ReMRSA forward pass at denoising step \(u\)**

Given query tokens \(\{q_{u,t}\}_t\), concept key-value banks \(\{K^{(i)},V^{(i)}\}_{i=1}^K\), step coefficients \(\beta_u,\tau_u,\eta_u,\delta_u\), and aggregation temperature \(\lambda\), compute token logits \(\ell_{u,t,n}^{(i)}\) for every concept and reference token. Normalize logits within each concept to obtain \(a_{u,t,n}^{(i)}\), and use these weights to form concept summary vectors \(c_{u,t}^{(i)}\). Aggregate concept-conditioned similarity with the log-sum-exp rule to obtain \(s_{u,t}^{(i)}\). Next, estimate context prototypes \(\bar{k}^{(i)}\) and smoothed query features \(\bar{q}_{u,t}\), then compute background affinities \(b_{u,t}^{(i)}\) and deconfounded scores \(\tilde{s}_{u,t}^{(i)}\). Apply a softmax across concepts with temperature \(\tau_u\) to produce routing weights \(r_{u,t}^{(i)}\). Form adaptive concept gains \(\gamma_u^{(i)}\), combine the routed concept summaries, and add the residual vanilla MRSA path weighted by \(\delta_u\). Return the mixed output \(z_{u,t}\), which replaces the vanilla MRSA value at the corresponding layer and token.

# Experiments

## Experimental setup

The evaluation targets contamination-aware multi-concept composition in FreeCustom [freecustom2024]. Each test case contains multiple customized concepts and a prompt requiring them to appear together in one image. To expose the failure mode of interest, the benchmark is split into a hard matched-background/shared-context subset and a cleaner mismatched-background/clean-context subset. The hard subset contains cases in which references share contextual statistics such as scene layout, dominant color, or object co-occurrence, making cross-reference borrowing more likely. The clean subset retains the same compositional task but reduces contextual overlap. This split follows the practical intuition discussed in recent compositional and binding benchmarks: aggregate scores can hide whether a system succeeds for the right reason [geneval2024, compbench2024, t2ibind2024].

Region-level contamination error is the primary metric, with lower values indicating better concept binding. The metric is computed per case and then averaged within each method and subset. We also report aggregate contamination across all cases. Because every method is evaluated on the same cases and seeds, comparisons are paired at the case level. The experiments were run on an NVIDIA A800-SXM4-80GB GPU using the same image size, inference schedule, and guidance settings across methods. Table 1 summarizes the main hyperparameters used by the reported models.

**Table 1. Hyperparameters for vanilla MRSA and ReMRSA variants. All methods share the same generation setup; only the adaptive attention coefficients vary.**

| Hyperparameter | Value |
|---|---:|
| Image resolution | 1024 × 1024 |
| Inference steps | 50 |
| Guidance scale | 7.5 |
| Denoising schedule phases | early / middle / late |
| Aggregation temperature \(\lambda\) | 6.0 |
| Early \(\beta_u\) | 0.20 |
| Middle \(\beta_u\) | 0.45 |
| Late \(\beta_u\) | 0.10 |
| Early \(\tau_u\) | 4 |
| Middle \(\tau_u\) | 8 |
| Late \(\tau_u\) | 3 |
| Early \(\eta_u\) | 0.15 |
| Middle \(\eta_u\) | 0.25 |
| Late \(\eta_u\) | 0.10 |
| Residual coefficient \(\delta_u\) | 0.10 / 0.05 / 0.20 |
| Listed random seeds | 11, 23, 37, 49, 83 |
| GPU | NVIDIA A800-SXM4-80GB |

Although the evaluation is paired by design, the available artifact supports only descriptive analysis rather than inferential claims across repeated completed runs. Accordingly, any comparison below is stated as descriptive, and differences are treated as not statistically significant in the absence of valid replicated estimates. This framing keeps the paper centered on what the experiments show: the adaptive variants expose a strong tradeoff structure, but the full deconfounded routing model does not improve the primary metric relative to vanilla MRSA.

## Main quantitative results

Table 2 presents the aggregate contamination results. Vanilla FreeCustomMRSA remains the strongest reference among the core methods, with an overall contamination error of 0.063727. The full ReMRSA variant, reported here as ForegroundDeconfoundedConflictRoutedMRSA, reaches 0.066970, which is higher and therefore worse under the lower-is-better metric. This difference is descriptive and not statistically significant under the available evidence. The result matters because it directly answers the paper’s central question: adding concept-conditioned weighting, deconfounding, routing, and scheduling does not by itself yield a better contamination score than the vanilla FreeCustom attention rule.

**Table 2. Main contamination results on the FreeCustom composition benchmark. Lower is better. Means are reported from the available evaluation summary; inferential comparison is not significant under the current evidence. Best value in each column is bold.**

| Method | Overall contamination error |
|---|---:|
| VanillaFreeCustomMRSA | 0.063727 |
| ForegroundOnlySimilarityWithoutConflictRouting | 0.066168 |
| BackgroundBlindConflictRoutingWithoutDeconfounding | 0.067201 |
| StaticExclusivityWithoutThreePhaseSchedule | 0.067161 |
| ForegroundDeconfoundedConflictRoutedMRSA | 0.066970 |
| DenseUncappedResidualBoostedMRSA | **0.063039** |
| EarlyOverSeparationScheduledMRSA | 0.065922 |

The strongest non-vanilla aggregate result comes from DenseUncappedResidualBoostedMRSA, which slightly improves the overall score relative to vanilla. Yet this apparent gain does not establish a superior general method, because the subset analysis reveals that it improves hard cases partly by sacrificing performance on cleaner compositions. Building on this observation, aggregate ranking alone would have obscured the actual mechanism at work. The practical takeaway is that adaptive MRSA should be evaluated as a conditional tradeoff, not as a single-score replacement for vanilla attention.

## Subset analysis and tradeoff structure

The matched-background/shared-context subset is where the adaptive variants become most informative. Vanilla FreeCustomMRSA records hard-slice contamination values of 0.103694, 0.103994, 0.100578, 0.104976, and 0.104119 across the listed seeds. The full ReMRSA variant is consistently higher on the same subset, with 0.110118, 0.110516, 0.109637, 0.107964, and 0.109986, again indicating no improvement. This directional consistency matters more than the small aggregate difference because the method was designed precisely for shared-context ambiguity. As shown in Figure 2, the hard-slice comparison makes clear that the full deconfounded routing mechanism strengthened separation pressure without translating that pressure into lower contamination.

In contrast, some ablations reduce hard-slice contamination more effectively than the full model. DenseUncappedResidualBoostedMRSA reaches relatively low matched/shared-context values in several seeds, including 0.094044 and 0.097295, while EarlyOverSeparationScheduledMRSA also shows occasional reductions such as 0.095095 and 0.101947. These cases suggest that stronger concept separation can help when contextual confusion is severe. However, the same models degrade on the clean mismatched-background/control slice, where vanilla remains near 0.0238-0.0244 and the adaptive variants can rise substantially above that range. The residual-boosted variant, for example, includes clean-slice values such as 0.026115 and 0.029874, while the early-over-separation schedule reaches as high as 0.031204. The differences between these methods and vanilla are descriptive and not statistically significant with the current evidence, but the pattern is consistent enough to support a substantive interpretation: mechanisms that suppress leakage in hard scenes can also over-separate concepts in scenes that already provide sufficient disambiguation.

![Hard-slice versus clean-slice contamination tradeoff across vanilla MRSA, full ReMRSA, and ablations. Lower values indicate better region-faithful concept binding. The figure highlights that gains on matched-background/shared-context cases often coincide with degradation on mismatched-background/clean-context cases.](charts/hard_clean_tradeoff.png)

This tradeoff clarifies why the full deconfounded routing model underperforms even though its design is well motivated. The method combines several interventions that each push attention toward exclusivity, and their interaction appears to be too aggressive when the scene does not require strong arbitration. By contrast, the residual-boosted variant preserves more of the vanilla signal and therefore avoids the worst clean-slice degradation while still helping some hard examples. In other words, the useful ingredient in adaptive MRSA may not be deconfounded routing alone, but a carefully constrained balance between selective competition and access to the original pooled reference path.

## Ablation analysis

Table 3 summarizes the mechanism-level ablation comparison. Similarity-only weighting performs worse than vanilla, which suggests that simply amplifying concept-level similarity is insufficient once contextual overlap is present. Routing without deconfounding is also worse, indicating that competition built on a confounded score can sharpen the wrong decision rather than correct it. Likewise, static exclusivity underperforms the scheduled variant family, supporting the idea that temporal adaptation matters when concept ownership and detail recovery occur at different denoising stages. None of these differences is statistically significant under the current evidence, but the direction of results is consistent with the qualitative mechanism analysis.

**Table 3. Ablation summary for adaptive MRSA components. Lower is better. Bold indicates the best result among ablations. Under the available evidence, comparisons are descriptive and not statistically significant.**

| Ablation family | Overall contamination error |
|---|---:|
| Similarity only | 0.066168 |
| Routing without deconfounding | 0.067201 |
| Static exclusivity | 0.067161 |
| Full deconfounded routing | 0.066970 |
| Residual-boosted adaptive MRSA | **0.063039** |
| Early over-separation schedule | 0.065922 |

The ablations also refine the original hypothesis. If background deconfounding and conflict-aware routing were sufficient, the full model would have dominated the simpler variants on the hard slice while preserving control performance elsewhere. That did not occur. Instead, the strongest evidence supports a narrower conclusion: adaptive scaling changes the operating point of MRSA, and the best variants are those that soften the intervention through residual preservation or milder scheduling. This insight is directly relevant to FreeCustom composition because it suggests that contamination control is less about maximizing exclusivity and more about matching the degree of arbitration to scene ambiguity.

## Discussion

The main empirical message is straightforward. Adaptive MRSA is useful for diagnosing composition failure modes, but the full deconfounded routing design does not outperform vanilla FreeCustomMRSA on the primary contamination metric. This finding differs from a common expectation in personalized generation that stronger reference selectivity should automatically improve concept binding [customdiffusion2023, composer2023, freecustom2024]. In our setting, selectivity helps only conditionally. When background overlap is severe, stronger routing can reduce wrong-reference borrowing; when the scene is already clean, the same mechanism can suppress benign sharing that preserves detail and identity.

In contrast to prior work that reports broad improvements in compositional or personalized generation, our results isolate a narrower but practically important regime: shared-context ambiguity [elite2023, subjectdiffusion2024, compbench2024]. The hard-slice analysis suggests that aggregate scores can be misleading because they average over examples that need very different levels of attention arbitration. This helps explain why vanilla MRSA remains difficult to beat. Its uniform mixing is imperfect, but it also provides a robust default when references are not strongly confounded. ReMRSA makes that hidden tradeoff visible by perturbing the attention mechanism in controlled ways.

A surprising result is that the residual-boosted ablation outperforms the full adaptive model in aggregate. Mechanistically, this implies that the base MRSA path retains useful fine-grained information that deconfounded routing can accidentally attenuate. Similar observations have been made in attention-control work where aggressive intervention improves binding at the cost of realism or identity texture [attendandexcite2023, boxdiff2023]. Our experiments extend that lesson to multi-reference personalized composition: the objective is not to maximize exclusivity, but to intervene selectively where contextual overlap is genuinely harmful.

More broadly, these findings suggest a practical strategy for future FreeCustom systems. Rather than replacing vanilla MRSA with a single globally stronger router, a successful design may need uncertainty-aware or scene-adaptive activation that intervenes only when concept confusion is detected. Such a view aligns adaptive attention with diagnostic estimation rather than blanket control. Even without a positive overall gain, the present study contributes a clearer picture of where MRSA fails and why some seemingly principled fixes overcorrect.

![Per-method contamination profiles on the matched-background/shared-context subset. The chart visualizes that the full deconfounded routing model remains above vanilla across the reported seeds, while residual-preserving variants sometimes reduce hard-case contamination at the expense of other slices.](charts/hard_slice_profiles.png)

# Limitations

- The available evaluation supports descriptive comparisons but not strong inferential conclusions. The reported comparisons therefore should be read as directional patterns, and all method differences in the main text are stated as not statistically significant under the current evidence.

- The benchmark emphasizes region-level contamination error as the primary endpoint for composition quality. This focus is appropriate for the MRSA failure mode studied here, but it does not capture other relevant aspects of personalized generation such as identity fidelity, prompt adherence, or perceptual image quality.

- One important comparator, GlobalSimilarityWeightedMRSA, is absent from the reported tables. As a result, the experiments isolate several adaptive ingredients but do not fully separate the effect of simple similarity reweighting from the effect of the full deconfounded routing design.

- The reported artifact was produced on an NVIDIA A800-SXM4-80GB GPU with a 50-step generation schedule, and the available summary does not yet support fully replicated multi-run statistical estimation. This limits the precision of uncertainty quantification for the present paper.

# Conclusion

ReMRSA adapts multi-reference self-attention in FreeCustom through concept-conditioned similarity scaling, background deconfounding, conflict-aware routing, and stage-wise scheduling. In the reported evaluation, the full adaptive model does not beat vanilla FreeCustomMRSA on region-level contamination error, while residual-preserving ablations reveal a clearer separation–fidelity tradeoff on matched-background/shared-context cases.

These results suggest that adaptive MRSA is most valuable when it acts selectively rather than uniformly. Future work should test uncertainty-aware routing and broader composition metrics so that contamination control can be improved without sacrificing clean-context fidelity.