# Simulated Peer Review

## Reviewer A (Methodology Expert)

### Summary
The paper is tightly centered on adapting multi-reference self-attention for FreeCustom-style multi-concept composition, with an emphasis on contamination under shared-context ambiguity. The draft is unusually transparent about negative and inconclusive results, which is commendable. However, the current submission reads more like a methodological note plus failed pilot than a complete empirical paper. The main methodological issue is that several proposed components are only loosely specified, and the reported evidence does not validate the full AMRA design.

### Strengths
- **Good topic alignment overall.** The manuscript stays mostly focused on adaptive MRSA for FreeCustom and on contamination-aware composition.
- **Problem framing is appropriate.** The stress-test framing around matched-background/shared-context contamination is well motivated.
- **Mechanism decomposition is valuable.** The paper distinguishes similarity weighting, deconfounding, routing, and scheduling rather than presenting a monolithic method.
- **Transparent reporting.** The authors clearly state that the pilot evidence is non-confirmatory and that the main model underperforms vanilla.

### Weaknesses
#### 1) Topic alignment
The paper is **mostly on-topic**, but there are a few places where it drifts:
- In the **Abstract** and **Conclusion**, execution failure, missing baseline, and effectively \(n=1\) are foregrounded almost as outcomes. These are valid limitations, but they should not be framed as substantive contributions.
- The statement in the Introduction that the paper “makes three contributions” includes:
  - “it reports the current pilot evidence transparently.”
  This is not really a scientific contribution; it is good reporting practice.
- The paper sometimes shifts from a method paper to an experiment-status report. That weakens the methodological narrative.

#### 2) Method under-specification
Key components are described conceptually, but not operationally:
- How exactly is the concept-conditioned similarity \(s_t^{(i)}\) computed?
- How is the background/context affinity term \(b_t^{(i)}\) estimated?
- What token features define “foreground evidence” versus “shared-context evidence”?
- How is stage-wise scheduling applied across denoising steps in practice?
- What are the exact update equations for ablation variants?

Without these details, the method is not reproducible and is hard to evaluate as a technical contribution.

#### 3) Claim-evidence alignment
The paper is unusually conservative, which helps, but several claims still overreach relative to the evidence.

##### Title claim
**Title:** “AMRA: Adaptive Multi-Reference Attention for FreeCustom Composition”
- Supported that the paper proposes such a method.
- Not supported as an effectiveness claim, but the title does not explicitly claim improvement, so this is acceptable.

##### Abstract claims
1. **“AMRA ... combines similarity-aware weighting, foreground deconfounding, conflict-aware routing, and stage-wise scheduling.”**
   - Supported by Method description, not by Results as an effectiveness claim.
2. **“The evaluation targets region-level contamination error...”**
   - Supported by Experiments section.
3. **“the evidence is not confirmatory: execution status is failed, reported cells are effectively \(n=1\), and one critical baseline is missing.”**
   - Supported by the provided run evidence; indeed actual experiment execution count is 1.
4. **“the full proposed variant does not improve over vanilla FreeCustomMRSA under the stated metric direction.”**
   - Supported by reported overall and slice-wise values in Experiments.
5. **“the ablation design is informative and identifies shared-context composition as the appropriate stress test for future reruns.”**
   - Partially supported. The stress-test claim is somewhat supported by the reported slice tradeoffs.
   - “Ablation design is informative” is qualitative and not directly evidenced beyond descriptive comparisons.

##### Conclusion claims
1. **“We presented AMRA ... to reduce multi-concept contamination.”**
   - This is phrased as intended purpose; acceptable.
2. **“The method is well motivated for shared-context composition...”**
   - Supported conceptually, but not empirically proven.
3. **“the ablation suite provides a useful mechanism-level decomposition.”**
   - Partially supported by the ablation structure, though usefulness is asserted rather than demonstrated.
4. **“the full proposed variant underperforms vanilla ... on overall, hard-slice, and clean-slice contamination error...”**
   - Supported by the reported numbers.
5. **“identifies the correct stress test”**
   - Somewhat overstated. The evidence suggests the hard slice is informative, but “correct” is too strong given incomplete experimentation.

#### 4) Critical discrepancy with actual evidence
The manuscript says:
- “although five seeds are listed, reported cells are effectively \(n=1\)”
This is directionally honest, but the provided evidence states:
- **Actual experiment executed 1 time**
- `run-1.json` contains **empty metrics**
So the reported per-seed numeric tables in the paper are not traceable to the supplied run artifact. That is a **major methodological credibility issue** unless those numbers came from an external prior run not documented here.

#### 5) Completeness
The paper includes core sections: Title, Abstract, Introduction, Method, Experiments, Conclusion. But it is still incomplete for a NeurIPS-style submission:
- **No Related Work section**
- **No Limitations / broader impacts / ethics section**
- **No explicit Discussion section**
- **No Appendix or implementation details section**
- Likely **well below 5,000 words**

### Actionable revisions
1. **Reframe contributions.** Replace “transparent pilot reporting” as a contribution with a real technical contribution or remove the numbered contribution framing.
2. **Fully specify the method.** Define \(s_t^{(i)}\), \(b_t^{(i)}\), scheduling rules, residual variant, and all ablation equations.
3. **Resolve evidence provenance.** If the reported numbers come from prior runs, provide exact run IDs, logs, and artifacts. If not, remove the unsupported tables.
4. **Separate limitations from contributions.** Keep execution failure and missing baseline in a limitations paragraph, not as central takeaways.
5. **Add missing sections.** Include Related Work, Discussion, and implementation details.
6. **Tone down overclaims.** Replace “identifies the correct stress test” with “suggests that matched/shared-context is a useful stress test.”

---

## Reviewer B (Domain Expert)

### Summary
This paper addresses an important problem in personalized text-to-image generation: compositional binding failures when multiple customized concepts share context or background. The focus on FreeCustom and MRSA is domain-relevant, and the idea of deconfounding background similarity is sensible. Unfortunately, the current draft is not yet a convincing domain contribution because the empirical support is incomplete, one critical comparator is missing, and the reported execution evidence indicates the main run failed.

### Strengths
- **Strong domain relevance.** Multi-concept composition quality is a meaningful open problem in customized diffusion systems.
- **Right failure mode.** Background/context leakage across concept references is a realistic and practically important issue.
- **Stress-test choice is good.** A matched-background/shared-context slice is exactly the kind of setup where MRSA-based methods may fail.
- **Negative results are honestly reported.** This is refreshing and useful.

### Weaknesses
#### 1) Topic alignment
The draft is mostly aligned with the stated topic:
- adaptive MRSA for FreeCustom,
- dynamic scaling of attention based on concept similarity,
- improving multi-concept composition quality.

However, there are two alignment issues:
- The paper emphasizes **region-level contamination error** almost exclusively, while the title/topic mentions **composition quality** more broadly. Composition quality in this domain usually includes identity fidelity, prompt adherence, image quality, and binding accuracy. The paper narrows to contamination only, which is acceptable, but should be explicit in the title or abstract.
- The discussion of runtime/execution failure risks reading like an operations report. Environment issues should not be treated as paper content beyond reproducibility notes.

#### 2) Missing essential baseline
The paper itself notes the absence of **GlobalSimilarityWeightedMRSA**, and the code confirms it is a registered condition. This baseline is crucial because:
- It isolates whether adaptive gains come from **simple similarity weighting** versus the full deconfounded conflict-routing design.
- Without it, the paper cannot substantiate the claim that deconfounding/routing specifically matter.

This omission is especially damaging because the main scientific question is whether more structured attention modulation improves over vanilla and simpler adaptive alternatives.

#### 3) Claim-evidence alignment
For the domain claims:

##### Introduction/Abstract/topic-level claims
- **Shared-context composition is the key stress test.**
  - Reasonably supported by slice design and by tradeoffs observed in the reported tables.
  - But still not definitive due to incomplete baseline coverage and failed run status.
- **AMRA is well motivated for contamination reduction.**
  - Supported conceptually, not empirically.
- **The full AMRA variant does not improve over vanilla.**
  - Supported by the reported values, assuming those values are authentic.
- **Ablations identify which ingredients matter.**
  - Not really supported. The ablations are described, but since the full run failed and only effectively one execution exists, the paper cannot reliably infer which ingredients matter.

#### 4) Evaluation breadth is too narrow
For a domain paper on composition quality, one metric is insufficient:
- No identity preservation metric
- No prompt/image alignment metric
- No human evaluation
- No qualitative figure comparisons
- No analysis of failure cases by concept type, pose overlap, or semantic relation

A contamination metric is useful, but not enough to assess whether improved separation hurts overall customization quality.

#### 5) Figures
The draft contains **zero figures** in the provided text.
- This is a major presentation issue.
- Per your checklist: **zero figures = desk reject**.
For this topic, figures are essential: example compositions, attention maps, failure cases, and tradeoff visualizations.

#### 6) Citation distribution
The provided draft includes **no citations at all**.
This violates the requirement that Method, Experiments, and Discussion cite relevant work. For this topic, there should be references to:
- FreeCustom / custom concept composition methods,
- attention routing or token ownership methods,
- diffusion personalization and compositionality benchmarks,
- statistical evaluation methods if used.

### Actionable revisions
1. **Add the missing GlobalSimilarityWeightedMRSA baseline** and rerun all methods under the same setup.
2. **Broaden evaluation beyond contamination error**:
   - identity fidelity,
   - prompt compliance / concept binding,
   - image quality,
   - human side-by-side preference if possible.
3. **Add at least 2 figures**:
   - qualitative examples comparing vanilla, simple similarity weighting, and AMRA,
   - a plot showing hard-slice vs clean-slice tradeoffs.
4. **Clarify scope**: if the paper is really about contamination-aware composition rather than overall composition quality, reflect that explicitly in title/abstract.
5. **Add citations throughout**, especially in Method and Experiments.
6. **Provide qualitative evidence** that deconfounding reduces wrong-reference borrowing rather than merely changing the contamination metric.

---

## Reviewer C (Statistics / Rigor Expert)

### Summary
From a rigor standpoint, the paper is not yet publishable. The biggest issue is a mismatch between claimed experimental detail and the supplied run evidence: the actual experiment appears to have executed only once and produced no metrics, yet the manuscript reports multi-seed numerical results. Even setting that aside, the statistical reporting is inadequate: no confidence intervals, no error bars, no valid \(n>1\) at the reported table-cell level, no significance tests, and no clear accounting of independence units.

### Strengths
- The paper explicitly acknowledges several rigor limitations rather than hiding them.
- The code imports tooling for bootstrap CIs, Wilcoxon tests, Cohen’s \(d\), and rank-biserial effect sizes, indicating the authors were at least aware of proper statistical procedures.
- The manuscript correctly avoids making a strong positive efficacy claim.

### Weaknesses
#### 1) Statistical validity
This is the weakest aspect.

##### Confidence intervals / error bars
- **Not reported.**
- The manuscript explicitly says no uncertainty estimates are available.
- This fails the checklist requirement.

##### Multiple seeds / \(n>1\)
- The paper says five seeds are listed but “reported cells are effectively \(n=1\).”
- The supplied evidence says **actual experiment executed 1 time**.
- Therefore, there is **no credible multi-seed evidence** in the current artifact.
- This is a **critical discrepancy**.

##### Significance tests
- **None reported.**
- Given the tiny and possibly non-independent sample structure, significance testing may not even be appropriate yet, but if claims compare methods, inferential support is needed once reruns are complete.

##### Independence / unit of analysis
- It is unclear whether reported values are:
  - per generated image,
  - per case,
  - per seed aggregate,
  - or mixed summaries.
- The paper should define the experimental unit and whether bootstrap resampling occurs over cases, prompts, or seeds.

#### 2) Claim-evidence verification
Below is a stricter claim check for title, abstract, and conclusion.

### Claim-evidence matrix

| Claim | Supported by specific result? | Evidence location | Verdict |
|---|---|---|---|
| AMRA is an adaptive MRSA variant for FreeCustom | Method defines it | Method | Supported as description |
| AMRA improves multi-concept composition quality | Not explicitly claimed in title, but implied by topic | No confirmatory result | Not established |
| Evaluation targets region-level contamination error | Yes | Experiments narrative | Supported |
| Full proposed variant does not improve over vanilla | Yes, in manuscript tables/text | Experiments paragraph | Supported if numbers are valid |
| Evidence is non-confirmatory due to failed execution, \(n=1\), missing baseline | Yes | Experiments + supplied run metadata | Supported |
| Ablations identify shared-context composition as appropriate stress test | Weak support only | Descriptive slice comparisons | Partially supported |
| AMRA reduces contamination | No, contradicted for full variant | Results show worse than vanilla | Unsupported / contradicted |
| Ablation suite provides useful mechanism-level decomposition | No quantitative proof of usefulness | No figure/table analyzing component contributions statistically | Weakly supported |

The strongest unsupported/contradicted phrasing is in the **Conclusion**:
- “to reduce multi-concept contamination”
This is framed as purpose, but in context can be read as an outcome claim. Since the full method underperforms vanilla, wording should be careful.

#### 3) Reproducibility
Partially specified, but incomplete.

##### Present
- GPU type: NVIDIA A800-SXM4-80GB
- Some hyperparameters from code:
  - inference steps,
  - guidance scale,
  - image size,
  - several AMRA-specific coefficients,
  - seeds list,
  - cases per regime per seed,
  - checkpoint paths,
  - dataset root.

##### Missing or insufficient in the paper text
- Exact dataset composition and splits
- Number of total cases actually evaluated
- Selection criteria for matched/shared-context vs mismatched/clean slices
- Whether prompts are fixed across methods
- Whether all methods share identical generations except for attention mechanism
- Software versions
- Runtime budget
- Failure diagnostics
- Exact compute hours / memory use
- Whether seeds listed were actually run to completion

The paper itself does not present most of these details; they are only visible in code snippets provided externally.

#### 4) Completeness and format
- Likely **far below 5,000–6,500 words**.
- Missing Related Work and likely Discussion.
- No tables explicitly shown, although results are narrated.
- No figures.
- No appendix-level implementation details.

#### 5) Writing quality
- The prose is generally fluent and not bullet-driven.
- I did **not** see bullet-point lists in Method/Results/Discussion within the paper draft.
- Hedging such as “we do not claim” is not excessive; the caution level is appropriate.
- **Title length check:**  
  “AMRA: Adaptive Multi-Reference Attention for FreeCustom Composition” = 8 words after the acronym label, or 9 if counting “AMRA”; comfortably within the **<=14 words** requirement.

#### 6) Figures
- **No figures included.**
- By the stated rule: **desk reject**.

#### 7) Citation distribution
- **No citations present anywhere.**
- This fails the requirement outright.
- Method, Experiments, and Discussion must cite prior work and evaluation/statistical references.

### Actionable revisions
1. **Do not report numerical results that cannot be traced to archived artifacts.** This is the top priority.
2. **Rerun experiments with genuine replication**:
   - multiple seeds completed,
   - same case set across methods,
   - archived raw outputs and metrics.
3. **Report uncertainty properly**:
   - mean ± standard deviation or standard error,
   - bootstrap confidence intervals over cases,
   - error bars in plots.
4. **Use appropriate paired tests** if the same prompts/cases are evaluated across methods:
   - Wilcoxon signed-rank or paired permutation tests,
   - with effect sizes.
5. **Define the unit of analysis** and sample size clearly.
6. **Include an actual results table** with \(n\), CI, and significance markers.
7. **Add figures**; without them, the paper is not submission-ready.
8. **Document reproducibility in the paper**, not only in code.

---

# Cross-Cutting Checklist Assessment

## 1. Topic Alignment
**Mostly aligned**, with mild drift.
- On-topic: adaptive MRSA for FreeCustom; concept similarity; contamination-aware multi-concept composition.
- Drift flagged:
  - execution failure and missing baseline are discussed at length and can read like contributions;
  - environment/runtime status should be limitations, not scientific outcomes.

## 2. Claim-Evidence Alignment
### Supported
- evaluation uses contamination error;
- evidence is non-confirmatory;
- full proposed variant does not beat vanilla;
- missing baseline and effective \(n=1\) limit conclusions.

### Unsupported or weakly supported
- “reduce multi-concept contamination” as an outcome for the proposed full method;
- “ablation suite identifies which ingredients matter”;
- “identifies the correct stress test” is too strong.

### Critical discrepancy
- Actual provided run artifact shows **1 execution and empty metrics**, while the manuscript reports per-seed numeric results. This must be reconciled.

## 3. Statistical Validity
- **Confidence intervals / error bars:** No
- **Multiple seeds / \(n>1\):** No credible evidence in supplied artifact
- **Significance tests:** No
- **Appropriate rigor level:** Insufficient

## 4. Completeness
- Missing major sections:
  - Related Work
  - Discussion
  - likely implementation/reproducibility appendix
- Likely below NeurIPS body length expectation of **5,000–6,500 words**

## 5. Reproducibility
- **Partially specified externally**, not adequately in-paper.
- Hyperparameters and seeds exist in code, but paper lacks full dataset and protocol detail.
- Compute resources only partially specified.
- Actual execution provenance is unclear.

## 6. Writing Quality
- Flowing prose: **Yes**
- Bullet lists in Method/Results/Discussion: **No**
- Excessive hedging: **No**
- Title <= 14 words: **Yes**

## 7. Figures
- **Zero figures in draft**
- By your criterion: **desk reject**

## 8. Citation Distribution
- **Fails badly**
- No citations in Intro, Method, Experiments, or Discussion.

---

# Overall Recommendation
**Current status: Reject / major revision required before serious review.**

The paper is honest, on-topic, and potentially interesting as a negative-result methodological study. But in its current form it fails on core submission standards: no figures, no citations, incomplete sections, inadequate reproducibility in the text, missing baseline, no statistical support, and—most seriously—a mismatch between reported results and the supplied execution artifact.

If you want, I can next convert this into:
1. a **meta-review summary**, or  
2. a **line-by-line revision checklist for the authors**.