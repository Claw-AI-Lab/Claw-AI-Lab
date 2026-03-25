Here are 3 feasible, compute-conscious hypotheses grounded in the synthesis and the prior lessons learned.

---

## Hypothesis 1: Pairwise similarity-aware sharpening helps most on high-overlap concept pairs

### Concrete, testable claim
Adding a **training-free similarity-conditioned gate** to FreeCustom’s MRSA—where each reference’s attention contribution is sharpened or suppressed based on **pairwise concept similarity**—will improve **multi-concept fidelity** and reduce **concept leakage** versus vanilla MRSA, especially for **high-similarity concept pairs**.

### Methodology
Implement a simple inference-time controller on top of MRSA:

1. For each concept/reference pair, compute a **global similarity score** using cheap pretrained embeddings:
   - text embedding similarity from prompt concept names
   - image embedding similarity from CLIP image features on references
   - optionally average them into a single conflict score

2. Convert similarity into a gate:
   - if similarity is high, increase selectivity by lowering temperature / sharpening softmax for that concept’s reference attention
   - optionally suppress competing references with a normalized inverse-similarity penalty

3. Keep everything else identical to FreeCustom:
   - same base model
   - same masks
   - same sampling schedule
   - no finetuning

4. Evaluate on concept-pair bins:
   - low similarity
   - medium similarity
   - high similarity

5. Report:
   - per-concept CLIP similarity to its intended reference
   - image-text alignment to full prompt
   - leakage/interference score from region-level concept misassignment, or a proxy from attention-region overlap if manual labels are limited
   - human pairwise preference on a small subset if possible

### Why this is achievable with limited compute
- No training.
- Only adds cheap embedding extraction and scalar gating at inference.
- Can be tested on a modest benchmark of concept pairs and a few seeds.
- Uses pretrained CLIP-like encoders already common in diffusion evaluation.

### Rationale based on proven techniques
- FreeCustom and MC² already show that **attention control at inference** is effective.
- The synthesis identifies a specific gap: current MRSA is not explicitly **similarity-conditioned**.
- Similarity-aware sharpening is a conservative extension of proven attention reweighting, not a new learned module.
- Prior inconclusive temporal studies suggest starting from **simple static routing** before more complex temporal control.

### Measurable prediction
Compared with vanilla MRSA:
- **High-similarity pairs** should show the clearest gain:
  - higher per-concept reference fidelity
  - lower leakage/interference
- **Low-similarity pairs** should show little change, which is expected.

A practical target:
- 5–10% relative reduction in leakage proxy on high-similarity pairs
- small but consistent improvement in per-concept CLIP/reference alignment

### Failure condition
Hypothesis fails if:
- gains do not concentrate in high-similarity pairs, or
- global similarity gating causes over-suppression and lowers concept fidelity overall, or
- results are inconsistent across seeds and not statistically distinguishable from vanilla MRSA.

Also: if outputs/metrics are suspiciously identical across conditions, treat as pipeline failure, not negative evidence.

### Resource requirements estimate
- **Engineering**: 2–4 days
- **Compute**: 1 GPU sufficient
- **Run cost**:
  - ~100–300 generations total is enough for a first valid study
  - 3 seeds per condition minimum
- **Extra memory/compute overhead**: negligible beyond baseline diffusion inference

---

## Hypothesis 2: Region-aware similarity gating outperforms global gating for ambiguous compositions

### Concrete, testable claim
A **region-aware adaptive MRSA** that applies similarity-conditioned scaling **per spatial token/region** will outperform a global similarity gate on multi-concept prompts where concepts compete locally, reducing attribute leakage without weakening non-conflicting regions.

### Methodology
Build a lightweight regional controller:

1. Start from the global similarity-aware MRSA in Hypothesis 1.
2. Add a **region confidence term** using existing attention/mask information:
   - use FreeCustom weighted masks, cross-attention maps, or simple text-token-to-region maps
   - for each spatial region/token, estimate which concept is currently dominant
3. Apply adaptive scaling only where there is likely conflict:
   - high concept similarity + low region confidence + overlapping ownership
   - then sharpen/suppress competing reference attention locally
4. Leave non-conflicting regions unchanged.

Suggested formula:
- gate(reference r, token i) = f(global similarity, local ownership confidence, mask overlap)

Baselines:
- vanilla MRSA
- global similarity gate only
- region-aware similarity gate

Evaluation:
- focus on prompts with likely local competition:
  - same-category identities
  - concepts with shared colors/textures
  - overlapping foreground/background cues
- metrics:
  - region-wise concept alignment
  - boundary contamination / attribute leakage across masks
  - per-concept fidelity
  - prompt consistency

### Why this is achievable with limited compute
- Still training-free.
- Reuses attention maps/masks already produced during inference.
- No need for new segmentation training; simple heuristics are enough for a first pass.
- Only moderate implementation complexity beyond Hypothesis 1.

### Rationale based on proven techniques
- The synthesis strongly suggests **similarity alone is too coarse**.
- Concept Conductor, FreeCustom, and MC² all indicate that **where** a concept is injected matters as much as **how much**.
- Region-aware routing is a practical bridge between static MRSA and more complicated multi-trajectory isolation methods.
- This is also safer than fully temporal methods, which prior studies did not validate due to pipeline issues.

### Measurable prediction
Compared with global gating:
- stronger gains on cases with spatial competition
- lower boundary leakage and fewer swapped attributes
- equal or better total prompt alignment

A practical target:
- 5%+ reduction in region-level contamination proxy over global gating
- no more than 1–2% loss in whole-image prompt alignment

### Failure condition
Hypothesis fails if:
- local gating gives no benefit over global gating, or
- noisy region confidence causes instability and degrades concept fidelity, or
- improvements only appear on aggregate metrics but not on mechanism-aligned local leakage metrics.

Also fails if the implementation cannot demonstrate actual condition differences in saved outputs/logged gates.

### Resource requirements estimate
- **Engineering**: 4–7 days
- **Compute**: 1 GPU sufficient
- **Run cost**:
  - similar generation budget to Hypothesis 1, but with extra logging/storage for attention maps
  - ~150–400 generations total for a clean ablation
- **Storage**: moderate increase if attention maps are saved

---

## Hypothesis 3: Multi-view similarity estimates are more reliable than text-only or image-only gating

### Concrete, testable claim
Using a **multi-view similarity score**—combining text similarity, reference-image similarity, and attention-map agreement—will produce more effective adaptive MRSA gating than using any single similarity source alone.

### Methodology
Compare several similarity estimators while keeping the same adaptive MRSA policy:

1. **Text-only similarity**
   - cosine between text embeddings of concept names/prompts

2. **Image-only similarity**
   - cosine between CLIP image embeddings of references

3. **Attention-derived similarity**
   - similarity in early attention ownership patterns or overlap statistics

4. **Multi-view similarity**
   - weighted sum of the above, with fixed hand-tuned weights
   - no training required; tune on a very small validation split

Then:
- plug each similarity score into the same MRSA gating rule
- evaluate on concept sets with known failure modes:
  - semantically related but visually distinct
  - visually similar but textually distinct identities
  - references with distracting shared backgrounds

### Why this is achievable with limited compute
- No model training.
- Most cost is just computing embeddings and logging attention maps.
- The comparison is modular: same generation pipeline, only swap similarity source.
- Feasible on a small benchmark because the main variable is the similarity signal, not a large architecture change.

### Rationale based on proven techniques
- The synthesis explicitly flags **feature-level grounding** as a major gap.
- FreeGraftor and FreeEdit suggest that richer feature matching improves reference-guided control.
- Text-only similarity misses identity differences; image-only similarity can overreact to shared context/background.
- Multi-view aggregation is a practical, established way to stabilize noisy control signals.

### Measurable prediction
Compared with text-only and image-only gating:
- multi-view similarity should improve robustness across different failure modes
- lower variance across prompt/reference types
- fewer obvious misgates caused by shared background or lexical similarity

A practical target:
- best or tied-best leakage metric on at least 2 of 3 similarity strata
- lower variance across test subsets than single-view methods
- improved correlation between predicted conflict score and observed leakage rate

### Failure condition
Hypothesis fails if:
- multi-view does not outperform at least one single-view baseline on interference-sensitive cases, or
- added signals are too noisy and reduce reliability, or
- attention-derived similarity contributes no measurable value beyond text+image.

### Resource requirements estimate
- **Engineering**: 3–5 days
- **Compute**: 1 GPU + CPU for embedding preprocessing
- **Run cost**:
  - ~150–300 generations total for clean comparison
  - 3 seeds per condition minimum
- **Overhead**: low to moderate, mostly from feature extraction and diagnostics

---

# Recommended order of execution

Given the prior failed temporal/conflict studies, the most reliable path is:

1. **Hypothesis 1 first**
   - simplest
   - clearest baseline extension
   - lowest risk

2. **Hypothesis 3 second**
   - helps validate whether the similarity signal itself is good enough

3. **Hypothesis 2 third**
   - more likely to yield stronger gains, but depends on having trustworthy routing and diagnostics

---

# Minimal experimental safeguards based on prior failures

These should be mandatory before claiming results:

- run **at least 3 seeds per condition**
- disable or audit caching
- save output hashes per condition/seed
- log actual gate values applied during inference
- assert that ablation flags change forward-pass behavior
- report per-seed metrics, not only aggregates
- include at least one mechanism metric:
  - gate activation stats
  - region conflict counts
  - reference-attention share by concept

---

# Best two hypotheses if you want the strongest practical shortlist

If you only want 2 to pursue immediately, I recommend:

1. **Global pairwise similarity-aware MRSA gating**
2. **Region-aware similarity gating**

They are the most directly aligned with the topic, easiest to test without training, and grounded in the strongest literature signals.

If you want, I can next convert these into a **table**, a **1-page experiment plan**, or a **full ablation matrix**.