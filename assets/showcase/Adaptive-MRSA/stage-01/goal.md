# SMART Research Goal

- **Topic**: World Model — adaptive multi-reference self-attention (MRSA) for FreeCustom to improve multi-concept composition quality in image generation/editing

- **Novel Angle**:  
  Most personalization and subject-driven generation methods already support **single-concept identity preservation** or **multi-reference conditioning**, but they typically use either:
  1. **fixed attention fusion** across references,  
  2. **uniform reference weighting**, or  
  3. **offline learned routing/gating** that does not adapt token-by-token to **inter-concept similarity/conflict** during denoising.

  The underexplored gap is: **when multiple personalized concepts are composed together, semantically or visually similar concepts often over-dominate attention while dissimilar concepts are underused, causing concept leakage, identity blending, or attribute collapse**. Existing methods generally treat all references as equally informative or rely on static architectural heuristics. What has *not* been well studied is **dynamic scaling of self-attention weights based on concept similarity estimated at inference/denoising time**, especially in **FreeCustom-style training-light or plug-in personalization pipelines**.

  This is timely **now (2024–2026)** because recent work in diffusion transformers / stronger attention-based generators has made **attention manipulation a primary control interface**, and personalization is shifting from expensive finetuning toward **modular, reference-driven, low-cost customization**. The opportunity comes from:
  - stronger backbone world/image generation models with more interpretable attention blocks,
  - rising interest in **multi-concept personalization/composition** rather than single-subject generation,
  - practical demand for **compute-light adapters** rather than full retraining.

  **Why this is likely not already covered**:  
  Recent personalization papers focus on identity retention, reference injection, adapter design, or training-free guidance, but not specifically on **adaptive multi-reference self-attention reweighting as a function of concept similarity/conflict** for **multi-concept composition**. Standard approaches ask “how do we inject multiple references?”; this project asks the sharper question: **“how should each reference attend at each layer/token/timestep when concepts are partially redundant, conflicting, or complementary?”** That is a narrower and more novel mechanism-level contribution.

  **Trend validation (2024–2026):**
  1. **DreamMatcher (2024)** — highlights the growing focus on more faithful subject-driven generation and reference-to-generation alignment. This supports the relevance of improving reference usage rather than just adding more references.  
  2. **AnyDoor / related subject-driven controllable generation follow-ups (2024-era adoption and extensions)** — show the importance of flexible reference-conditioned generation/editing, but typically do not explicitly solve adaptive inter-reference competition in multi-concept setups.  
  3. **Diffusion Transformer / attention-centric generative model trends (2024–2025)** — the shift toward transformerized diffusion/world models makes attention-level intervention much more natural and impactful than in earlier U-Net-only pipelines.

  These trends create a clear opening for a method that is:
  - **plug-in**,  
  - **attention-native**,  
  - **multi-concept aware**, and  
  - **feasible without large-scale retraining**.

  ### Benchmark
  - **Name**: Subject-driven multi-concept personalization benchmark built from **DreamBooth-style subject sets** + compositional prompts; optionally **CustomConcept101**-style personalized concept collections if available in the selected codebase
  - **Source**: Public personalized generation datasets/reference sets used in DreamBooth, subject-driven generation, and multi-concept composition follow-up evaluations
  - **Metrics**:
    - CLIP-T / text alignment
    - CLIP-I / image-reference similarity
    - DINO / identity similarity
    - Multi-concept composition accuracy via VLM judge or concept classifier
    - Human preference study (small-scale)
  - **Current SOTA (if known)**:
    - No universally accepted single SOTA leaderboard exists for **multi-concept personalized composition** across all datasets.
    - Strong recent baselines typically include **DreamBooth-based composition**, **Custom Diffusion / Mix-of-show style methods**, and **training-light reference-based methods** depending on benchmark protocol.
    - Because the benchmark landscape is fragmented, publishability will rely on **clear protocol definition + strong ablations + consistent gains over modern baselines**.

  **Primary evaluation benchmark for this paper**:
  - **Dataset/Benchmark**: A curated **multi-concept composition split** built from DreamBooth subjects (2-concept and 3-concept prompts), with held-out pairings and prompts.
  - **Why this benchmark**: It is reproducible, widely recognized, and directly stresses concept interference, which is the central failure mode targeted by adaptive MRSA.
  - **How results will be measured**:
    - per-concept identity preservation,
    - prompt faithfulness,
    - composition success rate,
    - attention conflict diagnostics.

- **Scope**:  
  A single-paper scope is to design and test **one lightweight MRSA module** for FreeCustom-like multi-reference generation, with:
  - one similarity estimator (e.g., CLIP/DINO embedding similarity),
  - one adaptive attention scaling rule,
  - integration into 1–2 attention layers or blocks,
  - evaluation on **2-concept and 3-concept composition** only,
  - comparison against 3–4 strong baselines and ablations.

  The project should **not** attempt:
  - full foundation model pretraining,
  - video world modeling,
  - large-scale human evaluation,
  - broad architecture search.

- **SMART Goal**:  
  **Specific**: Develop an **Adaptive MRSA** module for FreeCustom that dynamically rescales attention contributions from multiple reference concepts based on pairwise concept similarity/conflict, and integrate it into a diffusion/world-model attention pipeline for multi-concept image generation.  
  **Measurable**: On a curated DreamBooth multi-concept benchmark, achieve:
  - at least **+3.0 percentage points** improvement in multi-concept composition success rate over a fixed-weight multi-reference baseline,
  - at least **+2.0 points** in identity-preservation similarity (DINO or CLIP-I),
  - with **≤10% inference overhead** and **no full-model retraining**.
  **Achievable**: Implemented as a lightweight plug-in attention reweighting mechanism using frozen pretrained encoders (e.g., CLIP/DINO) and a public FreeCustom-compatible backbone, trainable on a **single GPU** through small adapter tuning or calibration only.  
  **Relevant**: Addresses a current bottleneck in personalized generative world/image models: reliable composition of multiple custom concepts without identity blending.  
  **Time-bound**:  
  - Week 1–2: reproduce FreeCustom baseline and create multi-concept benchmark split  
  - Week 3–4: implement adaptive MRSA and similarity-aware weighting  
  - Week 5: run ablations (similarity metric, layer placement, timestep scheduling)  
  - Week