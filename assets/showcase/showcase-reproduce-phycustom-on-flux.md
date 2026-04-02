# Reproducing PhyCustom on FLUX

**Project:** `reproduce-phycustom-on-flux` · **Track:** Reproduce

---

## 📄 Paper Title

> **Reproducing PhyCustom-Style Physical Property Customization on FLUX**

---

## 💡 Idea

PhyCustom studies whether customized concepts can preserve **physical properties** under interventions such as viewpoint, lighting, context, and carrier-object changes. This project asks a stricter reproduction question: when the idea is ported to **FLUX**, do the claimed gains still hold under **intervention-based evaluation**, or do they disappear once we measure descriptor confusion, prompt-background leakage, and physical-transfer robustness rather than only generic realism? The generated code implements multiple FLUX-native adaptation variants, including attention-only LoRA, output-space prompt-swapped regularization, hidden-state proxy regularization, and hybrid LoRA plus selective block unfreezing.

---

## ⚙️ Pipeline Journey

| | |
| :--- | :--- |
| **Track** | Reproduce — PhyCustom-style regularization on FLUX |
| **Topic** | Reproduce PhyCustom on FLUX and test whether intervention-consistency gains survive the architecture transfer |
| **Anchor** | PhyCustom-style physical-property customization benchmark on the local `PhyDiff` dataset |
| **Stages** | Available artifacts currently cover **S11 → S12**: code generation → sanity check |
| **Data** | Local `PhyDiff` dataset with `objects` and `verbs` concept families, prompt YAMLs, and concept images |
| **Model Anchor** | Local `FLUX.1-dev` checkpoint |
| **Compute Plan** | Single CUDA GPU, bf16 by default, 512 resolution full run, reduced smoke-test schedule |
| **Experimental Grid** | Full plan registers **14 conditions**: 3 baselines + 5 proposed methods + 6 ablations |
| **Artifacts** | `experiment_spec.md`, generated `main.py`, `EXPERIMENT_PLAN.yaml`, sanity outputs, metrics summary JSON |

### Stage Breakdown

| Phase | Stages | Description |
| :--- | :--- | :--- |
| **L2 · Experiment Design** | S9 | Reproduction plan built around H1-H3: PhyCustom regularization advantage, output-space vs hidden-state portability, and adaptation-capacity placement |
| **L3 · Coding** | S11 → S12 | Generated single-file FLUX experiment harness, passed sanity check, emitted output grids and summary JSON under smoke-test settings |

---

## 🧪 Registered Reproduction Agenda

### Main Hypotheses

1. **H1**: PhyCustom-style regularization should improve intervention consistency over plain LoRA and LoRA with prior preservation.
2. **H2**: Prompt/output-space decoupling should be a more portable FLUX-native bridge than hidden-state proxy regularization.
3. **H3**: Adaptation capacity and placement should matter: attention-only, MLP-inclusive, and hybrid unfreezing may separate prompt binding from deeper physical transfer.

### Implemented Method Family

- **Baselines (3)**: few-shot prompt binding, prior-preservation variant, and other lightweight FLUX adaptation references from the registered plan
- **Proposed methods (5)**: attention-only PhyCustom intervention consistency, prompt-swapped output-space decoupling, hidden-state proxy hooks, attention+MLP LoRA, and hybrid LoRA plus selective block unfreezing
- **Ablations (6)**: mechanism-isolation variants registered in `EXPERIMENT_PLAN.yaml`

---

## 🎯 Pilot Smoke-Test Output

The project has not yet produced full reproduction results, but the sanity run already emitted one pilot summary entry for the smoke-test condition:

| Method | Primary Metric (↓) | Descriptor Confusion | Prompt Leakage | Reference Fidelity Error | Physical Transfer Error | FID |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| AttentionOnlyLoRA + PhyCustom Regularization | **0.0591** | 0.0059 | 0.0451 | 0.3543 | 0.0656 | 14.94 |

These numbers should be read only as a **pipeline sanity signal**, not as a scientific conclusion.

---

## 🔑 Key Code Snippets

### FLUX-side LoRA attachment

The reproduction is implemented directly on the FLUX transformer, with method-specific target modules controlling what adaptation capacity is exposed:

```python
def attach_lora(self, pipe: FluxPipeline, method: Dict):
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias='none',
        target_modules=method['target_modules'],
    )
    pipe.transformer = get_peft_model(pipe.transformer, lora_config)
    return pipe
```

### Intervention-aware prompt construction

Instead of evaluating only one prompt per concept, the code builds a small intervention suite that probes viewpoint, lighting, context, carrier-object transfer, and descriptor swap behavior:

```python
def build_intervention_pairs(self, concept: ConceptRecord) -> Dict[str, str]:
    base = concept.base_prompt
    if concept.family == 'verbs':
        mismatched = base.replace('burn', 'melt').replace('melt', 'shatter').replace('shatter', 'expand')
    else:
        mismatched = base + ', made of transparent crystal'
    return {
        'base': base,
        'viewpoint_change': base + ', from a high-angle viewpoint',
        'lighting_change': base + ', with warm sunset lighting',
        'scene_context_change': base + ', inside a kitchen scene',
        'carrier_object_change': base + ', transferred to an unseen carrier-object setting',
        'descriptor_swap': mismatched,
    }
```

### PhyCustom-style regularized training step

The core reproduction path combines the diffusion objective with same-concept pull, different-concept margin, and leakage penalties, with warmup to reduce instability:

```python
if method['name'] == 'AttentionOnlyLoRA_FLUX_WithPhyCustomInterventionConsistencyRegularization':
    images = self.generate_images(pipe, [intervention_pairs['base'], intervention_pairs['viewpoint_change'], intervention_pairs['descriptor_swap']], seed + step)
    embeddings = self.encode_output_embeddings(images)
    sameconcept_pull_loss = self.compute_sameconcept_pull_loss(embeddings)
    diffconcept_margin_loss = self.compute_diffconcept_margin_loss(embeddings, method['margin'])
    leakage_penalty = self.compute_leakage_penalty(embeddings[0:1], intervention_pairs['base'], intervention_pairs['scene_context_change'])
    total_loss = diffusion_loss + warmup * (
        method['lambda_pull'] * sameconcept_pull_loss
        + method['lambda_margin'] * diffconcept_margin_loss
        + method['lambda_leak'] * leakage_penalty
    )
```

### Intervention-based evaluation metrics

The evaluation code measures whether generated outputs preserve the intended concept under controlled perturbations, rather than only scoring generic realism:

```python
intervention_consistency_score = F.cosine_similarity(generated_embeddings[0:1], generated_embeddings[1:4]).mean().item()
descriptor_swap_confusion_error = max(
    0.0,
    F.cosine_similarity(generated_embeddings[0:1], generated_embeddings[4:5]).mean().item() - intervention_consistency_score,
)
prompt_background_leakage_error = max(
    0.0,
    F.cosine_similarity(generated_embeddings[0:1], context_text_embed).mean().item() - text_alignment + 0.05,
)
reference_fidelity_error = 1.0 - torch.mm(generated_embeddings[0:1], reference_embeddings.T).max().item()
discovery_aligned_endpoint_physical_transfer_error = 1.0 - F.cosine_similarity(generated_embeddings[0:1], generated_embeddings[3:4]).mean().item()
```

---

## 💻 Code

👉 Generated [`main.py`](reproduce-phycustom/main.py)

---

**Reference:** Wu et al., *PhyCustom: Towards Realistic Physical Customization in Text-to-Image Generation*, arXiv [2512.02794](https://arxiv.org/abs/2512.02794), 2025.

---

*Generated by Claw AI Lab pipeline · Reproduce · showcase currently reflects code generation and sanity-stage pilot outputs*
