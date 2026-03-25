# Experiment Specification

## Topic
[World Model] Adaptive multi-reference self-attention (MRSA) for FreeCustom: dynamically scaling attention weights based on concept similarity to improve multi-concept composition quality

## Project Structure
Multi-file experiment project with 2 file(s): `main.py`, `models.py`

## Entry Point
`main.py` — executed directly via sandbox

## Outputs
- `main.py` emits metric lines in `name: value` format
- Primary metric key: `primary_metric`

## Topic-Experiment Alignment
MISALIGNED: The code appears only partially related to the topic. It imports MRSA components (`MultiReferenceSelfAttention`, `hack_self_attention_to_mrsa`) and uses CLIP features, which is superficially consistent with a multi-reference attention idea. However, the shown experiment code mainly builds dataset cases and computes CLIP-based image/text similarities; it does not show any implementation of the key claimed mechanism: adaptive scaling of attention weights based on concept similarity to improve multi-concept composition. There is no visible logic that uses similarity scores to dynamically modulate self-attention weights during generation, no comparison between adaptive-MRSA and fixed/non-adaptive MRSA, and no clear evaluation of composition quality tied to the adaptive mechanism. The regimes in the dataset indexer (`matched_background_or_shared_context` vs `mismatched_background_or_clean_context`, semantic overlap heuristics) are heuristic labels, but the displayed code does not show that they drive meaningfully different attention behaviors or controlled experimental conditions. The presence of CLIP and transformer-based modules satisfies the transformer-related requirement only at a tooling level, not as evidence that the specific adaptive MRSA method is actually tested.

## Constraints
- Time budget per run: 1800s
- Max iterations: 3
- Uses local data/codebases from project config
- Validated: Code validation: 3 warning(s)

## Generated
2026-03-23T18:49:17+00:00
