## Decision
REFINE

## Justification
REFINE is mandatory here because the minimum criteria for PROCEED are not met. The analysis explicitly reports a **failed run**, **effective n=1 per cell**, a **missing key baseline**, and an **analysis quality rating of 3/10**. There is also a serious **metric interpretation inconsistency**: under the stated primary metric direction (“lower is better”), the proposed method appears worse than vanilla on the overall metric and the key hard slice.

This does **not** warrant PIVOT because the underlying hypothesis and benchmark framing still appear sensible. The matched/shared-context slice is a reasonable stress test, and the ablation design is directionally useful. The problem is execution and evidentiary quality, not a clearly falsified research premise.

## Evidence
- **Primary metric defined:** `region_level_contamination_error`, **lower is better**
- **Run status:** **failed**
- **Seed reporting:** 5 seeds noted, but **effectively n=1 per reported cell**
- **Quality rating:** **3/10** (< required 4/10 for PROCEED)
- **Missing baseline:** `GlobalSimilarityWeightedMRSA` not clearly reported quantitatively
- **Proposed method underperforms vanilla on reported numbers:**
  - Overall: **0.066970 vs 0.063727** → proposed is worse
  - Matched/shared-context: **0.110118 vs 0.103694** → proposed is worse
  - Clean subset: **0.025301 vs 0.023800** → proposed is worse
- **Interpretation integrity issue:** narrative claims improvement, but reported values contradict that if lower is better
- **No valid uncertainty estimates:** no SD/CI/significance possible from current reporting

## Next Actions
1. **Rerun all registered conditions successfully**
   - Include at minimum: vanilla, at least two baselines, and the proposed method
   - Ensure `GlobalSimilarityWeightedMRSA` is present

2. **Fix reporting integrity**
   - Audit metric formula, sign, normalization, and table generation
   - Add sanity-check examples showing the metric behaves as intended

3. **Repair statistical design**
   - Report true aggregation across seeds and prompts
   - Use **≥3 seeds per condition** in final tables
   - Provide mean, SD, bootstrap CI, and paired deltas vs vanilla

4. **Check ablation integrity**
   - Verify per-seed outputs are not duplicated across conditions
   - Preserve seed-level logs and condition IDs

5. **Freeze a confirmatory evaluation plan**
   - Predefine the primary endpoint as the matched/shared-context slice
   - State comparison set, success criteria, and analysis plan before rerunning

6. **Add metric triangulation**
   - Include complementary automatic metrics and a small human eval
   - Report compute/runtime overhead and tuning parity across methods

7. **Only reconsider PROCEED after all minimum criteria are satisfied**
   - 2+ baselines plus proposed method
   - defined primary metric
   - ≥3 seeds per condition
   - no duplicated per-seed values across conditions
   - analysis quality ≥4/10