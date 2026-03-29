# P4-S4: Composite Loss Function (`losses.py`)

**Status:** COMPLETE | **Date:** 2026-03-28 | **Tests:** 29/29 PASS

---

## What Was Built

Multi-task composite loss for CFA-paired training with 4 components.

| File | Purpose |
|------|---------|
| `training/scripts/model/losses.py` | StreamGuardLoss, CWE_LABEL_MAP, LABEL_CWE_MAP |
| `tests/test_p4s4_losses.py` | 29 tests (8 mandatory + 21 additional) |

## Loss Formula

```
L_total = lambda_ce  * L_CE
        + lambda_cfa * L_CFA
        + 0.2        * L_CWE
        + lambda_sev * L_severity
```

| Component | Implementation | Default Weight |
|-----------|---------------|----------------|
| L_CE | CrossEntropyLoss on (B,2) logits vs binary labels | lambda_ce = 1.0 |
| L_CFA | relu(cosine_sim + margin).mean() on (orig, CFA) embedding pairs | lambda_cfa = 0.5, margin = 0.5 |
| L_CWE | CrossEntropyLoss(label_smoothing=0.1) on (B,12) CWE logits | fixed 0.2 |
| L_severity | HuberLoss(delta=1.0) on severity predictions vs CVSS proxy | lambda_sev = 0.1 |

## L_CFA Contrastive Loss (R-02 Critical)

```
CORRECT: relu(cosine_sim + 0.5)
  sim = +0.8 → relu(1.3) = 1.3    ← HIGH penalty (not separated)
  sim = -0.3 → relu(0.2) = 0.2    ← small penalty (almost there)
  sim = -0.8 → relu(-0.3) = 0.0   ← no penalty (correctly separated)

WRONG:   relu(cosine_sim - 0.5)    ← DO NOT USE
  sim = +0.8 → relu(0.3) = 0.3    ← weak penalty (nearly useless)
  sim = -0.3 → relu(-0.8) = 0.0   ← NO penalty (should fire!)
```

The wrong sign makes L_CFA near-useless — it only fires when embeddings are extremely similar (sim > 0.5), which rarely happens early in training. The model converges on L_CE alone and the CFA research contribution disappears.

**Verified by test:** `TestLCFASign::test_cfa_sign_correct` checks that sim=0.8, margin=0.5 produces L_CFA=1.3 (not 0.3).

## Diagnostic: frac_separated

Every forward pass with CFA pairs reports `frac_separated`: the fraction of pairs where `cosine_sim < -margin`. This tracks training progress:
- Epoch 1: ~0.0 (pairs start random, not yet separated)
- Convergence: ~0.8-1.0 (most pairs in opposite hemispheres)

## CWE Label Map (12 Target CWEs)

```python
CWE_LABEL_MAP = {
    "CWE-89":  0,   "CWE-78":  1,   "CWE-79":  2,   "CWE-119": 3,
    "CWE-120": 4,   "CWE-121": 5,   "CWE-122": 6,   "CWE-125": 7,
    "CWE-134": 8,   "CWE-190": 9,   "CWE-416": 10,  "CWE-476": 11,
}
```

`LABEL_CWE_MAP` is the inverse (for eval.py per-CWE F1 reporting).

## Masking Behavior

| Input | Condition | Effect |
|-------|-----------|--------|
| cwe_labels | -1 or >= 12 | Excluded from L_CWE (valid mask) |
| severity_labels | -1 | Excluded from L_severity (R-23) |
| outputs_cfa | None | L_CFA skipped entirely |
| labels | None | L_CE skipped |

All components are optional except that at least one must be active for a non-zero total.

## Unequal Batch Size Handling

When orig and CFA batches have different sizes (e.g., batch has mix of pairs and singletons), `min_len = min(emb_v.size(0), emb_vp.size(0))` truncates both to the smaller size. No crash, no padding artifacts.

## Risk Mitigations

| Risk | Status | Evidence |
|------|--------|----------|
| R-02 (wrong L_CFA sign) | MITIGATED | Correct formula relu(sim + margin) verified in 3 dedicated tests; wrong formula would give 0.3 not 1.3 |
| R-23 (severity on -1 labels) | MITIGATED | `valid_mask = sev_dev >= 0` filter; 3 tests verify skip/partial/full |

## Test Summary

```
tests/test_p4s4_losses.py — 29/29 PASS

TestLCEDecreases:          1 test  (L_CE decreases over 10 SGD steps)
TestLCFASign:              2 tests (CRITICAL: relu(0.8+0.5)=1.3, wrong formula=0.3)
TestLCFASeparated:         1 test  (L_CFA=0.0 when cosine_sim=-1.0)
TestLCFANotSeparated:      1 test  (L_CFA=1.5 when cosine_sim=+1.0)
TestFracSeparated:         2 tests (0.0 when close, 1.0 when opposite)
TestSeveritySkip:          3 tests (all -1 excluded, partial mask, all valid)
TestUnequalBatchSizes:     2 tests (orig>cfa, cfa>orig)
TestGradientFlow:          2 tests (grad flows all 4 components, all keys present)
TestCWEMaps:               4 tests (12 entries, inverse map, 0-11 range, known CWEs)
TestCWELabelSmoothing:     2 tests (0.1 default, 0.0 override)
TestCWEMask:               2 tests (all -1 excluded, partial valid)
TestLambdaWeights:         1 test  (2x lambda_ce → 2x total)
TestMinimalInputs:         2 tests (no labels → 0.0, only labels → L_CE only)
TestDefaults:              4 tests (lambdas, margin, label_smoothing, huber_delta)
```

## Combined Test Results (P4-S1 through S4)

```
tests/test_p4s1_model.py      — 37/37 PASS (0 regressions)
tests/test_p4s2_callee.py     — 25/25 PASS (0 regressions)
tests/test_p4s3_dataloader.py — 21/21 PASS (0 regressions)
tests/test_p4s4_losses.py     — 29/29 PASS
Total: 112/112 PASS
```
