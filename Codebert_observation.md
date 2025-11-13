# CodeBERT v1.7 Production Training Observations

## Current Configuration
- **Base model:** `microsoft/codebert-base`
- **Training script:** `training/train_transformer.py`
- **Key hyperparameters:**
  - Seeds: 42, 2025, 7
  - Epochs: 10 (production), 8 (balanced preset)
  - Batch size: 32 (prod), 24 (preset)
  - Gradient accumulation: 2
  - Learning rate: `2e-5` with sqrt batch scaling (≈2.45e-5 preset, ≈2.83e-5 prod)
  - Weight multiplier: 1.5 (preset), 1.6 (prod)
  - Loss: CrossEntropy + label smoothing 0.05
  - WeightedRandomSampler enabled, no code features yet

## Accuracy Snapshot (latest run)
| Seed | Run Type | Best F1 (vuln) | Accuracy @ 0.5 | Recommended Threshold | Confusion Matrix (TN/FP/FN/TP) |
|------|----------|----------------|----------------|-----------------------|-------------------------------|
| 42   | Production | 0.6323 | 0.5714 | 0.40 (F1≈0.637) | 554 / 991 / 180 / 1007 |
| 2025 | Production | 0.6332 | 0.5550 | 0.50 | 474 / 1071 / 141 / 1046 |
| 7    | Production | 0.6379 | 0.5667 | 0.50 | 505 / 1040 / 144 / 1043 |
| 42   | Balanced Preset | 0.6275 | 0.5531 | 0.50 | 490 / 1055 / 162 / 1025 |

**Mean metrics (production seeds):**
- F1 ≈ 0.634 ± 0.002
- Accuracy ≈ 0.564 ± 0.007 (at threshold 0.5)
- Precision ranges 0.49–0.51; recall stays ≥0.82, confirming a recall-heavy regime.

## Key Observations
1. **Recall dominates accuracy.** With both WeightedRandomSampler and weight multipliers ≥1.5, the model predicts “vulnerable” on ~65–70% of validation samples. Accuracy (~0.56) lags because FP ≫ TN (e.g., 991 vs 554 for seed 42).
2. **Threshold tuning helps immediately.** Post-training sweeps recommend thresholds between 0.40 and 0.50; raising the decision threshold to 0.55–0.60 should roughly halve FP while keeping F1 ≥0.60.
3. **Balanced preset mirrors prod behaviour.** The 24×8 preset stabilizes faster but lands in the same F1 band (≈0.63) with slightly lower accuracy, so it remains a good “smoke test” baseline.
4. **Opportunities for higher precision:**
   - Reduce `weight_multiplier` in production runs (e.g., 1.3 or 1.0) to back off the vulnerability bias.
   - Enable `--use-code-features` to incorporate static metrics; past experiments show +2–3 precision points.
   - Consider `--focal-loss --focal-gamma=1.5` plus dropout 0.3 if FP remain high even after thresholding.

## Recommended Next Steps
1. Rerun Cell 15 with `weight_multiplier=1.3` and `--use-code-features` to measure the precision gain at threshold 0.5.
2. Use Cell 15.5 after each production run to log the updated accuracy/precision/recall and pick an operating threshold before deployment.
3. If accuracy is still unacceptable (>0.60 target unmet), add a `threshold_override` inference step (0.55–0.60) or train a focal-loss variant for comparison.
