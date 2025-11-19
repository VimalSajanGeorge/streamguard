# StreamGuard Production Training Plan
**Version:** 1.7 Phase 1 → Production
**Created:** 2025-11-07
**Total Story Points:** 61

---

## Executive Summary

This plan transitions StreamGuard from validated v1.7 safety suite to full production-grade training pipeline for:
- **Transformer** (Enhanced SQL Intent Detection)
- **GNN** (Taint-Flow Graph Analysis)
- **Fusion** (Combined Predictions)

All models will be trained on NVIDIA A100 with mixed precision, adaptive configuration, collapse detection, and comprehensive reproducibility tracking.

---

## Story Point Breakdown

### Phase 1: Safety Infrastructure (Story Points: 18)

#### 1.1 JSON Serialization Safety (SP: 3)
**Task:** Implement `safe_jsonify()` helper
**Acceptance Criteria:**
- ✅ Converts PyTorch tensors to Python types (.detach().cpu().numpy().tolist())
- ✅ Handles NumPy arrays, pathlib.Path, device objects
- ✅ Recursively processes nested dicts and lists
- ✅ Raises clear errors for unserializable types

**Files Affected:**
- `training/utils/json_safety.py` (new)
- All training scripts (import and use)

---

#### 1.2 Atomic JSON Writes (SP: 2)
**Task:** Implement atomic write-replace pattern
**Acceptance Criteria:**
- ✅ Writes to temporary file first (`{path}.tmp`)
- ✅ Uses `os.replace()` for atomic move
- ✅ Preserves original file on write failure
- ✅ Works across all platforms (Windows/Linux)

**Files Affected:**
- `training/utils/json_safety.py` (extend)

---

#### 1.3 AMP-Safe Gradient Clipping (SP: 3)
**Task:** Fix gradient clipping under mixed precision
**Acceptance Criteria:**
- ✅ Calls `scaler.unscale_(optimizer)` BEFORE gradient clipping
- ✅ Clips gradients with `torch.nn.utils.clip_grad_norm_()`
- ✅ Properly scales gradients after clipping
- ✅ Logs gradient norms before/after clipping

**Files Affected:**
- `training/train_transformer.py:training_loop`
- `training/train_gnn.py:training_loop`
- `training/train_fusion.py:training_loop`

---

#### 1.4 Tensor .numpy() Audit (SP: 2)
**Task:** Fix all unsafe tensor-to-numpy conversions
**Acceptance Criteria:**
- ✅ All `.numpy()` calls use `.detach().cpu().numpy()`
- ✅ No direct tensor → numpy without gradient detach
- ✅ Logging functions use safe conversions
- ✅ Metric computations are gradient-safe

**Command:**
```bash
grep -rn "\.numpy()" training/ --include="*.py" | grep -v "detach"
```

---

#### 1.5 Adaptive GPU Config Loader (SP: 2)
**Task:** Load GPU-specific training configs with fallback
**Acceptance Criteria:**
- ✅ Reads `training_config.json` if exists
- ✅ Provides safe defaults for A100, V100, T4
- ✅ Auto-detects GPU memory and adjusts batch size
- ✅ Logs config source (file vs. auto-detected)

**Files Affected:**
- `training/utils/adaptive_config.py` (new)

---

#### 1.6 LR Finder Cache System (SP: 3)
**Task:** Implement 168-hour LR cache with quick-mode finder
**Acceptance Criteria:**
- ✅ Cache key = hash(model_config, dataset_hash, seed)
- ✅ Cache stored in `training/cache/lr_finder/{model_type}/`
- ✅ Cache TTL = 168 hours (7 days)
- ✅ Quick-mode LR Finder: 100 steps, 1e-7 → 1 range
- ✅ Saves LR curve plot and recommended LR

**Files Affected:**
- `training/utils/lr_cache.py` (new)
- `training/utils/lr_finder.py` (enhance)

---

#### 1.7 Collapse Detection System (SP: 3)
**Task:** Auto-stop training on model collapse
**Acceptance Criteria:**
- ✅ Detects: zero gradients, constant predictions, NaN loss
- ✅ Threshold: 3 consecutive collapse events
- ✅ Saves `collapse_report.json` with diagnostics
- ✅ Auto-stops training and logs warning

**Files Affected:**
- `training/utils/collapse_detector.py` (new)

---

### Phase 2: Production Training Cells (Story Points: 29)

#### 2.1 Cell 51: Transformer Production Training (SP: 8)
**Task:** Full production training for Transformer v1.7
**Features:**
- 🔹 Adaptive GPU config loading
- 🔹 LR Finder with caching (168h TTL)
- 🔹 Mixed precision training (AMP)
- 🔹 AMP-safe gradient clipping (max_grad_norm=1.0)
- 🔹 Collapse detection with auto-stop
- 🔹 Safe JSON metadata writes
- 🔹 3-seed reproducibility loop (42, 2025, 7)
- 🔹 Triple weighting for class imbalance
- 🔹 Checkpoint saving with enhanced metadata

**Hyperparameters:**
- LR: 1e-5 → 5e-5 (from LR Finder)
- Batch Size: 64 (adaptive to GPU)
- Sequence Length: 512
- Epochs: 10
- Warmup: 10% of steps
- Weight Decay: 0.01

**Outputs:**
```
training/outputs/transformer_v17/
├── seed_42/
│   ├── model_checkpoint.pt
│   ├── training_metadata.json
│   ├── metrics_history.csv
│   ├── lr_curve.png
│   └── collapse_report.json (if triggered)
├── seed_2025/
└── seed_7/
```

---

#### 2.2 Cell 52: GNN Production Training (SP: 8)
**Task:** Full production training for GNN v1.7 Phase-1
**Features:**
- 🔹 PyTorch Geometric LR Finder (graph-aware)
- 🔹 WeightedRandomSampler for graph class imbalance
- 🔹 Focal Loss with γ=1.5
- 🔹 Mixed precision training (AMP)
- 🔹 AMP-safe gradient clipping
- 🔹 Collapse detection (gradient + prediction variance)
- 🔹 Safe JSON metadata
- 🔹 3-seed loop
- 🔹 Graph statistics logging (nodes/edges/depth)

**Hyperparameters:**
- LR: 1e-4 → 1e-3 (from PyG LR Finder)
- Batch Size: 64 graphs
- Graph Size: 10-50 nodes (adaptive)
- Epochs: 15
- Weight Decay: 1e-4
- Focal Loss γ: 1.5

**Outputs:**
```
training/outputs/gnn_v17/
├── seed_42/
│   ├── model_checkpoint.pt
│   ├── training_metadata.json
│   ├── metrics_history.csv
│   ├── graph_statistics.json
│   └── lr_curve.png
├── seed_2025/
└── seed_7/
```

---

#### 2.3 Cell 53: Fusion Production Training (SP: 8)
**Task:** Full production training for Fusion model
**Features:**
- 🔹 Discriminative Learning Rates:
  - Transformer backbone: LR × 0.1
  - GNN backbone: LR × 0.5
  - Fusion layer: LR × 1.0
- 🔹 Gradient monitoring per component
- 🔹 Mixed precision with gradient accumulation
- 🔹 Collapse detection for fusion layer
- 🔹 Safe JSON metadata
- 🔹 3-seed loop
- 🔹 OOF (Out-of-Fold) prediction validation

**Hyperparameters:**
- Base LR: 1e-5 (Transformer), 5e-5 (GNN), 1e-4 (Fusion)
- Batch Size: 32 (memory-constrained)
- Epochs: 12
- Weight Decay: 0.01
- Gradient Accumulation: 2 steps

**Outputs:**
```
training/outputs/fusion_v17/
├── seed_42/
│   ├── model_checkpoint.pt
│   ├── training_metadata.json
│   ├── metrics_history.csv
│   ├── gradient_stats.json
│   └── oof_predictions.npy
├── seed_2025/
└── seed_7/
```

---

#### 2.4 Cell 54: Unified Metrics & Export (SP: 5)
**Task:** Aggregate metrics across all models and seeds
**Features:**
- 📊 Loads metrics from all 3 models × 3 seeds = 9 runs
- 📊 Computes mean/std for F1, Recall, Precision, Accuracy
- 📊 Generates comparison plots (box plots, line charts)
- 📊 Exports `production_summary.json` with all stats
- 📊 Generates `production_report.md` with tables

**Outputs:**
```
training/outputs/production_summary/
├── production_summary.json
├── production_report.md
├── metrics_comparison.png
├── f1_by_seed.png
└── training_stability.csv
```

**JSON Schema:**
```json
{
  "timestamp": "2025-11-07T12:00:00",
  "models": {
    "transformer_v17": {
      "mean_f1": 0.92,
      "std_f1": 0.01,
      "seeds": [42, 2025, 7],
      "best_seed": 2025,
      "best_f1": 0.93
    },
    "gnn_v17": { ... },
    "fusion_v17": { ... }
  },
  "best_overall_model": "fusion_v17",
  "best_overall_f1": 0.95
}
```

---

### Phase 3: Testing & Validation (Story Points: 10)

#### 3.1 Transformer Production Testing (SP: 3)
- ✅ Run 3-seed training (42, 2025, 7)
- ✅ Verify no tensor serialization errors
- ✅ Confirm collapse detection triggers correctly (mock test)
- ✅ Validate LR cache hit/miss behavior
- ✅ Check AMP stability (no NaNs)

---

#### 3.2 GNN Production Testing (SP: 3)
- ✅ Run 3-seed training with WeightedSampler
- ✅ Verify Focal Loss γ=1.5 converges
- ✅ Test PyG LR Finder on small graph dataset
- ✅ Validate graph statistics logging
- ✅ Check gradient clipping under AMP

---

#### 3.3 Fusion Production Testing (SP: 3)
- ✅ Run 3-seed training with discriminative LR
- ✅ Verify gradient flow to all components
- ✅ Test OOF prediction generation
- ✅ Validate fusion layer convergence
- ✅ Check gradient accumulation correctness

---

#### 3.4 Metadata Validation (SP: 2)
**Command:**
```python
import json
for f in Path("training/outputs").rglob("*.json"):
    try:
        with open(f) as fp:
            json.load(fp)  # Must not fail
    except TypeError as e:
        print(f"❌ {f}: {e}")
```

- ✅ All JSONs must be primitive-only (no tensors/numpy/devices)
- ✅ No `Object of type Tensor is not JSON serializable` errors

---

#### 3.5 AMP Stability Validation (SP: 2)
**Command:**
```python
# Check for NaN gradients in logs
grep -r "NaN" training/outputs/*/metrics_history.csv
```

- ✅ No NaN losses in any epoch
- ✅ GradScaler state saved correctly
- ✅ Gradient norms logged and bounded

---

### Phase 4: Documentation & Integration (Story Points: 4)

#### 4.1 Production Summary Aggregator (SP: 3)
**Task:** Implement `aggregate_production_metrics.py`
**Features:**
- Reads all `training_metadata.json` files
- Computes aggregate statistics
- Generates comparison plots
- Exports `production_summary.json`

---

#### 4.2 Notebook Documentation (SP: 2)
**Task:** Add markdown cells explaining production workflow
**Sections:**
1. Production Training Overview
2. Safety Features Enabled
3. Hyperparameter Baselines
4. Expected Outputs
5. Troubleshooting Guide

---

## Timeline Estimate

| Phase | Story Points | Estimated Hours | Days (8h/day) |
|-------|--------------|-----------------|---------------|
| Phase 1: Safety Infrastructure | 18 | 18-27 | 2-3 |
| Phase 2: Production Cells | 29 | 29-44 | 4-6 |
| Phase 3: Testing & Validation | 10 | 10-15 | 1-2 |
| Phase 4: Documentation | 4 | 4-6 | 0.5-1 |
| **TOTAL** | **61** | **61-92** | **8-12 days** |

---

## Success Criteria

### ✅ Must Have
1. All 3 models train successfully with 3 seeds each (9 total runs)
2. Zero tensor serialization errors in JSON metadata
3. AMP training stable (no NaNs)
4. LR Finder cache working (hit rate > 80% on re-runs)
5. Collapse detection triggers on mock collapse test
6. `production_summary.json` generated with valid schema

### 🎯 Nice to Have
1. SageMaker integration scripts
2. AWS Spot Instance support
3. CloudWatch metrics export
4. Distributed training (multi-GPU)

---

## Risk Mitigation

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| OOM errors on A100 | High | Medium | Adaptive batch sizing, gradient accumulation |
| LR Finder instability | Medium | Low | Conservative LR caps, quick-mode fallback |
| Model collapse | High | Low | Auto-stop, checkpoint rollback, LR reduction |
| JSON serialization bugs | Medium | Medium | Comprehensive safe_jsonify(), unit tests |
| Slow training on large graphs | Medium | Medium | Graph size capping, batch size tuning |

---

## Next Steps

1. ✅ Implement safety infrastructure (Phase 1)
2. ✅ Create production training cells (Phase 2)
3. ✅ Run validation tests (Phase 3)
4. ✅ Generate production summary (Phase 4)
5. 🚀 Deploy to SageMaker/AWS Batch

---

**Plan Status:** 🟢 Ready for Execution
**Last Updated:** 2025-11-07
