# StreamGuard Notebook Integration Checklist

This checklist guides you through integrating the production training cells into your `StreamGuard_Complete_Training.ipynb` notebook.

---

## Pre-Integration Validation

### Step 1: Run Safety Utilities Tests

Before integrating into the notebook, verify all safety utilities work:

```bash
# Run test suite
python training/tests/test_safety_utilities.py
```

**Expected Output:**
```
================================================================================
STREAMGUARD SAFETY UTILITIES TEST SUITE
================================================================================

test_adjust_batch_size_for_memory (test_safety_utilities.TestAdaptiveConfig) ... ok
test_detect_gpu (test_safety_utilities.TestAdaptiveConfig) ... ok
test_load_adaptive_config_fusion (test_safety_utilities.TestAdaptiveConfig) ... ok
...

================================================================================
TEST SUMMARY
================================================================================
Tests run: 25
Failures: 0
Errors: 0
Skipped: 5

✅ ALL TESTS PASSED!
```

**If Tests Fail:**
- Review error messages
- Check PyTorch installation: `pip install torch`
- Verify file paths are correct

---

## Integration Steps

### Step 2: Backup Your Notebook

```bash
# Create backup
cp StreamGuard_Complete_Training.ipynb StreamGuard_Complete_Training_BACKUP_$(date +%Y%m%d).ipynb
```

---

### Step 3: Add Production Training Cells

You have two options:

#### Option A: Add as Python Script Calls (Recommended)

**Advantages:**
- Cleaner notebook
- Easier to version control
- Scripts can be edited independently

**Cell 51 (Markdown):**
```markdown
## Cell 51: Transformer v1.7 Production Training

Features:
- 3-seed reproducibility (42, 2025, 7)
- LR Finder with 168h caching
- Mixed precision (AMP)
- Collapse detection
- Triple weighting for class imbalance

**Hyperparameters:**
- LR: 1e-5 → 5e-5 (from LR Finder)
- Batch size: 64 (adaptive)
- Epochs: 10

**Outputs:** `training/outputs/transformer_v17_production/`
```

**Cell 51 (Code):**
```python
# Run Transformer Production Training
!python training/scripts/cell_51_transformer_production.py
```

**Cell 52 (Markdown):**
```markdown
## Cell 52: GNN v1.7 Production Training

Features:
- PyTorch Geometric graph loading
- WeightedRandomSampler for imbalance
- Focal Loss (γ=1.5)
- Graph statistics logging

**Hyperparameters:**
- LR: 1e-4 → 1e-3 (PyG LR Finder)
- Batch size: 64 graphs
- Epochs: 15

**Outputs:** `training/outputs/gnn_v17_production/`
```

**Cell 52 (Code):**
```python
# Run GNN Production Training
!python training/scripts/cell_52_gnn_production.py
```

**Cell 53 (Markdown):**
```markdown
## Cell 53: Fusion v1.7 Production Training

Features:
- Discriminative learning rates
  - Transformer: LR × 0.1
  - GNN: LR × 0.5
  - Fusion: LR × 1.0
- Loads pretrained models from Cells 51-52

**Hyperparameters:**
- Base LR: 1e-5
- Batch size: 32
- Gradient accumulation: 2 steps

**Outputs:** `training/outputs/fusion_v17_production/`
```

**Cell 53 (Code):**
```python
# Run Fusion Production Training
!python training/scripts/cell_53_fusion_production.py
```

**Cell 54 (Markdown):**
```markdown
## Cell 54: Unified Metrics & Export

Aggregates results from all 9 training runs:
- 3 models (Transformer, GNN, Fusion)
- 3 seeds per model (42, 2025, 7)

**Outputs:**
- `production_summary.json` - Complete metrics
- `production_report.md` - Formatted report
- `metrics_comparison.png` - Visualization
```

**Cell 54 (Code):**
```python
# Aggregate Metrics
!python training/scripts/cell_54_metrics_aggregator.py

# Display summary
import json
with open('training/outputs/production_summary/production_summary.json', 'r') as f:
    summary = json.load(f)

print("\n" + "="*80)
print("PRODUCTION TRAINING SUMMARY")
print("="*80)

for model, stats in summary['models'].items():
    print(f"\n{model}:")
    print(f"  Mean F1: {stats['mean_f1']:.4f} ± {stats['std_f1']:.4f}")
    print(f"  Best F1: {stats['best_f1']:.4f} (seed {stats['best_seed']})")

print(f"\nBest Overall: {summary['best_overall_model']} (F1 = {summary['best_overall_f1']:.4f})")
```

---

#### Option B: Inline Code (Alternative)

Copy the entire script content into notebook cells. This is not recommended as it makes the notebook very large.

---

### Step 4: Update Data Paths

Before running, update data paths in each script to match your setup:

**Edit `training/scripts/cell_51_transformer_production.py`:**
```python
# Line ~60
TRAIN_DATA_PATH = Path("data/processed/codexglue/train.jsonl")  # ← Update this
VAL_DATA_PATH = Path("data/processed/codexglue/val.jsonl")      # ← Update this
```

**Edit `training/scripts/cell_52_gnn_production.py`:**
```python
# Line ~60
TRAIN_DATA_PATH = Path("data/processed/graphs/train")  # ← Update this
VAL_DATA_PATH = Path("data/processed/graphs/val")      # ← Update this
```

**Edit `training/scripts/cell_53_fusion_production.py`:**
```python
# Line ~100
transformer_checkpoint = Path("training/outputs/transformer_v17_production/seed_42/model_checkpoint.pt")
gnn_checkpoint = Path("training/outputs/gnn_v17_production/seed_42/model_checkpoint.pt")
# These should auto-populate from Cells 51-52
```

---

### Step 5: Test Individual Cells

Run cells one at a time to catch errors early:

1. **Test Cell 51 First** (Transformer - most stable)
   ```python
   !python training/scripts/cell_51_transformer_production.py
   ```

   **Check outputs:**
   ```bash
   ls training/outputs/transformer_v17_production/seed_42/
   # Expected: model_checkpoint.pt, training_metadata.json
   ```

2. **Test Cell 52** (GNN - requires graph data)
   ```python
   !python training/scripts/cell_52_gnn_production.py
   ```

3. **Test Cell 53** (Fusion - requires Cells 51-52 complete)
   ```python
   !python training/scripts/cell_53_fusion_production.py
   ```

4. **Test Cell 54** (Aggregation - requires all previous cells)
   ```python
   !python training/scripts/cell_54_metrics_aggregator.py
   ```

---

## Validation Checklist

After integration, verify:

### ✅ Cell 51 (Transformer)
- [ ] Training completes for all 3 seeds
- [ ] `model_checkpoint.pt` created for each seed
- [ ] `training_metadata.json` is valid JSON (no tensor errors)
- [ ] LR cache hit on second run
- [ ] No collapse events triggered (unless expected)
- [ ] `production_summary.json` created

**Validation Commands:**
```bash
# Check outputs exist
ls training/outputs/transformer_v17_production/seed_*/

# Validate JSON
python -c "import json; json.load(open('training/outputs/transformer_v17_production/seed_42/training_metadata.json'))"

# Check LR cache
ls models/transformer/.lr_cache/
```

---

### ✅ Cell 52 (GNN)
- [ ] Training completes for all 3 seeds
- [ ] `graph_statistics.json` created
- [ ] Focal Loss converges properly
- [ ] WeightedSampler balances classes
- [ ] `production_summary.json` created

**Validation Commands:**
```bash
# Check graph stats
cat training/outputs/gnn_v17_production/seed_42/graph_statistics.json

# Verify focal loss in logs (should see loss values)
grep -i "focal" training/outputs/gnn_v17_production/seed_42/*.log
```

---

### ✅ Cell 53 (Fusion)
- [ ] Loads pretrained models successfully
- [ ] Discriminative LR applied correctly
- [ ] Gradient monitoring shows different norms per component
- [ ] `gradient_stats.json` created

**Validation Commands:**
```bash
# Check gradient stats
cat training/outputs/fusion_v17_production/seed_42/gradient_stats.json

# Verify LR multipliers in metadata
python -c "
import json
meta = json.load(open('training/outputs/fusion_v17_production/seed_42/training_metadata.json'))
print('LR Multipliers:', meta['config']['transformer_lr_mult'], meta['config']['gnn_lr_mult'])
"
```

---

### ✅ Cell 54 (Aggregation)
- [ ] Aggregates all 9 runs successfully
- [ ] `production_summary.json` valid
- [ ] `production_report.md` created
- [ ] `metrics_comparison.png` generated (if matplotlib available)

**Validation Commands:**
```bash
# View summary
cat training/outputs/production_summary/production_report.md

# Validate JSON schema
python -c "
import json
summary = json.load(open('training/outputs/production_summary/production_summary.json'))
assert 'models' in summary
assert 'best_overall_model' in summary
print('✅ Valid schema')
"
```

---

## Common Issues & Solutions

### Issue 1: `ModuleNotFoundError: No module named 'training'`

**Cause:** Python path not set correctly

**Solution:**
```python
# Add at top of cell
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
```

---

### Issue 2: `FileNotFoundError: Data path not found`

**Cause:** Data paths not updated

**Solution:**
Update paths in each script (see Step 4 above)

---

### Issue 3: `CUDA out of memory`

**Cause:** Batch size too large for GPU

**Solution:**
```python
# Reduce batch size in config override
# Add to script before load_adaptive_config():
BASE_CONFIG['batch_size'] = 32  # Down from 64
```

---

### Issue 4: `TypeError: Tensor not JSON serializable`

**Cause:** Not using `safe_jsonify()` or `atomic_write_json()`

**Solution:**
All scripts already use safe serialization. If you see this error:
1. Check you're using production scripts (not old versions)
2. Verify `training/utils/json_safety.py` exists
3. Run test suite to validate

---

### Issue 5: Training diverges / NaN loss

**Cause:** Learning rate too high

**Solution:**
1. Check `collapse_report.json` for diagnostics
2. Manually reduce LR:
   ```python
   BASE_CONFIG['learning_rate'] = 1e-6  # Lower than LR Finder suggestion
   ```

---

## Monitoring During Training

### Real-Time Progress

Each cell outputs progress bars and metrics:

```
================================================================================
TRAINING WITH SEED: 42
================================================================================

--- Epoch 1/10 ---
Epoch 1: 100%|████████| 156/156 [02:15<00:00, loss=0.4521, grad_norm=2.3451]
Train Loss: 0.4521
Val Loss: 0.3821
Val F1: 0.8934
Val Accuracy: 0.9012

[+] Seed 42 complete. Best F1: 0.8934
```

### What to Watch For

**Good Signs:**
- ✅ Loss decreasing steadily
- ✅ Gradient norm stable (1-10 range)
- ✅ F1 score improving
- ✅ LR cache hits on re-runs

**Warning Signs:**
- ⚠️ Loss spiking (check gradient clipping)
- ⚠️ Gradient norm > 100 (might explode)
- ⚠️ F1 not improving after 3 epochs
- ⚠️ "Collapse detected" messages

**Critical Issues:**
- ❌ NaN/Inf loss
- ❌ Out of memory errors
- ❌ Training stops with collapse report

---

## Post-Integration Testing

### Full Pipeline Test

Run all 4 cells in sequence:

```bash
# Run complete pipeline
python training/scripts/cell_51_transformer_production.py && \
python training/scripts/cell_52_gnn_production.py && \
python training/scripts/cell_53_fusion_production.py && \
python training/scripts/cell_54_metrics_aggregator.py
```

**Expected Duration:**
- Cell 51 (Transformer): ~30-60 minutes (depends on dataset size)
- Cell 52 (GNN): ~40-70 minutes
- Cell 53 (Fusion): ~20-40 minutes
- Cell 54 (Aggregation): ~1-2 minutes

**Total:** ~2-3 hours for complete pipeline

---

### Verify Reproducibility

Run Cell 51 twice and compare:

```bash
# First run
python training/scripts/cell_51_transformer_production.py

# Second run (should use cached LR)
python training/scripts/cell_51_transformer_production.py

# Compare checksums
md5sum training/outputs/transformer_v17_production/seed_42/model_checkpoint.pt
```

**Expected:** Checksums should be identical (seeds guarantee reproducibility)

---

## Final Checklist

Before considering integration complete:

- [ ] All 4 cells run without errors
- [ ] All output directories created
- [ ] All JSON files validate (no tensor errors)
- [ ] LR cache working (hit on re-run)
- [ ] No collapse events (unless expected)
- [ ] `production_summary.json` contains valid metrics
- [ ] Reproducibility verified (same seed = same results)
- [ ] Documentation added to notebook (markdown cells)
- [ ] Backup notebook created

---

## Next Steps After Integration

### 1. Performance Tuning
- Experiment with different LR ranges
- Adjust class weights if imbalance severe
- Tune Focal Loss γ parameter

### 2. Monitoring Setup
- Add CloudWatch metrics (if using AWS)
- Set up alerts for collapse events
- Log metrics to MLflow/Weights&Biases

### 3. Production Deployment
- Export best models to ONNX
- Create inference API endpoint
- Set up A/B testing framework

---

## Support

If you encounter issues during integration:

1. **Check Logs:** Each seed saves detailed logs in `training_metadata.json`
2. **Review Tests:** Run `python training/tests/test_safety_utilities.py`
3. **Consult Guides:**
   - `PRODUCTION_TRAINING_GUIDE.md` - Usage instructions
   - `PRODUCTION_TRAINING_PLAN.md` - Detailed specifications
4. **Debug Mode:** Add `verbose=True` to safety utilities for detailed output

---

**Status:** Ready for Integration
**Last Updated:** 2025-11-07
