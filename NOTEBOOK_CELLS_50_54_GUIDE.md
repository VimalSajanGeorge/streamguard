# How to Add Cells 50-54 to StreamGuard Notebook

**Current Notebook:** `StreamGuard_Complete_Training.ipynb` (25 cells: 0-24)
**Goal:** Add production training cells 50-54
**Method:** Copy-paste the cell templates below into Jupyter

---

## Quick Start

1. Open `StreamGuard_Complete_Training.ipynb` in Jupyter
2. Scroll to the end (after cell 24)
3. Add 5 new cells using the templates below
4. Run Cell 50 first to validate everything is ready
5. Run Cells 51-54 sequentially

---

## Cell 50: Pre-Flight Validation

**Cell Type:** Code

**Purpose:** Runs all safety checks before starting expensive A100 training

```python
# CELL 50: Pre-Flight Validation
# ============================================================
# Runs all critical checks:
# - GPU detection
# - Safety utilities tests
# - Smoke tests (overfit verification)
# - Data availability
# - Memory safety
# ============================================================

import sys
from pathlib import Path

# Ensure we're in the right directory
if not Path("training/scripts/pre_flight_validation.py").exists():
    print("[!] ERROR: Not in streamguard root directory")
    print("[!] Current dir:", Path.cwd())
else:
    # Run pre-flight validation
    print("Running pre-flight validation checks...\n")
    !python training/scripts/pre_flight_validation.py
```

**Expected Output:**
```
🚀 ALL CHECKS PASSED - READY FOR PRODUCTION TRAINING!

Next steps:
  1. Run: python training/scripts/cell_51_transformer_production.py
  ...
```

**If Fails:** Review the failed checks and fix them before proceeding.

---

## Cell 51: Transformer v1.7 Production Training

### Cell 51a: Markdown Header

**Cell Type:** Markdown

```markdown
## Cell 51: Transformer v1.7 Production Training

**Features:**
- 3-seed reproducibility (seeds: 42, 2025, 7)
- LR Finder with 168-hour caching + fallback
- Mixed precision (AMP) training
- AMP-safe gradient clipping
- Collapse detection with auto-stop
- Atomic JSON writes (no tensor serialization errors)
- Triple weighting for class imbalance

**Hyperparameters:**
- Learning Rate: 1e-5 → 5e-5 (from LR Finder)
- Batch Size: 64 (adaptive for A100)
- Max Seq Length: 512
- Epochs: 10
- Weight Decay: 0.01
- Gradient Clipping: 1.0

**Class Weighting:**
- Safe (0): 1.0x
- Vulnerable (1): 3.0x (conservative imbalance handling)

**Outputs:**
- `training/outputs/transformer_v17_production/seed_42/`
- `training/outputs/transformer_v17_production/seed_2025/`
- `training/outputs/transformer_v17_production/seed_7/`
- `training/outputs/transformer_v17_production/production_summary.json`

**Expected Duration:** 30-60 minutes on A100 (depending on dataset size)
```

### Cell 51b: Training Code

**Cell Type:** Code

```python
# CELL 51: Transformer Production Training
# ============================================================

import time
start_time = time.time()

# Run training script
!python training/scripts/cell_51_transformer_production.py

# Report duration
duration = time.time() - start_time
print(f"\n{'='*80}")
print(f"Total Duration: {duration/60:.1f} minutes")
print(f"{'='*80}")

# Quick validation - check outputs exist
from pathlib import Path
import json

output_dir = Path("training/outputs/transformer_v17_production")
summary_file = output_dir / "production_summary.json"

if summary_file.exists():
    with open(summary_file) as f:
        summary = json.load(f)

    print("\n✅ TRANSFORMER TRAINING COMPLETE")
    print(f"   Mean F1: {summary['mean_f1']:.4f} ± {summary['std_f1']:.4f}")

    for result in summary['results']:
        print(f"   Seed {result['seed']}: F1 = {result['best_f1']:.4f}")
else:
    print("\n❌ Training failed - check logs above")
```

---

## Cell 52: GNN v1.7 Production Training

### Cell 52a: Markdown Header

**Cell Type:** Markdown

```markdown
## Cell 52: GNN v1.7 Production Training

**Features:**
- PyTorch Geometric graph data loading
- WeightedRandomSampler for class imbalance (1.2x multiplier)
- Focal Loss (γ=1.5) for hard negative mining
- Mixed precision (AMP) training
- AMP-safe gradient clipping
- Collapse detection
- Graph statistics logging
- 3-seed reproducibility

**Hyperparameters:**
- Learning Rate: 1e-4 → 1e-3 (from PyG LR Finder)
- Batch Size: 64 graphs
- Node Features: 768-dim (CodeBERT placeholders)
- Hidden Dim: 256
- Epochs: 15
- Focal Loss γ: 1.5
- Weight Decay: 1e-4

**Graph Structure:**
- Simple sequential graphs (line i → line i+1)
- Bidirectional edges for information flow
- ~21,854 training graphs

**Outputs:**
- `training/outputs/gnn_v17_production/seed_42/`
  - `model_checkpoint.pt`
  - `training_metadata.json`
  - `graph_statistics.json`
- `training/outputs/gnn_v17_production/production_summary.json`

**Expected Duration:** 40-70 minutes on A100

**Prerequisites:**
- Graph data must exist in `data/processed/graphs/train/`
- If missing, run: `python training/preprocessing/create_simple_graph_data.py`
```

### Cell 52b: Training Code

**Cell Type:** Code

```python
# CELL 52: GNN Production Training
# ============================================================

import time
from pathlib import Path

# Check graph data exists
graph_train_dir = Path("data/processed/graphs/train")
if not graph_train_dir.exists() or len(list(graph_train_dir.glob("*.pt"))) == 0:
    print("[!] Graph data not found. Creating graphs...")
    print("[*] This will take ~5-10 minutes for ~21,854 graphs\n")

    !python training/preprocessing/create_simple_graph_data.py

    print("\n[+] Graph data creation complete")
else:
    num_graphs = len(list(graph_train_dir.glob("*.pt")))
    print(f"[+] Found {num_graphs} training graphs")

# Run GNN training
start_time = time.time()

!python training/scripts/cell_52_gnn_production.py

duration = time.time() - start_time
print(f"\n{'='*80}")
print(f"Total Duration: {duration/60:.1f} minutes")
print(f"{'='*80}")

# Validation
import json
output_dir = Path("training/outputs/gnn_v17_production")
summary_file = output_dir / "production_summary.json"

if summary_file.exists():
    with open(summary_file) as f:
        summary = json.load(f)

    print("\n✅ GNN TRAINING COMPLETE")
    print(f"   Mean F1: {summary['mean_f1']:.4f} ± {summary['std_f1']:.4f}")

    for result in summary['results']:
        print(f"   Seed {result['seed']}: F1 = {result['best_f1']:.4f}")
else:
    print("\n❌ Training failed - check logs above")
```

---

## Cell 53: Fusion v1.7 Production Training (Optional)

### Cell 53a: Markdown Header

**Cell Type:** Markdown

```markdown
## Cell 53: Fusion v1.7 Production Training

**Status:** ⏸️ **DEFERRED** - See `CELL_53_DEFERRAL_NOTE.md`

**Reason:** Cell 53 has a placeholder training loop and is NOT a critical blocker.
Focus on validating Cell 51 & 52 first, then implement Cell 53 if needed.

**When to Implement:**
- After Cell 51 & 52 complete successfully on A100
- After verifying mean F1 > 0.85 for both models
- If fusion is needed to boost performance further

**Alternative:** Use simple ensemble instead:
```python
# Ensemble prediction (no training needed)
final_pred = 0.6 * transformer_pred + 0.4 * gnn_pred
```

**For now, SKIP this cell and proceed directly to Cell 54.**
```

### Cell 53b: Placeholder Code

**Cell Type:** Code

```python
# CELL 53: Fusion Production Training (DEFERRED)
# ============================================================

print("⏸️  Cell 53 (Fusion) is DEFERRED")
print()
print("Reason: Not a critical blocker. Validate Cell 51 & 52 first.")
print()
print("See CELL_53_DEFERRAL_NOTE.md for:")
print("  - Why it's deferred")
print("  - When to implement it")
print("  - How to implement it")
print()
print("For now, proceed to Cell 54 (Metrics Aggregation).")
```

---

## Cell 54: Unified Metrics & Export

### Cell 54a: Markdown Header

**Cell Type:** Markdown

```markdown
## Cell 54: Unified Metrics & Export

**Purpose:** Aggregates results from all training runs

**Aggregation:**
- 3 models: Transformer, GNN, Fusion (if implemented)
- 3 seeds per model: 42, 2025, 7
- Total: 6-9 training runs

**Outputs:**
- `training/outputs/production_summary/production_summary.json`
  - Complete metrics for all models and seeds
  - Mean ± std for reproducibility verification
- `training/outputs/production_summary/production_report.md`
  - Human-readable formatted report
- `training/outputs/production_summary/metrics_comparison.png`
  - Visualization (if matplotlib available)

**Success Criteria:**
- Mean F1 > 0.85 across seeds
- Std F1 < 0.05 (good reproducibility)
- No collapse events
- All JSON files valid (no tensor errors)

**Expected Duration:** 1-2 minutes
```

### Cell 54b: Aggregation Code

**Cell Type:** Code

```python
# CELL 54: Metrics Aggregation
# ============================================================

import time
start_time = time.time()

# Run aggregation
!python training/scripts/cell_54_metrics_aggregator.py

duration = time.time() - start_time
print(f"\nAggregation Duration: {duration:.1f} seconds")

# Display summary
import json
from pathlib import Path

summary_file = Path("training/outputs/production_summary/production_summary.json")

if summary_file.exists():
    with open(summary_file) as f:
        summary = json.load(f)

    print("\n" + "="*80)
    print("PRODUCTION TRAINING SUMMARY")
    print("="*80)

    # Check which models completed
    models_available = list(summary.get('models', {}).keys())

    if models_available:
        for model_name, stats in summary['models'].items():
            print(f"\n{model_name.upper()}:")
            print(f"  Mean F1: {stats['mean_f1']:.4f} ± {stats['std_f1']:.4f}")
            print(f"  Best F1: {stats['best_f1']:.4f} (seed {stats['best_seed']})")

            # Reproducibility check
            if stats['std_f1'] < 0.05:
                print(f"  ✅ Good reproducibility (std < 0.05)")
            else:
                print(f"  ⚠️  High variance (std > 0.05)")

        print(f"\n{'='*80}")
        print(f"Best Overall: {summary['best_overall_model'].upper()}")
        print(f"Best F1: {summary['best_overall_f1']:.4f}")
        print(f"{'='*80}")

        # Final assessment
        best_f1 = summary['best_overall_f1']
        if best_f1 > 0.90:
            print("\n🎉 EXCELLENT RESULTS (F1 > 0.90)")
        elif best_f1 > 0.85:
            print("\n✅ GOOD RESULTS (F1 > 0.85)")
        elif best_f1 > 0.80:
            print("\n⚠️  ACCEPTABLE RESULTS (F1 > 0.80)")
        else:
            print("\n❌ NEEDS IMPROVEMENT (F1 < 0.80)")

        # View detailed report
        report_file = Path("training/outputs/production_summary/production_report.md")
        if report_file.exists():
            print(f"\n📊 Detailed report: {report_file}")

    else:
        print("\n[!] No model results found")
        print("[!] Ensure Cell 51 and/or Cell 52 completed successfully")

else:
    print("\n❌ Aggregation failed - check logs above")
```

---

## Complete Workflow Summary

### Step-by-Step Execution

1. **Cell 50** - Pre-Flight Validation (~2-3 min)
   - ✅ All checks must pass
   - Creates graph data if missing

2. **Cell 51** - Transformer Training (~30-60 min)
   - 3 seeds × 10 epochs each
   - Verify mean F1 > 0.85

3. **Cell 52** - GNN Training (~40-70 min)
   - 3 seeds × 15 epochs each
   - Verify mean F1 > 0.85

4. **Cell 53** - Fusion Training (DEFERRED)
   - Skip for now
   - Implement later if needed

5. **Cell 54** - Metrics Aggregation (~1-2 min)
   - Combines all results
   - Generates summary report

**Total Time:** ~2-3 hours for complete pipeline (Cells 51-52-54)

---

## Troubleshooting

### Issue: Pre-Flight Checks Fail

**Solution:**
1. Check which check failed (GPU, tests, data, etc.)
2. Follow the recommendations in the output
3. Re-run Cell 50 until all checks pass

### Issue: Cell 51/52 Training Fails Mid-Run

**Check:**
```python
# View training metadata for diagnostics
import json
from pathlib import Path

meta_file = Path("training/outputs/transformer_v17_production/seed_42/training_metadata.json")
if meta_file.exists():
    with open(meta_file) as f:
        meta = json.load(f)
    print(json.dumps(meta, indent=2))
```

**Common Causes:**
- Out of memory → Reduce batch size in script
- Model collapse → Check `collapse_report.json`
- Data issues → Verify data paths in script

### Issue: LR Finder Takes Too Long

**Check LR Cache:**
```python
from pathlib import Path

cache_dir = Path("models/transformer/.lr_cache")
cache_files = list(cache_dir.glob("*.json"))

print(f"LR cache files: {len(cache_files)}")
for f in cache_files:
    print(f"  {f.name}")
```

**Note:** First run = ~10 min for LR Finder, subsequent runs (within 7 days) = instant

### Issue: Graph Data Missing

**Create Graphs:**
```python
!python training/preprocessing/create_simple_graph_data.py
```

This will create ~21,854 .pt files in `data/processed/graphs/train/`

---

## Validation Checklist

After running all cells, verify:

### ✅ Cell 51 (Transformer)
- [ ] All 3 seeds completed
- [ ] `production_summary.json` exists
- [ ] Mean F1 > 0.85
- [ ] Std F1 < 0.05
- [ ] LR cache hit on re-run
- [ ] No tensor serialization errors in JSON

### ✅ Cell 52 (GNN)
- [ ] All 3 seeds completed
- [ ] `graph_statistics.json` created
- [ ] Mean F1 > 0.85
- [ ] Std F1 < 0.05
- [ ] No collapse events

### ✅ Cell 54 (Aggregation)
- [ ] `production_summary.json` valid
- [ ] `production_report.md` created
- [ ] Best overall F1 identified
- [ ] All metrics sensible

---

## Next Steps After Notebook Integration

### If Results are Good (F1 > 0.90)
1. Export best models to ONNX for deployment
2. Create inference API
3. Set up model monitoring

### If Results are Acceptable (F1 = 0.85-0.90)
1. Implement Cell 53 (Fusion) to boost performance
2. Experiment with hyperparameters
3. Try data augmentation

### If Results Need Improvement (F1 < 0.85)
1. Review data quality
2. Check for class imbalance issues
3. Examine collapse reports
4. Consider different architectures

---

## Files Reference

| File | Purpose |
|------|---------|
| `training/scripts/pre_flight_validation.py` | Cell 50 - Runs all checks |
| `training/scripts/cell_51_transformer_production.py` | Cell 51 - Transformer training |
| `training/scripts/cell_52_gnn_production.py` | Cell 52 - GNN training |
| `training/scripts/cell_53_fusion_production.py` | Cell 53 - Fusion (deferred) |
| `training/scripts/cell_54_metrics_aggregator.py` | Cell 54 - Aggregation |
| `BLOCKER_FIXES_SUMMARY.md` | Summary of all blocker fixes |
| `CELL_53_DEFERRAL_NOTE.md` | Why Cell 53 is deferred |
| `NOTEBOOK_INTEGRATION_CHECKLIST.md` | Detailed integration guide |

---

**Last Updated:** 2025-11-08
**Ready to Use:** Yes ✅
**Status:** Production-Ready for A100 Training
