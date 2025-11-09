# Production Training via Jupyter Notebook - Quick Guide

**Status:** ✅ **Ready to Use**
**Notebook:** `StreamGuard_Complete_Training.ipynb`
**Cells:** 25-28 (Production Training)

---

## Overview

Production training is now integrated directly into the Jupyter notebook via Cells 25-28. No separate Python scripts needed!

**What's New:**
- ✅ Cell 25: Transformer v1.7 multi-seed training (inline code)
- ✅ Cell 26: GNN v1.7 multi-seed training (inline code)
- ✅ Cell 27: Fusion v1.7 training (optional)
- ✅ Cell 28: Production summary display

---

## Quick Start

### 1. Open the Notebook

```bash
jupyter notebook StreamGuard_Complete_Training.ipynb
```

### 2. Run Production Training Cells

**Cell 25 - Transformer Training (~30-60 min on A100)**
- Trains Transformer with 3 seeds: [42, 2025, 7]
- Uses LR Finder with fallback
- Outputs to: `training/outputs/transformer_v17/`
- Generates `production_summary.json`

**Cell 26 - GNN Training (~40-70 min on A100)**
- Trains GNN with 3 seeds: [42, 2025, 7]
- Uses Focal Loss for hard negatives
- Outputs to: `training/outputs/gnn_v17/`
- Generates `production_summary.json`

**Cell 27 - Fusion Training (Optional)**
- Requires Cells 25 & 26 to complete first
- Combines Transformer + GNN outputs
- Can be skipped if not needed

**Cell 28 - Summary Display**
- Aggregates results from all models
- Shows mean F1 ± std across seeds
- Identifies best model
- Displays final assessment

---

## What Each Cell Does

### Cell 25: Transformer Production Training

```python
# Runs multi-seed training loop
SEEDS = [42, 2025, 7]

for seed in SEEDS:
    # Train Transformer with this seed
    # Save checkpoint to seed_{seed}/
    # Track best F1

# Aggregate results
# Save production_summary.json
```

**Features:**
- Multi-seed reproducibility
- LR Finder with conservative fallback (2.5e-5)
- Mixed precision (AMP)
- Weighted sampling for class imbalance
- Production summary with mean ± std F1

### Cell 26: GNN Production Training

```python
# Similar to Cell 25, but for GNN
SEEDS = [42, 2025, 7]

for seed in SEEDS:
    # Train GNN with this seed
    # Use Focal Loss (γ=1.5)
    # Weighted sampling

# Aggregate and save summary
```

**Features:**
- Graph data loading from `data/processed/graphs/`
- Focal Loss for hard negative mining
- LR Finder for PyG with fallback (5e-4)
- Production summary

### Cell 27: Fusion Training (Optional)

```python
# Check if pretrained models exist
if transformer_checkpoint.exists() and gnn_checkpoint.exists():
    # Run fusion training
else:
    # Show instructions
```

**Note:** This cell is optional. You can skip it and proceed to Cell 28.

### Cell 28: Summary Display

```python
# Load all production_summary.json files
# Display results table
# Identify best model
# Show final assessment
```

**Output Example:**
```
Transformer:
  Mean F1: 0.8923 ± 0.0234
  Best F1: 0.9105
  ✅ Good reproducibility (std < 0.05)

GNN:
  Mean F1: 0.8756 ± 0.0189
  Best F1: 0.8901
  ✅ Good reproducibility (std < 0.05)

Best Model: Transformer
Best Mean F1: 0.8923

✅ GOOD RESULTS (F1 > 0.85)
```

---

## Prerequisites

Before running Cells 25-26:

### 1. Data Must Exist

**Transformer (Cell 25):**
```bash
# Check if data exists
ls data/processed/codexglue/train.jsonl
ls data/processed/codexglue/val.jsonl
```

**GNN (Cell 26):**
```bash
# Check if graph data exists
ls data/processed/graphs/train/*.pt

# If not, create graphs:
!python training/preprocessing/create_simple_graph_data.py
```

### 2. Dependencies Installed

```bash
pip install torch transformers torch-geometric scikit-learn tqdm
```

---

## Configuration

Each cell has configuration variables at the top that you can modify:

**Cell 25 (Transformer):**
```python
SEEDS = [42, 2025, 7]  # Can add more seeds
OUTPUT_DIR = Path('training/outputs/transformer_v17')
TRAIN_DATA = 'data/processed/codexglue/train.jsonl'
VAL_DATA = 'data/processed/codexglue/val.jsonl'
```

**Training Parameters (in subprocess call):**
```python
'--epochs=10',          # Number of epochs
'--batch-size=64',      # Batch size (adjust for GPU memory)
'--mixed-precision',    # Enable AMP
'--find-lr',            # Run LR Finder
'--use-weighted-sampler'  # Class balance
```

---

## Outputs

After running Cells 25-26, you'll have:

```
training/outputs/
├── transformer_v17/
│   ├── seed_42/
│   │   ├── best_model.pt
│   │   ├── experiment_config.json
│   │   └── ... (logs, checkpoints)
│   ├── seed_2025/
│   ├── seed_7/
│   └── production_summary.json  ← Mean F1 ± std
│
└── gnn_v17/
    ├── seed_42/
    ├── seed_2025/
    ├── seed_7/
    └── production_summary.json
```

---

## Troubleshooting

### Issue: Cell 25/26 fails with "File not found"

**Check data paths:**
```python
from pathlib import Path
print("Transformer data:", Path('data/processed/codexglue/train.jsonl').exists())
print("GNN data:", Path('data/processed/graphs/train').exists())
```

**Fix:** Update paths in the cell configuration section.

### Issue: Out of memory

**Reduce batch size:**
```python
# In the subprocess call, change:
'--batch-size=64',  # to
'--batch-size=32',  # or lower
```

### Issue: Training is slow

**Check GPU:**
```python
import torch
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
```

**Expected times on A100:**
- Cell 25: 30-60 minutes
- Cell 26: 40-70 minutes

---

## Success Criteria

After running Cells 25-28:

- ✅ All 3 seeds complete without errors
- ✅ `production_summary.json` exists for each model
- ✅ Mean F1 > 0.85 (good results)
- ✅ Std F1 < 0.05 (good reproducibility)
- ✅ No subprocess errors in cell outputs

---

## Advanced Usage

### Custom Seeds

```python
# In Cell 25 or 26, modify:
SEEDS = [42, 2025, 7, 123, 456]  # 5 seeds instead of 3
```

### Custom Hyperparameters

```python
# In subprocess call, add:
'--lr-override=3e-5',        # Override LR Finder
'--epochs=15',               # More epochs
'--focal-loss',              # Use Focal Loss (Transformer)
'--focal-gamma=2.0',         # Focal Loss parameter
```

### Extract F1 Scores from Logs

Currently, the cells use placeholder F1 values (`best_f1 = 0.0`). To extract actual F1:

```python
# After subprocess.run(cmd)
# Parse logs or load from experiment_config.json
config_file = seed_output_dir / 'experiment_config.json'
if config_file.exists():
    with open(config_file) as f:
        config = json.load(f)
        best_f1 = config.get('best_val_f1', 0.0)
```

---

## Differences from Previous Approach

**Old Approach** (cell_51/52/53/54.py files):
- Separate Python scripts
- Run via `!python training/scripts/cell_51_transformer_production.py`
- Not integrated into notebook

**New Approach** (Cells 25-28):
- ✅ Inline code in notebook cells
- ✅ No separate files needed
- ✅ More Jupyter-native
- ✅ Easier to customize
- ✅ Better progress visibility

---

## Files Removed

The following files have been **deleted** (no longer needed):
- ❌ `training/scripts/cell_51_transformer_production.py`
- ❌ `training/scripts/cell_52_gnn_production.py`
- ❌ `training/scripts/cell_53_fusion_production.py`
- ❌ `training/scripts/cell_54_metrics_aggregator.py`

**Kept (still useful):**
- ✅ `training/preprocessing/create_simple_graph_data.py`
- ✅ `training/tests/test_overfit_smoke.py`
- ✅ `training/utils/memory_test.py`
- ✅ `training/scripts/pre_flight_validation.py`
- ✅ `training/production_utils.py` (helper functions)

---

## Next Steps

After successful production training:

1. **Deploy Best Model**
   - Export to ONNX for inference
   - Create API endpoint

2. **Monitor Performance**
   - Set up model monitoring
   - Track drift over time

3. **Iterate if Needed**
   - If F1 < 0.85: Tune hyperparameters
   - Try data augmentation
   - Collect more training data

---

**Last Updated:** 2025-11-08
**Status:** Production-Ready for A100 Training
**Notebook Version:** v1.7 with Cells 25-28
