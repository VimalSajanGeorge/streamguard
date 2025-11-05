# StreamGuard Jupyter Notebook Usage Guide

**Last Updated:** 2025-01-05
**Notebook Version:** v1.7 (GNN Phase 1 Complete)
**Target Environment:** Google Colab (T4/V100/A100 GPU)

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Environment Setup](#environment-setup)
3. [Cell Execution Sequence](#cell-execution-sequence)
4. [Testing Suite (GNN v1.7)](#testing-suite-gnn-v17)
5. [Production Training Workflows](#production-training-workflows)
6. [Output Interpretation](#output-interpretation)
7. [Troubleshooting](#troubleshooting)
8. [Resource Requirements](#resource-requirements)

---

## Quick Start

### Prerequisites

- Google account with Colab access
- GitHub account (for accessing StreamGuard repository)
- Basic familiarity with Jupyter notebooks

### 3-Minute Setup

1. **Open Notebook**
   ```
   https://colab.research.google.com/
   File → Upload notebook → StreamGuard_Complete_Training.ipynb
   ```

2. **Connect to GPU Runtime**
   ```
   Runtime → Change runtime type → Hardware accelerator: GPU (T4)
   ```

3. **Run Initial Setup Cells** (Cells 1-10)
   - Cell 1: Mount Google Drive
   - Cells 2-10: Install dependencies and clone repository

4. **Navigate to Data Collection or Training**
   - For data collection: Start at Cell 11
   - For training only: Jump to Cell 21 (if data already exists)

---

## Environment Setup

### Cell 1: Mount Google Drive

**Purpose:** Persist notebooks, checkpoints, and logs across sessions

```python
from google.colab import drive
drive.mount('/content/drive')
```

**Expected Output:**
```
Mounted at /content/drive
```

**Troubleshooting:**
- If mount fails, restart runtime and try again
- Ensure you're logged into the correct Google account

---

### Cells 2-5: Install Dependencies

**Purpose:** Install PyTorch, PyTorch Geometric, and other ML libraries

**Execution Time:** ~3-5 minutes

**Key Dependencies:**
- PyTorch 2.1.0+cu121
- PyTorch Geometric (PyG)
- Transformers
- networkx, torch-scatter, torch-sparse

**Warning:** Do NOT interrupt these cells mid-installation

---

### Cells 6-10: Clone Repository & Setup

**Cell 6:** Clone StreamGuard repository
```bash
git clone https://github.com/VimalSajanGeorge/streamguard.git
```

**Cell 7:** Navigate to repository
```bash
cd /content/streamguard
```

**Cell 8:** Install project dependencies
```bash
pip install -r requirements.txt
```

**Cell 9:** Verify installation
```bash
python -c "import torch; import torch_geometric; print('✓ All imports successful')"
```

**Cell 10:** Create output directories
```bash
mkdir -p data/raw/{github,opensource,cves}
mkdir -p data/processed
mkdir -p training/outputs/{transformer,gnn,fusion}
```

---

## Cell Execution Sequence

### Phase 1: Data Collection (Cells 11-20)

**Skip this phase if you already have collected data**

| Cell | Purpose | Execution Time | Output |
|------|---------|----------------|--------|
| 11-13 | Collect GitHub vulnerability data | ~30-60 min | `data/raw/github/*.json` |
| 14-16 | Collect Open Source CVE data | ~20-40 min | `data/raw/opensource/*.json` |
| 17-19 | Process and merge datasets | ~10-15 min | `data/processed/train.jsonl` |
| 20 | Verify data quality | ~1-2 min | Dataset statistics |

**Critical Data Collection Notes:**

1. **GitHub API Rate Limits:**
   - Authenticated: 5,000 requests/hour
   - Unauthenticated: 60 requests/hour
   - Set `GITHUB_TOKEN` environment variable for authenticated access

2. **CVE Database Access:**
   - Requires internet connectivity
   - Downloads NVD feeds (can be slow)

3. **Data Persistence:**
   - Raw data saved to `data/raw/`
   - Processed data saved to `data/processed/`
   - Always verify `train.jsonl`, `val.jsonl`, `test.jsonl` exist before training

---

### Phase 2: Graph Preprocessing (Cells 21-23)

**Purpose:** Convert code datasets to PyTorch Geometric graphs

| Cell | Purpose | Execution Time | Output |
|------|---------|----------------|--------|
| 21 | Convert training data to graphs | ~15-30 min | `data/processed/graphs/train/` |
| 22 | Convert validation data to graphs | ~5-10 min | `data/processed/graphs/val/` |
| 23 | Convert test data to graphs | ~5-10 min | `data/processed/graphs/test/` |

**Expected Directory Structure After Cell 23:**
```
data/processed/graphs/
├── train/
│   ├── graph_0.pt
│   ├── graph_1.pt
│   └── ...
├── val/
│   └── ...
└── test/
    └── ...
```

**Verification:**
```bash
python -c "import torch; g = torch.load('data/processed/graphs/train/graph_0.pt'); print(g)"
```

**Expected Output:**
```
Data(x=[256, 768], edge_index=[2, 512], y=1)
```

---

### Phase 3: Training

#### 3A. Transformer Training (Cells 24-27)

**Cell 24:** Transformer LR Finder (Optional)
```bash
python training/train_transformer.py \
  --find-lr \
  --lr-finder-subsample 128 \
  --epochs 1 \
  --quick-test
```
**Execution Time:** ~5-10 minutes
**Output:** `training/outputs/transformer/lr_finder_cache.json`

---

**Cell 25:** Transformer Quick Test (Sanity Check)
```bash
python training/train_transformer.py \
  --use-weighted-sampler \
  --focal-loss \
  --weight-multiplier 1.5 \
  --epochs 5 \
  --quick-test
```
**Execution Time:** ~10-15 minutes
**Output:** `training/outputs/transformer/quick_test/best_model.pt`

**Success Criteria:**
- Training loss decreases consistently
- Validation F1 > 0.70 by epoch 5
- No collapse warnings

---

**Cell 26:** Transformer Full Training (Production)
```bash
python training/train_transformer.py \
  --find-lr \
  --use-weighted-sampler \
  --focal-loss \
  --weight-multiplier 1.2 \
  --epochs 20 \
  --early-stopping-patience 5 \
  --output-dir training/outputs/transformer/production
```
**Execution Time:** ~2-4 hours (T4 GPU)
**Output:**
- `training/outputs/transformer/production/best_model.pt`
- `training/outputs/transformer/production/metrics_history.csv`
- `training/outputs/transformer/production/best_model_metadata.json`

**Expected Results:**
- Validation F1 > 0.85
- Test F1 > 0.80
- Recall for vulnerable class > 0.75

---

#### 3B. GNN Training (Cells 28-31.6) — GNN v1.7 Phase 1

**Cell 28:** GNN LR Finder (Optional)
```bash
python training/train_gnn.py \
  --find-lr \
  --lr-finder-subsample 64 \
  --lr-finder-max-iter 100 \
  --epochs 1 \
  --quick-test
```
**Execution Time:** ~3-5 minutes
**Output:** `training/outputs/gnn/lr_finder_cache.json`

**GNN-Specific Notes:**
- Default LR cap: 1e-3 (lower than Transformer's 1e-2)
- GNN LR Finder uses PyG DataLoader with graph batching
- Cache key includes model architecture hash

---

**Cell 29:** GNN Quick Test (Sanity Check)
```bash
python training/train_gnn.py \
  --use-weighted-sampler \
  --focal-loss \
  --weight-multiplier 1.5 \
  --epochs 10 \
  --quick-test
```
**Execution Time:** ~5-10 minutes
**Output:** `training/outputs/gnn/quick_test/best_model.pt`

**Success Criteria:**
- Training loss decreases consistently
- Validation F1 > 0.65 by epoch 10
- No collapse warnings (checks start at epoch 2)

---

**Cell 30:** GNN Full Training (Production)
```bash
python training/train_gnn.py \
  --find-lr \
  --use-weighted-sampler \
  --focal-loss \
  --weight-multiplier 1.2 \
  --epochs 30 \
  --early-stopping-patience 7 \
  --auto-batch-size \
  --output-dir training/outputs/gnn/production
```
**Execution Time:** ~3-6 hours (T4 GPU)
**Output:**
- `training/outputs/gnn/production/best_model.pt`
- `training/outputs/gnn/production/metrics_history.csv`
- `training/outputs/gnn/production/best_model_metadata.json`
- `training/outputs/gnn/production/collapse_diagnostic_*.json` (if collapse detected)

**Expected Results:**
- Validation F1 > 0.80
- Test F1 > 0.75
- Recall for vulnerable class > 0.70

---

**NEW: Cell 31** — GNN LR Finder Quick Test (v1.7)
```bash
python training/train_gnn.py \
  --find-lr \
  --lr-finder-subsample 64 \
  --lr-finder-max-iter 100 \
  --epochs 1 \
  --quick-test
```
**Purpose:** Test LR Finder implementation with minimal data
**Execution Time:** ~2-3 minutes
**Expected Output:**
```
[+] LR Finder completed
[+] Suggested LR: 0.000234 (validated)
[+] Cached to: gnn_abc123_lr_cache.json
```

---

**NEW: Cell 31.5** — GNN Tiny Overfit Test (v1.7)
```bash
python training/train_gnn.py \
  --use-weighted-sampler \
  --focal-loss \
  --weight-multiplier 1.5 \
  --epochs 10 \
  --quick-test
```
**Purpose:** Verify triple weighting + collapse detection on tiny dataset
**Execution Time:** ~3-5 minutes
**Expected Output:**
```
Epoch 10/10 - Train Loss: 0.12 - Val F1: 0.85
[+] No collapses detected
[+] Triple weighting auto-adjusted multiplier: 1.5 → 1.2
```

**Success Criteria:**
- Model should overfit (train F1 > 0.95)
- No collapse warnings
- Prediction distribution shows both classes predicted

---

**NEW: Cell 31.6** — GNN Short Full-Data Test (v1.7)
```bash
python training/train_gnn.py \
  --use-weighted-sampler \
  --focal-loss \
  --weight-multiplier 1.2 \
  --epochs 3 \
  --auto-batch-size \
  --output-dir training/outputs/gnn/short_test
```
**Purpose:** Validate Phase 1 features on full dataset (3 epochs only)
**Execution Time:** ~20-30 minutes
**Expected Output:**
```
Epoch 3/3 - Train Loss: 0.45 - Val F1: 0.72
[+] CSV metrics saved to: metrics_history.csv
[+] Checkpoint saved with enhanced metadata
```

**Success Criteria:**
- Training completes without errors
- Validation F1 improves across epochs
- `metrics_history.csv` contains all Phase 1 metrics
- `best_model_metadata.json` includes LR Finder analysis

---

#### 3C. Fusion Model Training (Cells 32-34)

**Cell 32:** Fusion Model Quick Test
```bash
python training/train_fusion.py \
  --transformer-ckpt training/outputs/transformer/production/best_model.pt \
  --gnn-ckpt training/outputs/gnn/production/best_model.pt \
  --epochs 5 \
  --quick-test
```
**Execution Time:** ~10-15 minutes
**Dependencies:** Must have trained Transformer and GNN models first

---

**Cell 33:** Fusion Model Full Training
```bash
python training/train_fusion.py \
  --transformer-ckpt training/outputs/transformer/production/best_model.pt \
  --gnn-ckpt training/outputs/gnn/production/best_model.pt \
  --epochs 15 \
  --early-stopping-patience 5 \
  --output-dir training/outputs/fusion/production
```
**Execution Time:** ~2-3 hours
**Expected Results:**
- Validation F1 > 0.88 (better than individual models)
- Test F1 > 0.85

---

**Cell 34:** Model Evaluation & Comparison
```bash
python scripts/evaluate_all_models.py \
  --transformer-ckpt training/outputs/transformer/production/best_model.pt \
  --gnn-ckpt training/outputs/gnn/production/best_model.pt \
  --fusion-ckpt training/outputs/fusion/production/best_model.pt \
  --test-data data/processed/test.jsonl
```
**Execution Time:** ~5-10 minutes
**Output:** Comparative metrics table (F1, Precision, Recall, Accuracy)

---

## Testing Suite (GNN v1.7)

### Three-Tier Testing Strategy

| Test Cell | Dataset Size | Epochs | Purpose | Pass Criteria |
|-----------|-------------|--------|---------|---------------|
| 31 | 64 samples | 1 | LR Finder validation | Suggests LR in [1e-6, 1e-3] |
| 31.5 | 64 samples | 10 | Overfit + collapse detection | Train F1 > 0.95, no collapse |
| 31.6 | Full dataset | 3 | Phase 1 feature integration | Val F1 improves, CSV logged |

### Running the Full Test Suite

Execute cells in order:
1. Run Cell 31 → Verify LR Finder suggests reasonable LR
2. Run Cell 31.5 → Verify triple weighting works (check logs for auto-adjustment)
3. Run Cell 31.6 → Verify full training pipeline (check `metrics_history.csv`)

**Total Test Time:** ~30-40 minutes

---

## Production Training Workflows

### Workflow 1: Full Pipeline (First-Time Setup)

**Estimated Total Time:** ~8-12 hours

```
1. Run Cells 1-10 (Setup) — 10 min
2. Run Cells 11-20 (Data Collection) — 90 min
3. Run Cells 21-23 (Graph Preprocessing) — 30 min
4. Run Cell 26 (Transformer Training) — 3 hours
5. Run Cell 30 (GNN Training) — 4 hours
6. Run Cell 33 (Fusion Training) — 2 hours
7. Run Cell 34 (Evaluation) — 10 min
```

**Recommendation:** Use A100 GPU for production training (2-3x faster than T4)

---

### Workflow 2: Quick Validation (Pre-Existing Data)

**Estimated Total Time:** ~1-2 hours

```
1. Run Cells 1-10 (Setup) — 10 min
2. Run Cell 25 (Transformer Quick Test) — 15 min
3. Run Cells 31, 31.5, 31.6 (GNN Test Suite) — 40 min
4. Run Cell 32 (Fusion Quick Test) — 15 min
```

**Purpose:** Verify code changes don't break training pipeline

---

### Workflow 3: Hyperparameter Tuning

**Use Case:** Optimize `weight_multiplier`, `focal_gamma`, `lr`

```
1. Run Cell 31 (LR Finder) — Find optimal LR range
2. Modify Cell 31.6 with different hyperparameters
3. Run Cell 31.6 multiple times (3 epochs each)
4. Compare `metrics_history.csv` across runs
5. Select best hyperparameters for full training (Cell 30)
```

**Example Hyperparameter Grid:**
- `weight_multiplier`: [1.0, 1.2, 1.5, 2.0]
- `focal_gamma`: [1.5, 2.0, 2.5]
- `lr`: Use LR Finder suggestion

---

## Output Interpretation

### Training Logs

**Key Metrics to Monitor:**

```
Epoch 5/20 - Train Loss: 0.42 - Val Loss: 0.38 - Val F1: 0.82
  Binary F1 (Vulnerable): 0.79
  Precision: 0.85 | Recall: 0.73
  Prediction Distribution: Safe=234, Vulnerable=89
```

**Healthy Training Indicators:**
- ✓ Train loss decreases monotonically
- ✓ Val loss decreases (with minor fluctuations)
- ✓ Val F1 improves or plateaus
- ✓ Prediction distribution shows both classes (no collapse)
- ✓ Binary F1 (Vulnerable) > 0.70

**Warning Signs:**
- ⚠ Val loss increases while train loss decreases (overfitting)
- ⚠ Prediction distribution: Safe=323, Vulnerable=0 (collapse)
- ⚠ Binary F1 (Vulnerable) < 0.50 (model ignoring minority class)

---

### Metrics History CSV

**File:** `training/outputs/{model}/metrics_history.csv`

**Columns:**
```
epoch,train_loss,val_loss,val_acc,val_f1,val_binary_f1_vulnerable,val_precision,val_recall,lr,predicted_vulnerable,predicted_safe
1,0.65,0.58,0.72,0.74,0.68,0.82,0.67,0.0002,89,234
2,0.52,0.51,0.78,0.79,0.75,0.85,0.73,0.0002,95,228
...
```

**Visualization (Optional):**
```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('metrics_history.csv')
plt.plot(df['epoch'], df['val_f1'], label='Val F1')
plt.plot(df['epoch'], df['val_binary_f1_vulnerable'], label='Binary F1 (Vuln)')
plt.legend()
plt.show()
```

---

### Checkpoint Metadata

**File:** `training/outputs/{model}/best_model_metadata.json`

**Sample Contents:**
```json
{
  "seed": 42,
  "git_commit": "2333986",
  "timestamp": "2025-01-05T14:32:10",
  "lr_finder_used": true,
  "suggested_lr": 0.00023,
  "triple_weighting": true,
  "weighted_sampler": true,
  "focal_loss": true,
  "weight_multiplier": 1.2,
  "class_weights": [1.0, 2.4],
  "prediction_distribution": {
    "predicted_vulnerable": 89,
    "predicted_safe": 234,
    "actual_vulnerable": 95,
    "actual_safe": 228
  },
  "collapse_history": [],
  "lr_finder_analysis": {
    "suggested_lr": 0.00023,
    "min_loss": 0.42,
    "smoothed_gradient_peak_lr": 0.00019,
    "validation_status": "capped"
  }
}
```

**Key Fields:**
- `lr_finder_used`: Whether LR Finder was run
- `triple_weighting`: Whether all 3 balancing techniques are active
- `collapse_history`: List of epochs where collapse was detected (should be empty)
- `lr_finder_analysis`: Detailed LR Finder results

---

### Collapse Diagnostic Report

**File:** `training/outputs/{model}/collapse_diagnostic_epoch_{N}.json`

**Generated When:** 2 consecutive epochs with all predictions in one class

**Sample Contents:**
```json
{
  "collapse_epoch": 8,
  "consecutive_collapses": 2,
  "collapse_history": [7, 8],
  "final_metrics": {
    "val_f1": 0.45,
    "val_binary_f1_vulnerable": 0.0,
    "prediction_distribution": {
      "predicted_vulnerable": 0,
      "predicted_safe": 323
    }
  },
  "hyperparameters": {
    "weight_multiplier": 3.0,
    "focal_gamma": 2.5,
    "lr": 0.001
  },
  "recommendations": [
    "Reduce weight_multiplier (currently 3.0 → try 1.5)",
    "Check class weights for extreme values",
    "Consider reducing learning rate"
  ]
}
```

**Action Items:**
1. Read diagnostic report to understand collapse cause
2. Adjust hyperparameters (usually reduce `weight_multiplier`)
3. Re-run training with updated settings

---

## Troubleshooting

### Common Issues

#### Issue 1: Runtime Disconnects During Long Training

**Symptom:** Training stops after 2-3 hours with "Runtime disconnected"

**Cause:** Colab free tier has time limits (~12 hours for standard, ~24 hours for Pro)

**Solutions:**
1. **Use Colab Pro/Pro+** — Longer runtime limits
2. **Enable periodic callbacks:**
   ```python
   # Add to training script
   from google.colab import output
   output.eval_js('setInterval(() => { document.querySelector("colab-toolbar-button#connect").click() }, 60000)')
   ```
3. **Save checkpoints frequently** — Training resumes from last checkpoint

---

#### Issue 2: CUDA Out of Memory (OOM)

**Symptom:** `RuntimeError: CUDA out of memory`

**Solutions:**
1. **Reduce batch size:**
   ```bash
   python training/train_gnn.py --batch-size 16  # Default is 32
   ```
2. **Enable `--auto-batch-size`** (GNN only):
   ```bash
   python training/train_gnn.py --auto-batch-size
   ```
3. **Use gradient accumulation** (Transformer only):
   ```bash
   python training/train_transformer.py --accumulation-steps 2
   ```
4. **Switch to smaller model:**
   ```bash
   python training/train_gnn.py --hidden-dim 128  # Default is 256
   ```

---

#### Issue 3: Training Collapses (All Predictions Same Class)

**Symptom:** Collapse diagnostic report generated

**Root Causes:**
- `weight_multiplier` too high (>2.0)
- Learning rate too high
- Focal gamma too aggressive (>3.0)

**Solutions:**
1. **Reduce weight_multiplier:**
   ```bash
   # Before: --weight-multiplier 2.5
   # After:
   python training/train_gnn.py --weight-multiplier 1.2
   ```
2. **Use LR Finder:**
   ```bash
   python training/train_gnn.py --find-lr
   ```
3. **Disable focal loss temporarily:**
   ```bash
   python training/train_gnn.py --use-weighted-sampler  # No --focal-loss
   ```

---

#### Issue 4: ModuleNotFoundError in Notebook

**Symptom:** `ModuleNotFoundError: No module named 'training'`

**Cause:** Not running from `/content/streamguard` directory

**Solution:**
```bash
cd /content/streamguard
python training/train_gnn.py ...
```

**Verification:**
```bash
pwd  # Should print: /content/streamguard
```

---

#### Issue 5: LR Finder Suggests Very Low LR (<1e-5)

**Symptom:** LR Finder suggests LR below 1e-5

**Possible Causes:**
- Loss curve is noisy
- Dataset has extreme class imbalance
- Model architecture is unstable

**Solutions:**
1. **Use manual LR override:**
   ```bash
   python training/train_gnn.py --lr-override 0.0001
   ```
2. **Increase LR Finder iterations:**
   ```bash
   python training/train_gnn.py --find-lr --lr-finder-max-iter 200
   ```
3. **Check loss curve plot:**
   ```bash
   # LR Finder saves plot to: training/outputs/{model}/lr_finder_plot.png
   ```

---

#### Issue 6: GitHub API Rate Limit During Data Collection

**Symptom:** `403 Forbidden: API rate limit exceeded`

**Solutions:**
1. **Set GitHub token:**
   ```python
   import os
   os.environ['GITHUB_TOKEN'] = 'ghp_YOUR_TOKEN_HERE'
   ```
2. **Wait 1 hour** — Rate limits reset hourly
3. **Use pre-collected data** — Skip Cells 11-16, use existing `data/raw/`

---

## Resource Requirements

### GPU Recommendations

| Task | Minimum GPU | Recommended GPU | Execution Time (Est.) |
|------|------------|-----------------|----------------------|
| LR Finder | T4 | T4 | 3-5 min |
| Quick Test | T4 | T4 | 10-15 min |
| Transformer Full Training | T4 | V100 | 3h (T4), 1.5h (V100) |
| GNN Full Training | V100 | A100 | 4h (V100), 2h (A100) |
| Fusion Full Training | V100 | A100 | 2h (V100), 1h (A100) |

### Memory Requirements

| Model | Batch Size | GPU Memory | Recommendation |
|-------|-----------|------------|----------------|
| Transformer | 32 | ~10 GB | T4 (16GB) sufficient |
| GNN | 32 | ~12 GB | V100 (16GB) recommended |
| Fusion | 16 | ~14 GB | A100 (40GB) for large batches |

### Storage Requirements

| Component | Size (Est.) | Location |
|-----------|------------|----------|
| Raw Data | ~500 MB | `data/raw/` |
| Processed Graphs | ~2 GB | `data/processed/graphs/` |
| Checkpoints (all models) | ~1.5 GB | `training/outputs/` |
| Logs & Metrics | ~50 MB | `training/outputs/*/metrics_history.csv` |

**Total:** ~4 GB (ensure Google Drive has sufficient space)

---

## Advanced Usage

### Running with Custom Data

1. **Prepare JSONL files:**
   ```json
   {"code": "def foo(): ...", "label": 1, "metadata": {...}}
   ```

2. **Place in data directory:**
   ```
   data/processed/custom_train.jsonl
   data/processed/custom_val.jsonl
   data/processed/custom_test.jsonl
   ```

3. **Run preprocessing:**
   ```bash
   python scripts/preprocess_graphs.py \
     --input data/processed/custom_train.jsonl \
     --output data/processed/graphs/custom_train/
   ```

4. **Train with custom data:**
   ```bash
   python training/train_gnn.py \
     --train-data data/processed/graphs/custom_train/ \
     --val-data data/processed/graphs/custom_val/ \
     --test-data data/processed/graphs/custom_test/
   ```

---

### Resuming Interrupted Training

**If training stopped mid-run:**

1. **Check for latest checkpoint:**
   ```bash
   ls -lh training/outputs/gnn/production/checkpoints/
   ```

2. **Resume from checkpoint:**
   ```bash
   python training/train_gnn.py \
     --resume training/outputs/gnn/production/checkpoints/checkpoint_epoch_15.pt \
     --epochs 30  # Total epochs (will resume from 15 → 30)
   ```

---

### Debugging Training Issues

**Enable verbose logging:**
```bash
python training/train_gnn.py \
  --log-level DEBUG \
  --output-dir training/outputs/gnn/debug
```

**Generate detailed reports:**
```bash
python scripts/analyze_training_run.py \
  --metrics-csv training/outputs/gnn/production/metrics_history.csv \
  --metadata-json training/outputs/gnn/production/best_model_metadata.json
```

---

## Checklist: Before Production Training

Before running full training (Cells 26, 30, 33), verify:

- [ ] GPU runtime is connected (check top-right corner)
- [ ] Data files exist (`train.jsonl`, `val.jsonl`, `test.jsonl`)
- [ ] Graphs are preprocessed (`data/processed/graphs/train/`)
- [ ] Quick tests passed (Cells 25, 31.5)
- [ ] Output directory has sufficient space (~2 GB free)
- [ ] Google Drive is mounted (`/content/drive`)
- [ ] No pending runtime warnings (yellow warnings at top)

**Recommended:** Run test suite (Cells 31, 31.5, 31.6) before production training

---

## Next Steps After Training

1. **Evaluate on Test Set:**
   ```bash
   python scripts/evaluate_model.py \
     --model-ckpt training/outputs/gnn/production/best_model.pt \
     --test-data data/processed/graphs/test/
   ```

2. **Generate Confusion Matrix:**
   ```bash
   python scripts/plot_confusion_matrix.py \
     --model-ckpt training/outputs/gnn/production/best_model.pt \
     --test-data data/processed/graphs/test/
   ```

3. **Deploy Model (Optional):**
   - See `docs/DEPLOYMENT.md` for inference server setup
   - Export to ONNX for production: `scripts/export_to_onnx.py`

---

## Reference Documentation

- **Training Quick Start:** `docs/TRAINING_QUICK_START.md`
- **GNN Phase 2/3 Roadmap:** `docs/GNN_PHASE_2_3_ROADMAP.md`
- **Architecture Overview:** `docs/agent_architecture.md`
- **CLI Reference:** `python training/train_gnn.py --help`

---

## Support & Feedback

- **Issues:** https://github.com/VimalSajanGeorge/streamguard/issues
- **Documentation:** https://github.com/VimalSajanGeorge/streamguard/tree/main/docs

---

**Last Updated:** 2025-01-05
**Author:** StreamGuard ML Team
**Notebook Version:** v1.7 (GNN Phase 1 Complete)
