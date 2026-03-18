# Story 8: Colab Production Training Guide

**Notebook**: `StreamGuard_Production_Training_Colab.ipynb`
**Purpose**: Run all 3 ablation configs (B/C/D) on GPU, generate CFA proof artifact.
**Data**: 27,752 train / 3,469 val / 3,470 test samples (7 CWEs, 34,691 total)

---

## Prerequisites

### 1. Prepare the Data Zip (Local Machine)

```bash
cd C:\Users\Vimal Sajan\streamguard
python -c "import shutil; shutil.make_archive('training_data_final', 'zip', '.', 'training/data/final')"
```

This creates `training_data_final.zip` (~3.5 GB) containing:
- `training/data/final/train.h5` (3.0 GB, 27,752 samples)
- `training/data/final/val.h5` (365 MB, 3,469 samples)
- `training/data/final/test.h5` (365 MB, 3,470 samples)
- `training/data/final/split_stats.json`

### 2. Upload to Google Drive

Upload `training_data_final.zip` to:
```
My Drive/StreamGuard/training_data_final.zip
```

### 3. Upload the Notebook

Upload `StreamGuard_Production_Training_Colab.ipynb` to Google Colab.

### 4. Select GPU Runtime

In Colab: **Runtime > Change runtime type > Hardware accelerator > GPU**

- **T4** (free tier): ~5-8 hours per config, ~16-26 hours total
- **A100** (Colab Pro): ~2-3 hours per config, ~6-10 hours total
- **V100**: ~3-5 hours per config, ~10-16 hours total

---

## Cell-by-Cell Execution Guide

### Cell 1: Verify GPU Runtime
- Confirms GPU is available and reports VRAM/disk
- If "NO GPU DETECTED", go to Runtime > Change runtime type > GPU
- Minimum disk: 15 GB free (data=4GB + checkpoints=2GB + model weights=1GB)

### Cell 2: Mount Google Drive
- Mounts Drive and verifies the data zip exists
- If zip not found, re-check the Drive path

### Cell 3: Clone Repo + Install Deps
- First cell clones the git repo (or pulls if already cloned)
- Second cell installs only the packages needed for training (minimal)
- PyG install can take 2-3 minutes -- this is normal

### Cell 4: Unzip Training Data
- Extracts HDF5 files to Colab local SSD (`/content/`)
- Verifies sample counts match expected values
- Skips extraction if data already present (idempotent)

### Cell 5: Dry-Run Validation
- Runs Config C with 100 samples, 1 epoch
- Validates: imports, GPU usage, CFA batching, loss computation
- Should complete in < 2 minutes
- Check: `CFA_batches > 0`, no NaN, `Peak GPU memory` reported

### Cell 6: Config B (Baseline)
- 20 epochs, no CFA loss, cross-entropy only
- Checkpoint auto-backed to Drive after completion
- Expected val_F1 range: 0.55-0.72

### Cell 7: Config C (CFA Proof)
- 20 epochs, CFA contrastive loss enabled
- This is the key config -- CFA delta over Config B = the paper's proof
- Expected val_F1 range: 0.60-0.78
- Watch: `train_L_CFA` should decrease from ~0.5 to < 0.2

### Cell 8: Config D (TPG Ablation)
- 20 epochs, CFA + TPG edges
- Expected: similar to Config C (partial TPG coverage in M1)
- Validates TPG pathway without degradation

### Cell 9: Restore from Drive (ONLY if runtime restarted)
- Skip if all 3 configs ran in one session
- Copies checkpoints from Drive back to local SSD

### Cell 10: Ablation Comparison + Proof
- Evaluates all 3 checkpoints on test set (3,470 samples)
- Prints comparison table
- Generates `proof/cfa_proof_result.json`
- Reports PROOF POSITIVE / NEGATIVE

### Cell 11: Detailed Per-CWE Evaluation
- Per-config evaluation with full CWE breakdown
- Saves individual JSON files to `proof/`

### Cell 12: Copy Results to Drive
- Copies checkpoints, proof JSON, MLflow runs to Drive
- Lists all saved files with sizes

### Cell 13: Summary
- Final status report with proof result
- Next steps for post-POC work

---

## Strategy for T4 (Free Tier)

T4 has 12-hour runtime limits. Each config takes 5-8 hours. Strategy:

**Session 1**: Cells 1-6 (setup + Config B)
**Session 2**: Cells 1-4, Cell 9, Cell 7 (restore + Config C)
**Session 3**: Cells 1-4, Cell 9, Cell 8, Cells 10-13 (restore + Config D + proof)

Each session:
1. Run Cells 1-4 (re-setup: mount, clone, install, unzip)
2. Run Cell 9 (restore previous checkpoints from Drive)
3. Run the next config cell
4. Checkpoint is auto-backed to Drive at the end of each config

---

## What to Monitor During Training

### Per-Epoch Output Format
```
Epoch  5/20 | train_loss=0.4521 (CE=0.3210 CFA=0.2622) |
val_F1=0.6543 prec=0.7012 rec=0.6134 | pairwise_acc=0.4523 |
CFA_batches=2156/3942 | 1245.3s
```

### Key Indicators

| Metric | Healthy | Problematic |
|--------|---------|-------------|
| `train_loss` | Decreasing over epochs | Stuck or increasing |
| `train_L_CFA` | Drops from ~0.5 to < 0.2 | Stays > 0.4 after 10 epochs |
| `val_F1` | Increasing, plateaus after epoch 10-15 | 0.0 after 5 epochs |
| `CFA_batches` | > 50% of total | 0 (pair batching broken) |
| `pairwise_acc` | Increasing for C/D configs | Stuck at 0.0 |
| Epoch time | Stable across epochs | Growing (memory leak) |

### Early Warning Signs

1. **val_F1 = 0.0 after epoch 3**: Model predicting all-safe. Normal for first 2-3 epochs with warmup, but should improve by epoch 4-5.
2. **NaN in loss**: Immediately stop. Check data integrity with `pre_training_audit.py`.
3. **CFA_batches = 0**: CFAAwareBatchSampler broken. Check pair_id metadata.
4. **Early stopping at epoch 5-6**: Learning rate may be too high, or data issue.
5. **GPU OOM**: Reduce batch size. Add `--batch-size 4` (gradient accumulation compensates).

---

## Troubleshooting

### Runtime Disconnect Mid-Training

**Problem**: Colab disconnects after 30-90 minutes of inactivity or at 12-hour limit.

**Prevention**:
- Keep the browser tab active (don't minimize)
- Use Colab Pro for longer runtimes
- Each config cell backs up checkpoint to Drive on completion

**Recovery**:
1. Reconnect/restart runtime
2. Run Cells 1-4 (re-setup)
3. Run Cell 9 (restore checkpoints)
4. Continue from next config cell

**Note**: If a training run was interrupted mid-epoch, the interrupted config must be re-run from scratch. Only completed configs have checkpoints.

### OOM (Out of Memory)

**Problem**: `CUDA out of memory` error.

**Fix**: Modify the config cell to reduce batch size:
```python
config_x['batch_size_graphs'] = 4  # default is 8
config_x['gradient_accumulation'] = 8  # increase to compensate
```

Effective batch size stays at 32 (4 x 8 = 32 vs 8 x 4 = 32).

### ModuleNotFoundError

**Problem**: `ModuleNotFoundError: No module named 'training'`

**Fix**: Ensure `sys.path.insert(0, WORK_DIR)` is at the top of the cell. This is already in every training cell.

### MLflow WinError / Permission Error

**Problem**: MLflow fails to write metrics.

**Fix**: Not applicable on Colab (Linux). This was a Windows-only issue. On Colab, MLflow writes to `/content/streamguard/mlruns/` without issues.

### val_F1 Stays at 0.0

**Possible causes**:
1. **Not enough epochs**: With warmup, the model may need 3-5 epochs to start predicting vulnerable samples. Wait until epoch 5 before worrying.
2. **Label imbalance**: vuln ratio is ~0.39 (39% vulnerable). The model may initially predict all-safe. This is expected and should correct as training progresses.
3. **Learning rate too low**: Unlikely with defaults, but can increase `lr_ggnn_fusion` to 3e-4 if stuck.

### Config D Same as Config C

**Expected for M1**. TPG edges are sparse in SARD data. The result validates that the TPG pathway doesn't degrade performance. Full TPG benefit comes in M2 with expanded taint rules.

---

## Expected Results (M1 POC)

### Realistic F1 Ranges

| Config | Expected F1 | Notes |
|--------|------------|-------|
| B: No CFA | 0.55 - 0.72 | Graph-only baseline |
| C: +CFA | 0.60 - 0.78 | CFA contrastive improves separation |
| D: +TPG | 0.60 - 0.78 | Similar to C (partial TPG) |

### Why F1 Won't Reach 0.90+

1. **No CodeBERT signal**: HDF5 has no raw code tokens. CodeBERT receives zero vectors. This is the #1 limiter in M1.
2. **SARD-only data**: Synthetic Juliet patterns are repetitive. Real-world CVE data would add diversity.
3. **No severity signal**: `lambda_sev=0.0` (no CVSS scores in SARD).

These are known M1 limitations documented in Story 8. F1 > 0.90 requires M2 features.

### Proof Threshold

**proof_positive = True** when `Config C F1 - Config B F1 >= 0.03` (3 percentage points).

This is a conservative threshold. The key evidence is:
1. CFA delta is positive
2. L_CFA decreases over training (embedding separation is happening)
3. Pairwise accuracy > 0 (model distinguishes CFA vuln/safe members)

---

## After Colab: Local Download

1. Download from Google Drive:
   - `StreamGuard/checkpoints/best_model_B_plus_ggnn.pt`
   - `StreamGuard/checkpoints/best_model_C_plus_cfa.pt`
   - `StreamGuard/checkpoints/best_model_D_plus_tpg.pt`
   - `StreamGuard/proof/cfa_proof_result.json`
   - `StreamGuard/proof/eval_*.json`
   - `StreamGuard/mlruns_m1_training.zip`

2. Place files locally:
   ```
   streamguard/
     training/checkpoints/best_model_*.pt
     proof/cfa_proof_result.json
     proof/eval_*.json
   ```

3. Verify locally:
   ```bash
   python -m training.scripts.model.ablation_cfa_vs_baseline \
       --test-h5 training/data/final/test.h5 \
       --checkpoint-dir training/checkpoints/
   ```

4. View MLflow results:
   ```bash
   # Extract mlruns_m1_training.zip to repo root
   mlflow ui --port 5000
   # Open http://localhost:5000
   ```
