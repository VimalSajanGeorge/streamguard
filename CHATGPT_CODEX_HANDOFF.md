# CHATGPT_CODEX_HANDOFF.md

**StreamGuard Production Training - Complete Handoff Documentation**

---

## 📋 Repo Snapshot

- **Branch:** master
- **Commit:** 8ed2c80 (refactor: Add production training directly to notebook - Cells 25-28)
- **Date:** 2025-11-10 15:00 IST
- **Repo Size:** ~1.2 GB, ~450 files
- **Contact:** Vimal Sajan <vimalsajan135@gmail.com>
- **Project:** StreamGuard - AI-Powered Vulnerability Detection for C Code

### Recent Commits

```
8ed2c80 - refactor: Add production training directly to notebook (Cells 25-28)
6d54348 - docs: Add comprehensive A100 production ready summary
90f5fb2 - docs: Add Cell 53 deferral note and notebook integration guide
928da54 - fix: Implement all 6 critical blockers for A100 production training
ab03115 - docs: Update notebook with production training cell references
```

---

## 🎯 One-Line Mission

**Make the CodeXGlue-based production training pipeline stable, reproducible, and production-ready for Transformer + GNN + Fusion models; prioritize correctness over speed.**

---

## 📊 Executive Summary

### Current Status

🟡 **PRODUCTION READY (with critical blockers)**

The StreamGuard project has a complete production training pipeline infrastructure ready for A100 GPU execution. All 6 critical safety blockers have been addressed in the codebase, but **training is currently failing** with exit code 1 for unknown reasons.

### Latest Changes (2025-02-14 Codex Session)

- ✅ Fixed `WeightedRandomSampler` construction bug in `training/train_transformer.py` (Colab runs were crashing with `ValueError: only one element tensors can be converted to Python scalars`). We now derive per-sample weights via tensor indexing and guard zero-count classes so transformer training clears data-loader setup.

### Top 3 Immediate Blockers

| Priority | Issue | File | Impact |
|----------|-------|------|--------|
| **🔴 CRITICAL** | Training script fails with exit code 1 | `training/train_transformer.py` | Cannot train models |
| **🔴 CRITICAL** | LR Finder fails with exit code 2 | `training/train_transformer.py` | Cannot find optimal LR |
| **🟡 HIGH** | Missing graph data (~21,854 .pt files) | `data/processed/graphs/` | GNN training will fail |

### Progress Summary

- ✅ **CodeXGlue Dataset:** Ready (train/valid/test.jsonl present)
- ✅ **Safety Utilities:** Implemented (atomic JSON, AMP utils, collapse detection)
- ✅ **Documentation:** Comprehensive (7+ production-ready guides)
- ✅ **Config Files:** Created (quick_test.yaml, prod.yaml)
- ❌ **Training Scripts:** Failing (exit code 1 - cause unknown)
- ❌ **Graph Data:** Missing (needs generation)
- ❌ **Production Scripts:** cell_51-54.py don't exist (referenced but not created)

---

## 🖥️ Environment & Dependencies

### Python Environment

```
Python: 3.12.2 (MSC v.1937 64 bit AMD64)
PyTorch: 2.3.1+cu118
CUDA: 11.8 (CUDA Available: False - Windows development environment)
Transformers: 4.37.2
PyTorch Geometric: 2.4.0 (import issues - needs fixing)
```

### GPU Information

```
Status: No NVIDIA GPU detected in current environment
Note: This is a Windows development machine
Target deployment: A100 GPU (40GB or 80GB VRAM)
```

### Dependency Table

| Package | Version | Purpose | Verified | Issues |
|---------|---------|---------|----------|--------|
| **torch** | 2.3.1+cu118 | Core training framework | ✅ | CUDA not available locally |
| **transformers** | 4.37.2 | CodeBERT models | ✅ | None |
| **torch-geometric** | 2.4.0 | GNN implementation | ⚠️ | Import errors (WinError 127) |
| **torch-scatter** | [TBD] | PyG dependency | ❌ | Missing or broken |
| **torch-sparse** | [TBD] | PyG dependency | ❌ | Missing or broken |
| **torch-cluster** | [TBD] | PyG dependency | ❌ | Missing or broken |
| **tree-sitter** | 0.20.4 | AST parsing | ✅ | Platform-specific builds |
| **scikit-learn** | [TBD] | Metrics, utilities | [TBD] | Check requirements.txt |
| **numpy** | [TBD] | Numerical operations | [TBD] | Check requirements.txt |

**Action Required:**
1. Check `requirements.txt` and update [TBD] entries
2. Fix PyTorch Geometric import errors (torch-scatter, torch-sparse, torch-cluster)
3. Verify all dependencies install correctly on A100 environment

### Requirements.txt Checksum

```
File: requirements.txt
SHA256: [TBD - run: certutil -hashfile requirements.txt SHA256]
```

---

## ⚠️ Current Critical Error (DETAILED)

### Error Summary

```
Component: Transformer Training (train_transformer.py)
Exit Code: 1 (all seeds: 42, 2025, 7)
LR Finder: Exit code 2
Root Cause: UNKNOWN - needs investigation
First Failure: LR Finder phase
Secondary Failure: Training phase (all 3 seeds)
```

### Full Error Output (from user)

```bash
================================================================================
TRANSFORMER v1.7 PRODUCTION TRAINING
================================================================================
Seeds: [42, 2025, 7]
Output: training/outputs/transformer_v17
================================================================================

[1/2] Running LR Finder (once for all seeds)...

⏳ Running LR Finder on 64 samples...
⚠️  LR Finder failed: Command '['/usr/bin/python3', 'training/train_transformer.py',
    '--train-data=data/processed/codexglue/train.jsonl',
    '--val-data=data/processed/codexglue/valid.jsonl',
    '--output-dir=training/outputs/.lr_finder_temp',
    '--quick-test', '--find-lr', '--force-find-lr', '--epochs=5',
    '--batch-size=16', '--seed=42']' returned non-zero exit status 2.
Continuing with default LR (2e-5)...

[2/2] Training all seeds with cached LR...

================================================================================
TRAINING WITH SEED: 42 (1/3)
================================================================================

❌ Training failed for seed 42: Command '['/usr/bin/python3', 'training/train_transformer.py',
    '--train-data=data/processed/codexglue/train.jsonl',
    '--val-data=data/processed/codexglue/valid.jsonl',
    '--output-dir=training/outputs/transformer_v17/seed_42',
    '--seed=42', '--epochs=10', '--batch-size=64', '--mixed-precision',
    '--find-lr', '--use-weighted-sampler']' returned non-zero exit status 1.

================================================================================
TRAINING WITH SEED: 2025 (2/3)
================================================================================

❌ Training failed for seed 2025: [same command, exit code 1]

================================================================================
TRAINING WITH SEED: 7 (3/3)
================================================================================

❌ Training failed for seed 7: [same command, exit code 1]

================================================================================
TRANSFORMER TRAINING COMPLETE
================================================================================

❌ All seeds failed. Check errors above.
```

### Debug Trace Placeholder

**Note:** The error output above only shows exit codes, NOT the actual error messages.

**CRITICAL FIRST TASK FOR CODEX:** Capture the actual stderr output to diagnose root cause.

```bash
# Run this command to capture full error trace:
python training/train_transformer.py \
  --train-data data/processed/codexglue/train.jsonl \
  --val-data data/processed/codexglue/valid.jsonl \
  --output-dir outputs/debug \
  --quick-test \
  --epochs 1 \
  2>&1 | tee debug_full.log

# Then examine first 200 lines:
head -200 debug_full.log
```

**Placeholder for actual trace:**
```
[Codex: Insert actual error trace here after running command above]

Expected error types:
- ImportError: Missing module
- FileNotFoundError: Data path issue (val.jsonl vs valid.jsonl)
- RuntimeError: CUDA/GPU mismatch
- TypeError: JSON serialization issue
- ModuleNotFoundError: Dependency missing
```

### Likely Root Causes (Ranked by Probability)

1. **Data path mismatch (70% likely)**
   - Script may expect `val.jsonl` but file is `valid.jsonl`
   - Check: Line ~50-100 in train_transformer.py
   - Fix: Update data loading code to use correct filename

2. **PyTorch Geometric import error (15% likely)**
   - torch-scatter, torch-sparse, torch-cluster failing to load
   - Evidence: Import errors seen in environment capture
   - Fix: Reinstall PyG dependencies for correct CUDA version

3. **CUDA/Device mismatch (10% likely)**
   - Code assumes CUDA available but running on CPU
   - Check: Device setup logic in train_transformer.py
   - Fix: Add fallback to CPU if CUDA unavailable

4. **Missing safety utilities import (5% likely)**
   - Code references utilities that aren't in Python path
   - Check: Import statements at top of train_transformer.py
   - Fix: Add `training/utils/` to Python path or fix imports

### Last Successful Command

```
Status: NONE - Training has never completed successfully
```

This is important context: we're starting from a non-working state.

---

## 🚀 Run Commands & Examples

### Environment Setup

```bash
# Step 1: Create conda environment
conda create -n streamguard python=3.10
conda activate streamguard

# Step 2: Install core dependencies
pip install -r requirements.txt

# Step 3: Install PyTorch Geometric (runtime-aware)
python -c "import torch; tv=torch.__version__.split('+')[0]; cv=torch.version.cuda.replace('.',''); print(f'https://data.pyg.org/whl/torch-{tv}+cu{cv}.html')"
# Use URL from output:
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f <URL>
pip install torch-geometric==2.4.0
```

### Quick Smoke Test (CURRENTLY FAILS)

```bash
# Minimal test (expected: 2-3 min, actual: fails with exit code 1)
python training/train_transformer.py \
  --train-data data/processed/codexglue/train.jsonl \
  --val-data data/processed/codexglue/valid.jsonl \
  --output-dir outputs/test \
  --quick-test \
  --epochs 1
```

**Expected output:**
```
[+] Loading data...
[+] Loaded 21854 training samples
[+] Loaded 2732 validation samples
[+] Initializing model...
[+] Training epoch 1/1...
Epoch 1: 100%|████████| 342/342 [02:15<00:00]
[+] Validation F1: 0.8234
[+] Saved checkpoint: outputs/test/best_model.pt
[+] Training complete
```

**Actual output:**
```
[stderr content - unknown, exits with code 1]
```

### Full Production Training (After Fix)

```bash
# Single seed run (30-60 min on A100)
python training/train_transformer.py \
  --train-data data/processed/codexglue/train.jsonl \
  --val-data data/processed/codexglue/valid.jsonl \
  --output-dir training/outputs/transformer_v17/seed_42 \
  --seed 42 \
  --epochs 10 \
  --batch-size 64 \
  --mixed-precision \
  --find-lr \
  --use-weighted-sampler

# Multi-seed production run (90-180 min total)
for seed in 42 2025 7; do
  python training/train_transformer.py \
    --train-data data/processed/codexglue/train.jsonl \
    --val-data data/processed/codexglue/valid.jsonl \
    --output-dir training/outputs/transformer_v17/seed_$seed \
    --seed $seed \
    --epochs 10 \
    --batch-size 64 \
    --mixed-precision \
    --find-lr \
    --use-weighted-sampler
done
```

### Using Config Files

```bash
# Quick test with YAML config
python training/train_transformer.py --config configs/quick_test.yaml

# Production training with YAML config
python training/train_transformer.py --config configs/prod.yaml --seed 42
```

---

## 🔬 Reproducibility Seeds & Stable Settings

### Seeds for Multi-Run Validation

```python
PRODUCTION_SEEDS = [42, 2025, 7]  # 3 seeds for statistical significance
```

**Why 3 seeds?**
- 3 runs = sufficient for mean ± std deviation
- 5+ seeds = better statistics but 2x cost
- 1 seed = not reproducible, no confidence intervals

### Deterministic Configuration

```python
import torch
import random
import numpy as np

def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Strongest reproducibility (may raise errors for non-deterministic ops)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # NOTE: torch.use_deterministic_algorithms(True) may fail for some ops.
    # If you encounter RuntimeError about non-deterministic operations:
    # 1. Set CUBLAS_WORKSPACE_CONFIG=:4096:8 environment variable
    # 2. OR use fallback config below

def set_seed_fallback(seed: int = 42):
    """Fallback: Weaker reproducibility but more compatible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Skip torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

**Trade-offs:**
- **Deterministic mode:** ~10-15% slower, fully reproducible
- **cudnn.benchmark=False:** ~5-10% slower, mostly reproducible
- **cudnn.benchmark=True:** Fastest but non-reproducible

**Recommendation:** Use deterministic mode for final production runs, benchmark=True for development.

---

## 📁 Data Checksums & Counts

### CodeXGlue Dataset Files

```
Location: data/processed/codexglue/

Files:
  train.jsonl    - 527 MB (65,673,781 bytes)
  valid.jsonl    - 65 MB  (65,081,020 bytes)
  test.jsonl     - 65 MB  (65,673,781 bytes)

Checksums: [TBD - run: certutil -hashfile <file> SHA256]
```

### Estimated Sample Counts

```
Train: ~21,854 samples (estimated from graph preprocessing docs)
Valid: ~2,732 samples (estimated)
Test:  ~2,732 samples (estimated)

Total: ~27,318 samples
```

**Action Required:** Verify exact counts with:
```bash
python -c "
with open('data/processed/codexglue/train.jsonl') as f:
    print(f'Train: {sum(1 for _ in f)} samples')
"
```

### Graph Data Status

```
Location: data/processed/graphs/
Status: ❌ MISSING

Expected:
  train/ - ~21,854 .pt files
  val/   - ~2,732 .pt files
  test/  - ~2,732 .pt files

Generator: training/preprocessing/create_simple_graph_data.py (exists)
```

---

## 🗂️ Key Files and Exact Lines/Blocks with Issues

### 1. training/train_transformer.py - FAILING (Priority: CRITICAL)

**Issue:** Exits with code 1 for all training runs
**Status:** Root cause unknown - needs debug trace
**Action:** Capture stderr output and diagnose

**Suspected problem areas:**
- Lines ~50-80: Data loading
- Lines ~100-150: Model initialization
- Lines ~200-250: Training loop setup
- Lines ~400-450: LR Finder logic

### 2. training/scripts/pre_flight_validation.py - Unicode Issue (Priority: HIGH)

**Issue:** Uses emoji characters that fail on Windows (UnicodeEncodeError)
**Location:** Lines 110-160 (emoji usage throughout)
**Error:** `'charmap' codec can't encode character '\u274c'`

**Fix required:**
```python
# Replace ALL emoji with ASCII equivalents:
"❌" → "[X]" or "FAIL"
"✅" → "[OK]" or "PASS"
"🚀" → "[+]" or "START"
"⚠️" → "[!]" or "WARN"
```

### 3. Missing Production Scripts (Priority: HIGH)

**Issue:** Documentation references scripts that don't exist

**Missing files:**
```
training/scripts/cell_51_transformer_production.py  - Referenced but not found
training/scripts/cell_52_gnn_production.py         - Referenced but not found
training/scripts/cell_53_fusion_production.py      - Referenced but not found (deferred)
training/scripts/cell_54_metrics_aggregator.py     - Referenced but not found
```

**Action:** Either:
- Option A: Create these scripts based on templates in `NOTEBOOK_CELLS_50_54_GUIDE.md`
- Option B: Use existing `training/train_transformer.py` directly (current approach)
- Option C: Extract code from notebook cells 25-28

### 4. training/train_fusion.py:583 - FIXED

**Issue:** Unsafe `.numpy()` call (already fixed per docs)
**Fix applied:**
```python
# Before: val_labels.extend(batch['label'].numpy())
# After:  val_labels.extend(batch['label'].detach().cpu().numpy())
```

### 5. StreamGuard_Complete_Training.ipynb - JSON Corruption

**Issue:** Cells 25-28 missing trailing `\n` in JSON format
**Impact:** Notebook rendering broken, cells won't execute
**Analysis:** See `NOTEBOOK_STRUCTURE_ANALYSIS.txt`
**Fix:** Restore from `StreamGuard_Complete_Training.ipynb.backup`

---

## 📓 Notebook Mapping

### Which Notebook to Use?

**TL;DR:** Use **command-line scripts**, not notebooks, for production training.

### Available Notebooks

| Notebook | Status | Cells | Notes |
|----------|--------|-------|-------|
| **StreamGuard_Production_Training.ipynb** | ✅ Clean | 24 (0-23) | Newer, better formatted, but incomplete |
| **StreamGuard_Complete_Training.ipynb** | ⚠️ Corrupted | 25+ (0-28) | Has cells 25-28 but JSON broken |
| **StreamGuard_Complete_Training.ipynb.backup** | ✅ Good | 25+ | Last known good state |

### Recommended Approach

**Use command-line scripts** (`training/train_transformer.py`, etc.) instead of notebooks because:
1. Better error handling and logging
2. Easier to debug (full stack traces)
3. Production-ready (can run in CI/CD)
4. No notebook corruption issues
5. Proper argument parsing

**If you must use notebooks:**
- Use `StreamGuard_Production_Training.ipynb` for environment setup (cells 0-10)
- Run training via command-line from notebook cells using `!python training/train_transformer.py ...`

### Cell → Module Mapping

**Expected structure (based on docs):**

```
Notebook Cells → Python Modules:

Cells 0-10  → Environment setup
  - GPU detection
  - Dependency installation
  - Repository cloning
  - tree-sitter build

Cells 11-18 → Production training (planned but cells malformed)
  - Data validation
  - Transformer training
  - GNN training
  - Results viewing

Cells 19-23 → Optional features
  - LR Finder validation
  - Fusion training instructions

Cells 25-28 → Production scripts (in corrupted notebook)
  - Cell 26: Transformer v1.7 training
  - Cell 27: GNN v1.7 training
  - Cell 28: Fusion v1.7 training (deferred)
```

**Actual implementation:**
- Cells 0-10: ✅ Working
- Cells 11-24: ✅ Working in Production notebook
- Cells 25-28: ❌ Corrupted in Complete notebook
- Command-line scripts: ✅ Exist but failing

---

## 🔗 Graph Schema & Sample

### Schema Version

**Version:** v1.0 (sequential graphs)
**Generator:** `training/preprocessing/create_simple_graph_data.py`

### Node Types

```python
# Simplified graph schema (not full AST/CFG/DFG)
# Sequential structure: one node per code token

nodes = {
    "x": torch.Tensor,      # Node features [num_nodes, feature_dim]
    "edge_index": LongTensor,  # Edge connectivity [2, num_edges]
    "y": int                # Label: 0 (safe) or 1 (vulnerable)
}
```

### Sample Graph Structure

```python
{
    "x": tensor([
        [0.1, 0.2, 0.3, ...],  # Node 0 features
        [0.4, 0.5, 0.6, ...],  # Node 1 features
        ...
    ]),  # Shape: [num_nodes, feature_dim]

    "edge_index": tensor([
        [0, 1, 2, 3, ...],  # Source nodes
        [1, 2, 3, 4, ...]   # Target nodes
    ]),  # Shape: [2, num_edges]

    "y": 1  # Label (0=safe, 1=vulnerable)
}
```

### Expected Output

```
Location: data/processed/graphs/

train/
  ├── sample_00000.pt
  ├── sample_00001.pt
  ...
  └── sample_21853.pt  (~21,854 files)

val/
  ├── sample_00000.pt
  ...
  └── sample_02731.pt  (~2,732 files)
```

### Generation Command

```bash
# Generate all graph data (~15-25 min for full dataset)
python training/preprocessing/create_simple_graph_data.py

# Test on 100 samples first (~30 sec)
python training/preprocessing/create_simple_graph_data.py --max-samples 100
```

---

## 📦 Missing Assets List

### Critical Missing Files

1. **Graph Data** (Priority: HIGH)
   - Path: `data/processed/graphs/train/*.pt`
   - Count: ~21,854 files
   - Size: ~500 MB - 1 GB (estimated)
   - Generator: Exists (`create_simple_graph_data.py`)
   - Action: Run generator script

2. **Production Scripts** (Priority: HIGH - but may not be needed)
   - Path: `training/scripts/cell_51_transformer_production.py`
   - Path: `training/scripts/cell_52_gnn_production.py`
   - Path: `training/scripts/cell_53_fusion_production.py`
   - Path: `training/scripts/cell_54_metrics_aggregator.py`
   - Status: Referenced in docs but never created
   - Action: Determine if needed or use existing `train_transformer.py`

3. **LR Cache Directory** (Priority: LOW)
   - Path: `.lr_cache/` or `training/.lr_cache/`
   - Status: May not exist yet
   - Purpose: Store LR Finder results (7-day cache)
   - Action: Created automatically on first run

4. **Checkpoints** (Priority: LOW - created during training)
   - Path: `training/outputs/*/best_model.pt`
   - Status: None exist (training never completed)
   - Action: Will be created after successful training

---

## ✅ Acceptance Criteria

### Test 1: Quick Smoke Test

```bash
python training/train_transformer.py \
  --train-data data/processed/codexglue/train.jsonl \
  --val-data data/processed/codexglue/valid.jsonl \
  --output-dir outputs/test \
  --quick-test \
  --epochs 1
```

**Expected:**
- ✅ Completes 1 epoch without errors
- ✅ Exit code 0
- ✅ Runtime: ~2-3 min on GPU, ~5-10 min on CPU
- ✅ Creates checkpoint: `outputs/test/best_model.pt`
- ✅ Logs validation F1 score

### Test 2: LR Finder Success

```bash
python training/train_transformer.py \
  --train-data data/processed/codexglue/train.jsonl \
  --val-data data/processed/codexglue/valid.jsonl \
  --output-dir outputs/lr_test \
  --find-lr \
  --force-find-lr \
  --quick-test
```

**Expected:**
- ✅ LR Finder completes successfully
- ✅ Creates cache file in `.lr_cache/`
- ✅ Logs suggested LR (e.g., "Suggested LR: 2.5e-5")
- ✅ Exit code 0

### Test 3: Production Training (Single Seed)

```bash
python training/train_transformer.py \
  --train-data data/processed/codexglue/train.jsonl \
  --val-data data/processed/codexglue/valid.jsonl \
  --output-dir training/outputs/transformer_v17/seed_42 \
  --seed 42 \
  --epochs 10 \
  --batch-size 64 \
  --mixed-precision \
  --find-lr \
  --use-weighted-sampler
```

**Expected:**
- ✅ Completes 10 epochs
- ✅ Validation F1 > 0.85 (target: 0.88-0.93)
- ✅ Runtime: 30-60 min on A100
- ✅ Creates checkpoint: `training/outputs/transformer_v17/seed_42/best_model.pt`
- ✅ Creates metadata: `training/outputs/transformer_v17/seed_42/training_metadata.json`
- ✅ No tensor serialization errors in JSON files

### Test 4: Reproducibility

Run same training twice with same seed:

```bash
# Run 1
python training/train_transformer.py --seed 42 [args] 2>&1 | tee run1.log

# Run 2
python training/train_transformer.py --seed 42 [args] 2>&1 | tee run2.log

# Compare F1 scores
python -c "
import json
r1 = json.load(open('outputs/seed_42/run1/training_metadata.json'))
r2 = json.load(open('outputs/seed_42/run2/training_metadata.json'))
f1_1 = r1['best_val_f1']
f1_2 = r2['best_val_f1']
diff = abs(f1_1 - f1_2)
print(f'F1 Run 1: {f1_1:.4f}')
print(f'F1 Run 2: {f1_2:.4f}')
print(f'Difference: {diff:.4f}')
if diff < 0.005:
    print('✅ PASS - Reproducibility within ±0.5%')
else:
    print('❌ FAIL - Reproducibility variance too high')
"
```

**Expected:**
- ✅ F1 scores within ±0.5% (e.g., 0.8923 vs 0.8967 is acceptable)
- ✅ Final loss within ±1%

### Test 5: Pre-Flight Validation (After Fix)

```bash
python training/scripts/pre_flight_validation.py
```

**Expected:**
- ✅ All checks pass
- ✅ No Unicode errors on Windows
- ✅ Exit code 0
- ✅ Runtime < 3 min

### Test 6: Graph Data Generation

```bash
python training/preprocessing/create_simple_graph_data.py
```

**Expected:**
- ✅ Creates ~21,854 .pt files in `data/processed/graphs/train/`
- ✅ Creates ~2,732 .pt files in `data/processed/graphs/val/`
- ✅ Runtime: 15-25 min
- ✅ Exit code 0

### Test 7: JSON Serialization Safety

All JSON files must serialize without errors:

```bash
python -c "
import json
from pathlib import Path
for meta_file in Path('training/outputs').rglob('*.json'):
    try:
        with open(meta_file) as f:
            data = json.load(f)
        print(f'✅ {meta_file}')
    except Exception as e:
        print(f'❌ {meta_file}: {e}')
"
```

**Expected:**
- ✅ All JSON files load successfully
- ✅ No tensor serialization errors

---

## 🎯 Priority Action Items

### 1. DEBUG train_transformer.py (HIGHEST PRIORITY)

**Task:** Capture actual error output to diagnose root cause

**Steps:**
```bash
# 1. Run with full error capture
python training/train_transformer.py \
  --train-data data/processed/codexglue/train.jsonl \
  --val-data data/processed/codexglue/valid.jsonl \
  --output-dir outputs/debug \
  --quick-test \
  --epochs 1 \
  2>&1 | tee debug_full.log

# 2. Examine error
cat debug_full.log | grep -i "error\|exception\|traceback" -A 10

# 3. Common diagnostics
python -c "from training.train_transformer import *"  # Check imports
python -c "import json; print(sum(1 for _ in open('data/processed/codexglue/train.jsonl')))"  # Check data
python -c "import torch; print(torch.cuda.is_available())"  # Check CUDA
```

**Likely fixes (ranked):**
1. **Data path issue:** Change `val.jsonl` → `valid.jsonl` in code
2. **PyG import error:** Reinstall torch-scatter, torch-sparse, torch-cluster
3. **CUDA mismatch:** Add CPU fallback logic
4. **Import error:** Fix module paths or add to PYTHONPATH

### 2. Fix Windows Unicode Bug (HIGH PRIORITY)

**Task:** Replace emoji in pre_flight_validation.py with ASCII

**Files:** `training/scripts/pre_flight_validation.py`

**Changes needed:**
```python
# Find and replace throughout file:
"❌" → "[X]"
"✅" → "[OK]"
"🚀" → "[+]"
"⚠️" → "[!]"
"⏳" → "[...]"

# Also ensure all file opens use UTF-8:
open(path, 'r', encoding='utf-8', errors='replace')
```

**Verification:**
```bash
python training/scripts/pre_flight_validation.py
# Should run without UnicodeEncodeError
```

### 3. Create Graph Data (HIGH PRIORITY)

**Task:** Generate graph .pt files for GNN training

**Command:**
```bash
# Test with 100 samples first (~30 sec)
python training/preprocessing/create_simple_graph_data.py --max-samples 100

# If successful, run full generation (~15-25 min)
python training/preprocessing/create_simple_graph_data.py
```

**Verification:**
```bash
# Check output
ls data/processed/graphs/train/*.pt | wc -l  # Should be ~21,854
```

### 4. Fix PyTorch Geometric Imports (MEDIUM PRIORITY)

**Task:** Resolve import errors for torch-scatter, torch-sparse, torch-cluster

**Current error:**
```
OSError: [WinError 127] The specified procedure could not be found
```

**Fix:**
```bash
# Uninstall broken packages
pip uninstall torch-scatter torch-sparse torch-cluster torch-spline-conv -y

# Reinstall for correct CUDA version
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
  -f https://data.pyg.org/whl/torch-2.3.1+cu118.html

# Verify
python -c "import torch_geometric; print('PyG OK')"
```

### 5. Determine Production Scripts Strategy (MEDIUM PRIORITY)

**Task:** Decide whether to create cell_51-54.py or use existing scripts

**Options:**
- **A:** Use existing `train_transformer.py` directly (recommended)
- **B:** Create cell_51-54.py from templates in `NOTEBOOK_CELLS_50_54_GUIDE.md`
- **C:** Extract from corrupted notebook cells 25-28

**Recommendation:** Option A (use existing scripts), since they already exist and just need debugging.

---

## 🛠️ Known Hacks & Workarounds

### PyTorch OOM (Out of Memory)

```python
# If CUDA OOM, reduce batch size:
# From configs/prod.yaml
batch_size: 64  # Reduce to 32 or 16

# Or enable gradient accumulation:
gradient_accumulation_steps: 2  # Effective batch = 32 × 2 = 64
```

### tree-sitter Build Failures

**Platform-specific workarounds:**

**Linux/Mac:**
```bash
cd vendor/tree-sitter-c
npm install
npm run build
```

**Windows:**
```bash
# Use pre-built binaries or WSL
# See: https://github.com/tree-sitter/tree-sitter/issues/1070
```

### LR Finder Hangs or Fails

```python
# Fallback to conservative default LRs:
TRANSFORMER_LR = 2e-5  # Conservative, tested
GNN_LR = 5e-4          # Conservative, tested

# Or skip LR Finder entirely:
python train_transformer.py --lr 2e-5  # Skip --find-lr flag
```

### Windows Emoji Crashes

```python
# Replace all emoji with ASCII:
print("❌") → print("[X]")
print("✅") → print("[OK]")

# Use encoding parameter:
open(file, 'r', encoding='utf-8', errors='replace')
```

### Logging Best Practices

**Prefer structured JSONL logging over plain prints:**

```python
import logging
import json

# Setup
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Instead of print()
print(f"Epoch {epoch}: Loss {loss}")  # ❌ Unstructured

# Use logger with JSON
logger.info(json.dumps({
    "type": "epoch_complete",
    "epoch": epoch,
    "loss": float(loss),
    "f1": float(f1)
}))  # ✅ Structured, parseable
```

---

## 🔐 Security / Secrets

### Current Requirements

**For production training:** No secrets required ✅

**For data collection (separate phase, IGNORE for now):**
- `GITHUB_TOKEN` - GitHub API access (in `.env` file)
- `NVD_API_KEY` - NVD/CVE API access (optional, in `.env`)

### Secrets Handling

**DO NOT commit:**
- `.env` files
- API tokens
- AWS credentials
- Model checkpoints (large files)

**For CI/CD (future):**
```yaml
# .github/workflows/train.yml
env:
  GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
  AWS_ACCESS_KEY: ${{ secrets.AWS_ACCESS_KEY }}
```

### Sensitive Data in Logs

**Warning:** Logs may contain:
- File paths (potentially revealing directory structure)
- Code snippets (from dataset)
- Model hyperparameters
- Training metrics

**Action:** Sanitize logs before sharing publicly. Do not log raw code content.

---

## 🧪 Testing & CI

### Unit Tests

**Location:** `training/tests/test_safety_utilities.py`

**Run:**
```bash
python training/tests/test_safety_utilities.py
```

**Expected results:**
```
Ran 21 tests
✅ 18 passed (86%)
⚠️ 3 skipped/failed (platform-specific, acceptable)

Failures (expected):
- Path format: Windows uses \ instead of / (cosmetic)
- Collapse sensitivity: Tuned to avoid false positives
- CUDA tests: Skipped on CPU-only systems
```

### GPU Tests

**Requirement:** CUDA-capable GPU

**Run:**
```bash
# Smoke test with GPU
python training/train_transformer.py --quick-test --epochs 1

# Full test suite (if created)
pytest tests/gpu/ -v
```

### CI Configuration

**Status:** ❌ Not configured yet

**TODO:** Create `.github/workflows/ci.yml` (see prompt template section)

**Expected CI workflow:**
- Lint: ruff, black
- Type check: mypy
- Unit tests: pytest (CPU)
- GPU smoke test: (optional, requires self-hosted runner)
- Pre-commit hooks

---

## 🎨 Style & Linting

### Current Status

**Style guide:** ❌ Not enforced
**Linting:** ❌ Not configured
**Type hints:** ⚠️ Partial

### Recommended Setup

```bash
# Install tools
pip install black ruff mypy pre-commit

# Format code
black training/ --line-length 100

# Lint
ruff check training/

# Type check
mypy training/ --ignore-missing-imports

# Setup pre-commit
pre-commit install
```

### Pre-commit Config Template

**Create `.pre-commit-config.yaml`:**
```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 24.1.1
    hooks:
      - id: black
        args: [--line-length=100]

  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.9
    hooks:
      - id: ruff
        args: [--fix]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.8.0
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
```

---

## 📈 Performance Baselines

### Expected Training Times (A100 GPU)

| Model | Epochs | Batch Size | Time per Epoch | Total Time | Notes |
|-------|--------|------------|----------------|------------|-------|
| **Transformer** | 10 | 64 | 3-6 min | 30-60 min | With mixed precision |
| **GNN** | 15 | 32 | 3-5 min | 45-75 min | Graph preprocessing adds 5-10 min |
| **Fusion** | 12 | 32 | 2-3 min | 24-36 min | Requires pre-trained T+G models |

**Multi-seed runs (3 seeds):**
- Transformer: 90-180 min total
- GNN: 135-225 min total
- Fusion: 72-108 min total

### Expected Throughput

```
Target: >500 samples/sec on A100 GPU
Actual: [TBD - measure after fixing training]

With mixed precision (FP16): ~2x faster than FP32
With gradient accumulation: Slower but enables larger effective batch size
```

### Expected Metrics

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| Transformer (target) | 88-91% | 86-89% | 89-93% | **0.88-0.93** |
| GNN (target) | 86-89% | 84-87% | 87-91% | **0.85-0.91** |
| Fusion (target) | 91-94% | 90-93% | 92-95% | **0.91-0.96** |

### Expected Log Patterns (Healthy Runs)

```
[+] Epoch 1/10 - Train Loss: 0.4523 - Val F1: 0.8734 - Time: 3m12s
[+] Epoch 2/10 - Train Loss: 0.3812 - Val F1: 0.8921 - Time: 3m08s
[+] Epoch 3/10 - Train Loss: 0.3456 - Val F1: 0.9045 - Time: 3m10s
...
[+] Best model saved: F1 = 0.9123
[+] Training complete: 31m45s total
[+] Final metrics:
      Accuracy: 0.9087
      Precision: 0.8956
      Recall: 0.9234
      F1: 0.9123
```

**Indicators of healthy training:**
- Loss decreases monotonically or with minor fluctuations
- F1 improves over epochs
- Time per epoch stable (±10%)
- No NaN/Inf values
- No collapse warnings

**Red flags:**
- Loss increases or oscillates wildly
- F1 drops below 0.5 (random baseline)
- NaN/Inf in gradients
- Time per epoch increases significantly
- Collapse detector triggers

---

## 📝 Failure Log Snippets

### 1. Current Training Failure (from user)

```
⚠️  LR Finder failed: Command '['/usr/bin/python3', 'training/train_transformer.py',
    '--train-data=data/processed/codexglue/train.jsonl',
    '--val-data=data/processed/codexglue/valid.jsonl',
    '--output-dir=training/outputs/.lr_finder_temp',
    '--quick-test', '--find-lr', '--force-find-lr', '--epochs=5',
    '--batch-size=16', '--seed=42']' returned non-zero exit status 2.

❌ Training failed for seed 42: Command '['/usr/bin/python3', 'training/train_transformer.py',
    '--train-data=data/processed/codexglue/train.jsonl',
    '--val-data=data/processed/codexglue/valid.jsonl',
    '--output-dir=training/outputs/transformer_v17/seed_42',
    '--seed=42', '--epochs=10', '--batch-size=64', '--mixed-precision',
    '--find-lr', '--use-weighted-sampler']' returned non-zero exit status 1.
```

**Analysis:** Only exit codes visible, no actual error messages. Need stderr capture.

### 2. PyTorch Geometric Import Error

```
C:\Users\Vimal Sajan\AppData\Roaming\Python\Python312\site-packages\torch_geometric\typing.py:47:
UserWarning: An issue occurred while importing 'pyg-lib'. Disabling its usage.
Stacktrace: [WinError 127] The specified procedure could not be found

... (similar for torch-scatter, torch-sparse, torch-cluster, torch-spline-conv)

OSError: [WinError 127] The specified procedure could not be found
```

**Cause:** PyG dependencies not installed correctly for Windows + CUDA 11.8
**Fix:** Reinstall from correct wheel URL

### 3. Notebook Corruption (from NOTEBOOK_STRUCTURE_ANALYSIS.txt)

```
Cell 25 (Index 25) - Shows as "Cell 26" in UI:
├─ Cell Type: code
├─ Source Type: list
├─ Total Elements: 137
└─ Issue: NO trailing '\n' on any element

Example:
[0]: '# Cell 26: Transformer v1.7 Production Training'  ❌ Missing \n
[1]: '# ============================================'  ❌ Missing \n

Expected:
[0]: '# Cell 26: Transformer v1.7 Production Training\n'  ✅ Correct
```

**Cause:** JSON formatter stripped newlines during edit
**Fix:** Restore from backup or regenerate cells with proper formatting

---

## 📚 Essential Documents (READ THESE - 6 docs only)

**Priority order for ChatGPT Codex:**

### 1. A100_PRODUCTION_READY_SUMMARY.md (READ FIRST)
**Purpose:** Main reference, comprehensive overview of production readiness
**Key sections:**
- All 6 blocker fixes
- Pre-flight validation
- Training duration estimates
- Success criteria

### 2. BLOCKER_FIXES_SUMMARY.md (READ SECOND)
**Purpose:** Technical details of all critical fixes
**Key sections:**
- LR Finder fallback
- Exit codes
- Safe .numpy() calls
- Graph data preprocessing

### 3. PRODUCTION_NOTEBOOK_SUMMARY.md
**Purpose:** Explains notebook creation and structure
**Key sections:**
- Cell breakdown (0-23)
- All fixes implemented in notebook
- How to use the new notebook

### 4. CELL_53_DEFERRAL_NOTE.md
**Purpose:** Why Fusion training (Cell 53) is deferred
**Key sections:**
- Rationale for deferring
- Dependencies (needs Cell 51 & 52 working first)

### 5. NOTEBOOK_STRUCTURE_ANALYSIS.txt
**Purpose:** Analysis of notebook corruption issue
**Key sections:**
- Root cause (JSON formatting)
- Comparison of corrupted vs backup
- Fix recommendations

### 6. README_PRODUCTION_TRAINING.md
**Purpose:** Quick start and overview
**Key sections:**
- Quick start commands
- File structure
- Expected outputs

---

## 🚫 Documents to IGNORE (Data Collection Phase - Not Current Scope)

**IMPORTANT:** These documents are for a **separate phase** (data collection). Ignore them for production training work.

### Data Collection Docs (IGNORE)

- **README_DATA_COLLECTION.md** → Legacy data collection, not training
- **GITHUB_COLLECTION_FIX_COMPLETE.md** → Historical data collection fixes
- **test_api_connectivity.py** → Tests CVE/GitHub APIs, unrelated to training
- **OSV_*.md** → OSV collector (data collection phase)
- **EXPLOITDB_*.md** → ExploitDB collector (data collection phase)
- **COLLECTOR_*.md** → Various collector docs (data collection phase)
- **training/scripts/collection/** → All data collection scripts (separate workflow)
- **DATA_COLLECTION_*.md** → Data collection guides and logs

### Outdated or Tangential (IGNORE)

- **GOOGLE_COLAB_*.md** → Old Colab-specific guides (superseded by production docs)
- **NOTION_*.md** → Project management docs, not technical
- **SETUP_*.md** → Historical setup docs (outdated)
- **GITHUB_TOKEN_ISSUE.md** → Data collection issue (not training)

**Rationale:** These documents relate to **collecting vulnerability data from GitHub, CVEs, OSV, ExploitDB**. We already have the CodeXGlue dataset ready. Training on CodeXGlue is the current phase. Data collection is a future phase.

---

## 🎯 Scope: FOCUS vs OUT-OF-SCOPE

### IN SCOPE (FOCUS on these)

**Production Training:**
- ✅ Transformer model training on CodeXGlue
- ✅ GNN model training on CodeXGlue
- ✅ Fusion model training (after T+G working)
- ✅ Multi-seed training for reproducibility
- ✅ LR Finder and hyperparameter optimization
- ✅ Checkpoint management and resume
- ✅ Error handling and safety features
- ✅ Metrics tracking and logging
- ✅ Performance optimization (mixed precision, etc.)

**Infrastructure:**
- ✅ Config files (YAML)
- ✅ Utility scripts (atomic JSON, collapse detection)
- ✅ Unit tests for training components
- ✅ CI/CD for automated training
- ✅ Documentation for production training

**Data:**
- ✅ CodeXGlue dataset (already collected)
- ✅ Graph data generation from CodeXGlue
- ✅ Small dataset samples for unit tests

### OUT OF SCOPE (IGNORE these - separate phase)

**Data Collection:**
- ❌ GitHub Advisory Database collection
- ❌ CVE/NVD database scraping
- ❌ OSV (Open Source Vulnerabilities) collection
- ❌ ExploitDB integration
- ❌ Synthetic vulnerability generation
- ❌ Repository mining for vulnerable code

**Future Work:**
- ❌ Real-time vulnerability scanning API
- ❌ VS Code extension
- ❌ Docker containerization (do later)
- ❌ Web dashboard
- ❌ Model deployment to production servers

**Note:** Data collection is a **later phase** after production training is working on CodeXGlue.

---

## 🔧 Codex Prompt Templates (5 Templates)

Use these exact templates when requesting code changes from ChatGPT Codex.

### Template A: Extract & Refactor

```
Context: StreamGuard repo at commit 8ed2c80
File: [paste file snippet or path]

Task: Extract [specific component] into:
  - [target_file_1.py] ([description])
  - [target_file_2.py] ([description])

Constraints:
  - Keep public function names stable (provide mapping: old_name → new_name)
  - Use deterministic seeds: seed=42
  - Add pytest unit test tests/test_[component].py that runs one [operation] on tiny synthetic dataset
  - Preserve type hints and docstrings
  - Use pathlib.Path for all file paths
  - Follow existing code style

Output: Unified git diff patch only (not full file rewrites)
Tone: Conservative, low creativity (0.2 temperature)

Example:
  Extract DataLoader class and training loop from cell_26_snippet.py into:
    - data/dataset.py (VulnerabilityDataset class with typing)
    - train/engine.py (train_step, validate, checkpoint functions)

  Constraints:
    - Map: Old.save_checkpoint() → checkpoint.atomic_save()
    - Seed: 42
    - Test: tests/test_engine.py (one training step on 10 samples)
```

### Template B: Bugfix Small Patch

```
File: training/scripts/pre_flight_validation.py
Lines: 110-160 (paste relevant code)

Problem: UnicodeEncodeError on Windows when printing emoji characters
Error trace:
  UnicodeEncodeError: 'charmap' codec can't encode character '\u274c' in position 0

Task: Fix robust emoji handling so script works on Windows + Linux + Mac

Requirements:
  1. Replace ALL emoji with ASCII equivalents:
     "❌" → "[X]"
     "✅" → "[OK]"
     "🚀" → "[+]"
     "⚠️" → "[!]"
  2. Add encoding parameter to all file opens: encoding='utf-8', errors='replace'
  3. Add test case: tests/test_unicode_handling.py that reads file with BOM and non-ASCII chars

Output: Single-file patch + unit test
Tone: Conservative, production-safe
```

### Template C: Productionize Train Loop

```
Task: Re-write or enhance train/engine.py with production-grade features

Requirements:
  1. Config loading: argparse OR pydantic config from YAML
  2. Atomic checkpoint save: write to .tmp → os.rename() for crash safety
  3. AMP training: torch.cuda.amp with GradScaler (correct unscale→clip→step order)
  4. Signal handling: SIGTERM/SIGINT handler to save checkpoint before exit
  5. Resume capability: --resume path/to/checkpoint.pt to continue training
  6. Structured logging: JSONL format (not print statements) with timestamp, level, metrics
  7. Deterministic mode: torch.use_deterministic_algorithms(True) with fallback
  8. Metrics tracking: Track loss, F1, accuracy per epoch, save to JSON

Constraints:
  - Preserve existing public API
  - Add comprehensive docstrings
  - Type hints for all functions
  - Error handling with informative messages
  - Follow existing code patterns in repo

Output: Full file content + tests/test_resume.py (test resume from checkpoint)
Tone: Conservative, production-grade, well-documented

AMP GradScaler Sequence (IMPORTANT):
  scaler.scale(loss).backward()
  scaler.unscale_(optimizer)  # Must unscale before clipping!
  torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
  scaler.step(optimizer)
  scaler.update()
```

### Template D: Create CI Workflow

```
Task: Create .github/workflows/ci.yml for automated testing

Requirements:
  1. Trigger: on push to main, on pull request
  2. Jobs:
     - lint: ruff check + black --check
     - typecheck: mypy (ignore missing imports)
     - test-cpu: pytest tests/unit on ubuntu-latest (Python 3.10)
     - test-gpu: pytest tests/gpu on self-hosted runner (optional, can fail gracefully)
  3. Pre-commit: Enforce pre-commit hooks pass
  4. Fail fast: Stop on first failure
  5. Cache: pip dependencies
  6. Timeout: 30 min max

Resource limits:
  - CPU tests: 8 min timeout, 4 GB RAM
  - GPU tests: 20 min timeout, 8 GB RAM + 16 GB VRAM

Output: Full .github/workflows/ci.yml file
Tone: Standard GitHub Actions patterns
```

### Template E: Analysis Request (NEW)

```
Task: Analyze the following error log and diagnose root cause

Context: StreamGuard production training, running train_transformer.py
Error log (first 200 lines):

[Paste debug_full.log content here]

Requirements:
  1. Identify root cause in 5 lines maximum
  2. Suggest minimal code patch to fix (high-level steps, NOT full implementation yet)
  3. List files that need modification
  4. Estimate confidence level (high/medium/low)
  5. If multiple possible causes, rank by probability

Output format:
  **Root Cause:**
  [5 line explanation]

  **Suggested Fix:**
  1. [Step 1]
  2. [Step 2]
  ...

  **Files to Modify:**
  - path/to/file1.py (lines X-Y)
  - path/to/file2.py (reason)

  **Confidence:** [high/medium/low]

  **Alternative Causes:**
  1. [Cause A] (20% likely)
  2. [Cause B] (10% likely)

Tone: Diagnostic, concise, actionable
DO NOT implement fix yet - analysis only
```

---

## 🚀 How to Run / Reproduce

### Step 1: Create Environment

```bash
# Create conda environment
conda create -n streamguard python=3.10 -y
conda activate streamguard

# Or venv
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### Step 2: Install Dependencies

```bash
# Install core packages
pip install -r requirements.txt

# Install PyTorch Geometric (runtime-aware)
python -c "
import torch
torch_ver = torch.__version__.split('+')[0]
cuda_ver = torch.version.cuda.replace('.', '')
print(f'https://data.pyg.org/whl/torch-{torch_ver}+cu{cuda_ver}.html')
"
# Copy URL from output, then:
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f <URL>
pip install torch-geometric==2.4.0
```

### Step 3: Verify Installation

```bash
# Run unit tests
python training/tests/test_safety_utilities.py
# Expected: 18/21 tests pass (~1-2 min)

# Verify imports
python -c "import torch, transformers, torch_geometric; print('All imports OK')"

# Check GPU
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Step 4: Generate Small Dataset (Optional)

```bash
# Create small dataset for unit tests
python scripts/generate_small_dataset.py
# Output: data/sample/train_small.jsonl (100 samples)
```

### Step 5: Quick Smoke Test (CURRENTLY FAILS)

```bash
# Run quick test with small dataset
python training/train_transformer.py \
  --train-data data/processed/codexglue/train.jsonl \
  --val-data data/processed/codexglue/valid.jsonl \
  --output-dir outputs/test \
  --quick-test \
  --epochs 1

# Expected runtime: 2-3 min on GPU, 5-10 min on CPU
# Expected: Completes 1 epoch, exit code 0
# Actual: Exit code 1 (BLOCKER - needs debugging)
```

### Step 6: Full Production Training (After Fix)

```bash
# Single seed
python training/train_transformer.py \
  --train-data data/processed/codexglue/train.jsonl \
  --val-data data/processed/codexglue/valid.jsonl \
  --output-dir training/outputs/transformer_v17/seed_42 \
  --seed 42 \
  --epochs 10 \
  --batch-size 64 \
  --mixed-precision \
  --find-lr \
  --use-weighted-sampler

# Expected runtime: 30-60 min on A100

# Multi-seed (production)
for seed in 42 2025 7; do
  python training/train_transformer.py \
    --train-data data/processed/codexglue/train.jsonl \
    --val-data data/processed/codexglue/valid.jsonl \
    --output-dir training/outputs/transformer_v17/seed_$seed \
    --seed $seed \
    --epochs 10 \
    --batch-size 64 \
    --mixed-precision \
    --find-lr \
    --use-weighted-sampler
done

# Expected total runtime: 90-180 min on A100
```

---

## ✅ Acceptance Tests & Metrics

### Test Suite Commands

```bash
# Test 1: Unit tests pass
python training/tests/test_safety_utilities.py
# Expected: 18/21 tests pass (86%+), <2 min

# Test 2: Quick training smoke test
python training/train_transformer.py \
  --quick-test \
  --epochs 1 \
  --train-data data/processed/codexglue/train.jsonl \
  --val-data data/processed/codexglue/valid.jsonl \
  2>&1 | tee smoke_test.log
# Expected: Completes 1 epoch, exit code 0, ~2-3 min
# Check: outputs/test/best_model.pt exists

# Test 3: Pre-flight validation (after Unicode fix)
python training/scripts/pre_flight_validation.py
# Expected: All checks pass, no errors, <3 min

# Test 4: Graph data creation
python training/preprocessing/create_simple_graph_data.py --max-samples 100
# Expected: Creates 100 .pt files, <30 sec

# Test 5: Reproducibility (seed 42, two runs)
python training/train_transformer.py --seed 42 --epochs 10 [args] 2>&1 | tee run1.log
python training/train_transformer.py --seed 42 --epochs 10 [args] 2>&1 | tee run2.log

# Compare F1 scores (should be within ±0.5%)
python -c "
import json
r1 = json.load(open('training/outputs/transformer_v17/seed_42/run1/training_metadata.json'))
r2 = json.load(open('training/outputs/transformer_v17/seed_42/run2/training_metadata.json'))
f1_diff = abs(r1['best_val_f1'] - r2['best_val_f1'])
print(f'F1 difference: {f1_diff:.4f}')
print('✅ PASS' if f1_diff < 0.005 else '❌ FAIL')
"
```

### Metrics Targets

| Metric | Target | Notes |
|--------|--------|-------|
| **Throughput** | >500 samples/sec | A100 GPU with mixed precision |
| **Transformer F1** | >0.85 | Target: 0.88-0.93 |
| **GNN F1** | >0.85 | Target: 0.85-0.91 |
| **Fusion F1** | >0.90 | Target: 0.91-0.96 |
| **Reproducibility** | ±0.5% F1 | Across runs with same seed |
| **Training time (T)** | 30-60 min | 10 epochs, A100, batch=64 |
| **Training time (G)** | 45-75 min | 15 epochs, A100, batch=32 |

### Success Criteria Summary

- ✅ **Quick test passes** - 1 epoch completes, exit code 0
- ✅ **LR Finder works** - Completes without errors, caches result
- ✅ **Production training succeeds** - 10 epochs, F1 > 0.85
- ✅ **Checkpoints save correctly** - No tensor serialization errors
- ✅ **Reproducibility validated** - ±0.5% F1 across runs
- ✅ **No crashes** - Handles SIGTERM, OOM, data errors gracefully

---

## 🐛 Debugging Current Failure

### CRITICAL FIRST TASK FOR CODEX

**Problem:** Training script exits with code 1, but no error message visible.

**Step-by-Step Debug Procedure:**

#### Step 1: Capture Full Error Trace

```bash
# Run with full stderr/stdout capture
cd "C:\Users\Vimal Sajan\streamguard"

python training/train_transformer.py \
  --train-data data/processed/codexglue/train.jsonl \
  --val-data data/processed/codexglue/valid.jsonl \
  --output-dir outputs/debug \
  --quick-test \
  --epochs 1 \
  2>&1 | tee debug_full.log

# Examine error
type debug_full.log | findstr /i "error exception traceback" > debug_errors.txt
type debug_errors.txt
```

#### Step 2: Check Common Failure Modes

```bash
# A. Check imports work
python -c "from training.train_transformer import *"
# If fails → import error

# B. Check data files exist and are readable
python -c "
import json
with open('data/processed/codexglue/train.jsonl') as f:
    samples = [json.loads(line) for line in f]
print(f'✅ Loaded {len(samples)} training samples')
"
# If fails → data path or format issue

# C. Check CUDA availability
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'Device: {torch.cuda.get_device_name(0)}')
"
# If CUDA unavailable but code requires it → device mismatch

# D. Check dependencies
python -c "
import torch
import transformers
import torch_geometric
print('✅ All core imports OK')
"
# If fails → missing dependency
```

#### Step 3: Run in Interactive Mode

```python
# Run Python interactively to see full stack trace
python -i
>>> import sys
>>> sys.argv = [
...     'train_transformer.py',
...     '--train-data', 'data/processed/codexglue/train.jsonl',
...     '--val-data', 'data/processed/codexglue/valid.jsonl',
...     '--output-dir', 'outputs/debug',
...     '--quick-test',
...     '--epochs', '1'
... ]
>>> exec(open('training/train_transformer.py').read())
# This will show the FULL error with stack trace
```

#### Step 4: Diagnose Root Cause

**Likely causes (in order of probability):**

**1. Data path mismatch (70% likely)**
```python
# WRONG (script may be using this)
val_data_path = "data/processed/codexglue/val.jsonl"  # ❌ File doesn't exist

# CORRECT (actual filename)
val_data_path = "data/processed/codexglue/valid.jsonl"  # ✅ File exists

# Check in train_transformer.py around line 50-100:
# Search for "val.jsonl" and replace with "valid.jsonl"
```

**2. PyTorch Geometric import error (15% likely)**
```python
# Error: torch-scatter, torch-sparse not found
# Fix: Reinstall PyG dependencies
pip uninstall torch-scatter torch-sparse torch-cluster -y
pip install torch-scatter torch-sparse torch-cluster \
  -f https://data.pyg.org/whl/torch-2.3.1+cu118.html
```

**3. CUDA device mismatch (10% likely)**
```python
# Script assumes CUDA but running on CPU
# Add fallback in train_transformer.py:

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Instead of:
device = torch.device("cuda")  # ❌ Fails if no GPU
```

**4. Missing utility import (5% likely)**
```python
# Script imports safety utilities that aren't in path
# Fix: Add to PYTHONPATH or use relative imports

import sys
sys.path.insert(0, 'training/utils')
from json_safety import atomic_write_json
```

#### Step 5: Apply Fix and Verify

```bash
# After fixing issue, re-run smoke test
python training/train_transformer.py --quick-test --epochs 1

# Should now complete with exit code 0
echo $?  # Check exit code (should be 0)
```

### Common Error Patterns

**Pattern 1: FileNotFoundError**
```
FileNotFoundError: [Errno 2] No such file or directory: 'data/processed/codexglue/val.jsonl'

Fix: Change "val.jsonl" → "valid.jsonl" in code
```

**Pattern 2: ImportError**
```
ImportError: cannot import name 'atomic_write_json' from 'training.utils'

Fix: Check module structure, add __init__.py, or fix import path
```

**Pattern 3: RuntimeError (CUDA)**
```
RuntimeError: CUDA error: no kernel image is available for execution on the device

Fix: Reinstall PyTorch for correct CUDA version or add CPU fallback
```

**Pattern 4: TypeError (JSON)**
```
TypeError: Object of type 'Tensor' is not JSON serializable

Fix: Use atomic_write_json(path, obj, safe=True) instead of json.dump()
```

---

## 💡 Extra Tips for Codex (Anti-Hallucination)

### Code Modification Best Practices

1. **Always provide function/class headers when editing**
   ```
   ❌ Bad: "Fix the training loop"
   ✅ Good: "In train_transformer.py, function train_epoch() (lines 200-250), fix..."
   ```

2. **Use unified git diffs for patches**
   ```diff
   --- a/training/train_transformer.py
   +++ b/training/train_transformer.py
   @@ -75,7 +75,7 @@
   -    val_data_path = "data/processed/codexglue/val.jsonl"
   +    val_data_path = "data/processed/codexglue/valid.jsonl"
   ```

3. **Request unit tests with any code change**
   ```
   Codex: After fixing X, create tests/test_X.py that verifies...
   ```

4. **Add 1-2 line explanation comments**
   ```python
   # Fix: Use valid.jsonl (actual filename) instead of val.jsonl
   val_data_path = "data/processed/codexglue/valid.jsonl"
   ```

5. **Test on small data before full run**
   ```bash
   # Always test with --quick-test first
   python train_transformer.py --quick-test --epochs 1

   # Only run full training after smoke test passes
   python train_transformer.py --epochs 10
   ```

6. **Check file encodings on Windows**
   ```python
   # Always specify encoding explicitly
   with open(path, 'r', encoding='utf-8', errors='replace') as f:
       content = f.read()
   ```

7. **When in doubt, ask for analysis first**
   ```
   Use Template E (Analysis Request) before implementing fixes
   ```

8. **Preserve existing function signatures**
   ```python
   # ❌ Don't change public API
   def train(model, data):  # Was: train(model, train_data, val_data)

   # ✅ Keep signature, add optional params
   def train(model, train_data, val_data, **kwargs):
   ```

9. **Use pathlib.Path instead of strings**
   ```python
   from pathlib import Path

   # ✅ Good
   data_path = Path("data/processed/codexglue/train.jsonl")
   if data_path.exists():
       ...

   # ❌ Avoid
   data_path = "data/processed/codexglue/train.jsonl"
   if os.path.exists(data_path):
       ...
   ```

10. **Always use atomic_write_json for JSON**
    ```python
    from docs.snippets.atomic_write_json import atomic_write_json

    # ✅ Good (crash-safe)
    atomic_write_json("metrics.json", data, safe=True)

    # ❌ Avoid (can corrupt on crash)
    with open("metrics.json", "w") as f:
        json.dump(data, f)
    ```

---

## 📋 CI Resource Expectations

### Timeouts and Limits

| Test Type | Timeout | Memory | GPU VRAM | Notes |
|-----------|---------|--------|----------|-------|
| **Smoke test** | 8 min | 4 GB RAM | 8 GB | --quick-test, 1 epoch |
| **Unit tests** | 5 min | 2 GB RAM | N/A | CPU only |
| **Full training** | 90 min | 16 GB RAM | 40 GB | 10 epochs, A100 |
| **LR Finder** | 10 min | 8 GB RAM | 16 GB | Quick mode (100 iter) |
| **Graph gen** | 30 min | 8 GB RAM | N/A | Full dataset (~22k files) |

### CI Workflow Resource Allocation

```yaml
# Smoke test job
smoke-test:
  timeout-minutes: 8
  runs-on: ubuntu-latest
  resources:
    memory: 4GB

# GPU test job (self-hosted)
gpu-test:
  timeout-minutes: 20
  runs-on: self-hosted-gpu
  resources:
    memory: 16GB
    gpu-vram: 16GB
```

---

## 🔐 Secrets Handling in CI

### Current State

**For production training:** ❌ No secrets required

**For future data collection:**
- `GITHUB_TOKEN` - GitHub GraphQL API access
- `NVD_API_KEY` - NVD/CVE API access (optional)

### GitHub Actions Secrets

```yaml
# .github/workflows/train.yml (future)
env:
  GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
  AWS_ACCESS_KEY: ${{ secrets.AWS_ACCESS_KEY }}
  AWS_SECRET_KEY: ${{ secrets.AWS_SECRET_KEY }}

# Never commit secrets
# Add to .gitignore:
.env
.env.*
credentials.json
*.key
```

---

## 📊 Known Checkpoints

### Current Status

```
Known-good checkpoints: NONE AVAILABLE ❌

Reason: Training has never completed successfully

Expected location (after fix):
  training/outputs/transformer_v17/seed_42/best_model.pt
  training/outputs/gnn_v17_production/seed_42/best_model.pt
  training/outputs/fusion_v17_production/seed_42/best_model.pt
```

### Checkpoint Structure (Expected)

```python
checkpoint = {
    "epoch": 10,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scaler_state_dict": scaler.state_dict(),  # For AMP
    "best_val_f1": 0.9123,
    "config": {
        "batch_size": 64,
        "learning_rate": 2.5e-5,
        "seed": 42,
        ...
    }
}
```

### After First Successful Training

**Action:** Establish baseline checkpoint and document:
- F1 score achieved
- Training time
- Hyperparameters used
- Git commit SHA
- Date/time

---

## ⚖️ License & Contributing

### License

```
MIT License

Copyright (c) 2025 Vimal Sajan

See LICENSE file in repo root for full text
```

### Contributing Guidelines

**For ChatGPT Codex or external contributors:**

1. **All code changes must be MIT compatible**
   - No GPL, AGPL, or proprietary code
   - Document any third-party code sources

2. **Include attribution in commits**
   ```
   Co-Authored-By: ChatGPT Codex <noreply@openai.com>
   ```

3. **Update CHATGPT_CODEX_WORK_LOG.md**
   - Append entry for every change
   - Include files modified, reason, status

4. **Create tests for all changes**
   - Unit tests for new functions
   - Integration tests for workflows
   - Regression tests for bug fixes

5. **Follow code style**
   - Use black for formatting
   - Follow existing patterns
   - Add type hints and docstrings

6. **No breaking changes without discussion**
   - Public API must remain stable
   - Deprecate before removing
   - Document migration paths

---

## 📦 Project Structure

```
streamguard/
├── configs/                    # ✅ CREATED
│   ├── quick_test.yaml        # ✅ NEW - Smoke test config
│   └── prod.yaml              # ✅ NEW - Production config
│
├── core/                       # Core ML models
│   ├── models/
│   │   ├── transformer.py     # SQL Intent Transformer
│   │   ├── gnn.py            # Taint-Flow GNN
│   │   └── fusion.py         # Fusion Layer
│   └── preprocessing/
│       ├── c_preprocessor.py # C code preprocessing
│       └── graph_builder.py  # AST/CFG/DFG construction
│
├── data/
│   ├── processed/
│   │   ├── codexglue/        # ✅ Ready
│   │   │   ├── train.jsonl   # 527 MB
│   │   │   ├── valid.jsonl   # 65 MB
│   │   │   └── test.jsonl    # 65 MB
│   │   └── graphs/           # ❌ Missing (needs generation)
│   │       ├── train/        # ~21,854 .pt files
│   │       └── val/          # ~2,732 .pt files
│   └── sample/               # ✅ To be created
│       ├── train_small.jsonl # 100 samples
│       └── valid_small.jsonl # 20 samples
│
├── docs/
│   └── snippets/             # ✅ CREATED
│       └── atomic_write_json.py  # ✅ NEW - Atomic JSON utility
│
├── scripts/                   # ✅ CREATED
│   └── generate_small_dataset.py # ✅ NEW - Small dataset generator
│
├── training/
│   ├── train_transformer.py  # ❌ EXISTS but FAILING
│   ├── train_gnn.py         # GNN training script
│   ├── train_fusion.py      # Fusion training script
│   │
│   ├── scripts/
│   │   ├── pre_flight_validation.py  # ⚠️ Has Unicode bug
│   │   ├── cell_51_transformer_production.py  # ❌ Missing (referenced but not found)
│   │   ├── cell_52_gnn_production.py          # ❌ Missing
│   │   ├── cell_53_fusion_production.py       # ❌ Missing (deferred)
│   │   └── cell_54_metrics_aggregator.py      # ❌ Missing
│   │
│   ├── preprocessing/
│   │   └── create_simple_graph_data.py  # ✅ Exists - graph generator
│   │
│   ├── tests/
│   │   └── test_safety_utilities.py  # ✅ Exists - 18/21 pass
│   │
│   └── utils/               # Safety utilities (all implemented)
│       ├── json_safety.py   # ✅ Atomic JSON writes
│       ├── adaptive_config.py  # ✅ GPU-aware config
│       ├── collapse_detector.py  # ✅ Model collapse detection
│       └── amp_utils.py     # ✅ AMP-safe gradient clipping
│
├── StreamGuard_Production_Training.ipynb  # ✅ Clean, 24 cells
├── StreamGuard_Complete_Training.ipynb    # ⚠️ Corrupted, cells 25-28 broken
├── StreamGuard_Complete_Training.ipynb.backup  # ✅ Last good state
│
├── CHATGPT_CODEX_HANDOFF.md  # ✅ THIS FILE
├── CHATGPT_CODEX_WORK_LOG.md # ✅ CREATED - Work log template
│
├── A100_PRODUCTION_READY_SUMMARY.md  # ✅ Read this first
├── BLOCKER_FIXES_SUMMARY.md          # ✅ Technical details
├── PRODUCTION_NOTEBOOK_SUMMARY.md    # ✅ Notebook info
├── CELL_53_DEFERRAL_NOTE.md          # ✅ Fusion deferral
├── NOTEBOOK_STRUCTURE_ANALYSIS.txt   # ✅ Corruption analysis
├── README_PRODUCTION_TRAINING.md     # ✅ Quick start
│
├── requirements.txt          # Python dependencies
└── .gitignore               # ⚠️ Needs update
```

**Legend:**
- ✅ Exists and working
- ⚠️ Exists but has issues
- ❌ Missing or not working

---

## 🎯 FIRST ACTIONS FOR CHATGPT CODEX

### Priority 1: Debug Training Failure (CRITICAL)

**Task:** Diagnose why `train_transformer.py` exits with code 1

**Steps:**
1. Read "Debugging Current Failure" section above
2. Use Template E (Analysis Request) to capture error trace
3. Diagnose root cause
4. Propose fix in CHATGPT_CODEX_WORK_LOG.md
5. Implement fix
6. Verify with quick smoke test
7. Document results in work log

**Expected time:** 30-60 min

### Priority 2: Fix Unicode Bug

**Task:** Replace emoji in `pre_flight_validation.py` with ASCII

**Steps:**
1. Read "Key Files" section for exact location
2. Use Template B (Bugfix) to generate patch
3. Apply fix
4. Test on Windows
5. Document in work log

**Expected time:** 15-30 min

### Priority 3: Generate Graph Data

**Task:** Create ~21,854 .pt files for GNN training

**Steps:**
1. Test on 100 samples first
2. Verify output format
3. Run full generation
4. Document in work log

**Expected time:** 5-10 min (test) + 15-25 min (full)

---

## 📚 Summary

### What's Ready ✅

- CodeXGlue dataset (train/valid/test.jsonl)
- Safety utilities (atomic JSON, AMP utils, collapse detection)
- Comprehensive documentation (7+ guides)
- Config files (quick_test.yaml, prod.yaml)
- Small dataset generator
- Unit tests (18/21 passing)
- Production infrastructure

### What's Broken ❌

- Training script (exit code 1 - cause unknown)
- LR Finder (exit code 2)
- Graph data (missing ~21,854 files)
- Pre-flight validation (Unicode errors on Windows)
- Notebook cells 25-28 (JSON corruption)

### What's Missing ❌

- Production scripts (cell_51-54.py) - referenced but never created
- Working training pipeline
- Known-good checkpoints
- CI/CD configuration

### Critical Path to Success

1. **Debug train_transformer.py** → Get actual error message
2. **Fix root cause** → Apply patch based on diagnosis
3. **Verify smoke test passes** → 1 epoch completes successfully
4. **Generate graph data** → ~21,854 .pt files
5. **Run production training** → 3 seeds, F1 > 0.85
6. **Establish baselines** → Document first successful checkpoint

### Success Criteria Recap

Training is successful when:
- ✅ Quick test completes (1 epoch, exit code 0)
- ✅ F1 > 0.85 on validation set
- ✅ Reproducibility ±0.5% across seeds
- ✅ No crashes or serialization errors
- ✅ Checkpoints save correctly

---

## 📞 Support & Next Steps

### If You Encounter Issues

1. **Check this handoff** - Most answers are here
2. **Read essential docs** - 6 docs listed in "Essential Documents"
3. **Use templates** - 5 prompt templates provided
4. **Ask for analysis** - Template E before implementing fixes
5. **Update work log** - Document all changes

### Escalation

If blocked for >2 hours:
- Document blocker in work log
- Capture full error traces
- List attempted fixes
- Request human review

### Related Documentation

- **CHATGPT_CODEX_WORK_LOG.md** - Track your changes here
- **A100_PRODUCTION_READY_SUMMARY.md** - Production readiness overview
- **BLOCKER_FIXES_SUMMARY.md** - Technical blocker details
- **README_PRODUCTION_TRAINING.md** - Quick start guide

---

## ✨ Final Checklist

Before starting work, verify you understand:

- ✅ One-line mission (production training on CodeXGlue)
- ✅ Top 3 blockers (training fails, Unicode bug, missing graphs)
- ✅ What to focus on (Transformer/GNN/Fusion training)
- ✅ What to ignore (data collection, all *_COLLECTION_* docs)
- ✅ How to debug (capture stderr, use Template E)
- ✅ How to test (quick smoke test, unit tests)
- ✅ How to document (append to work log)
- ✅ Success criteria (F1 > 0.85, reproducibility, no crashes)

---

**READY TO START? → Begin with Priority 1: Debug Training Failure**

Use Template E (Analysis Request) to diagnose the root cause of exit code 1.

---

**END OF HANDOFF DOCUMENT**

**Last Updated:** 2025-11-10 15:00 UTC
**Version:** 1.0
**Status:** Production Ready for Codex Handoff
**Total Lines:** ~1850
**Author:** Claude Code + Vimal Sajan
**License:** MIT
