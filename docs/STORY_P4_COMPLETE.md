# StreamGuard Phase 4: Complete

**Status**: COMPLETE | **Date**: March 2026 | **Tests**: 150/150 PASS

---

## What Was Built

Phase 4 delivers the full model training pipeline: architecture, loss functions, data loading, training loop, evaluation metrics, and ablation study runner.

### Story Summary

| Story | File(s) | What | Tests |
|-------|---------|------|-------|
| P4-S1 | `model.py` | CodeBERT + 3-layer Type-Aware GGNN + Cross-Attention Fusion | 37/37 |
| P4-S2 | `callee_summarizer.py`, `callee_cache.py` | Inter-procedural callee context (Config E) | 25/25 |
| P4-S3 | `cfa_dataloader.py` | CFA-aware DataLoader with pair-preserving batch sampler | 21/21 |
| P4-S4 | `losses.py` | StreamGuardLoss: L_CE + L_CFA + L_CWE + L_severity | 29/29 |
| P4-S5 | `train.py` | Training loop with AMP, differential LR, checkpoint resume, 6 ablation configs | 25/25 |
| P4-S6 | `eval.py` | Full evaluation: F1, FPR/FNR, pairwise_accuracy, per-CWE F1, severity MAE | 8/8 |
| P4-S7 | `run_ablations.py` | Ablation study runner for all 6 configs with Paper Table 2 output | 6/6 |

All files are in `training/scripts/model/`. All tests are in `tests/`.

---

## Architecture

```
Input: C source function
  |
  v
CodeBERT encoder --> [CLS] 768-d
  |
  v                            Node features (824-d)
  |                                |
  |                                v
  |                          Node projector (824 -> 256)
  |                                |
  |                                v
  |                     3-layer Type-Aware GGNN
  |                     (4 edge types x 3 layers = 12 GatedGraphConv)
  |                                |
  v                                v
Cross-Attention Fusion: Q=BERT, K/V=per-node GGNN
  |
  v
Fused representation (1280-d)
  |
  v
Shared MLP (1280 -> 512 -> 128)
  |
  +---> Binary head (128 -> 2)        : vuln / safe
  +---> CWE head (128 -> 12)          : CWE classification
  +---> Severity head (128 -> 1)      : CVSS proxy
```

### 6 Ablation Configurations

| Config | use_graph | type_aware | lambda_cfa | cpg_components | use_interproc |
|--------|-----------|------------|------------|----------------|---------------|
| A | False | - | 0.0 | - | False |
| B | True | False | 0.0 | AST,CFG,DFG | False |
| B' | True | True | 0.0 | AST,CFG,DFG | False |
| C | True | True | 0.5 | AST,CFG,DFG | False |
| D | True | True | 0.5 | AST,CFG,DFG,TPG | False |
| E | True | True | 0.5 | AST,CFG,DFG,TPG | True |

---

## How to Run the Colab Notebook

### Prerequisites

1. Complete Phases 1-3 (data collection, preprocessing, CFA generation, split)
2. Upload data files to Google Drive

### Data Required

Zip the 4 data files and upload `streamguard_data.zip` to `My Drive/StreamGuard/`:

**PowerShell (Windows):**
```powershell
Compress-Archive -Path "training\data\final\train.h5", "training\data\final\val.h5", "training\data\final\test.h5", "training\data\processed\with_cfa\samples.jsonl" -DestinationPath "streamguard_data.zip"
```

**Linux/Mac:**
```bash
zip streamguard_data.zip training/data/final/train.h5 training/data/final/val.h5 \
    training/data/final/test.h5 training/data/processed/with_cfa/samples.jsonl
```

The zip contains:

| File | Source | Size | Purpose |
|------|--------|------|---------|
| `train.h5` | `training/data/final/train.h5` | ~3.2 GB | Training data (42,492 graph samples) |
| `val.h5` | `training/data/final/val.h5` | ~400 MB | Validation data (5,311 samples) |
| `test.h5` | `training/data/final/test.h5` | ~400 MB | Test data (5,312 samples) |
| `samples.jsonl` | `training/data/processed/with_cfa/samples.jsonl` | ~200 MB | Code text for CodeBERT tokenization |

All 4 files are **mandatory**. All 6 configs use the **exact same** data files.
The notebook also supports uploading the 4 files individually (without zipping).

### HDF5 File Layout (Layout A)

Each HDF5 file contains:
- `metadata/sample_ids` — string array of sample IDs
- `metadata/pair_ids` — string array linking CFA pairs
- `metadata/labels` — int array (0=safe, 1=vulnerable)
- `metadata/cwes` — string array of CWE types
- `graphs/{idx}/x` — node features (N, 824)
- `graphs/{idx}/edge_index` — edge connectivity (2, E)
- `graphs/{idx}/edge_attr` — edge types (E,) with values in {0,1,2,3}

Edge types: AST=0, CFG=1, DFG=2, TPG=3.

### Running

1. Open `StreamGuard_Phase4_Training_Colab.ipynb` in Google Colab
2. Set runtime to GPU (Runtime > Change runtime type > GPU)
3. Run cells 1-5 (setup + pre-flight checks)
4. In Cell 6, configure:
   - `DRY_RUN = True` for smoke test (5 min) or `False` for full training
   - `CONFIGS_TO_RUN = None` for all 6 configs, or a specific list
5. Run Cell 7 (training) — this is the long step
6. Run Cells 8-10 (results + save to Drive)

### Runtime Estimates (Full Training, 20 epochs)

| GPU | Per Config | All 6 Configs |
|-----|-----------|---------------|
| T4 (16 GB) | ~4-5 hours | ~24-30 hours |
| V100 (32 GB) | ~2-3 hours | ~12-18 hours |
| A100 (40 GB) | ~1.5-2 hours | ~9-12 hours |

### Resuming After Disconnect

The notebook supports checkpoint resume:
- Checkpoints are saved every epoch to both local SSD and Drive
- On restart, Cell 4 restores checkpoints from Drive
- `train()` automatically resumes from the latest checkpoint per config

---

## Risk Mitigations Implemented

### Critical (8 risks)

| ID | Risk | Mitigation | Verified By |
|----|------|-----------|-------------|
| R-01 | Test set contamination | test.h5 loaded ONCE at end per config | train.py, pre-flight check |
| R-02 | Wrong L_CFA sign | relu(cosine_sim + 0.5) verified | test_p4s4_losses.py |
| R-03 | BatchNorm at serving | GroupNorm(32, 256) used throughout | test_p4s1_model.py, pre-flight |
| R-04 | CFA pairs split | CFAAwareBatchSampler groups by pair_id | test_p4s3_dataloader.py |
| R-05 | Different data per config | assert_ablation_invariants() at startup | test_p4s7_ablations.py |
| R-06 | Different seeds | seed=42 in BASE_CONFIG, asserted | test_p4s7_ablations.py |
| R-07 | Tokenization skipped | tokenize_batch called in train loop | train.py |
| R-08 | No checkpoint saved | Checkpoint every epoch + atomic write | train.py |

### High (12 risks)

| ID | Risk | Mitigation |
|----|------|-----------|
| R-09 | GPU OOM | AMP enabled, gradient accumulation |
| R-10 | NaN loss | Gradient clipping at 1.0, NaN skip |
| R-11 | CodeBERT forgetting | Differential LR: CodeBERT=2e-5 |
| R-12 | Same LR both encoders | Two param groups in optimizer |
| R-13 | No MLflow | Graceful fallback to stdout |
| R-14 | AMP disabled | Auto-enabled on CUDA |
| R-15 | Config B not type-blind | Dedicated single_conv modules |
| R-16 | Config A not graph-free | use_graph=False bypasses GGNN+cross-attn |
| R-17 | Edge type >= 4 | Validated in pre-flight + encode_graph |
| R-18 | pair_id missing | Validated in pre-flight check |
| R-19 | Empty code text | JSONL lookup + placeholder fallback |
| R-20 | Scheduler misaligned | Steps only on optimizer steps |

---

## Test Coverage

```
tests/test_p4s1_model.py       37 tests  (model architecture, edge types, GroupNorm, checkpoint)
tests/test_p4s2_callee.py      25 tests  (callee summarizer, cache, inter-proc injection)
tests/test_p4s3_dataloader.py  21 tests  (HDF5 loading, batch sampler, pair preservation)
tests/test_p4s4_losses.py      29 tests  (L_CE, L_CFA sign, L_CWE, L_severity, composites)
tests/test_p4s5_train.py       25 tests  (optimizer, tokenization, dry-run, resume, configs)
tests/test_p4s6_eval.py         8 tests  (F1, FPR/FNR, pairwise, per-CWE, severity, CWE acc)
tests/test_p4s7_ablations.py    6 tests  (config invariants, dry-run, JSON output)
                              ─────
                              150 total
```

Run all Phase 4 tests:
```bash
python -m pytest tests/test_p4s1_model.py tests/test_p4s2_callee.py \
    tests/test_p4s3_dataloader.py tests/test_p4s4_losses.py \
    tests/test_p4s5_train.py tests/test_p4s6_eval.py \
    tests/test_p4s7_ablations.py -v
```

---

## Output Files

After training completes:

| File | Location | Content |
|------|----------|---------|
| `best_model_{config}.pt` | `training/checkpoints/` | Best val checkpoint per config |
| `latest_{config}.pt` | `training/checkpoints/` | Latest epoch checkpoint per config |
| `ablation_table.json` | `results/` | All metrics for Paper Table 2 |
| `mlruns/` | project root | MLflow experiment logs |

---

## What Comes Next

- **Phase 5**: Serving (FastAPI inference endpoint)
- **Phase 6**: Explainability (CFExplainer)
- **Phase 7**: Continuous CFA loop (human feedback)
