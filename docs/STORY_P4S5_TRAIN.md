# P4-S5: Training Loop (`train.py`)

**Status:** COMPLETE | **Date:** 2026-03-29 | **Tests:** 24/24 PASS

---

## What Was Built

Full Phase 4 M2 training loop rewrite with on-the-fly tokenization, CFA-aware forward passes, and 6 ablation configs.

| File | Purpose |
|------|---------|
| `training/scripts/model/train.py` | Complete training loop (M2) |
| `tests/test_p4s5_train.py` | 24 tests (6 mandatory + 18 additional) |

## Major Changes from M1

| Feature | M1 | M2 (Phase 4) |
|---------|----|----|
| CodeBERT input | None (zero vectors) | Tokenized on-the-fly from JSONL lookup |
| Forward pass | Single batch, old API | Separate orig/CFA batches from collate |
| AMP | Partial | Full GradScaler + autocast |
| Checkpoint | Best only | Latest every epoch + best (resume support) |
| Resume | Not supported | Load latest_*.pt on startup |
| --dry-run | --max-samples hack | Dedicated flag: 1 batch x 2 epochs |
| Configs | 3 (B, C, D) | 6 (A, B, B', C, D, E) |
| Test set | Used during training | ONCE at end with best checkpoint (R-01) |
| LR | Single group | Differential: CodeBERT=2e-5, GGNN=1e-4 |

## Architecture

```
train.py execution flow:

1. set_all_seeds(42) — full reproducibility
2. Load CodeBERT tokenizer
3. Load code lookup from JSONL (sample_id → code string)
4. Build CFADataset + CFAAwareBatchSampler for train/val
5. Create StreamGuardModel with freeze_codebert_layers=9
6. build_optimizer() → 2 param groups (CodeBERT=2e-5, GGNN=1e-4)
7. Linear warmup + cosine decay scheduler
8. GradScaler for AMP
9. Resume from latest checkpoint if exists

For each epoch:
  a. Rebuild sampler (reshuffles groups, set_epoch)
  b. For each batch:
     - Tokenize orig codes on-the-fly from code_lookup
     - Forward orig batch with input_ids + attention_mask
     - If CFA enabled + cfa_batch present: tokenize + forward CFA batch
     - StreamGuardLoss(orig, cfa, labels, cwe_labels)
     - Gradient accumulation + clip + step every grad_accum steps
  c. Save latest checkpoint (atomic)
  d. Evaluate on val set
  e. If val_f1 improves: save best checkpoint, reset patience
  f. Else: increment patience, early stop if exceeded

10. Load best checkpoint, evaluate test set ONCE, log to MLflow
```

## 6 Ablation Configurations

| Config | use_graph | type_aware | use_cfa | CPG | interproc |
|--------|-----------|-----------|---------|-----|-----------|
| A_baseline | False | True | False | AST,CFG,DFG | False |
| B_plus_ggnn | True | False | False | AST,CFG,DFG | False |
| B_prime_type_aware | True | True | False | AST,CFG,DFG | False |
| C_plus_cfa | True | True | True | AST,CFG,DFG | False |
| D_plus_tpg | True | True | True | AST,CFG,DFG,TPG | False |
| E_full | True | True | True | AST,CFG,DFG,TPG | True |

All configs share: seed=42, same HDF5 paths, same hyperparameters (R-05, R-06).

## On-the-Fly Tokenization (R-07, R-19)

HDF5 stores graph tensors but NOT raw code. Code text is loaded from JSONL at startup:

```
load_code_lookup() → dict[sample_id, code_string]
  Tries in order:
    1. training/data/processed/with_cfa/samples.jsonl
    2. training/data/processed/deduped/samples.jsonl
    3. training/data/processed/cleaned/samples.jsonl

tokenize_batch(codes, tokenizer, max_length=512, device)
  → (input_ids: (B, 512), attention_mask: (B, 512))
  Empty strings → "void placeholder(void){}" (prevents degenerate tokenization)
```

## Checkpoint Resume (R-08)

```
Every epoch:  latest_{config_name}.pt (atomic: .tmp + os.replace)
Best val F1:  best_model_{config_name}.pt (atomic)

On resume:
  if --resume AND latest_*.pt exists:
    load model_state_dict, optimizer_state_dict, scheduler_state_dict, scaler_state_dict
    continue from saved epoch + 1
```

## CLI

```bash
# Dry-run (smoke test)
python -m training.scripts.model.train \
    --config C_plus_cfa \
    --train-h5 training/data/final/train.h5 \
    --val-h5 training/data/final/val.h5 \
    --test-h5 training/data/final/test.h5 \
    --dry-run

# Full training
python -m training.scripts.model.train \
    --config D_plus_tpg \
    --train-h5 training/data/final/train.h5 \
    --val-h5 training/data/final/val.h5 \
    --test-h5 training/data/final/test.h5 \
    --epochs 20 --batch-size 8 --resume

# Custom LR
python -m training.scripts.model.train \
    --config E_full \
    --train-h5 training/data/final/train.h5 \
    --val-h5 training/data/final/val.h5 \
    --lr-codebert 2e-5 --lr-ggnn 1e-4
```

## Risk Mitigations

| Risk | Status | Evidence |
|------|--------|----------|
| R-01 (test set contamination) | MITIGATED | test.h5 evaluated ONCE after all training with best checkpoint; never used during training loop or early stopping |
| R-05 (different HDF5 across configs) | MITIGATED | All configs share same train_h5/val_h5/test_h5 paths from CLI |
| R-06 (different seeds) | MITIGATED | DEFAULT_CONFIG seed=42; all configs inherit it |
| R-07 (tokenization skipped) | MITIGATED | tokenize_batch called for every batch; empty strings get placeholder |
| R-08 (no per-epoch checkpoint) | MITIGATED | save_checkpoint called every epoch (latest) + on best F1 improvement |
| R-09 (GPU OOM) | MITIGATED | AMP enabled by default on CUDA; batch_size=8 + grad_accum=4 |
| R-10 (NaN loss / gradient explosion) | MITIGATED | clip_grad_norm_(max_norm=1.0) + NaN/Inf guard skips batch, prevents corrupt checkpoints |
| R-11 (CodeBERT catastrophic forgetting) | MITIGATED | lr_codebert=2e-5 (5x lower than GGNN LR); freeze_codebert_layers=9 |
| R-12 (differential LR not applied) | MITIGATED | build_optimizer creates 2 param groups; verified by test_two_param_groups |
| R-14 (AMP disabled on GPU) | MITIGATED | use_amp = device.type == "cuda" (True by default on GPU) |
| R-19 (code text missing) | MITIGATED | JSONL merge-all (3 files); logs empty-code warnings; placeholder for missing IDs |
| R-20 (scheduler step off-by-one) | MITIGATED | scheduler.step() called only inside `if (step+1) % grad_accum == 0` block |
| R-29 (val == test path) | MITIGATED | ValueError raised if os.path.abspath(val_h5) == os.path.abspath(test_h5) |

## Go/No-Go Gate Verification

| Gate | Check | Status |
|------|-------|--------|
| G-P4-05 | Differential LR: print optimizer param_groups | PASS — `[(2e-05, N), (0.0001, N)]` |
| G-P4-06 | Tokenization active: input_ids not None | PASS — tokenize_batch always returns valid tensors |
| G-P4-07 | AMP enabled on GPU | PASS — use_amp = True for CUDA |
| G-P4-09 | Checkpoint saves every epoch | PASS — dry-run verified |
| G-P4-10 | MLflow running and logging | PASS — dry-run verified |
| G-P4-11 | test.h5 NOT used during training loop | PASS — code review confirms test set only at end |
| G-P4-12 | All configs share same HDF5 paths | PASS — CLI args mandatory |
| G-P4-13 | All configs share seed=42 | PASS — DEFAULT_CONFIG["seed"] = 42 |
| G-P4-14 | Dry-run completes: 2 epochs, finite loss, checkpoint saved | PASS — test_dry_run_completes |

## Production Fixes (Post-Audit)

6 issues found and fixed during production audit:

| # | Severity | Issue | Fix |
|---|----------|-------|-----|
| 1 | CRITICAL | `load_code_lookup` stopped at first JSONL found → 8.8% samples missing code | Rewritten to merge ALL JSONL files (earlier files take priority for duplicate IDs) |
| 2 | CRITICAL | model.py had no `use_graph` / `type_aware_edges` params → Config A/B identical to B' | Added both params to `__init__` + `forward()`; Config A skips GGNN, Config B uses type-blind conv |
| 3 | HIGH | No NaN guard in training loop → corrupt checkpoints on NaN loss | `torch.isnan(total_loss)` check; skips batch + warns on NaN/Inf |
| 4 | MEDIUM | No logging when `code_lookup.get()` returns empty string | First-batch warning prints count and example of missing sample_ids |
| 5 | MEDIUM | No assertion that val_h5 != test_h5 → silent test contamination (R-29) | `os.path.abspath()` check raises ValueError |
| 6 | LOW | `frac_separated` not logged to MLflow | Added to per-epoch MLflow metrics + epoch print line |

## Test Summary

```
tests/test_p4s5_train.py — 25/25 PASS

TestSetAllSeeds:                 2 tests (same seed same output, different seeds differ)
TestBuildOptimizer:              3 tests (2 groups, CodeBERT LR lower, only trainable params)
TestTokenizeBatch:               2 tests (output shape (B,512), shorter max_length)
TestTokenizeBatchEmpty:          2 tests (empty string no crash, all-empty batch)
TestDryRun:                      1 test  (2 epochs, finite loss, checkpoint saved)
TestResume:                      1 test  (checkpoint loaded, training continues)
TestAblationConfigs:             5 tests (6 entries, names, required keys, A no graph, E interproc)
TestDefaultConfig:               3 tests (required keys, seed=42, differential LR)
TestLoadCodeLookup:              4 tests (loads JSONL, empty for missing, merges all, priority)
TestSaveCheckpoint:              1 test  (atomic write, no .tmp left, contents valid)
TestSelectWithCompleteGroups:    1 test  (pairs kept intact)
```

```
tests/test_p4s5_preflight.py — 12/12 PASS

TestPreflightMultiEpoch:         2 tests (loss decreases + no NaN, checkpoint epoch progression)
TestPreflightAblationDiff:       3 tests (Config A runs, Config B runs, A != B' embeddings)
TestPreflightGuards:             3 tests (val==test raises, NaN loss detected, merge-all)
TestPreflightModelIntegrity:     4 tests (all 6 configs instantiate, diff LR groups, R-02 sign)
```

## Combined Test Results (P4-S1 through S5)

```
tests/test_p4s1_model.py      — 37/37 PASS (0 regressions)
tests/test_p4s2_callee.py     — 25/25 PASS (0 regressions)
tests/test_p4s3_dataloader.py — 21/21 PASS (0 regressions)
tests/test_p4s4_losses.py     — 29/29 PASS (0 regressions)
tests/test_p4s5_train.py      — 25/25 PASS
tests/test_p4s5_preflight.py  — 12/12 PASS
Total: 149/149 PASS
```
