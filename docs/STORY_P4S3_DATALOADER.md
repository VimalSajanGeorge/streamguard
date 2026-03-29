# P4-S3: CFA-Aware DataLoader (`cfa_dataloader.py`)

**Status:** COMPLETE | **Date:** 2026-03-28 | **Tests:** 21/21 PASS

---

## What Was Built

CFA-aware data loading pipeline that guarantees pair integrity during training.

| File | Purpose |
|------|---------|
| `training/scripts/model/cfa_dataloader.py` | CFADataset, CFAAwareBatchSampler, collate, builder |
| `tests/test_p4s3_dataloader.py` | 21 tests |

## Architecture

```
CFADataset (HDF5 → PyG Data):
  1. Auto-detect HDF5 layout on init:
     Layout A: f['graphs'][str(idx)] + f['metadata'] parallel arrays
     Layout B nested: f[split][sample_id] with per-graph attrs
     Layout B flat: f[sample_id] at root with optional split attr
  2. Build in-memory index: [{sample_id, pair_id, label, cwe, source}, ...]
  3. __getitem__: lazy-load graph tensors from HDF5 → PyG Data + metadata dict

CFAAwareBatchSampler:
  1. Group dataset indices by pair_id (sentinels → singleton groups)
  2. Shuffle GROUPS (not indices) with seed + epoch → deterministic per-epoch
  3. Fill batches greedily; never split a group across batches
  4. Oversized groups (> batch_size) kept whole → slight batch overflow

cfa_collate_fn:
  Input:  [(Data, meta), ...] from one batch
  Logic:  First occurrence of pair_id → orig; second → CFA
  Output: (orig_batch, cfa_batch, orig_metas, cfa_metas)
          cfa_batch is None when batch has no CFA pairs
```

## Critical Invariant (R-04)

**CFA pairs must never be split across batches.** If pair members land in different batches, L_CFA trains on random unrelated pairs and contributes zero useful signal.

CFAAwareBatchSampler enforces this by grouping indices by `pair_id` before any shuffling. The test `test_pairs_never_split` exhaustively checks that no pair_id appears in more than one batch.

## Sentinel pair_id Handling (R-18)

`EMPTY_PAIR_SENTINELS = frozenset({"", "None", "null", "0", "none"})`

These values in `pair_id` fields are treated as "no pair" — each sample gets its own singleton group. This prevents false pair grouping from missing or placeholder pair_ids in HDF5 attrs.

## Dual HDF5 Layout Support

| Layout | Detection | Used By |
|--------|-----------|---------|
| A | `'metadata' in f and 'graphs' in f` | Phase 1 Stage 7 flat output |
| B nested | `split in f` | Phase 4 per-split files |
| B flat | fallback | Root-level sample groups |

## Risk Mitigations

| Risk | Status | Evidence |
|------|--------|----------|
| R-04 (pairs split across batches) | MITIGATED | CFAAwareBatchSampler groups by pair_id; tested in 2 tests with Layout A and B |
| R-18 (pair_id missing from HDF5) | ADDRESSED | 5 sentinel values handled; dataset reads pair_id from attrs/metadata; empty → singleton |

## Test Summary

```
tests/test_p4s3_dataloader.py — 21/21 PASS

TestDatasetIndex:           3 tests (required fields, int labels, pair_ids present)
TestLayoutBNested:          2 tests (loads, getitem returns Data+meta)
TestLayoutAFlat:            2 tests (loads, getitem returns Data+meta)
TestSamplerPairGrouping:    2 tests (CRITICAL: pairs never split — Layout A, Layout B)
TestSamplerSetEpoch:        2 tests (different epochs differ, same epoch same)
TestSamplerSingletons:      2 tests (singletons in batches, all samples covered)
TestCollateNoCFA:           1 test  (all singletons → cfa_batch is None)
TestCollateAlignment:       2 tests (paired batch aligned, mixed pairs+singletons)
TestBuildDataloader:        2 tests (returns tuple, iteration works)
TestSentinelPairIds:        2 tests (sentinel set membership, values correct)
TestCountBatches:           1 test  (count matches iteration)
```

## Combined Test Results (P4-S1 through S3)

```
tests/test_p4s1_model.py      — 37/37 PASS (0 regressions)
tests/test_p4s2_callee.py     — 25/25 PASS (0 regressions)
tests/test_p4s3_dataloader.py — 21/21 PASS
Total: 83/83 PASS
```
