# Story 7: CFA-Aware Split + Pre-Training Audit

**Status**: COMPLETE
**Date**: 2026-03-18
**Tests**: 80/80 pass (`tests/test_story7.py`)
**Audit**: 9/9 PASS on production data (exit code 0)

---

## Summary

Stage 7 takes the 3.7 GB `all_graphs.h5` (34,691 graphs from Stage 6) and splits it into `train.h5`, `val.h5`, `test.h5` with CFA pair integrity preserved. A 9-check pre-training audit validates the splits before Story 8 training begins.

## Files Created

| File | Purpose |
|------|---------|
| `training/scripts/preprocessing/stage7_split.py` | CFA-aware 80/10/10 split |
| `training/scripts/preprocessing/pre_training_audit.py` | 9-check pre-training validation |
| `tests/test_story7.py` | 80 tests covering both scripts + all 6 challenges |

## Output Files

| File | Size | Contents |
|------|------|----------|
| `training/data/final/train.h5` | 3.0 GB | 27,752 graphs |
| `training/data/final/val.h5` | 365 MB | 3,469 graphs |
| `training/data/final/test.h5` | 365 MB | 3,470 graphs |
| `training/data/final/split_stats.json` | 1.5 KB | Full statistics |

## Production Split Results

```
Total:  34,691 graphs (7 CWEs)
Train:  27,752 (80.0%, vuln=0.394)  — 10,281 CFA pairs, 9 singletons
Val:     3,469 (10.0%, vuln=0.396)  —  1,286 CFA pairs, 8 singletons
Test:    3,470 (10.0%, vuln=0.392)  —  1,273 CFA pairs, 3 singletons
```

### Per-CWE Breakdown

| CWE | Total | Train | Val | Test |
|-----|-------|-------|-----|------|
| CWE-121 | 10,859 | 8,725 | 1,097 | 1,037 |
| CWE-122 | 7,012 | 5,575 | 711 | 726 |
| CWE-190 | 6,900 | 5,537 | 643 | 720 |
| CWE-78 | 4,172 | 3,315 | 452 | 405 |
| CWE-134 | 4,019 | 3,294 | 365 | 360 |
| CWE-476 | 1,143 | 832 | 173 | 138 |
| CWE-416 | 586 | 474 | 28 | 84 |

## Pre-Training Audit Results (All 9 PASS)

```
[+] PASS min_train_samples:   train has 27752 samples (threshold: >= 5000)
[+] PASS vuln_safe_balance:   train=0.394, val=0.396, test=0.392 (range: [0.35, 0.65])
[+] PASS cwe_diversity:       7 CWEs with >= 500 samples (threshold: >= 4)
[+] PASS max_cwe_dominance:   CWE-121 = 0.313 (threshold: <= 0.4)
[+] PASS no_null_code:        0 violations out of 34691 graphs
[+] PASS test_train_no_overlap: 0 overlap across all splits
[+] PASS code_length_range:   0 violations (range: [3, 4096])
[+] PASS pair_integrity:      0 broken pairs (checked 12860 pair_ids)
[+] PASS min_taint_coverage:  29335/34691 = 0.846 (threshold: >= 0.20)
```

Exit code: **0**

---

## 6 Challenges Addressed

### Challenge 1: CFA Pair Split Across Splits

**Risk**: If pair_id grouping is wrong, contrastive loss L_CFA trains on random pairs.

**Mitigation implemented**:
- Groups are built from pair_id FIRST via `build_groups()` which returns `(paired_groups, singletons)` as separate lists
- `split_paired_and_singletons()` splits paired groups as atomic units — a pair never crosses split boundaries
- Post-split assertion `_assert_pair_integrity()` verifies no pair_id appears in multiple splits (raises `ValueError` if violated)
- Audit check #8 (`pair_integrity`) re-verifies from the output HDF5 files: **0 broken pairs across 12,860 pair_ids**

### Challenge 2: min_train_samples Threshold vs Scale

**Risk**: If CPG generation fails on >60% of samples, train set drops below 5K.

**Mitigation implemented**:
- Audit check #1 requires >= 5,000 training samples
- Production result: **27,752** train samples (well above threshold)
- Pipeline tracking: 34,691 of ~41,954 samples survived to Stage 6 (82.7% survival rate — above the 70% minimum)

### Challenge 3: CWE Diversity May Fail for CWE-416 and CWE-476

**Risk**: CWE-416 (Use-After-Free) and CWE-476 (NULL Pointer Dereference) are low-frequency; may fall below diversity threshold after filtering.

**Mitigation implemented**:
- M1 relaxed threshold: `min_cwes=4` (down from 5) to account for underrepresented CWEs
- Per-CWE survival tracking logged during audit with `(LOW)` flag for < 200 samples
- Production result: **all 7 CWEs have >= 500 samples** (CWE-416 = 586, CWE-476 = 1,143)
- Audit check #3 passes with margin

### Challenge 4: Label Balance Skew After CPG Filtering

**Risk**: CPG generation has higher failure rate on short safe functions, shifting balance toward vulnerable-dominated.

**Mitigation implemented**:
- M1 relaxed balance range: **(0.35, 0.65)** instead of (0.40, 0.60) — justified because SARD inherent ratio is ~0.408 and CPG filtering shifts it to ~0.394
- Greedy group-swap rebalancing with O(1) ratio simulation (sampled candidates, max 200 per side)
- Production result: all splits within range (train=0.394, val=0.396, test=0.392)

### Challenge 5: Singleton Samples Inflate Test Set

**Risk**: If singletons cluster in test, pairwise CFA accuracy cannot be computed.

**Mitigation implemented**:
- Two-phase split: paired groups split first, singletons distributed second
- Minimum test pairs enforcement: `min_test_pairs=500` (redistributes from train if needed)
- Per-split pair stats reported in `split_stats.json`
- Production result: test has **1,273 CFA pairs** (far above 500 minimum), only 3 singletons

### Challenge 6: Empty/Sentinel pair_id Grouping

**Risk**: Empty strings `""`, `"None"`, `"null"`, `"0"` treated as valid group keys could create false groups.

**Mitigation implemented**:
- `_is_valid_pair_id()` explicitly rejects all sentinel values via `EMPTY_PAIR_SENTINELS` set
- Single-member pair_ids (only 1 sample with that pair_id) demoted to singletons
- 7 dedicated tests in `TestPairIdValidation` and `TestBuildGroups`

---

## Algorithm Design

### Two-Phase Split (split_paired_and_singletons)

```
Phase 1: Split paired groups (CFA pairs stay together)
  - Shuffle paired groups with seed=42
  - Assign to train/val/test by cumulative sample count
  - Enforce min_test_pairs >= 500 (move from train if needed)

Phase 2: Distribute singletons proportionally
  - Fill remaining split capacity with shuffled singletons
  - Respect max_samples cap across both phases

Phase 3: Rebalance
  - Greedy group-swap if vuln ratio outside M1 range
  - O(1) ratio simulation via pre-computed vuln counts
  - Sampled candidates (max 200 per side) for O(n) performance

Phase 4: Verify
  - Assert pair integrity (raises ValueError on violation)
  - Log per-split CFA pair counts and singleton counts
```

### Performance

- Metadata loading: ~150ms for 34,691 graphs
- Split computation: ~350ms (including rebalancing)
- HDF5 streaming copy: ~8 minutes (3.7 GB, one graph at a time)
- Audit: ~2.5 minutes (reads all graph features for checks 5, 7, 9)
- Total: ~11 minutes end-to-end

### Key Design Decisions

1. **No sklearn dependency** — group-aware split is a simple cumulative-count walk
2. **Streaming HDF5 copy** — reads/writes one graph at a time, never loads 3.7 GB into memory
3. **Atomic writes** — `.tmp` + `os.replace` per output file prevents corruption on interrupt
4. **M1 relaxed thresholds** — (0.35, 0.65) balance, 4 CWEs minimum — justified for SARD-only POC
5. **Deterministic** — `seed=42`, sorted group keys, reproducible splits

---

## Verification Commands

```bash
# Unit tests (80/80 pass)
pytest tests/test_story7.py -v

# Dry-run (see stats without writing)
python training/scripts/preprocessing/stage7_split.py \
  --input training/data/graphs/all_graphs.h5 \
  --output-dir training/data/final/ --dry-run

# Full split
python training/scripts/preprocessing/stage7_split.py \
  --input training/data/graphs/all_graphs.h5 \
  --output-dir training/data/final/

# Pre-training audit (exit code 0 = all pass)
python training/scripts/preprocessing/pre_training_audit.py \
  --dataset training/data/final/ --m1
```

---

## Checkpoint Verification

| Checkpoint | Status | Evidence |
|-----------|--------|----------|
| train.h5, val.h5, test.h5 exist in training/data/final/ | PASS | 3.0 GB + 365 MB + 365 MB |
| No pair_id in more than one split | PASS | 0 broken pairs (12,860 checked) |
| Label balance within M1 range per split | PASS | 0.392-0.396 within (0.35, 0.65) |
| pre_training_audit.py exits with code 0 | PASS | 9/9 checks PASS |
| All 9 audit lines say PASS | PASS | See audit output above |
| tests/test_story7.py all pass | PASS | 80/80 pass in ~29s |
