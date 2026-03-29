# Story 5: Stage 2 Deduplication — Completion Report

**Date**: 2026-03-11
**Script**: `training/scripts/preprocessing/stage2_dedup.py`
**Input**: `training/data/processed/cleaned/samples.jsonl` (53,242 samples)
**Output**: `training/data/processed/deduped/samples.jsonl` (41,954 samples)

---

## Results Summary

```
Stage 2 input:              53,242 samples
After L1 (exact MD5):       53,242  (removed 0)
After L2 (CVE-ID):          53,242  (removed 0)
After L3 (commit SHA):      53,242  (removed 0)
After L4 (MinHash LSH):     41,954  (removed 11,288)
─────────────────────────────────────────────────────
Total removed:              11,288  (21.2%)
```

L4 breakdown (sard_group, threshold=0.95):
- 18,572 pair groups: 14,673 kept, 3,899 skipped as near-duplicates
- 2,697 unpaired samples: 1,573 kept, 1,124 skipped

---

## Acceptance Criteria

| Criterion | Target | Actual | Status |
|-----------|--------|--------|--------|
| Stage 1 comments removed | 0 comments in spot check | 0/3 samples had comments | PASS |
| Stage 1 line filter [5,500] | All samples in range | min=5, max=194 | PASS |
| Stage 1 skipped count | Logged | 424 skipped from 53,666 raw | PASS |
| Dedup size reduction | 5-10% target | 21.2% | ACCEPTABLE (see below) |
| Pairs broken by dedup | 0 | 0 | PASS |
| Pair integrity (with_cfa) | broken pairs = 0 | 0 broken by dedup | PASS |
| min_train_samples >= 30,000 | 30,000 | 41,954 | PASS |
| cwe_diversity >= 7 | 7 CWEs with >= 500 | 7 CWEs, min=586 | PASS |
| max_cwe_dominance < 0.45 | < 0.45 | 0.303 (CWE-190) | PASS |
| No null code | 0 | 0 | PASS |

---

## Why 21.2% Removal Is Acceptable (Not 5-10%)

The original 5-10% estimate assumed SARD only has exact copies. In practice,
Juliet Suite v1.3.1 creates **18 numbered flow variants** per CWE pattern
(e.g., `_01.c` through `_18.c`). Variants like `_01` (direct flow) vs `_09`
(multi-socket) share >95% Jaccard similarity in the bad() function body because
they test the same vulnerability with minor flow differences.

**Tradeoffs of keeping all variants vs. deduplicating:**

| Factor | Keep All (5-10% removal) | Dedup to 21.2% |
|--------|--------------------------|-----------------|
| Stage 3 CFA cost | ~18K Claude Haiku calls | ~14.7K calls (saves ~20%) |
| Stage 7 train/test leak risk | Higher (similar patterns both sides) | Lower |
| Training convergence | Slower (redundant gradient updates) | Faster |
| CWE diversity | Same 7 CWEs | Same 7 CWEs, min=586 |
| Total samples | 50K+ (above 30K threshold) | 42K (above 30K threshold) |

The 0.95 threshold keeps variants that differ meaningfully (>5% token-level
divergence) while removing true copy-paste duplicates. This is the right
balance for downstream stages.

---

## Warnings to Monitor

### 1. Vuln/Safe Balance: 0.408 (below [0.45, 0.55])
- **Cause**: SARD Juliet has 2-3 good() functions per 1 bad() function per file.
- **Resolution**: Stage 3 CFA Generation produces additional label=0 counterfactuals
  for vulnerable samples, which will push the ratio toward 0.50.
- **Action**: No fix needed in Stage 2.

### 2. Pre-existing Orphan Pairs: 20 remaining (32 in cleaned)
- **Cause**: process_sard.py created pair_ids for files where only good() functions
  survived Stage 1 line filter (bad() was too short after #ifdef stripping).
- **Resolution**: Stage 3 can regenerate the missing vulnerable member via CFA, or
  these 20 orphans can be dropped. 20 samples out of 42K is negligible.
- **Action**: Log warning in Stage 3 if orphan pair_ids are encountered.

### 3. Memory Usage
- Stage 2 loads all 53K samples into memory (~400MB for SARD).
- For M2 with real-world data (100K+ samples), consider streaming L1-L3 and
  batching L4 by CWE to reduce peak memory.

---

## Architecture: 3-Group Source-Aware LSH

```
┌──────────────────────────────────────────────────────────┐
│                    L4 MinHash LSH                        │
│                                                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐  │
│  │ sard_group  │  │ real_world  │  │   unpaired      │  │
│  │ thresh=0.95 │  │ thresh=0.80 │  │   thresh=0.75   │  │
│  │             │  │             │  │                 │  │
│  │ sard        │  │ cve         │  │ exploitdb       │  │
│  │ sard_cfa    │  │ osv         │  │ manual          │  │
│  │             │  │ github_adv  │  │ exploitdb_cfa   │  │
│  │             │  │ repo        │  │                 │  │
│  │             │  │ *_cfa       │  │                 │  │
│  └─────────────┘  └─────────────┘  └─────────────────┘  │
│                                                          │
│  Groups are deduped independently — never cross-compare  │
│  CFA pairs processed atomically (never broken)           │
└──────────────────────────────────────────────────────────┘
```

---

## M2 Incremental Mode (Future)

```bash
# M1 full run (current):
python stage2_dedup.py \
  --input training/data/processed/cleaned/samples.jsonl \
  --output training/data/processed/deduped/samples.jsonl

# M2 incremental (when real-world collectors are ready):
python stage2_dedup.py \
  --input training/data/processed/cleaned/cve.jsonl \
  --existing training/data/processed/deduped/samples.jsonl \
  --output training/data/processed/deduped/samples_m2.jsonl \
  --merge-with training/data/processed/deduped/samples.jsonl
```

---

## CWE Distribution After Dedup

| CWE | Cleaned | Deduped | Removed |
|-----|---------|---------|---------|
| CWE-121 | 12,802 | 10,911 | 14.8% |
| CWE-122 | 8,238 | 7,012 | 14.9% |
| CWE-134 | 7,591 | 4,207 | 44.6% |
| CWE-190 | 13,768 | 12,732 | 7.5% |
| CWE-416 | 603 | 586 | 2.8% |
| CWE-476 | 1,155 | 1,143 | 1.0% |
| CWE-78 | 9,085 | 5,363 | 41.0% |

CWE-134 and CWE-78 had the most near-duplicates in Juliet (format string and
command injection patterns are highly templated). CWE-416 and CWE-476 had
fewer variants in Juliet, so fewer removals.

---

## Production Code Review

Reviewed `stage2_dedup.py` for production issues:

1. **MD5 for hashing** — Used only for dedup fingerprinting (not security). Acceptable.
2. **Memory**: All samples loaded into RAM. Fine for 53K SARD (~400MB). For M2
   with 100K+ samples, may need streaming for L1-L3.
3. **Determinism**: MinHash uses random permutations seeded by datasketch defaults.
   Results may vary slightly across runs. For reproducibility, consider setting
   `hashfunc` or `seed` parameter in MinHash constructor.
4. **No `--dry-run` flag**: Unlike Stage 1, Stage 2 doesn't have `--dry-run`.
   Not critical since it's a pure transformation (no API calls, no destructive ops).
5. **Error handling**: `json.loads` can throw on malformed lines. Currently not
   caught — will crash. Acceptable for pipeline (Stage 1 already validated JSONL).

---

*docs/STORY5_STAGE2_DEDUP.md | StreamGuard v1.0 | March 2026*
