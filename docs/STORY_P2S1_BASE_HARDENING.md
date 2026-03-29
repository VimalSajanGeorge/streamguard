# P2-S1: Harden BaseCollector + Schema

**Status:** Complete
**Date:** 2026-03-19

## Objective

Harden `base_collector.py` and `schema.py` for Phase 2 production-scale data collection (40K+ samples, multi-source, long-running jobs).

## Changes Made

### 1. Schema — Max Token Limit (schema.py)

- Added `>4096 token` rejection in `validate_sample()` after the existing `<10 token` check
- Prevents excessively large code samples from entering the pipeline

### 2. FIX 3 (PI-08) — Jitter + 403 Abuse Detection (base_collector.py)

- Added `import random` and `wait_random` from tenacity
- Both `@retry` decorators now use `wait_exponential(...) + wait_random(min=0.5, max=3.0)` to prevent thundering herd
- `safe_get()`: detects GitHub 403 "secondary rate limit" responses, sleeps 120+random(0,30)s, retries
- Added `jitter_sleep(base=0.5)` utility method for callers to insert random delays between requests

### 3. FIX 1 (PI-01) — HTTP 503 Handling (base_collector.py)

- `safe_get()`: if 503, sleeps 30+random(0,10)s, retries once
- `safe_post()`: same 503 handling
- Handles transient server unavailability during long collection runs

### 4. FIX 2 (PI-07) — Fast Hash Persistence (base_collector.py)

- `save_sample()`: appends each code hash to `{source}_hashes.txt` sidecar file
- `_load_seen_hashes()` rewritten with two paths:
  - **Fast path:** reads `{source}_hashes.txt` (one hash per line) — O(n) file read, no JSON parsing
  - **Slow path:** rebuilds from JSONL if hash file missing; logs skipped malformed lines; writes sidecar for future fast loads
- Reduces restart time significantly at 40K+ scale (no JSON deserialization needed)

### 5. FIX 5 (PI-10) — Enhanced Disk Space Check (base_collector.py)

- Now checks both `self.output_dir` and `STREAMGUARD_DATA_DIR` (default: `./training/data`)
- Threshold raised from 2 GB to 5 GB
- Guards non-existent paths with `data_dir.exists()` check

### 6. FIX 4 (PI-09) — Inline Broken-Pair Detection (base_collector.py)

- Added `_pending_pairs: dict` to `__init__` — tracks pair_id → label during collection
- `save_sample()` inline pair validation (before JSONL write):
  - Sentinel pair_ids (`""`, `"None"`, `"null"`, `"0"`) are ignored
  - Same pair_id + same label → warning logged, pair_id cleared to `""`
  - Same pair_id + different label → valid pair completed, removed from pending
  - New pair_id → added to pending
- `finalize_pairs()` method for post-collection cleanup:
  - Pass 1: count pair_id occurrences in JSONL
  - Pass 2: atomic rewrite (`.tmp` + `os.replace`), clearing orphan pair_ids (count==1)
  - Returns count of cleared orphans

### 7. Verification Audit Script (audit_base.py)

New file: `training/scripts/collection/audit_base.py` — 6 checks (14 assertions):
1. Hash sidecar created with correct line count after saving samples
2. Fast-path hash reload on restart + duplicate rejection
3. Valid CFA pair (label=1 + label=0) saved with pair_id intact
4. Same-label duplicate pair_id detected and cleared inline
5. `finalize_pairs()` clears orphan pair_ids in JSONL atomically
6. Schema rejects samples with >4096 tokens

## Files Modified

| File | Change |
|---|---|
| `training/scripts/collection/schema.py` | +3 lines (max 4096 token check) |
| `training/scripts/collection/base_collector.py` | Rewritten with 5 production fixes |
| `training/scripts/collection/audit_base.py` | **New** — 6-check verification script |

## Test Results

| Suite | Result |
|---|---|
| `tests/collection/test_base_collector.py` | 10/10 pass |
| `tests/test_story2.py` | 27/27 pass |
| `training/scripts/collection/audit_base.py` | 14/14 PASS |

## Production Data Verification

| Check | Result |
|---|---|
| `sard_samples.jsonl` line count | 53,666 (Phase 1 output intact) |
| `sard_hashes.txt` generated | 53,666 lines (slow-path rebuild, sidecar persisted) |
| Broken pairs in SARD | 0 (18,687 complete pairs) |

## Key Design Decisions

- **Hash sidecar over pickle/sqlite:** Plain text (one hash per line) is simplest, portable, and fast enough for 40K+ scale. No binary format dependencies.
- **Inline pair detection + finalize_pairs():** Two-layer approach catches most issues during collection (inline) while `finalize_pairs()` handles edge cases (e.g., pair member rejected by validation after its mate was saved).
- **5 GB disk threshold:** Increased from 2 GB to account for multi-source collection generating larger intermediate files.
- **Jitter on retries:** `wait_exponential + wait_random` prevents synchronized retry storms when multiple collector instances hit the same API.
