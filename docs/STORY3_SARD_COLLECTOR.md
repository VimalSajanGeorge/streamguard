# Story 3: SARD Juliet Suite Collector

## What Was Done

### Goal
Implement a production-grade collector (`process_sard.py`) that processes the NIST Juliet Test
Suite C test cases into the StreamGuard canonical schema, producing labeled training samples
across 7 target CWE types.

**Result: 53,666 samples, 7 CWEs, 18,687 valid pairs, 0 broken pairs.**

---

### Files Changed

| File | Change |
|------|--------|
| `training/scripts/collection/process_sard.py` | Complete implementation with 4 critical fixes |
| `tests/test_story3.py` | New — 10 test cases, 47 assertions |
| `docs/STORY3_SARD_COLLECTOR.md` | This documentation |

---

### Critical Bug Fixed

`collect()` called `self._find_all_testcases_roots()` — but only `_find_juliet_root()` existed
(which didn't navigate to `testcases/`). The script would crash immediately with `AttributeError`.

**Fix:** Replaced with `_iter_testcases_roots()` — a generator that streams roots one at a time
(avoids loading all 64,104 paths into RAM, enables `--max-samples` early exit):

```python
def _iter_testcases_roots(self):
    found_any = False
    for candidate in self.sard_root.rglob("testcases"):
        if not candidate.is_dir():
            continue
        has_cwe = any(
            d.is_dir() and d.name.startswith("CWE")
            for d in candidate.iterdir()
        )
        if not has_cwe:
            continue
        found_any = True
        yield candidate
    if not found_any:
        raise FileNotFoundError(...)
```

This auto-discovers the correct `testcases/` directory regardless of nesting:
```
data/raw/sard/
└── 2022-08-11-juliet-c-cplusplus-v1-3-1-with-extra-support (1)/
    └── 61945-v1.0.0/
        └── src/
            └── testcases/   <-- auto-found via rglob
                ├── CWE121_Stack_Based_Buffer_Overflow/
                ├── CWE122_Heap_Based_Buffer_Overflow/
                └── ...
```

### Secondary Fix

Dry-run mode used an inline `from schema import validate_sample` (line 374) which fails when
running as a module (`training.scripts.collection.process_sard`). Moved `validate_sample` to
the module-level try/except import block.

---

### How the Collector Works

1. **Discovery** — `rglob("testcases")` finds the testcases root inside the Juliet zip structure
2. **CWE filtering** — Only processes 7 M1 target CWEs (CWE-78, 121, 122, 134, 190, 416, 476)
3. **File reading** — UTF-8 fast path → chardet fallback → latin-1 final fallback
4. **Parsing** — tree-sitter C parser extracts all `function_definition` AST nodes
5. **Name extraction** — Recursive `_find_identifier()` handles nested declarators
   (e.g. `pointer_declarator → function_declarator → identifier`)
6. **Classification** — Substring match: `"bad"` → label=1, `"good"` → label=0, helper names → skip
7. **CFA pairing** — Files with both bad+good functions get a shared `pair_id` (UUID4)
8. **Validation** — `validate_sample()` enforces schema (5–500 non-blank lines, 10+ tokens, etc.)
9. **Deduplication** — MD5 hash of whitespace-normalized code; duplicates silently dropped
10. **Checkpointing** — Atomic JSON checkpoint every 100 files; survives keyboard interrupt

### Production Issues Addressed

| # | Issue | Fix |
|---|-------|-----|
| 1 | SARD files `#include "std_testcase.h"` (custom headers, not compilable) | No compile-check; NIST labels functions by name convention |
| 2 | Some .c files use Windows-1252 / Latin-1 encoding | chardet fallback + latin-1 last resort |
| 3 | Tree-sitter declarator may be nested (`pointer_declarator → ...`) | Recursive `_find_identifier()` traversal |
| 4 | Label imbalance (1 bad, 1–5 good variants per file) | Logged as metric; Stage 2 MinHash dedup reduces near-duplicate good variants |
| 5 | Multi-part test cases (bad/good across multiple files) | `pair_id` assigned per file; unpaired functions get `pair_id=""` |
| 6 | Very short helper functions (< 5 lines) | `validate_sample()` rejects them automatically |
| 7 | `.cpp` files in Juliet | Only `*.c` processed; C++ skipped |

### Output

```
training/data/raw/sard/
├── sard_samples.jsonl        # valid samples (JSONL, one JSON object per line)
├── sard_failed.jsonl         # rejected samples with rejection reason (debug)
└── checkpoints/
    └── sard_checkpoint.json  # resume state
```

Each sample in `sard_samples.jsonl`:
```json
{
  "id":            "uuid4",
  "source":        "sard",
  "code":          "void CWE121_..._bad() { ... }",
  "label":         1,
  "cwe":           "CWE-121",
  "language":      "c",
  "collected_at":  "2026-03-10T18:00:00.000000Z",
  "pair_id":       "uuid4-or-empty",
  "file_path":     "data/raw/sard/.../CWE121_..._01.c",
  "function_name": "CWE121_Stack_Overflow__char_type_overrun_memcpy_01_bad",
  "cfa_origin":    "native",
  "severity_score": 0.0,
  "commit_sha":    "",
  "cve_id":        "",
  "aliases":       {},
  "metadata":      {}
}
```

### Tests Written (`tests/test_story3.py`)

| Test | What It Verifies |
|------|-----------------|
| 1 | `_iter_testcases_roots()` auto-discovers deep Juliet structure (generator) |
| 2 | `_extract_cwe()` maps all 7 target CWEs + rejects non-targets |
| 3 | `_classify()` labels bad/good/skip/unclassified functions correctly |
| 4 | Full end-to-end collection (17 assertions: schema, labels, CWEs, pairs, filtering) |
| 5 | Dry-run mode validates but does not write files |
| 6 | `--max-samples` stops collection at limit |
| 7 | Non-UTF-8 files decoded via chardet fallback |
| 8 | Checkpoint saves and round-trips correctly |
| 9 | `FileNotFoundError` raised when no testcases/ found |
| 10 | Deduplication: identical function code saved only once |

**Result: 47/47 assertions pass.**

---

## How to Run the Collector

### Prerequisites

```bash
pip install tree-sitter tree-sitter-c chardet loguru tenacity requests
```

Verify installation:
```bash
python -c "import tree_sitter_c; from tree_sitter import Language, Parser; print('OK')"
```

### Step 1 — Download Juliet Suite

Download from NIST SARD:
- URL: https://samate.nist.gov/SARD/test-suites/116
- File: `2022-08-11-juliet-c-cplusplus-v1-3-1-with-extra-support.zip` (approx. 300 MB)

Extract to:
```
data/raw/sard/
```

The extracted structure should be:
```
data/raw/sard/
└── 2022-08-11-juliet-c-cplusplus-v1-3-1-with-extra-support (1)/
    └── 61945-v1.0.0/
        └── src/
            └── testcases/
                ├── CWE121_Stack_Based_Buffer_Overflow/
                ├── CWE122_Heap_Based_Buffer_Overflow/
                └── ...
```

### Step 2 — Verify the structure (optional sanity check)

```bash
python -c "
from pathlib import Path
p = Path('data/raw/sard')
roots = list(p.rglob('testcases'))
cwe_roots = [r for r in roots if r.is_dir() and any(c.name.startswith('CWE') for c in r.iterdir())]
print('testcases roots found:', len(cwe_roots))
for r in cwe_roots:
    cwes = [c.name for c in r.iterdir() if c.is_dir() and c.name.startswith('CWE')]
    print(f'  {r}')
    print(f'  CWE dirs: {len(cwes)}')
"
```

### Step 3 — Dry run (always do this first)

```bash
cd C:\Users\Vimal Sajan\streamguard
python -m training.scripts.collection.process_sard \
    --sard-root data/raw/sard/ \
    --output-dir training/scripts/collection/data/raw/sard/ \
    --dry-run \
    --max-samples 20
```

Expected output:
```
DRY RUN MODE - will process but not save samples
...
SARD COLLECTION SUMMARY
Files processed:     ~10
Samples saved:       20
Label Distribution:
  vulnerable (label=1): ~10
  safe (label=0):       ~10
CWE Distribution:
  CWE-121: ...
```

### Step 4 — Full collection

```bash
python -m training.scripts.collection.process_sard \
    --sard-root data/raw/sard/ \
    --output-dir training/scripts/collection/data/raw/sard/
```

This will take ~5–15 minutes depending on disk speed. Progress is logged every 100 files.

If interrupted, resume by simply re-running the same command. The checkpoint will be loaded
automatically and previously saved samples will not be duplicated (dedup by MD5 hash).

### Step 5 — Run the Story 3 tests

```bash
python tests/test_story3.py
```

Expected: `47 passed, 0 failed out of 47`

### Step 6 — Verify output

```bash
python -c "
import json
from collections import Counter

samples = [json.loads(l) for l in open('training/scripts/collection/data/raw/sard/sard_samples.jsonl')]
print('Total:', len(samples))
print('Labels:', Counter(s['label'] for s in samples))
print('CWEs:', Counter(s['cwe'] for s in samples).most_common())

pairs = {}
for s in samples:
    if s['pair_id']:
        pairs.setdefault(s['pair_id'], []).append(s['label'])
valid = {k: v for k, v in pairs.items() if 0 in v and 1 in v}
broken = {k: v for k, v in pairs.items() if 0 not in v or 1 not in v}
print('Valid CFA pairs:', len(valid))
print('Broken pairs:', len(broken))
assert len(broken) == 0, 'BROKEN PAIRS FOUND!'
print('CHECKPOINT PASS')
"
```

Expected:
```
Total: 15,000–40,000 samples
Labels: Counter({0: ~20000, 1: ~10000})   # safe > vuln due to good variants
CWEs: [('CWE-121', ...), ('CWE-122', ...), ...]   # all 7 present
Valid CFA pairs: > 1000
Broken pairs: 0
CHECKPOINT PASS
```

---

## Possible Issues and Solutions

### Issue 1: `tree_sitter_c` not installed
**Symptom:** `ImportError: tree_sitter_c not installed` at startup
**Fix:**
```bash
pip install tree-sitter-c
# If that fails:
pip install tree-sitter==0.21.3 tree-sitter-c
```

### Issue 2: `tree_sitter.Language` API changed
**Symptom:** `TypeError` on `Language(tsc.language())` — tree-sitter 0.22+ changed the API
**Fix:** Pin to the compatible version:
```bash
pip install "tree-sitter>=0.21,<0.23" tree-sitter-c
```

### Issue 3: `chardet` version warning
**Symptom:** `RequestsDependencyWarning: urllib3 or chardet doesn't match a supported version`
**Fix:** This is a `requests` library warning, not a functional error. Safe to ignore. Or:
```bash
pip install "chardet>=3.0.2,<4" requests
```

### Issue 4: Juliet zip extracts to unexpected structure
**Symptom:** `FileNotFoundError: No testcases/ directory with CWE* subdirectories found`
**Diagnosis:**
```bash
python -c "
from pathlib import Path
for p in Path('data/raw/sard').rglob('*'):
    if p.is_dir() and p.name == 'testcases':
        print(p)
"
```
**Fix:** Ensure the zip is extracted directly into `data/raw/sard/` (not double-nested).
The collector uses `rglob` so it handles 1–3 levels of nesting automatically.

### Issue 5: Windows path with spaces or parentheses
**Symptom:** Shell quoting issues when running the CLI
**Fix:** Always quote paths with spaces:
```bash
python -m training.scripts.collection.process_sard \
    --sard-root "data/raw/sard/2022-08-11-juliet-c-cplusplus-v1-3-1-with-extra-support (1)" \
    --output-dir training/scripts/collection/data/raw/sard/
```
Or just pass the parent: `--sard-root data/raw/sard/` — the rglob will find testcases/ automatically.

### Issue 6: Very low sample count (< 5,000)
**Symptom:** Summary shows < 5,000 `samples_saved`
**Diagnosis:** Check `samples_failed` in summary. If high:
```bash
# Inspect rejected samples
python -c "
import json
with open('training/scripts/collection/data/raw/sard/sard_failed.jsonl') as f:
    for line in list(f)[:5]:
        item = json.loads(line)
        print(item['reason'])
"
```
**Common causes:**
- All functions < 5 non-blank lines → Juliet version mismatch, wrong directory
- All CWEs skipped → CWE directory names don't match `CWE_DIR_TO_CANONICAL` keys

### Issue 7: Label heavily skewed (e.g. 80% label=0)
**Symptom:** `samples_saved: 0 (label=1)` or very few
**Explanation:** Juliet files typically have 1 `bad()` function and 2–5 `good*()` variants.
This is expected. Stage 2 MinHash deduplication will reduce near-duplicate good variants.
The skew is logged in the summary as `Ratio: X% vuln / Y% safe`.
**Action:** No action needed at collection stage. Note the ratio and proceed.

### Issue 8: Broken CFA pairs (`assert len(broken) == 0` fails)
**Symptom:** Verification script reports broken pairs
**Status:** This issue is fully fixed in the current implementation.

**Root causes (both handled):**
1. Schema rejection — thin `_44_good()` wrapper functions (`{ goodG2B(); }`) have <10 tokens and fail `validate_sample()`. Fixed by pre-validating before assigning `pair_id`.
2. MD5 deduplication — the same sink function appears across hundreds of files; later files find it's a duplicate and skip saving it, leaving the other side orphaned. Fixed by `_is_duplicate()` pre-check before assigning `pair_id`.

**If it still occurs** (e.g. after code changes), inspect:
```bash
python -c "
import json
samples = [json.loads(l) for l in open('training/scripts/collection/data/raw/sard/sard_samples.jsonl')]
pairs = {}
for s in samples:
    if s['pair_id']:
        pairs.setdefault(s['pair_id'], []).append(s['label'])
broken = {k: v for k, v in pairs.items() if 0 not in v or 1 not in v}
print('Broken pairs:', len(broken))
for pid, labels in list(broken.items())[:3]:
    print(f'  pair_id={pid}: labels={labels}')
"
```

### Issue 9: Collection restarts from scratch despite checkpoint
**Symptom:** Logs show `0 existing hashes for dedup` on restart
**Explanation:** The checkpoint saves *position* metadata, but dedup is based on the
existing `sard_samples.jsonl` file (hashed on startup). As long as the JSONL file is intact,
restarts are safe — duplicates are filtered by MD5 hash. The checkpoint metadata is informational.
**Fix:** Ensure `sard_samples.jsonl` is not deleted between runs.

### Issue 10: UnicodeEncodeError in Windows terminal
**Symptom:** `UnicodeEncodeError: 'charmap' codec can't encode character`
**Cause:** Windows cp1252 console can't display UTF-8 box-drawing or emoji characters from loguru
**Fix:**
```bash
set PYTHONIOENCODING=utf-8
python -m training.scripts.collection.process_sard ...
# Or redirect output:
python -m training.scripts.collection.process_sard ... > output.log 2>&1
```

---

## Expected Final Output

After full collection, `sard_samples.jsonl` should have:

| Metric | Actual (Juliet 116, 7 CWEs) |
|--------|----------------------------|
| Total samples | 53,666 |
| Label=1 (vulnerable) | 22,371 (41.7%) |
| Label=0 (safe) | 31,295 (58.3%) |
| Valid CFA pairs | 18,687 |
| Broken pairs | 0 |
| CWEs present | All 7 (CWE-78, 121, 122, 134, 190, 416, 476) |

Stage 1 preprocessing (comment strip, `#ifdef` strip, length recheck) will reduce this further.
Near-duplicate deduplication (MinHash for `goodG2B1`/`goodG2B2` variants) is a **Stage 2** task.
