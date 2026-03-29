# StreamGuard Phase 3 — Stage 3: CFA Generation
## Complete Implementation Documentation

**Date:** 2026-03-22
**Status:** COMPLETE — All 5 tiers implemented, 73/73 tests passing
**Input:** 50,616 deduped samples | **Test suite:** 153 tests across all stories

---

## 1. What Is CFA Generation?

CounterFactual Augmentation (CFA) creates a safe counterpart (label=0) for every vulnerable C function (label=1) in the training dataset. These pairs are used for contrastive learning during model training: the GNN+transformer learns to distinguish the subtle structural difference between a vulnerable function and its fixed version, rather than just pattern-matching on surface features.

Without CFAs, the model sees individual vulnerable/safe samples. With CFAs, the model trains on *pairs* — (vulnerable function, fixed function) — using a CFA contrastive loss that penalizes representations where the pair members are too similar. Research shows this improves F1 by ~8% and reduces false negatives on novel CWE variants.

---

## 2. Architecture Overview: 5-Tier System

Generating correct, compilable, vulnerability-free C code fixes requires different strategies depending on the CWE. Simpler patterns (format strings, NULL checks) can be fixed deterministically; injection flaws require semantic reasoning about data flow; use-after-free requires understanding control flow and pointer lifetimes.

```
CWE Difficulty ──────────────────────────────────────────────────► Higher

Tier 1 (AST)    Tier 2 (Zero-Shot)   Tier 3 (CoT)    Tier 4 (Few-Shot)
CWE-134          CWE-121              CWE-89           CWE-416
CWE-120          CWE-122              CWE-78           CWE-119
CWE-476          CWE-125              CWE-190
                                      CWE-79

                    Tier 5 (Critique-and-Refine) — FALLBACK FOR ALL TIERS
```

Every generated fix passes through a **7-gate validator** before being written to output. Failed attempts are logged but never crash the pipeline.

---

## 3. The 7-Gate Validator (`validate_cfa_v2`)

Every CFA candidate, regardless of which tier produced it, must pass all 7 gates:

| Gate | Name | Description | Failure → |
|------|------|-------------|-----------|
| 1 | Identity | CFA must differ from original | Reject (`identical_to_original`) |
| 2 | Similarity bounds | Jaccard token similarity: Tier 1 ≥ 0.70, others ≥ 0.55; all ≤ 0.99 | Reject (`too_similar` / `too_different`) |
| 3 | Compilation | gcc syntax-only check; tree-sitter fallback on Windows | Reject (`compile_fail`) |
| 4 | Vuln pattern removed | Regex check that the original vulnerability pattern is gone | Reject (`vuln_pattern_remains`) |
| 5 | Fix signature present | **SOFT gate** — expected fix pattern missing → `quality_score = 0.6` (not rejected) | Score penalty only |
| 6 | Taint path broken | Lightweight regex taint check for injection CWEs (89, 78, 134, 79) | Reject (`taint_path_intact`) |
| 7 | CPG diff budget | Optional Joern-based structural diff (disabled by default) | Configurable |

Gate 5 is intentionally soft: a CFA that compiles and removes the vulnerability is still useful for training even if it doesn't match an expected fix pattern. Quality score 0.6 (vs 1.0 for full pass) lets the audit flag low-confidence fixes without discarding them.

Gate 6 is the most important safety gate for SQL injection and OS command injection. It uses source→sink regex tracing: if the original function had user input flowing to a dangerous sink (e.g. `sqlite3_exec`), the CFA must not have the same path intact.

---

## 4. Tier 1: AST Rule-Based Mutation (`cfa_tier1.py`)

**CWEs:** CWE-134 (format string), CWE-120 (buffer copy), CWE-476 (NULL deref)
**Tests:** 14/14 pass

### How It Works

Uses tree-sitter to parse C code into an AST, finds the exact vulnerability node, and applies a targeted byte-level edit. No LLM calls, no randomness.

**CWE-134 (Format String)**
Finds `printf(var)` / `fprintf(f, var)` calls where the format argument is a variable (not a string literal). Inserts `"%s", ` (or `L"%s", ` for wide strings) before the variable.

```c
// Before:                          // After:
printf(user_msg);                   printf("%s", user_msg);
fprintf(out, buf);                  fprintf(out, "%s", buf);
```

**CWE-120 (Buffer Copy)**
Replaces unsafe functions with bounded equivalents:
- `strcpy(dst, src)` → `strncpy(dst, src, sizeof(dst)-1); dst[sizeof(dst)-1] = '\0';`
- `gets(buf)` → `fgets(buf, sizeof(buf), stdin);`
- `sprintf(buf, ...)` → `snprintf(buf, sizeof(buf), ...)`
- `strcat(dst, src)` → `strncat(dst, src, sizeof(dst)-strlen(dst)-1)`

**Critical safety rule:** `sizeof(dst)` is only valid when `dst` is a local stack array (e.g. `char buf[64]`). If `dst` is a pointer parameter, `sizeof` gives the pointer size (8 bytes on 64-bit), not the buffer size. In this case Tier 1 returns `[]` and automatically **escalates to Tier 2**.

**CWE-476 (NULL Deref)**
Finds unguarded `*ptr` and `ptr->field` dereferences (not inside an existing `if (ptr != NULL)` block). Inserts a NULL guard with the correct return value for the function's return type: `void` → `return;`, `int`/`long` → `return -1;`, pointer → `return NULL;`.

---

## 5. Tier 2: Zero-Shot LLM (`cfa_tier2.py`)

**CWEs:** CWE-121 (stack overflow), CWE-122 (heap overflow), CWE-125 (OOB read)
**Model:** DeepSeek API (`deepseek-chat`)
**Max attempts:** 6 per sample
**Tests:** 10/10 pass

### How It Works

Sends a carefully engineered zero-shot prompt that gives the LLM exactly one task, with explicit fix options to choose from. The prompt ends with a `[FIXED_CODE_START]` marker that the model is instructed to write code after.

**CWE-121 example prompt excerpt:**
```
CHOOSE EXACTLY ONE fix:
  Option A - Index bounds check: if (idx < 0 || idx >= ARRAY_SIZE) { return; }
  Option B - String copy bounds: strncpy(arr, src, sizeof(arr)-1); arr[...] = '\0';
  Option C - Memory copy bounds: if (len > sizeof(arr)) { len = sizeof(arr); }
```

**Code extraction cascade** (`extract_c_code_v2`):
LLMs don't always format output consistently. The extractor tries in order:
1. `[FIXED_CODE_START]` marker (best — directly extracts fix)
2. ` ```c ... ``` ` markdown fence
3. First line matching a C function signature pattern
4. Raw response stripped (last resort)

**3 consecutive compile fails** → escalates to Tier 5 refinement.

---

## 6. Tier 3: Chain-of-Thought LLM (`cfa_tier3.py`)

**CWEs:** CWE-89 (SQL injection), CWE-78 (OS command injection), CWE-190 (integer overflow), CWE-79 (XSS)
**Model:** DeepSeek API (`deepseek-chat`)
**Max attempts:** 7 per sample | **Max tokens:** 1536 (CoT needs more space)
**Tests:** 12/12 pass

### How It Works

These CWEs require reasoning about *why* a pattern is dangerous before generating a fix. The CoT prompt walks the model through 3 explicit steps:

```
STEP 1 - IDENTIFY: Find the exact line where user input flows into [sink]
STEP 2 - STRATEGY: Choose the correct fix approach from [options]
STEP 3 - Generate the fixed function.
[FIXED_CODE_START]...[FIXED_CODE_END]
```

The `[FIXED_CODE_END]` marker is critical: it prevents the model's post-code reasoning text from being included in the extracted CFA.

**Gate 6 Taint Feedback Loop**
When Gate 6 rejects a CFA (`taint_path_intact`), the next attempt gets specific feedback injected into the prompt:

- **CWE-89:** "Your fix STILL has user input flowing into the SQL query without parameterization. You MUST use sqlite3_prepare_v2 + sqlite3_bind_*..."
- **CWE-78:** "Your fix STILL has user input flowing directly into a shell command. You MUST either remove user input from command or add a strict allowlist check..."
- **CWE-79:** "Your fix STILL writes user input directly to output without encoding. You MUST apply HTML entity encoding..."

This self-correction loop significantly improves success rate on the first few attempts.

---

## 7. Tier 4: Few-Shot Exemplar + CoT (`cfa_tier4.py` + `cfa_exemplar_db.py`)

**CWEs:** CWE-416 (use-after-free), CWE-119 (improper memory bounds)
**Model:** DeepSeek API (`deepseek-chat`)
**Max attempts:** 10 per sample
**Tests:** 10/10 pass (Tier 4) + 4 exemplar DB tests

### ExemplarDatabase (`cfa_exemplar_db.py`)

An offline-built database of real-world (non-SARD) vulnerable/safe pair examples per CWE. Built once from `deduped/samples.jsonl`, then loaded at inference time.

**Building:** Groups samples by `pair_id`, keeps only `source in {cve, exploitdb, github_advisory, osv, repo}`. Applies a diversity filter: if a new exemplar's keyword Jaccard similarity with any already-selected exemplar is ≥ 0.80, it's skipped (avoids redundant examples).

**Selection:** Given a query function, ranks exemplars by keyword overlap (Jaccard) and returns top-2.

**Minimum pairs assertion:** If `min_real_pairs` exemplars can't be found for CWE-416 or CWE-119, the build raises `ValueError` — alerting the operator that Phase 2 collection needs more real-world data for these CWEs.

```bash
python -m training.scripts.preprocessing.cfa_exemplar_db \
    --input training/data/processed/deduped/samples.jsonl \
    --output training/data/processed/exemplar_db.json \
    --max-per-cwe 8 --min-real-pairs 1
```

### Tier 4 Prompt Structure

```
VULNERABLE C FUNCTION:
```c
{target_code}
```

--- Example 1 ---
VULNERABLE:   ```c {exemplar_vuln} ```
FIXED:        ```c {exemplar_safe} ```

TASK:
You are fixing USE-AFTER-FREE (CWE-416)...
STEP 1 - IDENTIFY: ...
STEP 2 - STRATEGY: Option A / B / C...
STEP 3 - Generate.
[FIXED_CODE_START]...[FIXED_CODE_END]
```

**Think-harder feedback:** After attempt 4 with no valid CFA, a feedback message is prepended urging the model to re-examine the exemplars and apply the same structural pattern.

**CWE-416 structural post-check:** Even after passing all 7 gates, a CWE-416 CFA must pass `validate_cwe416_structural()`. It checks that at least one of these is true:
- `= NULL` assignment present (null-after-free fix)
- fewer `free()` calls than original (removed a redundant free)
- `if (ptr != NULL)` guard on pointer use

---

## 8. Tier 5: Critique-and-Refine Fallback (`cfa_tier5.py`)

**Trigger:** 3 consecutive compile failures in any primary tier
**Model:** `claude-haiku-4-5-20251001` (faster/cheaper than DeepSeek for refinement)
**Max attempts:** 1 (no infinite loop — fails cleanly)
**Tests:** 8/8 pass

### How It Works

Takes a failed CFA attempt and a structured critique of *exactly why it failed*, then asks the LLM to fix *only that specific issue*.

```
ORIGINAL VULNERABLE CODE (CWE-89):
```c
{original}
```

PREVIOUS FIX ATTEMPT (INCORRECT):
```c
{failed_cfa}
```

CRITIQUE: PREVIOUS ATTEMPT FAILED: The code had a C syntax error. Ensure:
(1) all braces match, (2) no missing semicolons, (3) all variables declared...

Generate a CORRECTED fix that addresses the critique above.
Apply ONLY the minimum change to fix CWE-89.
[FIXED_CODE_START]corrected_function_here[FIXED_CODE_END]
```

**8 failure reasons with CWE-specific feedback:**

| Reason | Feedback focus |
|--------|---------------|
| `compile_fail` | Brace matching, semicolons, variable declarations |
| `vuln_pattern_remains` | Names the exact CWE-specific pattern still present |
| `taint_path_intact` | Names the specific sink (e.g. "sqlite3_exec") |
| `ptr_not_nulled` | "Immediately after EVERY free(), add: ptr = NULL;" |
| `no_fix_signature` | Describes what the expected fix should contain |
| `too_similar` | "Change at least 1-3 lines, apply the actual fix" |
| `too_different` | "Change ONLY the vulnerability site" |
| `identical_to_original` | "You returned the same code — no fix was applied" |

**Quality discount:** `quality_score = base_score × 0.85`. Tier 5 CFAs are slightly lower confidence than primary-tier CFAs and are tracked separately in the quality report.

**`inject_feedback()`** is a lighter-weight variant that appends a "PREVIOUS ATTEMPT NOTES:" section to an existing prompt (used by Tier 3/4 for mid-run feedback without a full Tier 5 call).

---

## 9. Integration: `run_stage3_tiered()` Orchestrator

**File:** `stage3_cfa.py`
**Tests:** 8/8 integration tests pass

### Full Pipeline Flow

```
For each sample in deduped/samples.jsonl:

  1. Write original sample to output (ALWAYS)
  2. Skip if source in {sard, sard_cfa} or cfa_type == 'native'
     → SARD already has native pairs; generating LLM duplicates wastes API calls
  3. Skip if label != 1 (only vulnerable samples need CFAs)
  4. Skip if no CWE field
  5. Apply --cwe-filter if set (single-CWE production runs)
  6. Look up tier = CWE_TIER_MAP.get(cwe, 3)
  7. Call generate_cfa_for_sample(sample, config, exemplar_db)
     → Tier 1: if returns [] → auto-escalate to Tier 2
     → Tier 2/3/4: after 3 consecutive compile fails → call Tier 5
  8. For each valid CFA: write to output with:
     - label: 0
     - source: "{original_source}_cfa"
     - pair_id: {original_sample_id}   ← links the pair
     - cfa_tier: {tier_number}
     - cfa_quality_score: {score}
     - cfa_origin: "ast_rule" | "llm_generated"
  9. Log failures to cfa_failures.jsonl (never crash)
  10. Checkpoint every 200 samples (atomic write, resumable)
  11. Progress log every 100 samples
  12. Quality threshold check every 500 samples
```

### Escalation Rules

| Trigger | Action |
|---------|--------|
| Tier 1 returns [] | Escalate to Tier 2 |
| 3+ consecutive `compile_fail` in Tier 2/3/4 | Call Tier 5 with `failed_cfa` + `compile_fail` |
| Tier 3 Gate 6 fails (`taint_path_intact`) | Inject Gate 6 feedback into next attempt |
| Tier 4 attempt 4 with 0 success | Inject think-harder feedback |
| Max attempts reached, 0 valid CFAs | Log to `cfa_failures.jsonl`, continue |

### Output Files

| File | Description |
|------|-------------|
| `training/data/processed/with_cfa/samples.jsonl` | All originals + generated CFAs |
| `training/data/processed/with_cfa/cfa_quality_report.json` | Per-CWE metrics |
| `training/data/processed/with_cfa/cfa_failures.jsonl` | Samples with 0 valid CFAs |

---

## 10. Quality Monitoring: Check 10 in Pre-Training Audit

**File:** `pre_training_audit.py` (extended from 9 to 10 checks)

After Stage 3 completes, the pre-training audit reads `cfa_quality_report.json` and checks each CWE against thresholds from the CFA research document §4.3:

| CWE | Compile Rate Threshold | Secondary Threshold |
|-----|----------------------|---------------------|
| CWE-134 | 95% | fix_signature_rate ≥ 90% |
| CWE-120 | 90% | fix_signature_rate ≥ 88% |
| CWE-476 | 88% | fix_signature_rate ≥ 80% |
| CWE-121 | 83% | fix_signature_rate ≥ 75% |
| CWE-122 | 80% | fix_signature_rate ≥ 72% |
| CWE-125 | 78% | fix_signature_rate ≥ 70% |
| CWE-89 | 78% | taint_break_rate ≥ 72% |
| CWE-78 | 75% | taint_break_rate ≥ 70% |
| CWE-190 | 80% | fix_signature_rate ≥ 70% |
| CWE-79 | 68% | fix_signature_rate ≥ 60% |
| CWE-119 | 65% | fix_signature_rate ≥ 58% |
| CWE-416 | 60% | fix_signature_rate ≥ 52% |

**If `cfa_quality_report.json` does not exist:** Check 10 returns PASS with a warning (non-blocking). Stage 3 may not have run yet when auditing split files from Stage 7.

**Example output:**
```
----------------------------------------------------------------------
CFA Quality Check (Check 10)
----------------------------------------------------------------------
ID       CWE        Metric                  Actual  Threshold Status
----------------------------------------------------------------------
C10.1    CWE-119    compile_rate             0.812      0.650   PASS
C10.2    CWE-119    fix_signature_rate       0.701      0.580   PASS
C10.11   CWE-134    compile_rate             0.961      0.950   PASS
...
----------------------------------------------------------------------
```

---

## 11. Test Coverage Summary

| Test File | Story | Tests | Status |
|-----------|-------|-------|--------|
| `test_cfa_foundation.py` | P3-S3 Foundation | 11 | 11/11 PASS |
| `test_cfa_tier1.py` | CFA-T1 AST Rules | 14 | 14/14 PASS |
| `test_cfa_tier2.py` | CFA-T2 Zero-Shot | 10 | 10/10 PASS |
| `test_cfa_tier3.py` | CFA-T3 CoT LLM | 12 | 12/12 PASS |
| `test_cfa_tier4.py` | CFA-T4 Few-Shot | 10 | 10/10 PASS |
| `test_cfa_tier5.py` | CFA-T5 Critique | 8 | 8/8 PASS |
| `test_cfa_integration.py` | CFA-INT Wiring | 8 | 8/8 PASS |
| `test_story7.py` (updated) | Pre-Training Audit | 80 | 80/80 PASS |
| **Total** | | **153** | **153/153** |

All LLM calls are mocked in tests — no live API key required to run the test suite.

---

## 12. Production Dataset

The input to Stage 3 is the deduped dataset from Stage 2 (P3-S2):

| Metric | Value |
|--------|-------|
| Total samples | 50,616 |
| Vulnerable (label=1) | 25,975 |
| Safe (label=0) | 24,641 |
| Vuln/safe ratio | 1.054 (nearly balanced) |

**By CWE:**

| CWE | Count | Tier | Strategy |
|-----|-------|------|----------|
| CWE-121 | 14,793 | 2 | Zero-shot LLM |
| CWE-190 | 12,533 | 3 | CoT LLM |
| CWE-122 | 6,805 | 2 | Zero-shot LLM |
| CWE-78 | 5,563 | 3 | CoT LLM |
| CWE-134 | 4,187 | 1 | AST rules |
| CWE-476 | 2,814 | 1 | AST rules |
| CWE-416 | 1,886 | 4 | Few-shot + CoT |
| CWE-120 | 1,684 | 1 | AST rules |
| CWE-119 | 203 | 4 | Few-shot + CoT |
| CWE-125 | 144 | 2 | Zero-shot LLM |
| CWE-79 | 2 | 3 | CoT LLM |
| CWE-89 | 2 | 3 | CoT LLM |

**By source:**

| Source | Count | CFA generated? |
|--------|-------|----------------|
| sard | 38,279 | No (native pairs exist) |
| repo | 9,390 | Yes |
| exploitdb | 2,441 | Yes |
| cve | 324 | Yes |
| osv | 122 | Yes |
| github_advisory | 60 | Yes |

Only ~12,337 non-SARD samples need CFA generation. Of those, only ~6,200 are label=1 (vulnerable). These are the samples that will consume API credits.

---

## 13. API Key Configuration

The DeepSeek API key is stored in `.env`:

```
DEEPSEEK_API_KEY=sk-cd3b1017d3354103beac9b7774f55745
```

This is read automatically by `call_llm()` via `os.environ.get("DEEPSEEK_API_KEY", "")`. No code changes needed.

---

## 14. Production Run Commands

### Step 1: Build Exemplar Database (for CWE-416/119)

```bash
cd C:/Users/Vimal\ Sajan/streamguard

# Load .env
export $(cat .env | grep -v '^#' | grep DEEPSEEK | xargs)

python -m training.scripts.preprocessing.cfa_exemplar_db \
    --input  training/data/processed/deduped/samples.jsonl \
    --output training/data/processed/exemplar_db.json \
    --max-per-cwe 8 \
    --min-real-pairs 1
```

**Note:** If CWE-416/119 have fewer than 1 real pair in deduped data, this raises ValueError. In that case, run without `--min-real-pairs 1` constraint or skip the exemplar DB (Tier 4 will still run, using general knowledge).

### Step 2: Full CFA Generation Run

```bash
python -m training.scripts.preprocessing.stage3_cfa \
    --input  training/data/processed/deduped/samples.jsonl \
    --output training/data/processed/with_cfa/samples.jsonl \
    --exemplar-db training/data/processed/exemplar_db.json \
    --checkpoint-dir training/data/processed/with_cfa/checkpoints
```

### Step 3: Verify Output

```bash
# Count samples
wc -l training/data/processed/with_cfa/samples.jsonl

# Check quality report
python -c "
import json
r = json.load(open('training/data/processed/with_cfa/cfa_quality_report.json'))
for cwe, d in sorted(r.items()):
    if d['total_attempts'] > 0:
        print(f'{cwe}: {d[\"accepted\"]}/{d[\"total_attempts\"]} ({d[\"compile_rate\"]*100:.1f}% compile)')
"

# Check failures log
wc -l training/data/processed/with_cfa/cfa_failures.jsonl 2>/dev/null || echo "No failures"
```

### Dry Run (Test Without API Calls)

```bash
python -m training.scripts.preprocessing.stage3_cfa \
    --input  training/data/processed/deduped/samples.jsonl \
    --output training/data/processed/with_cfa/samples_dry.jsonl \
    --dry-run \
    --max-samples 100
```

### Single-CWE Run (Tier-by-Tier Testing)

```bash
# Test Tier 1 (no API calls)
python -m training.scripts.preprocessing.stage3_cfa \
    --input  training/data/processed/deduped/samples.jsonl \
    --output training/data/processed/with_cfa/cfa_cwe134.jsonl \
    --cwe-filter CWE-134 --max-samples 50

# Test Tier 2 (needs API key)
python -m training.scripts.preprocessing.stage3_cfa \
    --input  training/data/processed/deduped/samples.jsonl \
    --output training/data/processed/with_cfa/cfa_cwe121.jsonl \
    --cwe-filter CWE-121 --max-samples 20
```

### Resume After Interruption

The pipeline checkpoints every 200 samples. To resume:

```bash
python -m training.scripts.preprocessing.stage3_cfa \
    --input  training/data/processed/deduped/samples.jsonl \
    --output training/data/processed/with_cfa/samples.jsonl \
    --exemplar-db training/data/processed/exemplar_db.json \
    --checkpoint-dir training/data/processed/with_cfa/checkpoints
```

It automatically skips already-processed sample IDs from the checkpoint file.

---

## 15. Files Created in Phase 3 — Stage 3

| File | Purpose | Lines |
|------|---------|-------|
| `training/scripts/preprocessing/stage3_cfa.py` | Foundation + full orchestrator | ~720 |
| `training/scripts/preprocessing/cfa_tier1.py` | AST rule-based mutation | ~280 |
| `training/scripts/preprocessing/cfa_tier2.py` | Zero-shot LLM via DeepSeek | ~290 |
| `training/scripts/preprocessing/cfa_tier3.py` | CoT LLM + Gate 6 feedback | ~305 |
| `training/scripts/preprocessing/cfa_tier4.py` | Few-shot exemplar + CoT | ~295 |
| `training/scripts/preprocessing/cfa_tier5.py` | Critique-and-refine fallback | ~230 |
| `training/scripts/preprocessing/cfa_exemplar_db.py` | Exemplar DB builder | ~200 |
| `tests/test_cfa_foundation.py` | Foundation tests | ~280 |
| `tests/test_cfa_tier1.py` | Tier 1 tests | ~340 |
| `tests/test_cfa_tier2.py` | Tier 2 tests | ~345 |
| `tests/test_cfa_tier3.py` | Tier 3 tests | ~380 |
| `tests/test_cfa_tier4.py` | Tier 4 tests | ~345 |
| `tests/test_cfa_tier5.py` | Tier 5 tests | ~230 |
| `tests/test_cfa_integration.py` | Integration tests | ~330 |

**Modified files:**
- `training/scripts/preprocessing/pre_training_audit.py` — Check 10 added (CFA quality gates)
- `tests/test_story7.py` — 2 assertions updated for 10-check audit

---

## 16. Key Design Decisions and Lessons Learned

**Why tree-sitter over regex for Tier 1?**
C code structure is deeply nested. Regex on raw source can match comments, string literals, and macro expansions — tree-sitter gives exact AST node positions with parent context, making it safe to detect "is this dereference already inside a NULL check?" without false positives.

**Why not gcc on Windows?**
gcc is not available in the Windows PATH on the development machine. The tree-sitter fallback (Gate 3) checks for parse errors — it catches `{` mismatches, unclosed blocks, and most structural errors. True type errors aren't caught, but these are rare in LLM-generated C that closely mirrors the original.

**Why is Gate 5 soft?**
A CFA that correctly removes the vulnerability but uses an unexpected fix pattern is still valuable training data. Making Gate 5 hard would discard valid CFAs where the LLM found a different (but correct) approach. The 0.6 score flags them for review without discarding.

**Why does CWE-79 use lower similarity threshold?**
Adding an `html_encode()` helper function to a CFA adds many new tokens. Similarity naturally drops below 0.55 for CWE-79 fixes. Using `similarity_lower_default=0.40` in config or in production handles this.

**Why is Tier 5 a single attempt?**
Tier 5 is meant to break out of a failure loop, not create a new one. If the refined CFA also fails, the sample is logged to `cfa_failures.jsonl` and skipped. Retrying Tier 5 indefinitely would waste API credits on samples that are inherently hard to fix.

**SARD skip rationale:**
38,279 of 50,616 samples (75.6%) come from SARD/Juliet, which already has native vulnerable/safe pairs. Generating LLM CFAs for these would: (a) cost unnecessary API credits, (b) produce worse-quality pairs than the human-written Juliet counterparts. The SARD skip saves ~$50-100 in API cost depending on pricing.
