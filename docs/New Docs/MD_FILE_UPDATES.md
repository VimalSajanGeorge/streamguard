# StreamGuard MD File Updates — Exact Replacement Sections
# CFA 5-Tier Strategy Integration
# March 2026 | Replace these sections in-place. All other content stays unchanged.

---

## HOW TO USE THIS FILE

For each file below:
1. Open the target file
2. Find the exact section marked `── FIND: exact text to locate ──`
3. Replace that section entirely with the content under `── REPLACE WITH ──`
4. Save. Nothing else in the file changes.

---

---
# FILE 1: docs/PREPROCESSING.md
---

## CHANGE: Replace entire Stage 3 section

── FIND: locate this heading and replace everything until the next `---` separator ──

```
## Stage 3: CFA Generation

**Script**: `training/scripts/preprocessing/stage3_cfa.py`  
**Input**: `data/processed/deduped/samples.jsonl` (vulnerable samples only)  
**Output**: `data/processed/with_cfa/samples.jsonl` (originals + CFA pairs)
```

...all content through the closing `run_stage3` function and the `---` separator...

── REPLACE WITH the entire block below ──

---

## Stage 3: CFA Generation

**Script**: `training/scripts/preprocessing/stage3_cfa.py`  
**Input**: `data/processed/deduped/samples.jsonl` (vulnerable samples only)  
**Output**: `data/processed/with_cfa/samples.jsonl` (originals + CFA pairs)

> [!IMPORTANT]
> **Stage 3 uses a 5-Tier generation strategy.** Each CWE is routed to the
> generation method that matches its structural complexity. See
> `docs/CFA_Generation_Research.md` and `docs/CFA_Stories.md` for the full
> implementation specification. This section documents the authoritative
> configuration; those documents contain the detailed Claude Code prompts.

> [!CAUTION]
> **SARD samples (source starting with `"sard"` or `cfa_type == "native"`) MUST
> be skipped.** SARD's `_bad.c` / `_good.c` pairs are already native CFA pairs
> assigned in `process_sard.py`. Re-generating LLM CFAs for SARD destroys the
> existing `pair_id` linkage and replaces NIST-validated pairs with unvalidated
> LLM output. Add this check at the top of the generation loop before any other
> processing.

### CWE Tier Assignment

```python
# CWE_TIER_MAP — authoritative assignment, do not change without updating
# the exemplar DB and per-tier prompts in cfa_tier1.py through cfa_tier5.py

CWE_TIER_MAP = {
    # Tier 1: Deterministic AST mutation — no LLM, 100% structural validity
    "CWE-134": 1,   # Format String:        printf(var) → printf("%s", var)
    "CWE-120": 1,   # Buffer Copy:          strcpy → strncpy + sizeof
    "CWE-476": 1,   # NULL Deref:           add NULL guard before dereference

    # Tier 2: Zero-shot LLM with Option A/B/C structured prompts
    "CWE-121": 2,   # Stack Buffer Overflow: bounds check before array write
    "CWE-122": 2,   # Heap Buffer Overflow:  NULL check + size validation
    "CWE-125": 2,   # Out-of-Bounds Read:    bounds check before array read

    # Tier 3: 3-step Chain-of-Thought LLM
    "CWE-89":  3,   # SQL Injection:         parameterized query
    "CWE-78":  3,   # OS Command Injection:  execve with fixed args
    "CWE-190": 3,   # Integer Overflow:      pre-operation overflow check
    "CWE-79":  3,   # XSS (C web):           HTML-encode output

    # Tier 4: Dynamic few-shot exemplar + CoT — requires exemplar DB
    "CWE-416": 4,   # Use-After-Free:        ptr=NULL after free + NULL check
    "CWE-119": 4,   # Buffer Overflow (gen): bounds validation for buffer ops
}
```

### Tier Summary

| Tier | CWEs | Strategy | API Calls | Target Compile Rate |
|------|------|----------|-----------|---------------------|
| 1 | CWE-134, CWE-120, CWE-476 | AST rule-based (no LLM) | 0 | 100% |
| 2 | CWE-121, CWE-122, CWE-125 | Zero-shot structured prompt | ~6/sample | ≥ 83% |
| 3 | CWE-89, CWE-78, CWE-190, CWE-79 | 3-step CoT prompt | ~7/sample | ≥ 75% |
| 4 | CWE-416, CWE-119 | Few-shot exemplar + CoT | ~10/sample | ≥ 65% |
| 5 | All (fallback) | Critique-and-refine | ~2/sample | varies |

### 7-Gate Validation System

Every generated CFA must pass all applicable gates before being written to output.
Gates 1–4 are hard (rejection). Gate 5 is soft (quality score reduction). Gates 6–7
are conditional.

```python
# Gate 1: Identity — CFA must differ from original
# Gate 2: Similarity bounds — 0.55 <= SequenceMatcher ratio <= 0.99
#          (Tier 1 uses stricter lower bound: 0.70)
# Gate 3: Compilation — gcc -fsyntax-only returns 0
# Gate 4: Vuln pattern removed — CWE-specific regex NOT found in CFA
# Gate 5: Fix signature present — expected fix pattern IS found (SOFT: fail = quality_score 0.6)
# Gate 6: Taint path broken — source→sink regex path broken in CFA
#          (ACTIVE ONLY for: CWE-89, CWE-78, CWE-134, CWE-79)
# Gate 7: CPG diff budget — Joern diff shows ≤ CWE_CPG_LIMIT changed nodes
#          (OPTIONAL: disabled by default, enable with --enable-cpg-diff)

VULN_PATTERNS = {
    "CWE-120": [r'\bstrcpy\s*\(', r'\bgets\s*\(', r'\bstrcat\s*\('],
    "CWE-89":  [r'sprintf\s*\([^,]+,\s*[^"]*%s', r'mysql_query\s*\(.*\+'],
    "CWE-78":  [r'system\s*\(.*argv', r'popen\s*\(.*user'],
    "CWE-134": [r'printf\s*\(\s*\w+\s*\)', r'fprintf\s*\([^,]+,\s*\w+\s*\)'],
    "CWE-416": [r'free\s*\(\w+\)(?!\s*;\s*\w+\s*=\s*NULL)'],
    "CWE-190": [r'\w+\s*\+\s*\w+(?!.*INT_MAX)', r'\w+\s*\*\s*\w+(?!.*overflow)'],
    "CWE-476": [r'\*\w+(?!.*!=\s*NULL)(?!.*==\s*NULL)'],
    "CWE-121": [r'\bstrcpy\s*\(', r'\bwcscpy\s*\('],
    "CWE-122": [r'\bmemcpy\s*\(.*\+'],
    "CWE-125": [r'\w+\[\w+\](?!.*<.*len)', r'\w+\[\w+\](?!.*size)'],
}

FIX_SIGNATURES = {
    "CWE-134": [r'"%s"', r'"%d"', r'"%f"'],
    "CWE-120": [r'\bstrncpy\s*\(', r'\bstrncat\s*\(', r'\bsnprintf\s*\(', r'\bfgets\s*\('],
    "CWE-89":  [r'sqlite3_prepare', r'mysql_stmt', r'mysql_real_escape'],
    "CWE-78":  [r'\bexecve\s*\(', r'whitelist', r'\bstrncmp\s*\('],
    "CWE-190": [r'INT_MAX\s*-', r'INT_MAX\s*/', r'__builtin_.*overflow'],
    "CWE-416": [r'NULL\s*;', r'==\s*NULL', r'!=\s*NULL'],
    "CWE-476": [r'==\s*NULL\s*\)', r'!=\s*NULL'],
    "CWE-121": [r'\bsizeof\s*\(', r'strncpy', r'snprintf'],
    "CWE-122": [r'\bsizeof\s*\(', r'NULL\s*\)', r'strncpy'],
    "CWE-125": [r'>=\s*0', r'<\s*\w+_\w*(size|len|count)', r'!=\s*NULL'],
}

# CWE-specific CPG diff node budget (Gate 7, if enabled)
CWE_CPG_DIFF_LIMIT = {
    "CWE-134": 3, "CWE-476": 5, "CWE-120": 5,
    "CWE-121": 8, "CWE-122": 8, "CWE-125": 8,
    "CWE-89": 10, "CWE-78": 10, "CWE-190": 8, "CWE-79": 10,
    "CWE-416": 20,  # higher limit: multiple free() + NULL assignments
    "CWE-119": 12,
}
```

### Output Schema Additions

CFA samples written by Stage 3 now include two additional optional fields:

```python
cfa_sample = {
    **original_sample,
    "id":               str(uuid.uuid4()),
    "code":             cfa_code,
    "label":            0,                    # safe (label flipped)
    "pair_id":          pair_id,              # links to original vuln sample
    "cfa_type":         "llm_generated",      # or "ast_generated" for Tier 1
    "source":           f"{source}_cfa",
    # NEW FIELDS:
    "cfa_tier":         tier,                 # int 1-5, which strategy produced this
    "cfa_quality_score": quality_score,       # float 0.0-1.0 from 7-gate validation
}
```

### Quality Report

Stage 3 writes `data/processed/cfa_quality_report.json` on completion. This file
is read by `pre_training_audit.py` for per-CWE quality threshold checks.

```python
# Per-CWE metrics in cfa_quality_report.json:
{
    "CWE-X": {
        "total_vuln_samples": int,
        "cfa_attempts": int,
        "cfa_accepted": int,
        "compile_rate": float,           # gate3_passed / (attempts - identical)
        "pattern_removal_rate": float,   # gate4_passed / compile_passed
        "fix_signature_rate": float,     # gate5_not_flagged / compile_passed
        "taint_break_rate": float,       # gate6_passed / compile_passed (injection only)
        "avg_changed_lines": float,
        "avg_similarity": float,
        "rejection_reasons": {
            "identical": int,
            "too_similar": int,
            "too_different": int,
            "compile_fail": int,
            "vuln_pattern_remains": int,
            "no_fix_signature": int,
            "taint_path_intact": int,
        },
        "pairs_generated": int,
        "avg_pairs_per_vuln": float,
    }
}
```

### CLI

```bash
python training/scripts/preprocessing/stage3_cfa.py \
    --input  data/processed/deduped/samples.jsonl \
    --output data/processed/with_cfa/samples.jsonl \
    --exemplar-db data/processed/exemplar_db.json \  # required for Tier 4 CWEs
    --target-ratio 2.0 \
    --checkpoint-dir training/checkpoints/ \
    --dry-run \
    --max-samples N \
    --cwe-filter CWE-X        # process only one CWE (for testing)
```

### Module Structure

Stage 3 is split across these files — do not merge them:

| File | Purpose |
|------|---------|
| `stage3_cfa.py` | Orchestrator: `CWE_TIER_MAP`, `TIER_GENERATORS` registry, `run_stage3_tiered()` |
| `cfa_tier1.py` | Tier 1 AST rules: `fix_cwe134`, `fix_cwe120`, `fix_cwe476` |
| `cfa_tier2.py` | Tier 2 zero-shot: prompts + `extract_c_code_v2()` |
| `cfa_tier3.py` | Tier 3 CoT: prompts + Gate 6 taint check |
| `cfa_tier4.py` | Tier 4 few-shot: `build_tier4_prompt()` + `validate_cwe416_structural()` |
| `cfa_tier5.py` | Tier 5 fallback: `tier5_refine()` + `FEEDBACK_TEMPLATES` |
| `cfa_exemplar_db.py` | Offline tool: builds Tier 4 exemplar DB from real CVE pairs |

---

---
# FILE 2: CLAUDE.md
---

## CHANGE 1: Replace the Key Design Decisions table

── FIND this exact table header and replace the full table ──

```
## Key Design Decisions and Why

| Decision | Alternative | Reason |
|----------|-------------|--------|
| CodeBERT (not GraphCodeBERT) | ...
```

── REPLACE WITH ──

---

## Key Design Decisions and Why

| Decision | Alternative | Reason |
|----------|-------------|--------|
| CodeBERT (not GraphCodeBERT) | GraphCodeBERT | Vul-LMGNN ablation: CodeBERT+GGNN > GraphCodeBERT alone on C vulnerability tasks |
| 3-layer GGNN (not deeper) | 4-5 layers | Over-smoothing in deeper GNNs on CPG graphs (Vul-LMGNN ablation RQ4) |
| Cross-attention fusion (not concat) | Simple concatenation | Cross-attention allows BERT to attend to structural graph features directly. **Must be node-level** (BERT queries attend to N graph nodes via scatter softmax), not graph-embed level (single 256-d vector as K/V) — graph-embed cross-attention mathematically degrades to simple concatenation |
| TPG as 4th CPG component | AST+CFG+DFG only | Explicit taint paths critical for injection CWEs (CWE-89/78/79) |
| **5-Tier CFA generation (not single strategy)** | **Single LLM prompt per CWE** | **CWE difficulty varies 10x: CWE-134 (format string) needs 1-line deterministic fix; CWE-416 (use-after-free) needs lifetime tracking. Tier 1 AST rules give 100% structural validity for simple CWEs; Tier 4 few-shot exemplar CoT required for CWE-416. Single-strategy wastes API budget on simple CWEs and fails on complex ones. See `CFA_Stories.md`.** |
| CFA margin contrastive loss | BCE loss only | VISION: BCE alone allows spurious correlations; contrastive loss forces structural separation |
| tree-sitter for function extraction | regex | regex fails on nested braces, #ifdef blocks, function pointers |
| Joern subprocess (not library) | pycparser | pycparser can't produce CFG/DFG; Joern is the only open-source tool for full CPG |
| HDF5 cache for graphs | re-run Joern | Joern takes ~700ms/function; HDF5 reload is ~2ms |

---

## CHANGE 2: Replace the Critical Paths section

── FIND this exact section header and replace through the closing `---` ──

```
## Critical Paths — What Breaks Everything

These are the single points of failure in the whole system:

1. **`run.ossdataflow` in Joern script** ...
```

── REPLACE WITH ──

---

## Critical Paths — What Breaks Everything

These are the single points of failure in the whole system:

1. **`run.ossdataflow` in Joern script** — Without this, DFG edges don't exist. The GGNN has no data flow signal.
2. **`pair_id` linkage in CFA pairs** — If broken, contrastive loss trains on random pairs. F1 will not improve.
3. **Edge index bounds validation** — `edge_index.max() < num_nodes` must be checked before every training run.
4. **CFA-aware DataLoader** — Standard PyG DataLoader shuffles pairs apart. Custom loader required.
5. **Taint role labels on nodes** — If all nodes have `taint_role='none'`, the TPG component has zero signal.
6. **Edge type masking in `encode_graph()`** — Each of the 4 GatedGraphConv modules MUST receive only its own edge type (mask by `edge_attr == etype`). If any module receives the full unmasked `edge_index`, the per-type architecture silently degrades to type-blind. No error is thrown.
7. **`CWE_TIER_MAP` dispatch in `stage3_cfa.py`** — Assigning a CWE to the wrong tier degrades CFA quality silently. Tier 1 applied to a CWE without a clear AST fix rule returns empty CFAs. Tier 2/3 applied to CWE-416 produces ~40% failure rate and weak pairs. Never change tier assignments without re-running Stage 3 and re-verifying `cfa_quality_report.json` thresholds.
8. **SARD skip in Stage 3** — If the `source.startswith('sard')` check is missing, Stage 3 overwrites SARD's native `pair_id` linkage with new LLM-generated CFAs. The original NIST-validated `good()` functions become orphaned singletons. L_CFA trains on LLM quality rather than ground-truth quality.
9. **`pair_id` saved as HDF5 attribute in Stage 6** — If `grp.attrs['pair_id']` is missing, `CFAAwareBatchSampler` treats every sample as a singleton. L_CFA trains on random pairs. The entire CFA research contribution appears not to work.

---

## CHANGE 3: Replace the Ask Before Assuming section

── FIND this exact section and replace it ──

```
## Ask Before Assuming

If any specification is unclear, ask before implementing. Common ambiguity points:

- **Which Joern version?** ...
```

── REPLACE WITH ──

---

## Ask Before Assuming

If any specification is unclear, ask before implementing. Common ambiguity points:

- **Which Joern version?** → Use latest stable (4.x). The API changed significantly from 1.x. `joern-parse` in 4.x includes DFG by default; older versions require `run.ossdataflow` in REPL mode.
- **Which CodeBERT checkpoint?** → `microsoft/codebert-base` (not graphcodebert, not codet5)
- **CFA ratio target?** → 2 CFA pairs per vulnerable sample minimum; 3 is target. CWE-416 may achieve only 1.5 pairs/sample — this is acceptable.
- **Max CPG nodes?** → 200 after context slicing. Graph with > 200 nodes gets sliced.
- **Training batch size?** → 8 graphs per batch (GPU memory), gradient accumulation × 4 = effective batch 32
- **Which CFA tier for a given CWE?** → See `CWE_TIER_MAP` in `stage3_cfa.py`. Do NOT change assignments without re-running Stage 3 + verifying `cfa_quality_report.json`.
- **Do SARD samples get LLM CFA generation?** → NO. SARD already has native CFA pairs from `process_sard.py`. Stage 3 skips any sample where `source.startswith('sard')` or `cfa_type == 'native'`.
- **What is `cfa_tier` field?** → Integer 1-5. 1 = AST rule, 2 = zero-shot, 3 = CoT, 4 = few-shot, 5 = critique-refine. Written to JSONL by Stage 3. Stored in HDF5 attrs by Stage 6. Used for diagnostics and optional loss weighting.

---

---
# FILE 3: docs/DATA_PIPELINE.md
---

## CHANGE: Replace the Canonical Sample Schema section

── FIND this exact heading and replace the full code block (through the closing ```) ──

```python
def validate_sample(s: dict) -> tuple[bool, list[str]]:
    """Returns (is_valid, list_of_errors). All errors must be empty to save."""
    errors = []
    
    # Required fields
    if not s.get("id"):           errors.append("missing id")
    ...
    return len(errors) == 0, errors

def make_sample_id() -> str:
    return str(uuid.uuid4())

def make_timestamp() -> str:
    return datetime.utcnow().isoformat() + "Z"
```

── REPLACE WITH ──

```python
# training/scripts/collection/schema.py

from dataclasses import dataclass, field
from typing import Optional
import uuid, re
from datetime import datetime

VALID_CWE = {
    "CWE-89", "CWE-78", "CWE-79", "CWE-119", "CWE-120",
    "CWE-121", "CWE-122", "CWE-125", "CWE-134", "CWE-190",
    "CWE-416", "CWE-476"
}
VALID_SOURCES = {
    "sard", "exploitdb", "cve", "github_advisory",
    "osv", "repo", "manual",
    "sard_cfa", "exploitdb_cfa", "cve_cfa",        # CFA variants
    "github_advisory_cfa", "osv_cfa", "repo_cfa",
}

# Optional fields with default values. validate_sample() does NOT require these.
# They are populated by Stage 3 CFA generation and carried through all downstream stages.
OPTIONAL_FIELD_DEFAULTS = {
    "cfa_tier":           0,      # int 1-5: which CFA generation tier produced this sample
                                   # 0 = not a CFA sample (original or SARD native pair)
                                   # 1 = AST rule  2 = zero-shot  3 = CoT
                                   # 4 = few-shot  5 = critique-refine
    "cfa_quality_score":  1.0,    # float 0.0-1.0 from 7-gate validation
                                   # 1.0 = all gates passed  0.6 = Gate 5 soft fail
    "severity_score":     0.0,    # CVSS float, used by L_severity head in training
    "commit_sha":         "",     # 40-char SHA for cross-source dedup and Stage 7 split
    "cve_id":             "",     # for cross-source dedup and Stage 7 split grouping
    "cfa_type":           "",     # "native" (SARD pairs) | "llm_generated" | "ast_generated"
    "aliases":            {},     # {cve: "...", ghsa: "...", osv: "..."}
    "metadata":           {},     # source-specific extra fields
}

def validate_sample(s: dict) -> tuple[bool, list[str]]:
    """Returns (is_valid, list_of_errors). All errors must be empty to save.
    
    Only REQUIRED fields are enforced. Optional fields (cfa_tier, cfa_quality_score,
    severity_score, etc.) are never rejected — they default to 0 or "" if absent.
    """
    errors = []
    
    # Required fields
    if not s.get("id"):           errors.append("missing id")
    if not s.get("source"):       errors.append("missing source")
    if not s.get("code"):         errors.append("missing code")
    if s.get("label") not in [0, 1]: errors.append("label must be 0 or 1")
    if not s.get("cwe"):          errors.append("missing cwe")
    if not s.get("language"):     errors.append("missing language")
    if not s.get("collected_at"): errors.append("missing collected_at")
    
    # Value validation
    if s.get("source") and s["source"] not in VALID_SOURCES:
        errors.append(f"invalid source: {s['source']}")
    if s.get("cwe") and s["cwe"] not in VALID_CWE:
        errors.append(f"invalid CWE: {s['cwe']}")
    if s.get("language") and s["language"] != "c":
        errors.append("language must be 'c'")
    
    # Code sanity
    code = s.get("code", "")
    lines = code.splitlines()
    if len(lines) < 5:   errors.append(f"code too short: {len(lines)} lines")
    if len(lines) > 500: errors.append(f"code too long: {len(lines)} lines")
    tokens = code.split()
    if len(tokens) < 10: errors.append(f"too few tokens: {len(tokens)}")
    
    return len(errors) == 0, errors

def make_sample_id() -> str:
    return str(uuid.uuid4())

def make_timestamp() -> str:
    return datetime.utcnow().isoformat() + "Z"
```

Also replace the **Canonical Sample Schema** dict comment block that shows the schema fields.

── FIND this block ──

```python
{
    # Required fields
    "id":           str,     # UUID4
    "source":       str,     # "sard" | "exploitdb" | "cve" | "github_advisory" | "osv" | "repo" | "manual"
    "code":         str,     # complete C function, compilable
    "label":        int,     # 1=vulnerable, 0=safe
    "cwe":          str,     # "CWE-89" etc.
    "language":     str,     # always "c" for this project
    "collected_at": str,     # ISO8601 timestamp

    # Optional fields
    "cve_id":       str,     # "CVE-2023-1234" if known
    "pair_id":      str,     # links CFA pair: same UUID for (vuln, safe) pair
    "commit_sha":   str,     # git commit SHA if from repo/CVE
    "repo_url":     str,     # GitHub repo URL
    "file_path":    str,     # path within repo
    "reviewer_verified": bool, # manually verified label
    "metadata":     dict,    # source-specific extra fields
}
```

── REPLACE WITH ──

```python
{
    # ── REQUIRED FIELDS (validate_sample() enforces these) ──────────────────
    "id":           str,     # UUID4
    "source":       str,     # "sard" | "exploitdb" | "cve" | "github_advisory"
                             # | "osv" | "repo" | "manual" | "{source}_cfa"
    "code":         str,     # complete C function, compilable
    "label":        int,     # 1=vulnerable, 0=safe
    "cwe":          str,     # "CWE-89" etc. — must be in VALID_CWE
    "language":     str,     # always "c" for this project
    "collected_at": str,     # ISO8601 UTC timestamp

    # ── PAIRING FIELDS (set by collectors + Stage 3) ─────────────────────────
    "pair_id":      str,     # links CFA pair: same UUID for (vuln, safe) pair
                             # "" for singletons; SARD native pairs set in process_sard.py
    "cfa_type":     str,     # "native" (SARD) | "llm_generated" | "ast_generated" | ""

    # ── CFA QUALITY FIELDS (set by Stage 3, read by audit + training) ───────
    "cfa_tier":           int,   # 0=original, 1=AST, 2=zero-shot, 3=CoT, 4=few-shot, 5=refine
    "cfa_quality_score":  float, # 0.0-1.0 from 7-gate validation; 0.6 = Gate 5 soft fail

    # ── DEDUP + SPLIT FIELDS (set by collectors, used by Stage 2 + Stage 7) ─
    "commit_sha":   str,     # full 40-char git SHA (not abbreviated) or ""
    "cve_id":       str,     # "CVE-2023-1234" or ""
    "repo_url":     str,     # GitHub repo URL or ""
    "file_path":    str,     # path within repo or ""

    # ── TRAINING FIELDS ───────────────────────────────────────────────────────
    "severity_score":    float, # CVSS proxy 0.0-10.0; 0.0 if unknown (used by L_severity)
    "reviewer_verified": bool,  # manually verified label

    # ── EXTENSIBLE ────────────────────────────────────────────────────────────
    "aliases":      dict,    # {cve: "...", ghsa: "...", osv: "..."} cross-references
    "metadata":     dict,    # source-specific extra fields
}
```

---

---
# FILE 4: PRD.md
---

## CHANGE 1: Replace FR-13 row in the Functional Requirements table

── FIND this exact row ──

```
| FR-13 | CFA generation pipeline | P0 | ≥ 80% compile rate; ≥ 60% taint elimination |
```

── REPLACE WITH ──

```
| FR-13 | CFA generation pipeline (5-tier) | P0 | Per-CWE compile rates: Tier 1 CWEs ≥ 95%; Tier 2 CWEs ≥ 83%; Tier 3 CWEs ≥ 75%; Tier 4 CWEs ≥ 65%. Overall compile rate ≥ 80%. Pattern removal rate ≥ 70%. Fix signature rate ≥ 70% (Gate 5). Taint break rate ≥ 70% (Gate 6, injection CWEs). Results verified by `cfa_quality_report.json`. |
```

---

## CHANGE 2: Replace Section 6.2 CFA Augmentation Targets

── FIND this entire subsection ──

```
### 6.2 CFA Augmentation Targets

| CWE | Real Samples (min) | CFA Target | CFA Ratio |
|-----|--------------------|------------|-----------|
| CWE-120 | 3,000 | 9,000 | 3:1 |
...
| Others (7 types) | 800 each | 2,400 each | 3:1 |
```

── REPLACE WITH ──

### 6.2 CFA Augmentation Targets

| CWE | Strategy Tier | Real Samples (min) | CFA Target | CFA Ratio | Min Compile Rate |
|-----|---------------|--------------------|------------|-----------|-----------------|
| CWE-120 | Tier 1 (AST) | 3,000 | 9,000 | 3:1 | ≥ 95% |
| CWE-89 | Tier 3 (CoT) | 2,000 | 6,000 | 3:1 | ≥ 75% |
| CWE-416 | Tier 4 (Few-shot) | 1,500 | 3,000 | 2:1 | ≥ 65% |
| CWE-476 | Tier 1 (AST) | 1,500 | 4,500 | 3:1 | ≥ 90% |
| CWE-78 | Tier 3 (CoT) | 1,200 | 3,600 | 3:1 | ≥ 75% |
| CWE-122 | Tier 2 (Zero-shot) | 1,200 | 3,600 | 3:1 | ≥ 83% |
| CWE-134 | Tier 1 (AST) | 1,000 | 3,000 | 3:1 | ≥ 95% |
| CWE-121 | Tier 2 (Zero-shot) | 1,000 | 3,000 | 3:1 | ≥ 83% |
| CWE-190 | Tier 3 (CoT) | 800 | 2,400 | 3:1 | ≥ 78% |
| CWE-125 | Tier 2 (Zero-shot) | 800 | 2,400 | 3:1 | ≥ 80% |
| CWE-119 | Tier 4 (Few-shot) | 800 | 1,600 | 2:1 | ≥ 65% |
| CWE-79 | Tier 3 (CoT) | 500 | 1,000 | 2:1 | ≥ 68% |

> **Note on CWE-416 and CWE-119:** Tier 4 (few-shot) generates fewer pairs per sample than
> Tier 1-3 due to higher failure rate. Accept 2:1 ratio (not 3:1) for these CWEs.
> Supplement with SARD native pairs where available.

---

## CHANGE 3: Replace Section 6.3 Dataset Quality Gates

── FIND this entire subsection ──

```python
REQUIRED_CHECKS = {
    "min_train_samples":   30_000,      # absolute minimum
    "vuln_safe_balance":   (0.45, 0.55), # label balance range
    "cwe_diversity":       7,            # minimum CWE types with ≥ 500 samples
    "max_cwe_dominance":   0.45,         # relaxed for M1 — CWE-121/122 dominate Juliet
    "no_null_code":        0,            # zero samples with empty code field
    "test_train_no_overlap": 0,          # zero commit SHA shared between train/test
    "code_length_range":   (10, 4096),   # token count range (CodeBERT max)
    "pair_integrity":      0,            # zero broken CFA pairs
    "manual_verified_min": 800,          # minimum manually verified samples
}
```

── REPLACE WITH ──

```python
# pre_training_audit.py — All 21 checks must PASS before training begins.
# Checks 1-9: dataset structure (unchanged from original spec)
# Checks 10-21: per-CWE CFA quality (NEW — reads cfa_quality_report.json)

REQUIRED_CHECKS = {
    # ── DATASET STRUCTURE CHECKS (1-9) ──────────────────────────────────────
    "min_train_samples":     30_000,       # absolute minimum unique samples
    "vuln_safe_balance":     (0.25, 0.75), # NOTE: relaxed from (0.45,0.55) because
                                            # CFA augmentation adds label=0 pairs,
                                            # shifting balance toward 25-30% vuln
    "cwe_diversity":         7,             # minimum CWE types with ≥ 500 samples
    "max_cwe_dominance":     0.45,          # relaxed for M1 — CWE-121/122 dominate Juliet
    "no_null_code":          0,             # zero samples with empty code field
    "test_train_no_overlap": 0,             # zero commit SHA shared between train/test
    "code_length_range":     (10, 4096),    # token count range (CodeBERT max)
    "pair_integrity":        0,             # zero broken CFA pairs
    "manual_verified_min":   800,           # minimum manually verified samples

    # ── PER-CWE CFA QUALITY CHECKS (10-21) ──────────────────────────────────
    # Read from: data/processed/cfa_quality_report.json
    # Generated by: stage3_cfa.py on completion
    # If cfa_quality_report.json does not exist: WARN but do not FAIL
    "cfa_quality_CWE_134": {"compile_rate": 0.95, "fix_signature_rate": 0.90},
    "cfa_quality_CWE_120": {"compile_rate": 0.90, "fix_signature_rate": 0.88},
    "cfa_quality_CWE_476": {"compile_rate": 0.88, "fix_signature_rate": 0.80},
    "cfa_quality_CWE_121": {"compile_rate": 0.83, "fix_signature_rate": 0.75},
    "cfa_quality_CWE_122": {"compile_rate": 0.80, "fix_signature_rate": 0.72},
    "cfa_quality_CWE_125": {"compile_rate": 0.78, "fix_signature_rate": 0.70},
    "cfa_quality_CWE_89":  {"compile_rate": 0.78, "taint_break_rate":   0.72},
    "cfa_quality_CWE_78":  {"compile_rate": 0.75, "taint_break_rate":   0.70},
    "cfa_quality_CWE_190": {"compile_rate": 0.80, "fix_signature_rate": 0.70},
    "cfa_quality_CWE_79":  {"compile_rate": 0.68, "fix_signature_rate": 0.60},
    "cfa_quality_CWE_119": {"compile_rate": 0.65, "fix_signature_rate": 0.58},
    "cfa_quality_CWE_416": {"compile_rate": 0.60, "fix_signature_rate": 0.52},
}

# M1 relaxed thresholds (use --m1 flag when Phase 2 collection is incomplete)
REQUIRED_CHECKS_M1 = {
    **{k: v for k, v in REQUIRED_CHECKS.items() if not k.startswith("cfa_quality")},
    "min_train_samples":  5_000,
    "vuln_safe_balance":  (0.25, 0.75),
    "cwe_diversity":      5,
    "min_taint_coverage": 0.20,
    # CFA quality checks: same thresholds as M2 when cfa_quality_report.json exists
    # If report does not exist yet (Stage 3 not run): skip CFA checks silently
}
```

---

---
# FILE 5: docs/MODEL.md
---

## CHANGE: Add tier-weighted loss section after StreamGuardLoss class

── FIND this exact line (end of StreamGuardLoss class) ──

```
        losses["total"] = total.item()
        return total, losses
```

── ADD the following block IMMEDIATELY AFTER that closing ``` of the class code block ──

---

### Optional: Tier-Weighted Contrastive Loss

When `cfa_tier` metadata is available in the batch (populated by `CFAAwareBatchSampler`
from HDF5 attrs), you can optionally weight the L_CFA term by generation confidence.
Tier 1 (deterministic AST) pairs are the most reliable; Tier 5 (critique-refine) pairs
carry slight uncertainty. This is an **optional enhancement** — the default `StreamGuardLoss`
above works correctly without it and should be used for all ablation configs A–E.

```python
# Optional tier-weighted variant — only use in Config E (Full StreamGuard)
# Enable with: loss = StreamGuardLoss(use_tier_weighting=True)

TIER_CONFIDENCE = {
    1: 1.00,   # AST rule — deterministic, structurally guaranteed
    2: 0.90,   # zero-shot LLM — compile-validated
    3: 0.82,   # CoT LLM — taint-path validated for injection CWEs
    4: 0.75,   # few-shot LLM — exemplar-guided but still LLM
    5: 0.65,   # critique-refine — accepted after failure, lower confidence
    0: 1.00,   # native SARD pair — NIST-validated ground truth
}

class StreamGuardLossTierWeighted(StreamGuardLoss):
    """
    Extends StreamGuardLoss with CFA tier confidence weighting on L_CFA.
    All other losses (L_CE, L_severity) are unchanged.
    
    Use only when cfa_tier is available in batch metadata.
    Falls back to standard L_CFA when tier info is absent.
    """
    
    def forward(self, outputs_orig, outputs_cfa=None, labels=None,
                cwe_labels=None, severity_labels=None, cfa_tiers=None):
        """
        cfa_tiers: optional (B_pairs,) int tensor of tier values 0-5.
                   If None: falls back to standard unweighted L_CFA.
        """
        # Compute standard losses (L_CE, L_severity) from parent
        total, losses = super().forward(
            outputs_orig, outputs_cfa=None,  # skip CFA in parent
            labels=labels, cwe_labels=cwe_labels, severity_labels=severity_labels
        )
        
        # Override L_CFA with tier-weighted version
        if outputs_cfa is not None and outputs_orig is not None:
            emb_v  = outputs_orig["embedding"]
            emb_vp = outputs_cfa["embedding"]
            cosine_sim = F.cosine_similarity(emb_v, emb_vp, dim=-1)
            margin_loss = F.relu(cosine_sim - (-self.margin))
            
            if cfa_tiers is not None:
                weights = torch.tensor(
                    [TIER_CONFIDENCE.get(int(t), 0.80) for t in cfa_tiers],
                    dtype=torch.float32, device=emb_v.device
                )
                l_cfa = (margin_loss * weights).mean()
            else:
                l_cfa = margin_loss.mean()
            
            losses["L_CFA"] = l_cfa.item()
            total = total + self.lambda_cfa * l_cfa
        
        losses["total"] = total.item()
        return total, losses
```

> **When to use:** Only in Config E (Full StreamGuard) when `cfa_tier` is populated in
> HDF5 by Stage 6 and forwarded by `CFAAwareBatchSampler`. For ablation configs A–D,
> use the standard `StreamGuardLoss` to keep comparisons clean.

---

---
# END OF FILE UPDATES
---

# Quick Summary: What Changed and Why

| File | What Changed | Why |
|------|-------------|-----|
| PREPROCESSING.md | Stage 3 section fully replaced | Single-strategy → 5-tier dispatch with CWE_TIER_MAP, 7-gate validation, module structure, quality report spec |
| CLAUDE.md | Key Design Decisions table + Critical Paths + Ask Before Assuming | New row for 5-tier CFA; 3 new critical paths (7/8/9); new ambiguity answers |
| DATA_PIPELINE.md | Schema: added cfa_tier, cfa_quality_score + updated field docs | Stage 3 now writes these fields; downstream stages must preserve them |
| PRD.md | FR-13, Section 6.2, Section 6.3 | Per-CWE compile rate targets; relaxed label balance (CFA augmentation shifts to 25-30% vuln); 12 new audit checks |
| MODEL.md | Added optional StreamGuardLossTierWeighted after StreamGuardLoss | Optional enhancement for Config E; keeps ablation configs clean |
