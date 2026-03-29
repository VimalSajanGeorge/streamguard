
# docs/PREPROCESSING.md — 7-Stage Preprocessing Pipeline

> Read this before implementing: stage1_clean.py through stage7_split.py
> Also read: docs/DATA_PIPELINE.md for collector output format

---

## Overview

```
Raw JSONL (from collectors)
    │
    ├─ Stage 1: Clean & Normalize     → data/processed/cleaned/
    ├─ Stage 2: Deduplicate           → data/processed/deduped/
    ├─ Stage 3: CFA Generation        → data/processed/with_cfa/
    ├─ Stage 4: CPG Construction      → data/processed/cpg/
    ├─ Stage 5: Node Embedding        → data/processed/embedded/
    ├─ Stage 6: Graph Tensors         → data/graphs/
    └─ Stage 7: CFA-Aware Split       → data/final/{train,val,test}.h5

Run order: stage1 → stage2 → stage3 → stage4 → stage5 → stage6 → stage7
Each stage reads from previous stage output and writes to its own output dir.
```

---

## Stage 1: Clean & Normalize

**Script**: `training/scripts/preprocessing/stage1_clean.py`  
**Input**: `C:\Users\Vimal Sajan\streamguard\training\scripts\collection\data\raw\sard\sard_samples.jsonl`  
**Output**: `data/processed/cleaned/samples.jsonl`

### SARD-Specific Handling

> [!IMPORTANT]
> **SARD samples (source="sard") already have single functions** in the `code` field
> — extracted by `process_sard.py` via tree-sitter. Stage 1 must NOT re-run
> `extract_functions()` on these. Instead, it validates the code parses as a
> single `function_definition` and normalizes directly.

> [!WARNING]
> **Juliet `#ifdef` stripping**: Juliet wraps functions in `#ifndef OMITBAD` /
> `#ifndef OMITGOOD` preprocessor guards. These must be stripped BEFORE comment
> removal, otherwise Joern in Stage 4 produces incomplete CPGs. Expect ~15–20%
> rejection rate on short `good()` functions that fall below 5-line minimum after
> stripping.

```python
#!/usr/bin/env python3
"""Stage 1: Extract, normalize, and filter C functions."""

import re, os, json, chardet
from pathlib import Path
from tree_sitter import Language, Parser
import tree_sitter_c as tsc

C_LANGUAGE = Language(tsc.language())
parser = Parser(C_LANGUAGE)

MIN_LINES = 5
MAX_LINES = 500
MIN_TOKENS = 10
MAX_TOKENS = 4096

def extract_functions(source: str, filepath: str = "") -> list[dict]:
    """
    Extract all function definitions using tree-sitter (NOT regex).
    
    NOTE: This is used for non-SARD sources (CVE, ExploitDB, etc.) where
    code field contains a full file. For SARD samples (source='sard'),
    the code field is already a single function — skip re-extraction.
    """
    tree = parser.parse(source.encode("utf-8", errors="replace"))
    fns = []
    for node in tree.root_node.children:
        if node.type == "function_definition":
            code = source[node.start_byte:node.end_byte]
            decl = node.child_by_field_name("declarator")
            name = decl.text.decode("utf-8", errors="replace") if decl else "unknown"
            fns.append({
                "code": code,
                "name": name,
                "start_line": node.start_point[0],
                "end_line": node.end_point[0],
                "filepath": filepath,
            })
    return fns

def strip_preprocessor_guards(code: str) -> str:
    """
    Strip Juliet #ifdef/#ifndef/#endif blocks BEFORE normalization.
    Juliet uses these for flow variants; without stripping, Joern 
    produces incomplete CPGs in Stage 4.
    """
    return re.sub(r'#\s*(ifdef|ifndef|elif|else|endif)[^\n]*\n', '', code)

def normalize(code: str, source: str = "") -> str | None:
    """Normalize a C function. Returns None if it fails filters."""
    # 0. Strip #ifdef/#endif for SARD/Juliet sources (before comment removal)
    if source == "sard":
        code = strip_preprocessor_guards(code)
    
    # 1. Encoding: already UTF-8 at this point
    
    # 2. Remove comments
    code = re.sub(r'//[^\n]*', '', code)
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
    
    # 3. Normalize blank lines
    code = re.sub(r'\n{3,}', '\n\n', code)
    code = code.strip()
    
    # 4. Length filter
    lines = code.splitlines()
    if not (MIN_LINES <= len(lines) <= MAX_LINES):
        return None
    
    # 5. Token count filter (rough approximation)
    tokens = code.split()
    if not (MIN_TOKENS <= len(tokens) <= MAX_TOKENS):
        return None
    
    return code

def process_sample(sample: dict) -> dict | None:
    """Normalize a canonical sample. Returns None if invalid."""
    raw_code = sample.get("code", "")
    if not raw_code:
        return None
    
    source = sample.get("source", "")
    
    # Fix encoding
    if isinstance(raw_code, bytes):
        det = chardet.detect(raw_code)
        raw_code = raw_code.decode(det.get("encoding") or "utf-8", errors="replace")
    
    # SARD: code is already a single function — validate, don't re-extract
    if source == "sard":
        tree = parser.parse(raw_code.encode('utf-8', errors='replace'))
        has_function = any(
            n.type == 'function_definition' for n in tree.root_node.children
        )
        if not has_function:
            return None  # malformed — discard
        normalized = normalize(raw_code, source=source)
    else:
        normalized = normalize(raw_code, source=source)
    
    if normalized is None:
        return None
    
    return {**sample, "code": normalized}

def run_stage1(input_dirs: list[str], output_path: str):
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    
    processed, skipped = 0, 0
    with open(output, "w") as out_f:
        for dir_path in input_dirs:
            for jsonl_file in Path(dir_path).rglob("*.jsonl"):
                with open(jsonl_file) as f:
                    for line in f:
                        sample = json.loads(line.strip())
                        result = process_sample(sample)
                        if result:
                            out_f.write(json.dumps(result) + "\n")
                            processed += 1
                        else:
                            skipped += 1
    
    print(f"Stage 1 complete: {processed} processed, {skipped} skipped")
```

---

## Stage 2: Deduplicate

**Script**: `training/scripts/preprocessing/stage2_dedup.py`  
**Input**: `data/processed/cleaned/samples.jsonl`  
**Output**: `data/processed/deduped/samples.jsonl`

Four deduplication levels in order (each catches what the previous misses):

```python
import hashlib
from datasketch import MinHash, MinHashLSH

def md5_normalize(code: str) -> str:
    """Normalize whitespace before hashing."""
    return hashlib.md5(" ".join(code.split()).encode()).hexdigest()

def level1_exact(samples: list) -> list:
    """Remove exact duplicates after whitespace normalization."""
    seen = {}
    for s in samples:
        h = md5_normalize(s["code"])
        if h not in seen:
            seen[h] = s
    return list(seen.values())

def level2_cve_id(samples: list) -> list:
    """Same CVE from multiple sources → keep one per (cve_id, file_path)."""
    seen, result = set(), []
    for s in samples:
        cve = s.get("cve_id")
        key = (cve, s.get("file_path", "")) if cve else None
        if not key or key not in seen:
            result.append(s)
            if key: seen.add(key)
    return result

def level3_commit_sha(samples: list) -> list:
    """Same commit + file → keep one."""
    seen, result = {}, []
    for s in samples:
        sha = s.get("commit_sha")
        key = (sha, s.get("file_path", "")) if sha else None
        if not key or key not in seen:
            result.append(s)
            if key: seen[key] = True
    return result

def level4_minhash_lsh(samples: list, threshold: float = 0.85) -> list:
    """Near-dedup with MinHash LSH (Jaccard similarity threshold 0.85)."""
    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    result = []
    for i, s in enumerate(samples):
        m = MinHash(num_perm=128)
        # Use 4-gram character shingles for better accuracy than word tokens
        code = " ".join(s["code"].split())
        for j in range(len(code) - 3):
            m.update(code[j:j+4].encode())
        if not lsh.query(m):
            lsh.insert(str(i), m)
            result.append(s)
    return result

def run_stage2(input_path: str, output_path: str):
    samples = [json.loads(l) for l in open(input_path)]
    print(f"Stage 2 input: {len(samples)} samples")
    
    s = level1_exact(samples);     print(f"After L1 exact: {len(s)}")
    s = level2_cve_id(s);          print(f"After L2 CVE-ID: {len(s)}")
    s = level3_commit_sha(s);      print(f"After L3 SHA: {len(s)}")
    s = level4_minhash_lsh(s);     print(f"After L4 LSH: {len(s)}")
    
    with open(output_path, "w") as f:
        for sample in s:
            f.write(json.dumps(sample) + "\n")
    print(f"Stage 2 complete: {len(s)} unique samples")
```

---

## Stage 3: CFA Generation

**Script**: `training/scripts/preprocessing/stage3_cfa.py`
**Input**: `data/processed/deduped/samples.jsonl` (vulnerable samples only)
**Output**: `data/processed/with_cfa/samples.jsonl` (originals + CFA pairs)

> [!IMPORTANT]
> **Stage 3 uses a 5-Tier generation strategy.** Each CWE is routed to the
> generation method that matches its structural complexity. See
> `C:\Users\Vimal Sajan\streamguard\docs\New Docs\StreamGuard_CFA_Generation_Research.docx` and `C:\Users\Vimal Sajan\streamguard\docs\New Docs\StreamGuard_CFA_Stories.docx` for the full
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

## Stage 4: CPG Construction

**Script**: `training/scripts/preprocessing/stage4_cpg.py`  
**Input**: `data/processed/with_cfa/samples.jsonl`  
**Output**: `data/processed/cpg/*.json` (one per sample)

### SARD/Juliet-Specific Handling

> [!IMPORTANT]
> **Juliet stub header**: Juliet functions reference `std_testcase.h` headers
> (`printLine`, `printIntLine`, etc.) that don't exist locally. Joern will throw
> parse warnings without them. A stub header must be prepended before writing to
> the temp file for Joern processing.

> [!NOTE]
> **Low taint coverage on good() functions**: Juliet's `good()` (label=0) functions
> have very few taint paths by design — they ARE the fixed versions. So
> `min_taint_coverage` in the pre-training audit will be low across good samples.
> This is expected and correct. The M1 threshold of 0.20 already accounts for this.

```python
# Key components — see ARCHITECTURE.md §4 for full diagram

# Joern Scala scripts location: training/scripts/preprocessing/joern_scripts/

# generate_cpg.sc:
# @main def main(inputFile: String, outputCpg: String) = {
#   importCode(inputFile)
#   run.ossdataflow  # CRITICAL — enables DFG edges
#   save
# }

# ── Juliet Stub Header ──────────────────────────────────────────────
# Prepended to SARD samples before Joern processing.
# Without these, Joern CPG will be incomplete (missing call nodes).
JULIET_STUB = """
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
void printLine(const char* msg) {}
void printIntLine(int i) {}
void printDoubleLine(double d) {}
void printHexCharLine(char c) {}
void printLongLine(long l) {}
void printLongLongLine(long long ll) {}
void printUnsignedLine(unsigned u) {}
void printSizeTLine(size_t s) {}
"""

def write_temp_c(sample: dict, tmpdir: Path) -> Path:
    """Write sample code to temp .c file, with Juliet stub if SARD."""
    path = tmpdir / f"{sample['id']}.c"
    prefix = JULIET_STUB if sample.get('source') == 'sard' else ""
    path.write_text(prefix + "\n" + sample['code'], encoding='utf-8')
    return path

# ── Taint role labeling (Python post-process after Joern) ──────────
SOURCES     = {"scanf","gets","fgets","getenv","recv","read","fread","getchar",
               "fscanf","sscanf","getc","fgetc"}
SINKS       = {"strcpy","strcat","system","popen","mysql_query","sprintf",
               "printf","fprintf","execve","execl","execvp",
               "printLine","printIntLine","printHexCharLine","printDoubleLine"}
SANITIZERS  = {"strncpy","strncat","snprintf","validate","sanitize",
               "escape","filter","check"}

# Context slicing: 2-hop BFS from taint seeds → max 200 nodes
# See ARCHITECTURE.md §3 for context slicing algorithm
```

---

## Stage 5: Node Embedding

**Script**: `training/scripts/preprocessing/stage5_embed.py`  
**Input**: `data/processed/cpg/*.json`  
**Output**: `data/processed/embedded/*.npz` (node features per sample)

```python
# Node feature construction: 824-d vector
# [0:768]   CodeBERT [CLS] per node statement (max 64 tokens)
# [768:800] Node type one-hot (32 Joern node types)
# [800:808] Taint role one-hot (source/sink/sanitizer/propagation/none)
# [808:812] CPG component (AST=0, CFG=1, DFG=2, TPG=3)
# [812:824] Structural features (in_degree, out_degree, AST_depth, taint_dist × 3)

# IMPORTANT: Use batch_size=64 for CodeBERT inference, NOT 1
# Each node's code statement is max 64 tokens (not 512)
# Empty code nodes → zero vector fallback (do NOT skip)
```

---

## Stage 6: Graph Tensors

**Script**: `training/scripts/preprocessing/stage6_graphs.py`  
**Input**: `data/processed/cpg/*.json` + `data/processed/embedded/*.npz`  
**Output**: `data/graphs/all_graphs.h5`

```python
# PyG Data object:
# data.x          = node features (N, 824) float32
# data.edge_index = edge connectivity (2, E) int64
# data.edge_attr  = edge type (E,) int64  {AST=0, CFG=1, DFG=2, TPG=3}
# data.y          = label (1,) int64
# data.sample_id  = str
# data.pair_id    = str (empty if no CFA pair)
# data.cwe        = str

# MANDATORY validation before saving:
assert data.num_nodes >= 3, "trivial graph"
assert data.edge_index.max() < data.num_nodes, "edge index OOB"
assert not torch.isnan(data.x).any(), "NaN in features"
```

---

## Stage 7: CFA-Aware Split

**Script**: `training/scripts/preprocessing/stage7_split.py`  
**Input**: `data/graphs/all_graphs.h5`  
**Output**: `data/final/train.h5`, `data/final/val.h5`, `data/final/test.h5`

```python
# CRITICAL: Group by pair_id before any shuffling
# All members of a CFA pair MUST go to the same split
# After split: assert len(train_shas & test_shas) == 0
# Split ratio: 80/10/10 on groups (not individual samples)
```

---

## Pre-Training Validation

**Script**: `training/scripts/preprocessing/pre_training_audit.py`

Run this BEFORE any training. All 9 checks must PASS.

```python
REQUIRED_CHECKS = {
    "min_train_samples":    30_000,
    "vuln_safe_balance":    (0.45, 0.55),
    "cwe_diversity":        7,           # ≥ 7 CWE types with ≥ 500 samples each
    "max_cwe_dominance":    0.45,        # relaxed for M1 — CWE-121/122 dominate Juliet
    "no_null_code":         0,
    "test_train_no_overlap": 0,
    "code_length_range":    (10, 4096),
    "pair_integrity":       0,           # broken CFA pairs
    "min_taint_coverage":   0.20,        # relaxed for M1 — Juliet good() functions
                                         # have few taint paths by design (they are
                                         # the fixed versions)
}
```

---

*docs/PREPROCESSING.md | StreamGuard v1.0 | March 2026*
