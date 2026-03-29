# Phase 3 Stage 4: CPG Construction — Production Guide

**Date**: 2026-03-25
**Script**: `training/scripts/preprocessing/stage4_cpg.py`
**Tests**: `tests/test_story5.py` (81 Phase 1) + `tests/test_story_p3s4.py` (39 Phase 3) = 120 total
**Risk doc**: `docs/New Docs/StreamGuard_Phase3_Risk_Analysis.docx` (R-15 through R-34)

---

## Input / Output

| Item | Path | Description |
|------|------|-------------|
| **Input** | `training/data/processed/with_cfa/samples.jsonl` | Stage 3 CFA output (49,810 samples) |
| **Output** | `training/data/processed/cpg/{shard}/{sample_id}.json` | One CPG JSON per sample |
| **Failures** | `training/data/processed/cpg/cpg_failures.jsonl` | Failed samples log |
| **Stats** | `training/data/processed/cpg/cpg_stats.json` | Run statistics |
| **Checkpoint** | `training/data/processed/cpg/checkpoint.json` | Resume state (or `--checkpoint-dir`) |

**WARNING (R-33)**: The input MUST be `with_cfa/samples.jsonl`, NOT `deduped/samples.jsonl`. Using the deduped file skips CFA counterpart samples, which means L_CFA contrastive loss gets 0 pairs during training.

---

## Input Dataset Summary (49,810 samples)

### Source Distribution

| Source | Count | % |
|--------|-------|---|
| sard | 38,279 | 76.9% |
| repo | 9,390 | 18.9% |
| exploitdb | 894 | 1.8% |
| repo_cfa | 615 | 1.2% |
| cve | 324 | 0.7% |
| osv | 122 | 0.2% |
| cve_cfa | 65 | 0.1% |
| github_advisory | 60 | 0.1% |
| exploitdb_cfa | 60 | 0.1% |
| osv_cfa | 1 | 0.0% |

### CWE Distribution

| CWE | Count | % | Tier |
|-----|-------|---|------|
| CWE-121 | 14,573 | 29.3% | SARD-heavy |
| CWE-190 | 12,529 | 25.2% | SARD-heavy |
| CWE-122 | 6,679 | 13.4% | SARD-heavy |
| CWE-78 | 5,379 | 10.8% | SARD-heavy |
| CWE-134 | 4,104 | 8.2% | SARD-heavy |
| CWE-476 | 3,426 | 6.9% | Mixed |
| CWE-416 | 1,869 | 3.8% | Mixed |
| CWE-120 | 847 | 1.7% | Real-world only |
| CWE-119 | 242 | 0.5% | Real-world only |
| CWE-125 | 158 | 0.3% | Real-world only |
| CWE-79 | 2 | 0.0% | Vestigial |
| CWE-89 | 2 | 0.0% | Vestigial |

### Label & Pair Stats

- **Label distribution**: 24,428 vulnerable (label=1) / 25,382 safe (label=0)
- **Valid pairs**: 14,072 (pair_id linked, 0 orphans)
- **CFA samples**: 741 total (repo_cfa: 615, cve_cfa: 65, exploitdb_cfa: 60, osv_cfa: 1)

---

## Prerequisites

### Joern Installation

| Item | Requirement |
|------|-------------|
| Joern version | 4.0.498+ (4.x series) |
| Install location | `C:/Tools/joern-cli` (Windows) or `/opt/joern-cli` (Linux/Colab) |
| Required binaries | `joern-parse`, `joern-export` |
| JVM | Java 17+ (bundled with Joern) |

**Verify Joern:**
```bash
# Windows
C:/Tools/joern-cli/joern-parse.bat --help

# Linux/Colab
/opt/joern-cli/joern-parse --help
```

### System Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| RAM | 8 GB (2 workers) | 16 GB (4 workers) |
| Disk (input) | 200 MB | — |
| Disk (output CPGs) | 2 GB | 5 GB |
| CPU cores | 2 | 4+ |

**Memory formula**: Each worker spawns a JVM with 2 GB heap (`-Xmx2g`). Total = workers x 2 GB.
The script has an R-18 memory guard that warns if estimated usage exceeds 80% of system RAM.

---

## Pipeline (10 Steps per Sample)

```
sample.jsonl ─┐
              ├─ 1. write_temp_c()  ─── Source-aware stub prepending
              │     SARD: JULIET_STUB + strip Juliet includes
              │     Non-SARD: minimal standard includes (#include <stdio.h> etc.)
              │
              ├─ 2. joern-parse sample.c -o cpg.bin  ─── CPG with default overlays (incl DFG)
              │
              ├─ 3. joern-export cpg.bin -o export/ --repr cpg --format graphson
              │
              ├─ 4. Parse GraphSON TinkerGraph  ─── per-method subdirs, each has export.json
              │
              ├─ 5. Filter edges  ─── ONLY keep AST(0), CFG(1), DFG(2)
              │     DROP: CDG, DOMINATE, POST_DOMINATE, CONTAINS, ARGUMENT, REF, PARAMETER_LINK
              │
              ├─ 6. Label taint roles  ─── SOURCE, SINK, SANITIZER, NONE on CALL nodes only
              │
              ├─ 7. Build TPG edges  ─── BFS from SOURCE→SINK along DFG, type=3
              │     + Output-param injection (recv buf, fscanf args)
              │
              ├─ 8. Context slice  ─── 2-hop BFS from taint seeds, max 200 nodes
              │
              ├─ 9. Validate  ─── nodes >= 3, DFG > 0, no edge type >= 4
              │
              └─ 10. Save CPG JSON  ─── sharded: cpg/{first_2_chars}/{sample_id}.json
```

### Edge Types (4 only — CRITICAL)

| Type | Value | Joern Label | Notes |
|------|-------|-------------|-------|
| AST | 0 | `AST` | Abstract Syntax Tree |
| CFG | 1 | `CFG` | Control Flow Graph |
| DFG | 2 | `REACHING_DEF` (4.x) / `REACHES` (1.x) | Data Flow Graph |
| TPG | 3 | *(built by BFS, not from Joern)* | Taint Propagation Graph |

**Any edge with `type >= 4` causes CUDA index out-of-bounds crash during GGNN training.** The script has R-16 validation that rejects any CPG containing such edges.

### Source-Aware Handling

| Source | Stub Prepended | Include Stripping | Why |
|--------|---------------|-------------------|-----|
| `sard` | `JULIET_STUB` (8 print function stubs + stdio/stdlib/string) | `std_testcase.h`, `std_testcase_io.h`, `std_thread.h` removed | Juliet samples call `printLine()`, `printIntLine()` etc. which don't exist without stubs |
| All others | `HEADER_STUBS` (stdio.h, stdlib.h, string.h only) | None | Real-world samples are already self-contained |

### Taint Analysis Sets

**SOURCES** (data entry points): `scanf`, `gets`, `fgets`, `getenv`, `recv`, `read`, `fread`, `getchar`, `fscanf`, `sscanf`, `fgetc`, `getline`, `recvfrom`, `accept`, `listen`, `connect`, `fopen`, `atoi`, `strtol`, `strtoul`, `rand`, `fgetws`, `wscanf`, `fwscanf`, `cgi_param`, `query_string`

**SINKS** (damage points): `strcpy`, `strcat`, `system`, `popen`, `mysql_query`, `sprintf`, `printf`, `fprintf`, `execve`, `execl`, `execvp`, `memcpy`, `memmove`, `gets`, `vsprintf`, `vfprintf`, `wcscpy`, `wcscat`, `wmemcpy`, `swprintf`, `wprintf`, `fwprintf`, `free`, `realloc`, `malloc`, `calloc`, `printline`, `printintline`, `printhexcharline`, `mysql_real_query`, `sqlite3_exec`, `pg_exec`, `sql_exec`, `sqlite3_prepare`, `mysql_stmt_execute`, `execlp`, `execle`, `execvpe`, `pclose`, `dlopen`, `write`, `send`, `sendto`

**SANITIZERS**: `strncpy`, `strncat`, `snprintf`, `validate`, `sanitize`, `escape`, `filter`, `check`, `bound_check`, `verify`, `strlcpy`, `strlcat`, `wcsncpy`, `wcsncat`, `mysql_real_escape_string`, `sqlite3_snprintf`, `parameterize`, `prepared_statement`, `escapeshellarg`, `escapeshellcmd`, `sanitize_cmd`, `htmlspecialchars`, `html_escape`, `encode_html`

---

## Production Run Commands

### Local Windows Machine

```bash
# 1. Verify Joern is working
C:/Tools/joern-cli/joern-parse.bat --help

# 2. Dry run — validate input, show first 5 samples
python -m training.scripts.preprocessing.stage4_cpg --dry-run

# 3. Small test (10 samples) — verify full pipeline
python -m training.scripts.preprocessing.stage4_cpg --max-samples 10 --workers 1

# 4. Full production run (2 workers, ~175 hours)
python -m training.scripts.preprocessing.stage4_cpg --workers 2

# 5. Resume after interruption (automatic — reads checkpoint.json)
python -m training.scripts.preprocessing.stage4_cpg --workers 2

# 6. Custom paths
python -m training.scripts.preprocessing.stage4_cpg \
  --input training/data/processed/with_cfa/samples.jsonl \
  --output training/data/processed/cpg/ \
  --joern-dir C:/Tools/joern-cli \
  --workers 2 \
  --checkpoint-dir training/data/processed/cpg/
```

### Google Colab / Linux Server (Recommended for Speed)

```bash
# Install Joern 4.x
curl -L "https://github.com/joernio/joern/releases/latest/download/joern-install.sh" | bash
mv joern-cli /opt/joern-cli

# Verify
/opt/joern-cli/joern-parse --help

# Production run (4 workers on Colab, ~87 hours)
python -m training.scripts.preprocessing.stage4_cpg \
  --joern-dir /opt/joern-cli \
  --workers 4
```

### CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | `training/data/processed/with_cfa/samples.jsonl` | Input JSONL (Stage 3 output) |
| `--output` | `training/data/processed/cpg/` | Output directory for CPG JSONs |
| `--joern-dir` | `C:/Tools/joern-cli` | Joern installation directory |
| `--workers` | `2` | Parallel Joern workers (each uses 2 GB JVM) |
| `--dry-run` | `false` | Show what would be processed without running |
| `--max-samples` | `None` | Limit samples (for testing; disables checkpoint resume) |
| `--checkpoint-dir` | Same as `--output` | Directory for checkpoint.json |

---

## Time Estimates

| Setup | Workers | Rate | 49,810 Samples | Notes |
|-------|---------|------|----------------|-------|
| Local Windows (i7) | 1 | ~0.08/s | ~173 hours | JVM cold start per sample |
| Local Windows (i7) | 2 | ~0.16/s | ~87 hours | Default; safe for 16 GB RAM |
| Colab (CPU) | 4 | ~0.32/s | ~43 hours | No GPU needed for Joern |
| Linux server (8-core) | 4 | ~0.32/s | ~43 hours | Best balance of speed vs RAM |
| Linux server (16-core) | 8 | ~0.64/s | ~22 hours | Needs 16+ GB RAM |

**Bottleneck**: JVM cold start (~12s per joern-parse + joern-export cycle). This is CPU-bound, not GPU. Joern does not use GPU.

---

## Checkpoint / Resume

- Checkpoint saves automatically every **500 samples** and on exit (including Ctrl+C)
- File: `{output_dir}/checkpoint.json` (or `--checkpoint-dir`)
- Format: `{"completed_ids": ["id1", "id2", ...], "count": 500}`
- **Atomic writes** (R-34): `.tmp` + `os.replace()` + `.bak` backup
- To resume: just re-run the same command. The script loads the checkpoint and skips completed IDs.
- To start fresh: delete `checkpoint.json`

**NOTE**: `--max-samples` disables checkpoint resume (intended for testing only).

---

## Risk Mitigations (8 Risks from Phase 3 Analysis)

### R-15 (CRITICAL): DFG = 0 Detection
**What**: If Joern is misconfigured (`--nooverlays`), CPGs have zero data-flow edges. GGNN training fails silently.
**Mitigation**: After first 5 completed samples, if `dfg_edge_total == 0`, the script logs `CRITICAL` and calls `sys.exit(1)`. Prevents wasting days on useless CPGs.

### R-16 (CRITICAL): CDG Edge Audit
**What**: Storing CDG as `edge_type=4` causes CUDA index out-of-bounds during GGNN `edge_embedding(4)`.
**Mitigation**: Post-build assertion rejects any CPG with `edge.type >= 4`. Only `{0, 1, 2, 3}` are allowed.

### R-17 (HIGH): Subprocess Hang / Zombie JVM
**What**: Joern's JVM spawns child processes. Simple `subprocess.run()` timeout kills parent only — children survive.
**Mitigation**: `_run_subprocess_safe()` creates a new process group (POSIX: `os.setsid` + `os.killpg`; Windows: `taskkill /F /T`). Kills the entire process tree on timeout.

### R-18 (HIGH): JVM OOM
**What**: N workers x 2 GB JVM heap can exhaust system RAM.
**Mitigation**: Startup memory guard compares `workers * 2 GB` vs total system RAM. Warns if > 80% utilization. Auto-suggests reduced worker count.

### R-19 (HIGH): SARD Header Failures
**What**: SARD samples reference Juliet-internal headers (`std_testcase.h`) that don't exist on disk.
**Mitigation**: `write_temp_c()` detects SARD source, strips Juliet includes via `_strip_sard_includes()`, and prepends `JULIET_STUB` with all 8 Juliet print function stubs.

### R-20 (HIGH): Pair-Aware Discard
**What**: If one CFA pair member fails CPG, its partner becomes an orphan with no contrastive counterpart.
**Mitigation**: Post-run scan iterates all `pair_id` groups. If any member failed, its partner's CPG file is deleted. Count reported as `broken_pairs_removed` in stats.

### R-33 (HIGH): Wrong Input Path
**What**: Running Stage 4 on `deduped/samples.jsonl` instead of `with_cfa/samples.jsonl` means CFA samples never get CPGs. L_CFA contrastive loss = 0 during training.
**Mitigation**: Startup check: if input path contains "deduped" but not "with_cfa", logs a prominent warning. Also logs CFA sample count — warns if 0 CFA samples found.

### R-34 (LOW): Checkpoint Atomicity on Windows
**What**: `os.replace()` can fail on Windows if the target file is locked (e.g., antivirus scanning).
**Mitigation**: `.bak` backup before replace. On `PermissionError`, waits 500ms and retries. Falls back to direct write if atomic replace fails twice.

---

## Verification Checklist

Run these checks after the production run completes:

### VC-S4-01: Success Rate
```bash
python -c "
import json
stats = json.load(open('training/data/processed/cpg/cpg_stats.json'))
rate = stats['success_rate_pct']
print(f'Success rate: {rate}%')
assert rate >= 85, f'FAIL: success rate {rate}% < 85%'
print('PASS')
"
```
**Expected**: >= 85% (Phase 1 was ~90% on SARD-only)

### VC-S4-02: DFG Edge Presence
```bash
python -c "
import json
stats = json.load(open('training/data/processed/cpg/cpg_stats.json'))
avg_dfg = stats['avg_dfg_per_cpg']
print(f'Avg DFG edges/CPG: {avg_dfg}')
assert avg_dfg >= 10, f'FAIL: avg DFG {avg_dfg} < 10'
print('PASS')
"
```

### VC-S4-03: No Edge Type >= 4
```bash
python -c "
import json, os
from pathlib import Path
cpg_dir = Path('training/data/processed/cpg')
bad = 0
total = 0
for shard in cpg_dir.iterdir():
    if shard.is_dir() and len(shard.name) == 2:
        for f in shard.glob('*.json'):
            cpg = json.loads(f.read_text(encoding='utf-8'))
            total += 1
            for e in cpg['edges']:
                if e['type'] >= 4:
                    bad += 1
                    break
print(f'Checked {total} CPGs, {bad} with edge type >= 4')
assert bad == 0, f'FAIL: {bad} CPGs have invalid edge types'
print('PASS')
"
```

### VC-S4-04: Per-Source Success Rates
```bash
python -c "
import json
stats = json.load(open('training/data/processed/cpg/cpg_stats.json'))
for src in sorted(stats.get('per_source_succeeded', {})):
    ok = stats['per_source_succeeded'].get(src, 0)
    fl = stats['per_source_failed'].get(src, 0)
    rate = ok / (ok + fl) * 100 if (ok + fl) > 0 else 0
    flag = 'WARN' if rate < 70 else 'OK'
    print(f'  {src}: {ok}/{ok+fl} ({rate:.0f}%) [{flag}]')
"
```

### VC-S4-05: Taint Coverage
```bash
python -c "
import json
stats = json.load(open('training/data/processed/cpg/cpg_stats.json'))
tc = stats.get('taint_coverage_pct', 0)
print(f'Taint coverage: {tc}%')
if tc < 20:
    print('WARN: Below 20% target (expected for SARD-heavy dataset)')
else:
    print('PASS')
"
```

### VC-S4-06: Pair Integrity
```bash
python -c "
import json
stats = json.load(open('training/data/processed/cpg/cpg_stats.json'))
broken = stats.get('broken_pairs_removed', 0)
print(f'Broken pairs removed: {broken}')
if broken > 500:
    print('WARN: > 500 pair partners discarded')
else:
    print('PASS')
"
```

### VC-S4-07: Output File Count
```bash
python -c "
import os
from pathlib import Path
cpg_dir = Path('training/data/processed/cpg')
count = sum(1 for shard in cpg_dir.iterdir()
            if shard.is_dir() and len(shard.name) == 2
            for f in shard.glob('*.json'))
print(f'CPG files: {count}')
assert count >= 40000, f'FAIL: only {count} CPGs (expected ~42K+)'
print('PASS')
"
```

### VC-S4-08: Per-CWE Avg Node Count
```bash
python -c "
import json
stats = json.load(open('training/data/processed/cpg/cpg_stats.json'))
for cwe, avg in sorted(stats.get('per_cwe_avg_nodes', {}).items()):
    flag = 'WARN' if avg < 5 else 'OK'
    print(f'  {cwe}: avg {avg} nodes [{flag}]')
"
```
**Expected**: SARD CWE-121 avg should be 20-80 nodes. If < 5, Stage 1 clean may have produced empty functions (R-01 cross-ref).

---

## Output Format (CPG JSON)

```json
{
  "sample_id": "sard_000003_good",
  "label": 0,
  "cwe": "CWE-121",
  "pair_id": "sard_000003",
  "source": "sard",
  "nodes": [
    {
      "id": "42",
      "_label": "CALL",
      "code": "strcpy(dst, src)",
      "name": "strcpy",
      "line": 12,
      "type_full": "void",
      "taint_role": "SINK"
    }
  ],
  "edges": [
    {"src": "42", "dst": "43", "type": 1, "label": "CFG"},
    {"src": "40", "dst": "42", "type": 2, "label": "DFG"},
    {"src": "39", "dst": "42", "type": 3, "label": "TPG"}
  ],
  "stats": {
    "node_count": 87,
    "edge_count": 142,
    "ast_edges": 45,
    "cfg_edges": 31,
    "dfg_edges": 58,
    "tpg_edges": 8,
    "taint_sources": 1,
    "taint_sinks": 2,
    "taint_sanitizers": 0,
    "taint_propagation": 3
  }
}
```

---

## Downstream Pipeline

After Stage 4 completes:

```
Stage 4: CPG Construction (this stage)
    ↓  training/data/processed/cpg/{shard}/{id}.json
Stage 5: Node Embedding (stage5_embed.py)
    ↓  CodeBERT(768) + node_type(32) + taint(8) + cpg(4) + structural(12) = 824-d
    ↓  training/data/processed/embedded_data/embedded/{shard}/{id}.npz
Stage 6: Graph Tensor Assembly (stage6_graphs.py)
    ↓  HDF5 with PyG-compatible tensors
    ↓  training/data/graphs/all_graphs.h5  (~3.7 GB)
Stage 7: CFA-Aware Split + Audit (stage7_split.py + pre_training_audit.py)
    ↓  training/data/final/{train,val,test}.h5  (80/10/10 split)
Stage 8: Model Training (train.py)
```

Stages 5-6 require GPU (Colab T4 recommended). See `docs/STORY6_COLAB_GUIDE.md`.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| R-15 TRIGGERED: 0 DFG edges | Joern overlays disabled | Ensure `joern-parse` is NOT called with `--nooverlays`. CLI mode includes DFG by default. |
| All samples fail with timeout | JVM startup too slow on cold machine | Increase `JOERN_TIMEOUT` (default 60s) in code, or reduce `--workers` |
| `FileNotFoundError: joern-parse` | Wrong `--joern-dir` | Verify path: Windows = `C:/Tools/joern-cli/joern-parse.bat`, Linux = `/opt/joern-cli/joern-parse` |
| Memory exhaustion / swap thrashing | Too many workers | Reduce `--workers`. Formula: workers <= (RAM_GB * 0.8) / 2 |
| R-33 WARNING about deduped path | Using wrong input file | Switch to `--input training/data/processed/with_cfa/samples.jsonl` |
| Taint coverage < 20% | Expected for SARD-heavy dataset | SARD Juliet flow variants have no I/O sources (taint enters via function params). GNN trains on AST+CFG+DFG structure for these. |
| Broken pairs > 5% | Systematic Joern failures for specific CWE | Check `cpg_failures.jsonl` — if one CWE dominates failures, the code pattern may need HEADER_STUBS expansion |
| Checkpoint not resuming | `--max-samples` was used | Remove `--max-samples` for production (it disables resume) |
| `PermissionError` on checkpoint | Windows antivirus file lock | R-34 mitigation handles this (retry + fallback). If persistent, exclude output dir from antivirus |
| Progress says 0 samples/s | Normal for first ~30s | JVM cold start on first sample takes 10-15s. Rate stabilizes after 5+ samples. |
| `taskkill` not found (Windows) | Git Bash missing Windows tools | Run from PowerShell or CMD instead of Git Bash |

---

## Production Run Checklist

Before starting the full 49,810-sample run:

- [ ] **Joern verified**: `joern-parse --help` returns rc=0
- [ ] **Input file verified**: `wc -l training/data/processed/with_cfa/samples.jsonl` shows 49,810 lines
- [ ] **Disk space**: At least 5 GB free for output CPGs
- [ ] **RAM check**: Workers x 2 GB < 80% of total RAM
- [ ] **Test run passed**: `--max-samples 10 --workers 1` produces CPGs with DFG > 0
- [ ] **No stale checkpoint**: Delete old `checkpoint.json` if starting a new full run
- [ ] **Output dir clean**: `training/data/processed/cpg/` is empty or contains only data from previous partial run to resume

### During the Run

- Progress prints every 10 samples with ETA
- Checkpoint saves every 500 samples
- Safe to Ctrl+C — checkpoint is saved on exit
- Monitor `cpg_failures.jsonl` for clustering patterns
- R-15 auto-halts within first 5 samples if DFG is missing

### After Completion

- [ ] Run all 8 verification checks (VC-S4-01 through VC-S4-08)
- [ ] Check `cpg_stats.json` for success rate, taint coverage, broken pairs
- [ ] Verify per-source rates (real-world sources may have lower success than SARD)
- [ ] Proceed to Stage 5 (see `docs/STORY6_COLAB_GUIDE.md`)

---

## Tests

```bash
# Phase 1 tests (81 tests — edge types, taint, TPG, parsing, end-to-end)
python -m pytest tests/test_story5.py -v

# Phase 3 tests (39 tests — multi-source, risk mitigations)
python -m pytest tests/test_story_p3s4.py -v

# All Stage 4 tests together
python -m pytest tests/test_story5.py tests/test_story_p3s4.py -v
```

### Phase 3 Test Coverage

| Test Class | Count | Covers |
|------------|-------|--------|
| TestWriteTempCSard | 3 | SARD gets JULIET_STUB, Juliet includes stripped |
| TestWriteTempCNonSard | 3 | Non-SARD gets only standard includes |
| TestSardStripIncludes | 2 | std_testcase.h removal, standard includes preserved |
| TestCDGEdgeDropped | 2 | CDG/DOMINATE/POST_DOMINATE not in edge type map |
| TestReachingDefMapping | 2 | REACHING_DEF (4.x) and REACHES (1.x) both map to DFG=2 |
| TestExtendedTaintSets | 4 | SQL/cmd/XSS sources, sinks, sanitizers |
| TestPairAwareDiscard | 2 | Partner CPG removed when pair member fails |
| TestCheckpointResume | 3 | Save, load, skip already-done samples |
| TestContextSlice | 2 | 2-hop BFS, max 200 nodes |
| TestEndToEnd | 4 | Full pipeline with mock Joern for SARD and non-SARD |
| TestR15DFGZeroHalt | 2 | process_sample returns None on zero DFG |
| TestR16CDGEdgeAudit | 2 | edge type >= 4 rejected |
| TestR17SubprocessTimeout | 2 | _run_subprocess_safe basic and timeout |
| TestR33InputValidation | 3 | deduped path warning, with_cfa path OK |
| TestR34CheckpointRobustness | 3 | Checkpoint creates, backup, overwrite |

---

## Key Code References

| Component | File:Line | Purpose |
|-----------|-----------|---------|
| Edge type map | `stage4_cpg.py:65-76` | AST=0, CFG=1, DFG=2, TPG=3 |
| Taint sets | `stage4_cpg.py:90-139` | SOURCES, SINKS, SANITIZERS (12 CWEs) |
| Output param map | `stage4_cpg.py:149-164` | recv, fscanf, fread etc. argument indices |
| Subprocess kill | `stage4_cpg.py:170-218` | Process-group kill (R-17) |
| JULIET_STUB | `stage4_cpg.py:233-245` | 8 Juliet print function stubs |
| write_temp_c | `stage4_cpg.py:276-297` | Source-aware stub prepending |
| TPG BFS | `stage4_cpg.py:352-472` | BFS from SOURCE→SINK along DFG |
| Context slice | `stage4_cpg.py:475-525` | 2-hop BFS, max 200 nodes |
| GraphSON parser | `stage4_cpg.py:528-635` | TinkerGraph JSON parsing |
| Joern CLI runner | `stage4_cpg.py:687-765` | joern-parse + joern-export pipeline |
| process_sample | `stage4_cpg.py:782-854` | Full per-sample pipeline |
| run_stage4 | `stage4_cpg.py:918-1389` | Main orchestrator with all guards |
| CLI entrypoint | `stage4_cpg.py:1392-1448` | argparse CLI |
