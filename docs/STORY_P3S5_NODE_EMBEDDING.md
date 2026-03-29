# Story P3-S5: Stage 5 Node Embedding (Phase 3 Extension)

## Overview

P3-S5 extends the Phase 1 Stage 5 implementation (`stage5_embed.py`) to handle the full Phase 3 dataset including CFA-augmented samples. Phase 1 processed ~34,691 CPGs. Phase 3 targets 50K-80K CPGs (originals + CFA pairs).

The core embedding logic is unchanged. Four additions were made:
1. CFA source counting in output stats
2. Strict 824-d shape assertion at save time
3. Per-CWE embedding statistics
4. Node feature anomaly detection

---

## Files Modified / Created

| File | Action | Purpose |
|------|--------|---------|
| `training/scripts/preprocessing/stage5_embed.py` | Modified | C1-C4: source tracking, assertion, per-CWE stats, anomaly detection |
| `tests/test_story_p3s5.py` | Created | 5 tests for Phase 3 additions |
| `StreamGuard_Stage5_Stage6_Colab.ipynb` | Modified | Cell 4 (CFA zip support), Cell 5b (larger dataset), Cell 7 (copy per-CWE stats) |
| `docs/STORY_P3S5_NODE_EMBEDDING.md` | Created | This document |

---

## Changes in Detail

### C1: CFA Source Counting

CFA samples have `source` ending in `_cfa` (e.g., `sard_cfa`, `repo_cfa`, `cve_cfa`). The processing loop now counts `cfa_samples` vs `original_samples`. Both are included in `embed_stats.json`.

No code change was needed in `embed_cpg()` itself because CFA samples have identical CPG structure to originals — they flow through the same embedding path.

### C2: Strict 824-d Shape Assertion

Added before the full shape check:

```python
assert features.shape[1] == TOTAL_DIM, (
    f"Wrong feature dim: {features.shape} — expected {TOTAL_DIM} columns"
)
```

This catches any upstream CPG structure change that would cause a column count mismatch. Without this, the error would only surface in Phase 4 when `model.node_proj = nn.Linear(824, 256)` fails on the first forward pass — much harder to diagnose.

### C3: Per-CWE Embedding Statistics

After all samples are embedded, the pipeline computes and saves `embed_stats_by_cwe.json`:

```json
{
  "CWE-121": {
    "sample_count": 12345,
    "avg_node_count": 38.2,
    "avg_taint_node_pct": 7.8,
    "avg_dfg_degree": 42.1
  },
  ...
}
```

Fields:
- `avg_taint_node_pct`: percentage of nodes with taint roles (SOURCE/SINK/SANITIZER/PROPAGATION)
- `avg_node_count`: average graph size per CWE
- `avg_dfg_degree`: average number of DFG edges per sample (data-flow density)

### C4: Node Feature Anomaly Detection

Two anomaly checks run after embedding each sample:

1. **High value anomaly**: `features.max() > 100` logs a warning. CodeBERT [CLS] values typically range [-3, 15]. Values above 100 indicate a corrupted model or bad input.

2. **Missing type label**: `features[:, 768:800].sum(axis=1) == 0` for any node means no node type was encoded. This should never happen because `_build_node_type_onehot()` falls back to UNKNOWN. If it triggers, it means the feature construction loop has a bug.

Anomaly counts are tracked in `embed_stats.json` as `anomaly_high_value` and `anomaly_missing_type`.

---

## Risk Audit

Every risk from `StreamGuard_Phase3_Risk_Analysis.docx` relevant to Stage 5:

| Risk | Severity | Status | How Mitigated |
|------|----------|--------|---------------|
| R-21: Stage 5 must run on GPU | CRITICAL | Mitigated | Colab notebook updated for Phase 3 scale. `embed_stats.json` logs `samples_per_second` — gate G-07 checks `rate > 10`. CPU run takes 87 hours vs ~27 min on T4. |
| R-22: Empty code nodes skipped | HIGH | Mitigated (Phase 1) | `codebert_embeddings` initialized to zeros; only non-empty nodes get CodeBERT output. Empty nodes keep zero [0:768] + valid structural [768:824]. Test: `test_empty_code_node_gets_zero_codebert` in test_story6.py. |
| R-23: Feature dim != 824 | HIGH | Mitigated (C2) | Hard `assert features.shape[1] == 824` added at save time. Test: `test_assertion_on_wrong_feature_dim` in test_story_p3s5.py. |
| R-24: Colab disconnect, no checkpoint | MEDIUM | Mitigated (Phase 1) | `run_stage5` skips existing `.npz` files automatically. Re-running after disconnect resumes from where it left off. VC-S5-06 verifies this. |

### Upstream risks (not our responsibility but noted):

| Risk | What | Impact on Stage 5 |
|------|------|-------------------|
| R-20 | Half-pairs from Stage 4 | Broken pair_ids enter Stage 5. We embed them normally; Stage 7 catches orphans. |
| R-33 | Stage 4 reads wrong input (deduped/ not with_cfa/) | CFA samples get no CPGs. Stage 5 sees no CFA .json files. `cfa_samples` count in stats will be 0 — a clear signal. |

### Downstream risks (Stage 6 consuming our output):

| Risk | What | Our guarantee |
|------|------|---------------|
| R-25 | pair_id not in HDF5 | Not Stage 5's concern — Stage 6 reads pair_id from CPG JSON. |
| R-26 | edge_index OOB | Stage 5 output is `.npz` features only. Edge remapping is Stage 6. |
| R-27 | CDG edge_type=4 | Stage 5 uses edge types only for structural features (CPG component priority). Unknown types > 3 are clamped to index 3 (TPG) in `_build_cpg_component_onehot`. |

---

## Verification Checklist

All from Table 28 of the risk doc:

| Check | Command | Expected | Status |
|-------|---------|----------|--------|
| VC-S5-01 | `python -m pytest tests/test_story_p3s5.py -v` | 5/5 PASS | PASS |
| VC-S5-02 | Colab Cell 1: GPU check | GPU device name printed | Ready (notebook updated) |
| VC-S5-03 | `python -c "import numpy as np, glob; [print(np.load(f)['node_features'].shape) for f in list(glob.glob('training/data/processed/embedded_data/embedded/**/*.npz',recursive=True))[:5]]"` | All show (N, 824) | PASS (verified on Phase 1 data) |
| VC-S5-04 | `python -c "import numpy as np, glob; bad=[f for f in glob.glob('training/data/processed/embedded_data/embedded/**/*.npz',recursive=True) if np.isnan(np.load(f)['node_features']).any()]; assert len(bad)==0, bad"` | 0 NaN files | PASS |
| VC-S5-05 | See walkthrough below | Zero [0:768] + non-zero [768:824] for empty code node | PASS (covered by test) |
| VC-S5-06 | Interrupt + re-run on Colab | "Skipping N already processed" | Ready (checkpoint/resume in code) |
| VC-S5-07 | `embed_stats.json` | `samples_per_second > 10` | Ready (will verify on Colab run) |

**Note**: The risk doc's VC-S5-03 uses key `'features'` but our .npz key is `'node_features'`. The verification command above uses the correct key.

---

## Integration Checks

| Check | What | Status |
|-------|------|--------|
| INT-10 | cpg/ and embedded/ have matching shard structure (00/-ff/) | PASS — both use `sample_id[:2]` for sharding |
| INT-11 | All CPG sample_ids have .npz files | Verified: `succeeded + skipped == total_cpgs - failed` in stats |
| INT-12 | Node count matches: CPG JSON nodes == .npz rows | Enforced by `assert features.shape == (n_nodes, TOTAL_DIM)` |
| INT-13 | For sample with N CPG nodes, .npz has shape (N, 824) | Same assertion as INT-12 |
| INT-14 | Taint roles from Stage 4 correctly encoded in [800:808] | Verified in walkthrough below |
| INT-15 | Edge types match CPG JSON | Stage 5 reads edge types for structural features only; Stage 6 handles HDF5 edge types |

---

## Tests

5 tests in `tests/test_story_p3s5.py`, all passing:

| Test | What it verifies |
|------|-----------------|
| `TestFeatureShapeAssertion::test_correct_shape_passes` | Normal CPG produces (5, 824) |
| `TestFeatureShapeAssertion::test_assertion_on_wrong_feature_dim` | Monkeypatched 512-d output triggers AssertionError matching "Wrong feature dim" |
| `TestCFAEmbedding::test_cfa_embeds_same_as_original` | source='sard' and source='sard_cfa' with identical structure produce identical feature matrices |
| `TestPerCWEStats::test_cwe_stats_file_created` | embed_stats_by_cwe.json created with CWE keys and all 4 stat fields |
| `TestPerCWEStats::test_cfa_source_stats_tracked` | Stats dict has cfa_samples=1, original_samples=1 for mixed input |

68/68 existing Story 6 tests pass (zero regressions).

---

## How to Run

### Local smoke test (CPU, slow — for testing only)

```bash
cd "C:\Users\Vimal Sajan\streamguard"

# Dry run — verify setup without loading CodeBERT
python -m training.scripts.preprocessing.stage5_embed \
  --input training/data/processed/cpg/ \
  --output training/data/processed/embedded/ \
  --dry-run --max-samples 5

# Small test run (5 samples, CPU)
python -m training.scripts.preprocessing.stage5_embed \
  --input training/data/processed/cpg/ \
  --output training/data/processed/embedded/ \
  --device cpu --max-samples 5 --verify
```

### Production run (Colab T4 GPU)

1. **Zip CPG data locally**:
   ```bash
   python -c "import shutil; shutil.make_archive('cpg_data', 'zip', '.', 'training/data/processed/cpg')"
   ```
   If CFA CPGs are in a separate directory, also zip those as `cpg_data_cfa.zip`.

2. **Upload to Google Drive**: `My Drive/StreamGuard/cpg_data.zip` (and `cpg_data_cfa.zip` if applicable)

3. **Open notebook**: `StreamGuard_Stage5_Stage6_Colab.ipynb` in Colab

4. **Set runtime to GPU** (T4 or better): Runtime > Change runtime type > GPU

5. **Run all cells in order**. Stage 5 takes ~27 min for 35K samples, ~40-70 min for 50K-80K.

6. **Download results** from Drive:
   - `embedded_data.zip` → extract to `training/data/processed/embedded/`
   - `embed_stats.json`, `embed_stats_by_cwe.json`

7. **Verify locally**:
   ```bash
   python -m training.scripts.preprocessing.stage5_embed \
     --input training/data/processed/cpg/ \
     --output training/data/processed/embedded/ \
     --verify
   ```

### CLI flags

| Flag | Default | Purpose |
|------|---------|---------|
| `--input` | `training/data/processed/cpg/` | CPG JSON directory |
| `--output` | `training/data/processed/embedded/` | Output .npz directory |
| `--device` | auto-detect | `cuda` or `cpu` |
| `--batch-size` | 64 | CodeBERT batch size (64 optimal for T4) |
| `--dry-run` | off | Show what would be processed without loading CodeBERT |
| `--max-samples` | all | Limit processing to first N samples |
| `--verify` | off | After run, verify first 3 .npz files |

### Run tests

```bash
python -m pytest tests/test_story_p3s5.py -v    # 5 P3-S5 tests
python -m pytest tests/test_story6.py -v         # 68 Story 6 regression tests
```

---

## Output Files

| File | Location | Content |
|------|----------|---------|
| `{shard}/{sample_id}.npz` | `training/data/processed/embedded/` | Per-sample node features, shape (N, 824), key `node_features` |
| `embed_stats.json` | Same directory | Run statistics including cfa/original counts, anomaly counts |
| `embed_stats_by_cwe.json` | Same directory | Per-CWE metrics: sample_count, avg_node_count, avg_taint_node_pct, avg_dfg_degree |

---

## Node Embedding Walkthrough: One Real Sample

Sample: `0002653a-86d8-4adb-89ea-140cf803ca42`
- **CWE**: CWE-122 (Heap Buffer Overflow)
- **Label**: 0 (safe — the fixed version)
- **Source**: sard
- **Pair ID**: `00c8fa48-7e71-4f01-a3de-5b0fe94cbc49`
- **Nodes**: 32
- **Edges**: 222 (DFG: 114, AST: 62, CFG: 46)
- **Taint nodes**: 3 of 32 (2 SINK + 1 SANITIZER)

### Step 1: Load CPG JSON

The CPG was produced by Stage 4 (`stage4_cpg.py` via Joern). It contains:

```
Nodes (32):
  Node 0: _label=LITERAL    code="L'\\0'"        taint_role=NONE
  Node 1: _label=IDENTIFIER  code='data'          taint_role=NONE
  Node 2: _label=LITERAL    code='1'              taint_role=NONE
  ...
  Node 11: _label=CALL      code='source[100-1]'  taint_role=SINK
  Node 26: _label=CALL      code='free(data)'     taint_role=SINK
  Node 30: _label=CALL      code='wcsncat(...)'   taint_role=SANITIZER

Edges (222):
  DFG: 114 edges (data-flow reaching definitions)
  AST: 62 edges (syntax tree parent-child)
  CFG: 46 edges (control flow successor)
  TPG: 0 edges (no taint propagation paths — expected for safe code)
```

### Step 2: CodeBERT Encoding ([0:768])

For each of the 32 nodes, we extract the `code` field and feed it through CodeBERT (`microsoft/codebert-base`):

```
Tokenize: "source[100-1]" → ['source', '[', '100', '-', '1', ']'] (max 64 tokens)
Forward pass: model(**tokens).last_hidden_state[:, 0, :]  → 768-d [CLS] vector
AMP fp16 on GPU for 2x speedup, @torch.inference_mode() to skip gradients
```

Nodes with empty code (e.g., BLOCK nodes) get a zero vector — they are NOT skipped.

**Result for Node 0** (`LITERAL`, code=`L'\\0'`):
```
[0:768] CodeBERT: 768 non-zero values, range [-1.8363, 14.3473]
```

### Step 3: Node Type One-Hot ([768:800])

Each node's `_label` is mapped to a 32-d one-hot vector:

```
Node 0:  _label=LITERAL    → index 6  → [0,0,0,0,0,0,1,0,...,0]
Node 11: _label=CALL       → index 1  → [0,1,0,0,0,0,0,0,...,0]
Node 30: _label=CALL       → index 1  → [0,1,0,0,0,0,0,0,...,0]
```

Unknown types fall back to UNKNOWN (index 17). 15 reserved slots for future Joern versions.

### Step 4: Taint Role One-Hot ([800:808])

Each node's `taint_role` from Stage 4's taint analyzer:

```
Node 0:  taint_role=NONE       → [0,0,0,0,1,0,0,0]  (index 4)
Node 11: taint_role=SINK       → [0,1,0,0,0,0,0,0]  (index 1)
Node 26: taint_role=SINK       → [0,1,0,0,0,0,0,0]  (index 1)
Node 30: taint_role=SANITIZER  → [0,0,1,0,0,0,0,0]  (index 2)
```

### Step 5: CPG Component One-Hot ([808:812])

Dominant edge type the node participates in (TPG > DFG > CFG > AST):

```
Node 0:  participates in DFG edges → [0,0,1,0]  (DFG = index 2)
Node 11: participates in DFG edges → [0,0,1,0]
```

Since this is safe code with 0 TPG edges, no node gets TPG (index 3).

### Step 6: Structural Features ([812:824])

12-dimensional normalized structural metrics:

```
Node 0 structural features:
  [0] in_degree  = 0.1875   (6 / 32)
  [1] out_degree = 0.1875   (6 / 32)
  [2] AST_depth  = 0.6667   (depth 4 / max 6)
  [3] DFG dist to sink = 0.03125  (1 hop via DFG / 32 nodes)
  [4] CFG dist to sink = 0.28125  (9 hops via CFG / 32 nodes)
  [5] Any dist to sink = 0.03125  (1 hop via any edge / 32 nodes)
  [6-11] reserved = 0.0
```

### Step 7: Concatenate and Save

All 6 feature regions are concatenated into one 824-d vector per node:

```
Node 0: [CodeBERT(768) | NodeType(32) | TaintRole(8) | CPGComp(4) | Structural(12)] = 824-d
```

The full feature matrix shape is (32, 824) and is saved as:
```
training/data/processed/embedded/00/0002653a-86d8-4adb-89ea-140cf803ca42.npz
  Key: "node_features"
  Shape: (32, 824)
  dtype: float32
  NaN: False
  Range: [-2.6853, 15.0473]
```

### What the model sees

In Phase 4 training, the GGNN loads this matrix as `data.x` (shape `[32, 824]`). The first layer `node_proj = nn.Linear(824, 256)` projects each node's 824-d vector to a 256-d hidden representation. The GNN then propagates messages along the 222 edges. The taint information (SINK at nodes 11, 26; SANITIZER at node 30) gives the model a strong signal about vulnerability-relevant code paths.

---

## Phase 1 Production Results (Baseline)

| Metric | Value |
|--------|-------|
| Platform | Google Colab T4 GPU (16 GB VRAM) |
| Total CPGs | 34,691 |
| Failed | 0 |
| NaN rejected | 0 |
| Rate | ~21 samples/s |
| Total time | ~27 minutes |
| Avg nodes/sample | ~40 |
| Output size | 3,149.9 MB |

## Phase 3 Estimates

| Metric | Estimate |
|--------|----------|
| Total CPGs | 50,000-80,000 (originals + CFA) |
| Colab T4 time | ~40-70 minutes |
| Output size | ~5-7 GB |
| CFA samples | ~700-1,000 (from P3-S4 stats: 741 CFA) |
