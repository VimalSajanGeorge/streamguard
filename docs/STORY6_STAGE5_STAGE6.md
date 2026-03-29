# Story 6: Stage 5 (Node Embedding) + Stage 6 (Graph Tensor Assembly)

## Overview

Story 6 implements the final two preprocessing stages before model training:
- **Stage 5**: Produces 824-dimensional node feature vectors using CodeBERT + structural features
- **Stage 6**: Assembles CPG edges + embeddings into HDF5 graph tensors for GNN training

## Files

| File | Purpose |
|------|---------|
| `training/scripts/preprocessing/stage5_embed.py` | Stage 5 — node embedding (706 lines) |
| `training/scripts/preprocessing/stage6_graphs.py` | Stage 6 — graph tensor assembly (464 lines) |
| `training/scripts/preprocessing/__init__.py` | Package init |
| `tests/test_story6.py` | 68 unit/integration tests |
| `StreamGuard_Stage5_Stage6_Colab.ipynb` | Colab notebook for GPU execution |
| `docs/STORY7_COLAB_GUIDE.md` | Colab process guide |

---

## Stage 5: Node Embedding

### 824-d Feature Vector Layout

| Range | Dim | Source |
|-------|-----|--------|
| `[0:768]` | 768 | CodeBERT `[CLS]` embedding of node code |
| `[768:800]` | 32 | Node type one-hot (17 Joern types + 15 reserved) |
| `[800:808]` | 8 | Taint role one-hot (5 roles + 3 reserved) |
| `[808:812]` | 4 | CPG component one-hot (AST/CFG/DFG/TPG membership) |
| `[812:824]` | 12 | Structural features |

### Node Type Vocabulary (17 observed)

```
BLOCK, CALL, CONTROL_STRUCTURE, FIELD_IDENTIFIER, IDENTIFIER,
JUMP_TARGET, LITERAL, LOCAL, METHOD, METHOD_PARAMETER_IN,
METHOD_PARAMETER_OUT, METHOD_REF, METHOD_RETURN, MODIFIER,
RETURN, TYPE_DECL, TYPE_REF, UNKNOWN (+ 14 reserved slots)
```

### Taint Roles (5)

```
SOURCE, SINK, SANITIZER, PROPAGATION, NONE (+ 3 reserved)
```

### Structural Features (12-d)

| Index | Feature |
|-------|---------|
| 0 | in-degree (AST) |
| 1 | out-degree (AST) |
| 2 | in-degree (CFG) |
| 3 | out-degree (CFG) |
| 4 | in-degree (DFG) |
| 5 | out-degree (DFG) |
| 6 | AST depth (root=0) |
| 7 | AST depth normalized (depth / max_depth) |
| 8 | is leaf node (0/1) |
| 9 | total degree (all edge types) |
| 10 | BFS distance to nearest sink (finite, max=200) |
| 11 | has taint edge (0/1) |

### Key Design Decisions

- **Empty code nodes**: Get zero vector in CodeBERT region (never skipped — preserves graph topology)
- **CodeBERT inference**: `microsoft/codebert-base`, batch_size=64, max 64 tokens per node
- **AMP**: `torch.autocast('cuda', dtype=torch.float16)` on GPU for 2x speedup
- **`@torch.inference_mode()`**: No gradient computation overhead
- **Checkpoint/resume**: Skips existing `.npz` files automatically
- **Sharded output**: `{first_2_chars_of_id}/` directory structure (max 256 subdirs)
- **Performance optimization**: Pre-compute adjacency maps once O(E), not per-node

### Output Format

Per sample: `{shard}/{sample_id}.npz` containing:
- `features`: `np.float32` array of shape `(num_nodes, 824)`

---

## Stage 6: Graph Tensor Assembly

### What It Does

Joins CPG JSON (edges, labels, metadata) with Stage 5 embeddings (.npz) into a single HDF5 file with PyG-compatible tensors.

### HDF5 Structure

```
all_graphs.h5
├── attrs: feature_dim=824, total_graphs=N, created_at=...
├── {sample_id}/
│   ├── x          (num_nodes, 824) float32  — node features
│   ├── edge_index (2, num_edges) int64      — COO format
│   ├── edge_type  (num_edges,) int32        — {0=AST, 1=CFG, 2=DFG, 3=TPG}
│   ├── y          scalar int                — label (0=safe, 1=vuln)
│   └── attrs: cwe, language, source, num_nodes, num_edges
```

### 7 Validation Gates

| Gate | What it checks | Action on fail |
|------|---------------|----------------|
| 1 | Trivial graph (< 3 nodes) | Skip |
| 2 | Feature shape mismatch | Skip |
| 3 | NaN in features | Skip |
| 4 | Edge type outside {0,1,2,3} | Skip |
| 5 | Edge index out of bounds | Skip |
| 6 | Dangling edges (node not in graph) | Skip |
| 7 | No valid edges after filtering | Skip |

### Key Design Decisions

- **Atomic write**: Write to `.h5.tmp` then `os.replace()` to prevent corruption
- **Gzip compression**: Reduces HDF5 size by ~40%
- **Contiguous node indices**: Remaps node IDs to 0..N-1
- **Edge type clamping**: Only {0,1,2,3} — anything >=4 would cause CUDA crash in GGNN

---

## Production Run Results (Colab T4 GPU)

### Stage 5 Embedding

| Metric | Value |
|--------|-------|
| Platform | Google Colab, T4 GPU (16 GB VRAM) |
| Total CPGs | 34,691 |
| Processed | 34,691 |
| Failed | 0 |
| NaN rejected | 0 |
| Rate | **~21 samples/s** |
| Total time | **~27 minutes** |
| Avg nodes/sample | ~40 |
| Feature dim | 824 |
| Output size | 3,149.9 MB (3.1 GB) |
| Output path (local) | `training/data/processed/embedded_data/embedded/` |

### Verification (Stage 5)

```
0002653a: shape=(32, 824) dtype=float32 NaN=False min=-2.6853 max=15.0473
0003d22b: shape=(41, 824) dtype=float32 NaN=False min=-2.4202 max=15.0106
0003ebb5: shape=(30, 824) dtype=float32 NaN=False min=-2.5822 max=15.3387
0004e527: shape=(44, 824) dtype=float32 NaN=False min=-2.6113 max=15.0106
00087fa3: shape=(59, 824) dtype=float32 NaN=False min=-2.6853 max=15.0473
All 5 files: shape (N, 824) float32, no NaN
```

### Stage 6 Graph Assembly

| Metric | Value |
|--------|-------|
| Output file | `all_graphs.h5` |
| Output size | 3,744.1 MB (3.7 GB) |
| Output path (local) | `training/data/graphs/all_graphs.h5` |

### CPU vs GPU Comparison

| | CPU (local) | GPU (Colab T4) |
|---|------------|----------------|
| Rate | 0.11 samples/s | 21 samples/s |
| ETA (34,691 samples) | ~87 hours | ~27 minutes |
| Speedup | baseline | **~190x** |

---

## 8 Challenges Addressed

| # | Challenge | How addressed |
|---|-----------|--------------|
| 1 | Edge index OOB | Validation gate 5: skip if any edge points outside [0, N) |
| 2 | GPU OOM | batch_size=64 with AMP fp16; peak VRAM ~1-2 GB on T4 |
| 3 | 824-d mismatch | `TOTAL_DIM=824` constant enforced; shape check in Stage 6 gate 2 |
| 4 | HDF5 corruption | Atomic write: `.h5.tmp` + `os.replace()` |
| 5 | Zero TPG edges | Valid — some functions have no taint paths; cpg_component[3]=0 |
| 6 | NaN structural features | BFS returns finite max_dist=200, never np.inf; gate 3 rejects NaN |
| 7 | HDF5 scale | Gzip compression (~40% reduction); tested at 34K+ samples |
| 8 | Unknown node types | Maps to UNKNOWN slot in one-hot; 15 reserved slots for future types |

---

## CLI Usage

### Stage 5

```bash
# Dry run
python -m training.scripts.preprocessing.stage5_embed \
  --input training/data/processed/cpg/ \
  --output training/data/processed/embedded/ \
  --dry-run --max-samples 5

# Full run (GPU)
python -m training.scripts.preprocessing.stage5_embed \
  --input training/data/processed/cpg/ \
  --output training/data/processed/embedded/ \
  --device cuda --batch-size 64

# Verify
python -m training.scripts.preprocessing.stage5_embed \
  --input training/data/processed/cpg/ \
  --output training/data/processed/embedded/ \
  --verify
```

### Stage 6

```bash
# Full run
python -m training.scripts.preprocessing.stage6_graphs \
  --cpg-dir training/data/processed/cpg/ \
  --embed-dir training/data/processed/embedded/ \
  --output training/data/graphs/all_graphs.h5

# Verify
python -m training.scripts.preprocessing.stage6_graphs \
  --cpg-dir training/data/processed/cpg/ \
  --embed-dir training/data/processed/embedded/ \
  --output training/data/graphs/all_graphs.h5 \
  --verify
```

---

## Tests

68 tests in `tests/test_story6.py`, all passing:

| Test Class | Count | What it covers |
|-----------|-------|----------------|
| TestFeatureDimensions | 7 | 824-d total, sub-region sizes |
| TestNodeTypeOneHot | 6 | All 17 observed types + unknown mapping |
| TestTaintRoleOneHot | 7 | All 5 roles + unknown mapping |
| TestCPGComponentOneHot | 5 | Edge type presence detection |
| TestASTDepths | 3 | Depth computation, normalization |
| TestBFSDistance | 6 | Reachability, disconnected graphs, max cap |
| TestStructuralFeatures | 3 | Degree counts, leaf detection |
| TestEmbedCPG | 11 | End-to-end embedding with mock CodeBERT |
| TestOutputPath | 2 | Sharded directory structure |
| TestLoadSample | 8 | CPG+NPZ loading, all 7 validation gates |
| TestHDF5WriteRead | 5 | Atomic write, compression, read-back |
| TestValidationEdgeCases | 3 | Boundary conditions |
| TestStage5To6Pipeline | 2 | Full integration (embed → load → HDF5) |

---

## Space Consumption

| Data | Size | Local Path |
|------|------|------------|
| CPG JSONs (input) | 944 MB | `training/data/processed/cpg/` |
| CPG zip (for Colab) | 86 MB | `cpg_data.zip` (temp, can delete) |
| Embeddings (.npz) | 3,149.9 MB | `training/data/processed/embedded_data/embedded/` |
| HDF5 (all_graphs.h5) | 3,744.1 MB | `training/data/graphs/all_graphs.h5` |
| embed_stats.json | <1 MB | `training/data/processed/embedded_data/embedded/embed_stats.json` |
| graph_stats.json | <1 MB | `training/data/graphs/graph_stats.json` |

---

## Local Directory Layout

```
training/data/
├── processed/
│   ├── cpg/                          # Stage 4 output (input to Stage 5)
│   │   ├── 00/ ... ff/               # 256 shard dirs
│   │   └── {id}.json                 # CPG JSON files (34,691)
│   └── embedded_data/
│       └── embedded/                 # Stage 5 output
│           ├── 00/ ... ff/           # 256 shard dirs
│           ├── {id}.npz             # Embedding files (34,691)
│           └── embed_stats.json
└── graphs/
    ├── all_graphs.h5                 # Stage 6 output (3.7 GB)
    └── graph_stats.json
```

---

## Next Steps

1. **Verify locally** — run verify commands on downloaded data
2. **Stage 7**: CFA-Aware Split (train/val/test partitioning)
3. **Story 7**: Model Training (GNN + Transformer)
4. **Remaining CPGs**: ~7K samples not yet CPG-constructed; run Stage 4 then re-run Stage 5+6
