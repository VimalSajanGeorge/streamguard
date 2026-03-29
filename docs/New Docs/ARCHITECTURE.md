# docs/ARCHITECTURE.md — Full System Architecture

> Referenced by: CLAUDE.md §Overview, PRD.md §5
> Implements: All 6 system planes

---

## 1. Five-Plane Architecture

```
╔══════════════════════════════════════════════════════════════════╗
║  PLANE 1: DATA                                                   ║
║  7 sources → 6 collectors → Canonical JSONL → CFA Generator     ║
╚═════════════════════════════╦════════════════════════════════════╝
                              ║
╔═════════════════════════════▼════════════════════════════════════╗
║  PLANE 2: PREPROCESSING                                          ║
║  Stage 1–7: Clean→Dedup→CFA→CPG→Embed→Graph→Split               ║
║  Output: train.h5 / val.h5 / test.h5                            ║
╚═════════════════════════════╦════════════════════════════════════╝
                              ║
╔═════════════════════════════▼════════════════════════════════════╗
║  PLANE 3: MODEL                                                  ║
║  CodeBERT + 3-layer GGNN + Cross-Attention Fusion               ║
║  3 heads: binary / CWE / severity                               ║
║  Loss: L_CE + 0.5*L_CFA + 0.1*L_severity                       ║
╚═════════════════════════════╦════════════════════════════════════╝
                              ║
╔═════════════════════════════▼════════════════════════════════════╗
║  PLANE 4: SERVING                                                ║
║  FastAPI → Inference Worker → Prediction JSON                   ║
║  Joern pool + CodeBERT + GGNN + CFExplainer                     ║
╚═════════════════════════════╦════════════════════════════════════╝
                              ║
╔═════════════════════════════▼════════════════════════════════════╗
║  PLANE 5: FEEDBACK & CONTINUOUS LEARNING                         ║
║  Human corrections → CFA queue → Weekly fine-tune               ║
╚══════════════════════════════════════════════════════════════════╝
```

---

## 2. Data Plane Detail

```
CVE/NVD ──────────────────────────────────────────────────────────┐
GitHub Advisory ──────────────────────────────────────────────────┤
OSV ──────────────────────────────────────────────────────────────┤──► Canonical Store
ExploitDB ────────────────────────────────────────────────────────┤    (SQLite + JSONL)
SARD ─────────────────────────────────────────────────────────────┤    schema validated
Repo Miner ───────────────────────────────────────────────────────┤
Manual / Label Studio ────────────────────────────────────────────┘
                                    │
                                    ▼
                          ┌─────────────────────┐
                          │  CFA Generator       │
                          │  Per-CWE LLM prompts │
                          │  + validation gate   │
                          └──────────┬──────────┘
                                     │
                          ┌──────────▼──────────┐
                          │  CFA Pairs Store     │
                          │  pair_id linkage     │
                          │  label-flipped pairs │
                          └─────────────────────┘
```

### Canonical Sample Schema

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

---

## 3. Preprocessing Plane Detail

See `docs/PREPROCESSING.md` for full per-stage specification and code.

```
Stage 1: CLEAN
  - tree-sitter function boundary extraction
  - encoding fix (chardet → UTF-8)
  - comment removal
  - macro expansion (mcpp)
  - length filter: 5–500 lines, 10–4096 tokens

Stage 2: DEDUP
  - Level 1: MD5 exact hash
  - Level 2: CVE-ID dedup
  - Level 3: commit SHA dedup
  - Level 4: MinHash LSH (Jaccard threshold 0.85)

Stage 3: CFA GENERATION
  - Per-CWE LLM prompts (Anthropic Claude Haiku)
  - Validation: gcc syntax check + Joern diff + similarity bounds
  - Target: 2–3 CFA pairs per vulnerable sample

Stage 4: CPG CONSTRUCTION
  - Joern subprocess (4 workers, ~700ms/function)
  - Exports: AST + CFG + DFG (Joern native)
  - Custom: TPG (taint propagation post-processor in Python)
  - Context slicing: 2-hop BFS from taint seeds, max 200 nodes

Stage 5: NODE EMBEDDING
  - CodeBERT [CLS] per node statement (max 64 tokens)
  - Concatenate: type one-hot + taint role + CPG component + structural
  - Output: 824-d node feature vector

Stage 6: GRAPH TENSORS
  - PyTorch Geometric Data(x, edge_index, edge_attr, y)
  - HDF5 cache (key: sample_id → {x, edge_index, edge_attr, y})
  - Bounds validation: edge_index.max() < num_nodes

Stage 7: CFA-AWARE SPLIT
  - Group by pair_id before shuffling
  - 80/10/10 train/val/test on groups
  - Assert zero commit SHA overlap between train and test
```

---

## 4. Model Plane Detail

See `docs/MODEL.md` for full specification.

```
CodeBERT Encoder
─────────────────
Input:  token sequence (BPE, max 512)
Model:  microsoft/codebert-base (125M params, 12 layers)
Output: [CLS] embedding → 768-d

GGNN Encoder  
─────────────────────────────────────────────────────
Input:  node features X (N × 824), edge_index (2 × E), edge_attr (E,)
Layer:  4 × GatedGraphConv per layer (one per edge type: AST/CFG/DFG/TPG)
        × 3 layers = 12 GatedGraphConv modules total
        Per layer: mask edges by type → 4 parallel convolutions
        → concat(4 × 256 = 1024) → Linear(1024, 256) + BN + GELU + residual
        Per-type count normalization: h_type / sqrt(count) prevents AST dominance
        Batch norm after each layer
Readout: global_mean_pool + global_max_pool → concat → 512-d
MLP:     512 → 256-d
Output:  (graph_embed → 256-d, node_embeds → N × 256-d)

Cross-Attention Fusion
─────────────────────────────────────────────────────
Q = BERT [CLS] (768-d) projected to 256-d
K = V = GGNN per-node embeddings (N, 256-d) via scatter softmax
Attn = scatter_softmax(Q·Kᵀ / √256) · V, reduced per graph → 256-d
Fused = concat(BERT_768, Attn_256, GGNN_mean_256) = 1280-d
MLP: 1280 → 512 → 128
Note: This is TRUE node-level cross-attention (BERT attends to each
      graph node individually), not graph-embed cross-attention which
      mathematically degrades to simple concatenation.

Output Heads
─────────────────────────────────────────────────────
Binary:   Linear(128, 2) → logits → BCEWithLogitsLoss
CWE:      Linear(128, 12) → logits → CrossEntropyLoss(label_smoothing=0.1)
Severity: Linear(128, 1) → score [0,10] → HuberLoss(delta=1.0)
```

---

## 5. Storage Architecture

| Store | Technology | Contents | Access |
|-------|------------|----------|--------|
| Canonical samples | SQLite (WAL) + JSONL | All samples with schema | Write: collectors; Read: preprocessing |
| CPG store | Neo4j 5.x (optional) | AST/CFG/DFG/TPG nodes+edges | Write: Joern pipeline; Read: training |
| Embedding cache | HDF5 files | Pre-computed 824-d node features | Write: stage5; Read: stage6, training |
| CFA pairs | SQLite + JSONL | CFA pairs with pair_id | Write: stage3; Read: DataLoader |
| Model registry | MLflow + filesystem | Checkpoints + metrics | Write: train.py; Read: serving |
| Prediction log | SQLite | API predictions + feedback | Write: API; Read: dashboard, retraining |
| Callee summary cache | Redis | LLM callee summaries (hash → embed) | Write: inference; Read: inference |

---

## 6. Serving Plane Detail

See `docs/SERVING.md` for API spec, inference pipeline, and deployment.

```
Client Request (C function code)
        │
        ▼
FastAPI /v1/scan/function
        │
        ▼
Pre-processing (Joern subprocess pool, ~50ms)
  1. tree-sitter: function boundary detection
  2. Joern: CPG construction (subprocess, pre-warmed)
  3. Taint analyzer: TPG edges
  4. Inter-proc: callee lookup → Redis cache → LLM if miss
        │
        ▼
Embedding (~100ms)
  1. CodeBERT tokenize + encode → 768-d
  2. CPG node feature construction → 824-d
  3. PyG Data object assembly
        │
        ▼
Model Inference (~30ms, GPU)
  1. CodeBERT forward pass
  2. GGNN 3-layer propagation
  3. Cross-attention fusion
  4. Multi-head output
        │
        ▼
Post-processing (~20ms)
  1. Threshold (0.5)
  2. CWE label decode
  3. Illuminati node importance extraction
  4. CFExplainer counterfactual hint
        │
        ▼
Prediction JSON → Client
```

---

## 7. Continuous Learning Plane

```
Weekly batch (or 100+ corrections threshold):
        │
        ▼
Feedback store (corrected FP/FN labels)
        │
        ▼
CFA generator: new CFA pairs from corrections
        │
        ▼
Delta fine-tune (lr=5e-6, on new samples only)
        │
        ▼
Validation: new val F1 ≥ previous - 0.01
        │ PASS
        ▼
Candidate model → shadow mode (48 hours, 10% traffic)
        │ no degradation
        ▼
Promotion: 10% → 50% → 100% traffic rollout
        │ any 5-min window F1 drops > 3%
        ▼ (rollback)
Previous model restored
```

---

*docs/ARCHITECTURE.md | StreamGuard v1.0 | March 2026*
