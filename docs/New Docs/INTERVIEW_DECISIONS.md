# StreamGuard — Architecture & Implementation Decisions Log

**Session Date:** March 10, 2026
**Scope:** Pre-implementation technical interview covering M1 (CFA-GNN Proof on SARD)
**Status:** Decisions finalized — ready for Story 2 implementation

---

## 1. Current Project State

| Item | Status |
|------|--------|
| Story 1 (Environment/Deps) | **Complete** — Python, PyTorch, Joern, CodeBERT, tree-sitter all installed |
| Story 2 (Schema + Base Collector) | **Starting now** |
| SARD data download | Not started — pending download from NIST SARD main site |
| Joern DFG verification | Joern installs but `run.ossdataflow` DFG output unverified — **must verify before Story 5** |
| GPU for training | CPU-only dev machine. Will obtain cloud/GPU access (Colab/RunPod) before Story 8 |
| MLflow | Will use `mlflow.set_tracking_uri('file://./mlruns')` — no server needed |

---

## 2. Compute & Training Strategy

- **Development** (Stories 1–7): CPU-only machine for schema, preprocessing, CPG, embedding
- **Training** (Story 8): GPU access obtained before starting — Colab/Kaggle/RunPod acceptable
- **Implication**: All pipeline stages through Story 7 must be runnable on CPU without modification
- **AMP / batch_size**: Use `batch_size=8` + `gradient_accumulation=4` on GPU. On CPU test with `--max-samples 100 --epochs 1` only

---

## 3. SARD Data Download Decision

- **Source**: NIST SARD main site (`samate.nist.gov/SARD`) — full download, all 12 target CWE directories
- **Before committing to sample count targets**: Run MinHash dedup on all downloaded files first to establish true uniqueness baseline
- **Expected**: The 15% rejection estimate in the plan may be optimistic. Dedup may remove 50%+ of SARD's template-generated variants
- **M1 audit threshold**: `min_train_samples = 5,000` (relaxed from 30K). Even 3K samples is acceptable as pipeline proof-of-concept

---

## 4. Joern DFG Verification — Immediate Action Required

**Risk**: If `run.ossdataflow` is missing from the Joern script, the GGNN has zero data-flow signal and F1 will plateau at ~0.70. This is the #1 critical path per `CLAUDE.md`.

**Verification procedure** (do before Story 5):
```bash
# Write a 5-line C function with a clear taint path
echo 'void foo(char *s) { char buf[16]; strcpy(buf, s); }' > /tmp/test.c

# Run Joern with run.ossdataflow in the script
# Check output for edge types — must include DFG edges
# Count: if DFG edge count == 0, STOP and fix the Joern script
```

**Automated gate**: `stage4_cpg.py` will validate DFG edge count in the first 5 processed samples and fail fast with a clear error message if 0.

---

## 5. Ablation Study Design — Confirmed: Run All 5 Configs

Decision: Run **all 5 ablation configurations** in M1 since the architecture already supports them. Only compute cost, no additional engineering.

| Config | Description | Expected F1 | Purpose |
|--------|-------------|-------------|---------|
| A | CodeBERT sequence only (no GGNN) | ~0.79 | True baseline — proves GGNN adds value |
| B | CodeBERT + type-blind GGNN, 3-CPG, no CFA | ~0.83 | Reproduces Vul-LMGNN |
| B' | CodeBERT + type-aware GGNN, 3-CPG, no CFA | ~0.85 | Isolates type-awareness gain |
| C | Config B' + CFA contrastive training | ~0.87–0.89 | **CFA proof** — primary M1 goal |
| D | Config C + TPG (4-CPG) | ~0.89–0.91 | Validates N3 novelty (TPG) |
| D-blind | Config D but type-blind GGNN | ~0.87 | Isolates TPG value from type-awareness |

**CFA proof is positive if Config C F1 > Config B F1 by ≥ 3 percentage points.**

**Ablation validity clarification**: Without Config A and D-blind, the paper cannot cleanly isolate each contribution. Config A is needed to claim GGNN adds value. Config D-blind is needed to separate "adding TPG edges" from "type-aware message passing on TPG edges."

---

## 6. CFA Contrastive Loss — Margin as Hyperparameter

**Formula**: `L_CFA = mean(relu(cosine_sim(emb_v, emb_v') + margin))`
**Effect**: Forces `cosine_sim < -margin` (opposite hemispheres when `margin=0.5`)

**Decision**: Treat `margin` as a hyperparameter. Test `[0.2, 0.5, 0.8]` and select on validation F1.

- `margin=0.5` (plan default): Strong constraint — pushes pairs to opposite hemispheres. Motivated by VISION paper intent.
- `margin=0.2`: Softer — suitable if SARD pairs are only syntactically close (often only differ by bounds check)
- `margin=0.8`: Very aggressive — may cause training instability

**Implementation**: `margin` is a constructor argument in `StreamGuardLoss`, not hardcoded.

```python
class StreamGuardLoss(nn.Module):
    def __init__(self, margin: float = 0.5, ...):
        self.margin = margin
```

---

## 7. TPG Coverage for SARD — Combined Strategy

SARD uses NIST-specific wrapper functions (`printLine`, `printLongLong`, `printHexCharLine`, etc.) that are NOT in the standard SOURCES/SINKS lists. Without extending the lists, taint coverage on SARD will be near-zero.

**Decision**: Two-part strategy:

1. **Pre-inspection**: Before running Stage 4, scan SARD C files to identify the most common function names used as sources/sinks. Extend `SOURCES`/`SINKS` with SARD-specific wrappers.
2. **Accept 20% minimum taint coverage for M1**: The plan already relaxes `min_taint_coverage` to 0.20 for M1. Document as a known limitation.

**SARD-specific additions to investigate**:
```python
# These appear frequently in SARD synthetic code — verify against actual files
SARD_SOURCES_CANDIDATES = {"getData", "getenv", "fscanf", "sscanf"}
SARD_SINKS_CANDIDATES = {"printLine", "printLongLong", "printHexCharLine",
                          "printIntLine", "printUnsignedLine"}
```

**Note**: SARD sinks are mostly output functions (printing), not dangerous sinks like `strcpy`/`system`. The truly dangerous sinks (the actual vulnerability) are the buffer operations in `_bad.c` files. Both must be covered.

---

## 8. Cross-Attention Fusion — Confirmed Implementation

**The single biggest silent-failure risk in the entire system.** A bug here means the model passes gradients and produces outputs, but BERT is not actually conditioning on graph structure — it degrades to simple concatenation with no cross-modal attention.

### Confirmed correct implementation

```python
from torch_geometric.utils import softmax as pyg_softmax
from torch_scatter import scatter  # OR: from torch_geometric.nn.aggr

def cross_attention_fusion(self, bert_cls, h_nodes, h_graph, batch):
    # bert_cls:  (B, 768)   — one vector per graph
    # h_nodes:   (N_total, 256) — all nodes from all graphs concatenated
    # h_graph:   (B, 256)   — mean-pooled graph repr
    # batch:     (N_total,) — maps each node to its graph index

    Q = self.q_proj(bert_cls)       # (B, 256)
    K = self.k_proj(h_nodes)        # (N_total, 256)
    V = self.v_proj(h_nodes)        # (N_total, 256)

    Q_expanded = Q[batch]           # (N_total, 256) — broadcast BERT query to each node
    scores = (Q_expanded * K).sum(dim=-1, keepdim=True) * self.attn_scale  # (N_total, 1)

    # pyg_softmax normalises WITHIN each graph's node set (not across all 800 nodes)
    attn_w = pyg_softmax(scores, batch, num_nodes=bert_cls.size(0))  # (N_total, 1)

    # Weighted sum back to graph level
    attn_out = scatter(attn_w * V, batch, dim=0, reduce='sum')  # (B, 256)

    return torch.cat([bert_cls, attn_out, h_graph], dim=-1)  # (B, 1280)
```

### Why standard softmax breaks

When PyG batches 8 graphs with ~100 nodes each, nodes become a flat `(800, 256)` tensor. `F.softmax` over all 800 nodes means nodes from graph 1 compete with nodes from graph 8 for attention weight — semantically wrong. `pyg_softmax` normalises within each graph's subset using the `batch` index.

### Why GlobalAttention is insufficient

`torch_geometric.nn.GlobalAttention` is scatter-aware but uses a **node-driven gate** — no external query. The gate scores each node based only on its own features. This loses the key novelty: **BERT conditioning** — where the query is the BERT [CLS] representation, making the attention ask "which graph nodes are relevant to THIS function's sequence representation?"

### `torch-scatter` dependency decision

**`torch-scatter` is REMOVED from `requirements.txt`.**

`pyg_softmax` comes from `torch_geometric.utils` — already installed, CUDA-safe, no separate versioned wheel needed. For the weighted sum (`scatter`), use `torch_geometric.nn.aggr` or the built-in `torch_geometric.utils.scatter` to avoid the `torch-scatter` CUDA wheel dependency.

---

## 9. Cross-Attention Verification — 4 Required Smoke Tests

Before moving to losses, all 4 must pass:

| Test | Command / Check | What It Proves |
|------|-----------------|----------------|
| **Test 1**: Normalization | `attn_w.sum()` per graph == 1.0 | scatter_softmax is within-graph, not global |
| **Test 2**: BERT conditioning | Run same graph with 2 different `bert_cls` values → different `attn_w` distributions | BERT query actually conditions the attention |
| **Test 3**: Ablation comparison | Compare BERT-conditioned attn vs `GlobalAttention` (node-gate only) F1 | Confirms BERT→graph conditioning adds measurable value |
| **Test 4**: Gradient flow | After backward(), verify `bert_cls.grad` is non-None and non-zero | BERT encoder receives gradients through the cross-attention path |

Test 1 and 4 are run in unit tests. Tests 2 and 3 are run during Config B' vs C training comparison.

---

## 10. Type-Aware GGNN Concerns & Mitigations

Three concerns raised, all mitigated in the spec:

**Concern 1: Memory — 12 GatedGraphConv modules (4 types × 3 layers)**
- Mitigation: Reduce `batch_size` to 4 + `gradient_accumulation=8` if near VRAM limit
- Expected overhead: ~20MB extra activations per forward pass — acceptable on 8GB+ GPU

**Concern 2: AST edge dominance (~70% of all edges in SARD)**
- Mitigation: Per-type count normalization before aggregation: `h_type / sqrt(count)`
- This prevents the AST-dominated `edge_agg` Linear from ignoring TPG/DFG gradients

**Concern 3: Ablation validity — type-awareness conflated with TPG**
- Mitigation: Config D-blind (type-blind + TPG) explicitly isolates this
- The B' → D gap measures "TPG edges + type-awareness together"
- The D → D-blind gap measures "type-awareness alone on 4-CPG"
- The B → D-blind gap measures "TPG edges alone (type-blind)"

---

## 11. Joern Failure Handling Strategy

**Decision**: Two-layer defense:

**Layer 1 — Pre-filter**: Run `gcc -fsyntax-only` on every SARD file before sending to Joern.
- GCC is fast (~50ms per file)
- SARD uses `#include "std_testcase.h"` — these will fail `gcc -fsyntax-only`. The pre-filter uses a relaxed mode that ignores missing headers: `gcc -fsyntax-only -w -include /dev/null`
- Expect 10–20% of SARD files to fail the GCC pre-filter — log to `gcc_failures.jsonl`, continue

**Layer 2 — Small-batch Joern with fallback**:
- Send 50–100 files per Joern invocation (single JVM, amortized startup)
- Wrap in try/except around `subprocess.run()`
- If batch fails → fall back to per-file processing for that batch
- Per-file mode has `timeout=30s`, logs failures to `cpg_failures.jsonl`

**Why not full-batch (all 12K files at once)**: One malformed file causing a JVM crash kills all remaining samples with no partial output.

---

## 12. CFExplainer Hook — Corrected Design

**What the plan specifies**: `return_intermediates=True` flag that returns saved activation tensors (GGNN layer outputs, attention weights).

**Why this is insufficient**: CFExplainer (ISSTA 2024) works by finding minimal graph perturbations that flip the prediction. This requires gradient-based search, not saved activations. The hook needs **gradient access**, not activation storage.

**Corrected design for M2**:
```python
def forward(self, data, return_intermediates=False):
    # ... normal forward ...
    outputs = {"binary_logits": ..., "cwe_logits": ..., "severity": ...}

    if return_intermediates:
        # Expose for CFExplainer gradient-based perturbation search
        outputs["fused_embedding"] = fused      # gradient flows through this
        outputs["ggnn_node_embeddings"] = h_nodes  # per-node, requires_grad=True
        # Note: do NOT detach these — CFExplainer needs gradients

    return outputs
```

**For M1**: `return_intermediates=False` stub is sufficient. The correction is noted for M2 implementation.

---

## 13. Paper Timeline

- **Target venue**: ISSTA 2027 or ICSE 2027 (12+ months away — realistic)
- **M1 milestone**: CFA-GNN proof on SARD — 3–4 weeks
- **M2 milestone**: Full data collection (30K+ samples, 7 sources) + all 5 ablation configs
- **Paper writing**: After M2 ablation results confirm novelties N1–N4

No workshop paper shortcut planned — target full paper at primary venue.

---

## 14. Forward-Compatibility Changes (Appendix Conflicts 1–9)

All 9 conflicts identified in the plan appendix are addressed in implementation. Summary of changes by story:

| Story | Change | Lines Added |
|-------|--------|-------------|
| 2: Schema | 6 optional fields with defaults: `severity_score`, `commit_sha`, `cve_id`, `cfa_origin`, `aliases`, `metadata` | +15 |
| 2: Schema | `pair_id` and `file_path` added as optional fields | +5 |
| 4: Preprocessing | All stages use `--input`/`--output` CLI args (no hardcoded paths) | +5/stage |
| 6: Embed/Graph | Save `feature_dim=824` in HDF5 metadata | +3 |
| 7: Split | Add `commit_sha`/`cve_id` secondary group key (no-op for SARD) | +10 |
| 8: Model | `node_feature_dim` parameterized (not hardcoded 824) | +2 |
| 8: Model | `use_interproc=False` stub in `forward()` | +10 |
| 8: Model | `return_intermediates=False` flag | +10 |
| 8: Training | Full config dict saved in checkpoint | +15 |
| 8: DataLoader | `source_weights` and `cwe_order` args (None in M1) | +10 |
| 8: MLflow | `EXPERIMENT_NAME_M1 = "streamguard_m1_sard_proof"` | +3 |
| **Total** | | **~90 extra lines** |

**Estimated M2 rework saved**: 2–3 days of re-processing HDF5 data, restructuring model forward pass, fixing data leakage bugs.

---

## 15. Requirements.txt Changes

| Change | Reason |
|--------|--------|
| **Remove** `torch-scatter>=2.1.2` | Replaced by `torch_geometric.utils.softmax` (pyg_softmax). No separate CUDA-matched wheel needed. |
| Keep all other deps | Already present and correct per Story 1 verification |

---

## 16. Open Questions / Not Yet Decided

| Question | When to Decide |
|----------|----------------|
| Exact SARD sample count after dedup | After SARD download + MinHash dedup run (before Story 3) |
| CFA margin hyperparameter final value | After Config B and C training runs (Story 8) |
| Whether taint coverage 20% is achievable on SARD | After Stage 4 CPG run on 100 sample subset |
| Whether `gcc -fsyntax-only` SARD failure rate is acceptable | After pre-filter dry run on 500 SARD files |
| GPU environment specifics (VRAM, batch size adjustment) | Before starting Story 8 |

---

*Generated from technical interview session — March 10, 2026*
*Supersedes any informal notes. All decisions here are implementation-binding for M1.*
