# P4-S1: Full 4-CPG Model (`model.py`)

**Status:** COMPLETE | **Date:** 2026-03-28 | **Tests:** 37/37 PASS

---

## What Was Built

Upgraded `training/scripts/model/model.py` from Phase 1 proof to Phase 4 production.

| Component | Detail |
|-----------|--------|
| File | `training/scripts/model/model.py` |
| Test file | `tests/test_p4s1_model.py` (37 tests) |
| Depends on | Phase 3 Stage 5 (824-d features), Stage 6 (HDF5 graphs) |

## Architecture

```
Input: PyG Batch (x: N×824, edge_index: 2×E, edge_attr: E, batch: N)

CodeBERT encoder:  microsoft/codebert-base → [CLS] 768-d
Node projector:    Linear(824, 256)
Type-Aware GGNN:   4 GatedGraphConv per layer × 3 layers = 12 convolutions
  Per-layer: [mask by etype → conv → sqrt-count normalize] × 4 → concat → Linear(1024,256)
  GroupNorm(32, 256) + GELU + Dropout(0.1) + residual (from layer 2)
Readout:           mean_pool + max_pool → cat (B,512) → Linear(512,256)
Cross-Attention:   Q=BERT(768→256), K/V=h_nodes(N,256) via scatter softmax
Fused:             cat(BERT_768, Attn_256, GGNN_mean_256) = 1280-d
MLP:               1280 → LayerNorm → GELU → Dropout(0.3) → 512 → GELU → 128
Binary head:       Linear(128, 2)
CWE head:          Linear(128, 12)
Severity head:     Linear(128, 1)
```

## Phase 4 Changes from Phase 1

| Change | Phase 1 | Phase 4 | Why |
|--------|---------|---------|-----|
| Node feature dim | ~256 | 824 | Full feature vector from Stage 5 |
| Normalization | BatchNorm1d | GroupNorm(32, 256) | BatchNorm fails at batch_size=1 during serving |
| Edge types | 3 | 4 (+ TPG) | Novel N3: Taint Propagation Graph |
| Inter-proc stub | Not present | `use_interproc` + `interproc_proj` | Config E ablation |
| Layer freezing | Not present | `freeze_codebert_layers` param | Reduce memory / stabilize training |
| Fused dim (interproc) | N/A | 1536 (1280+256) | Zero-pad when no callee data |
| Checkpoint | Basic | Full config dict + atomic write | Serving reconstruction |
| Return keys | `binary_logits`, `severity` | `logits`, `severity_score` | Consistent with doc spec |
| Intermediates | `ggnn_node_embeddings` | `h_nodes` | Consistent naming |

## Forward Pass Return Dict

| Key | Shape | Description |
|-----|-------|-------------|
| `logits` | (B, 2) | Binary classification head |
| `cwe_logits` | (B, 12) | CWE type classification |
| `severity_score` | (B,) | CVSS proxy scalar |
| `embedding` | (B, 128) | Shared representation for L_CFA |

With `return_intermediates=True`, also includes:
| `h_nodes` | (N, 256) | Per-node GGNN embeddings |
| `graph_embed` | (B, 256) | Graph-level pooled embedding |
| `fused_embedding` | (B, 1280/1536) | Pre-MLP fused vector |
| `bert_cls` | (B, 768) | CodeBERT [CLS] vector |

## Checkpoint Format

`save_checkpoint()` writes atomic `.tmp` + `os.replace()`. Config dict includes:
- `ggnn_type: "per_edge_type_gated"` (identifies type-aware architecture)
- `cpg_components: ["AST", "CFG", "DFG", "TPG"]`
- `node_feature_dim: 824`, `num_edge_types: 4`, `ggnn_layers: 3`, `ggnn_hidden: 256`
- `use_interproc`, `freeze_codebert_layers`, `seed`, `ablation_config`

## Risk Mitigations

| Risk | Status | Evidence |
|------|--------|----------|
| R-03 (BatchNorm at serving) | MITIGATED | GroupNorm(32,256), verified in 4 tests |
| R-17 (CDG edge type >= 4) | MITIGATED | ValueError guard in encode_graph() |
| R-24 (Config E None callees) | MITIGATED | Zero-pad callee_ctx when absent |
| R-25 (Partial checkpoint) | MITIGATED | Atomic .tmp + os.replace() |
| R-31 (freeze too high) | MITIGATED | Param defaults to 0, tested at 9 |

## Bug Found & Fixed

When `use_interproc=True` but no `callee_embeddings` provided, the fused vector was 1280-d but MLP expected 1536-d (RuntimeError in matmul). Fixed by always zero-padding `callee_ctx` to 256-d when callee data is absent.

## Test Summary

```
tests/test_p4s1_model.py — 37/37 PASS

TestForwardPass:           5 tests (shapes, batch_size=1, keys, no NaN, large graph)
TestEdgeTypeIsolation:     2 tests (AST vs TPG, CFG vs DFG)
TestMissingEdgeTypes:      4 tests (no DFG, no TPG, only AST, zero edges)
TestGroupNorm:             4 tests (is GroupNorm, not BatchNorm, config, 3 layers)
TestCheckpoint:            4 tests (round-trip, ggnn_type, config fields, atomic write)
TestInterProc:             4 tests (no proj default, creates proj, forward no crash, fused dim)
TestReturnIntermediates:   5 tests (h_nodes, graph_embed, fused, bert_cls, not returned by default)
TestArchitectureConstants: 5 tests (dims, 12 convolutions, node_feature_dim, proj dims, head dims)
TestEdgeValidation:        2 tests (invalid raises, valid passes)
TestFreezeLayers:          2 tests (freeze 9, freeze 0)
```
