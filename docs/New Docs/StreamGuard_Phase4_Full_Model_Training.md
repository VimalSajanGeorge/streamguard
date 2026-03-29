# StreamGuard — Phase 4: Full Model + Training
## Complete Implementation Guide

**Version:** 1.0 | **Date:** March 2026 | **Status:** Implementation-Ready

| **Phase** | **Stories** | **Duration** | **Output** |
|-----------|------------|--------------|------------|
| 4 of 7 | P4-S1 through P4-S7 | 4–6 days (18–30 GPU hours training) | Best checkpoint + full ablation table |

---

## Table of Contents

1. [Phase 4 Overview & Context](#1-phase-4-overview--context)
2. [What Changes from Phase 1 Proof](#2-what-changes-from-phase-1-proof)
3. [Story Map & Build Order](#3-story-map--build-order)
4. [P4-S1 — Full 4-CPG Model (`model.py`)](#4-p4-s1--full-4-cpg-model)
5. [P4-S2 — Inter-Procedural Context (`callee_summarizer.py`)](#5-p4-s2--inter-procedural-context)
6. [P4-S3 — CFA-Aware DataLoader (`cfa_dataloader.py`)](#6-p4-s3--cfa-aware-dataloader)
7. [P4-S4 — Loss Functions (`losses.py`)](#7-p4-s4--loss-functions)
8. [P4-S5 — Training Loop (`train.py`)](#8-p4-s5--training-loop)
9. [P4-S6 — Evaluation & Metrics (`eval.py`)](#9-p4-s6--evaluation--metrics)
10. [P4-S7 — Ablation Study Runner (`run_ablations.py`)](#10-p4-s7--ablation-study-runner)
11. [Production Risks & Mitigations](#11-production-risks--mitigations)
12. [GPU Infrastructure & Performance](#12-gpu-infrastructure--performance)
13. [Go / No-Go Gates](#13-go--no-go-gates)
14. [Day-by-Day Execution Timeline](#14-day-by-day-execution-timeline)

---

## 1. Phase 4 Overview & Context

Phase 4 trains the full StreamGuard model on the complete 40K+ sample dataset produced by Phase 3. It is the phase that produces the numbers that go into the paper. Every decision made here affects the ablation table, the F1 scores, and whether the conference reviewers accept or reject.

### What Feeds Phase 4

| Input | Source | Required? |
|-------|--------|-----------|
| `training/data/final/train.h5` | Phase 3 Stage 7 | P0 — must exist |
| `training/data/final/val.h5` | Phase 3 Stage 7 | P0 — must exist |
| `training/data/final/test.h5` | Phase 3 Stage 7 | P0 — do NOT touch during training |
| `training/data/processed/exemplar_db.json` | Phase 3 Stage 3 | P1 — for inter-proc M2 |
| Pre-training audit PASS | Phase 3 Stage 7 | MANDATORY gate |

### What Phase 4 Produces

| Output | Location | Purpose |
|--------|----------|---------|
| Best model checkpoint | `training/checkpoints/best_model.pt` | Serving + evaluation |
| Per-epoch checkpoints | `training/checkpoints/epoch_{N}.pt` | Resume on crash |
| MLflow experiment | `mlruns/` | Ablation comparison |
| Ablation result table | `results/ablation_table.json` | Paper Table 2 |
| Per-CWE F1 table | `results/per_cwe_f1.json` | Paper Table 3 |
| Test set predictions | `results/test_predictions.jsonl` | Analysis |

### Critical Rules for Phase 4

> **NEVER touch test.h5 during training.** It is read exactly ONCE at the end, after all model selection is done on val.h5. Using test.h5 for any hyperparameter decision inflates F1 and the paper will be rejected.

> **Same seed across all ablation configs.** `torch.manual_seed(42)`, `numpy.random.seed(42)`, `random.seed(42)` at the top of every training script. Same `train.h5 / val.h5 / test.h5` for all configs. No exceptions.

> **Log everything to MLflow.** Every training run must be reproducible from the logged hyperparameters. Reviewers will ask "how can we reproduce Table 2?"

---

## 2. What Changes from Phase 1 Proof

Phase 1 (CFA-GNN Proof on SARD) built a minimal version to prove CFA works. Phase 4 upgrades to the full production system.

| Component | Phase 1 (Proof) | Phase 4 (Full) | Why Changed |
|-----------|----------------|----------------|-------------|
| Node feature dim | Simplified (~256-d) | **824-d** | Full feature vector from Stage 5 |
| Edge types | 3 (AST/CFG/DFG) | **4 (+ TPG)** | Novel N3: Taint Propagation Graph |
| GGNN norm | BatchNorm1d | **GroupNorm(32, 256)** | BatchNorm fails at batch_size=1 during serving |
| Dataset | SARD only (~8K) | **Full 40K+ samples** | Multi-source, all 12 CWEs |
| CodeBERT usage | Sequence only | **Sequence + Node embedding** | CodeBERT used at inference time too |
| Inter-proc context | Not implemented | **Callee summary injection (N5)** | ICSE 2024 improvement |
| Severity loss | Not used | **L_severity (HuberLoss)** | CVSS proxy head enabled |
| Ablation configs | 2 (B vs C) | **5 configs (A through E)** | Full paper ablation |
| Checkpoint | Basic state_dict | **Full config dict included** | Serving layer needs config to reconstruct |

---

## 3. Story Map & Build Order

```
P4-S1: Full 4-CPG Model (model.py)
  ↓  GATE: Forward pass with 824-d features + all 4 edge types works, no NaN
P4-S2: Inter-Procedural Context (callee_summarizer.py)
  ↓  GATE: Callee embed injected as node feature on call-site nodes
P4-S3: CFA-Aware DataLoader (cfa_dataloader.py)
  ↓  GATE: Pairs in same batch verified by printing first 5 batch pair_ids
P4-S4: Loss Functions (losses.py)
  ↓  GATE: L_CE + L_CFA + L_severity computed without NaN; L_CFA decreasing
P4-S5: Training Loop (train.py)
  ↓  GATE: 1 epoch completes, checkpoint saved, MLflow run logged
P4-S6: Evaluation & Metrics (eval.py)
  ↓  GATE: F1/FPR/FNR/pairwise_accuracy computed on val set
P4-S7: Ablation Runner (run_ablations.py)
  ↓  GATE: All 5 configs train; Config C F1 > Config B F1 (CFA works)
```

---

## 4. P4-S1 — Full 4-CPG Model

### 4.1 What to Build

| Item | Detail |
|------|--------|
| File | `training/scripts/model/model.py` |
| Test file | `tests/test_p4s1_model.py` |
| Depends on | Phase 1 model.py (extend, do not rewrite from scratch) |
| Key upgrade | 824-d node features, TPG as edge type 3, GroupNorm, inter-proc stub |

### 4.2 Critical Architecture Decision: GroupNorm vs BatchNorm

**This is the most important correctness decision in the entire model.**

Phase 1 used `BatchNorm1d(256)` in the GGNN. This must be replaced with `GroupNorm(32, 256)` in Phase 4.

**Why:**
- During serving (Phase 6), the inference worker scans one function at a time: `batch_size=1`.
- `BatchNorm1d` uses batch statistics. At `batch_size=1`, batch stats = single sample stats → normalization becomes identity mapping. The model behaves completely differently at inference time vs training time.
- `GroupNorm` normalizes within each sample's channels, not across the batch. `batch_size=1` works identically to `batch_size=8`.
- `256 channels / 32 groups = 8 channels per group` — this is within the stable 4–16 range.

```python
# WRONG — breaks at batch_size=1:
self.ggnn_bn = nn.BatchNorm1d(256)

# CORRECT — works at any batch size including 1:
self.ggnn_bn = nn.GroupNorm(num_groups=32, num_channels=256)
```

### 4.3 Type-Aware GGNN: Why Per-Edge-Type Convolutions

The GGNN uses **4 separate GatedGraphConv modules per layer** (Option B), one per edge type. A type-blind single GatedGraphConv would:
- Mix AST structural edges with DFG data-flow edges in the same GRU update
- Make the novel TPG component structurally present but semantically invisible (the GNN can't distinguish TAINT edges from CFG edges)
- Reduce Novel N3 contribution to zero

```
Per-layer (3 layers × 4 types = 12 GatedGraphConv modules total):

  AST edges  → GatedGraphConv_AST  → h_AST  (N, 256)
  CFG edges  → GatedGraphConv_CFG  → h_CFG  (N, 256)
  DFG edges  → GatedGraphConv_DFG  → h_DFG  (N, 256)
  TPG edges  → GatedGraphConv_TPG  → h_TPG  (N, 256)
                                    ↓
                          concat → (N, 1024)
                          Linear(1024, 256) + GroupNorm + GELU + residual
                                    ↓
                                  h  (N, 256)
```

**Edge count normalization:** TPG edges are rare (typically 5–15 per graph vs 50+ AST edges). Without normalization, the Linear aggregation is dominated by AST gradients. The per-type count sqrt normalization prevents this:

```python
count = max(mask.sum().item(), 1)
type_outputs.append(h_type / (count ** 0.5))
```

### 4.4 Cross-Attention Fusion: What "True" vs "Degenerate" Means

The architecture doc specifies **TRUE node-level cross-attention**. This is a research-critical distinction.

**Degenerate (wrong):**
```python
# K/V = single graph-level vector (already pooled)
# This is mathematically equivalent to concat(BERT, GGNN_mean)
# No attention, no graph structure used — just 3-way concatenation
Q = bert_cls   # (B, 768)
K = V = graph_embed  # (B, 256) — ALREADY POOLED
attn = softmax(Q @ K.T / sqrt(256)) @ V  # trivially = V for single K/V
```

**True (correct):**
```python
# K/V = per-node embeddings BEFORE pooling
# BERT query attends to EACH of the N graph nodes
# Attention selects which nodes are most relevant to the sequence representation
Q = bert_cls         # (B, 768) → projected to (B, 256)
K = V = h_nodes      # (N, 256) — N nodes across the batch
# scatter softmax within each graph → truly graph-structure-aware
```

This distinction is what makes the fusion "cross-attention" vs "concatenation with extra steps." The paper's novelty claim depends on this being implemented correctly.

### 4.5 Full Model Code

```python
# training/scripts/model/model.py
# Full Phase 4 version — extends Phase 1 proof model

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GatedGraphConv, global_mean_pool, global_max_pool
from torch_scatter import scatter
from transformers import AutoModel, AutoTokenizer
from loguru import logger


class StreamGuardModel(nn.Module):
    """
    StreamGuard: CodeBERT + 3-layer Type-Aware GGNN + Cross-Attention Fusion

    Architecture details (see ARCHITECTURE.md §4):
    - CodeBERT encoder:   microsoft/codebert-base → [CLS] 768-d
    - Node projector:     Linear(824, 256)   — 824-d from Stage 5
    - Type-Aware GGNN:    4 GatedGraphConv per layer × 3 layers = 12 convolutions
    - Cross-Attn Fusion:  Q=BERT (B,768→256), K/V=per-node (N,256) → scatter softmax
    - Fused repr:         concat(BERT_768, Attn_256, GGNN_mean_256) = 1280-d
    - Shared MLP:         1280 → LayerNorm → GELU → Dropout(0.3) → 512 → GELU → 128
    - Output heads:       Binary(2) + CWE(12) + Severity(1)

    Novel contributions:
    1. CodeBERT contextual node embeddings (vs Word2Vec in VISION)
    2. True node-level cross-attention (vs graph-embed concatenation)
    3. Per-edge-type GatedGraphConv (each type has own GRU weights)
    4. 4-component CPG including TPG as edge type 3
    5. Inter-procedural callee summary injection (stub for Config E)
    """

    CODEBERT_DIM  = 768
    NODE_FEAT_DIM = 824   # from Stage 5 preprocessing
    GGNN_HIDDEN   = 256
    GGNN_LAYERS   = 3
    NUM_EDGE_TYPES = 4    # AST=0, CFG=1, DFG=2, TPG=3
    FUSED_DIM     = CODEBERT_DIM + GGNN_HIDDEN + GGNN_HIDDEN  # 1280
    MLP_HIDDEN    = 512
    MLP_OUT       = 128
    NUM_CWE       = 12

    def __init__(
        self,
        codebert_model: str = "microsoft/codebert-base",
        node_feature_dim: int = 824,
        use_interproc: bool = False,     # Config E — set True for ablation E
        freeze_codebert_layers: int = 0, # 0 = don't freeze; 9 = freeze layers 0-8
    ):
        super().__init__()

        self.node_feature_dim = node_feature_dim
        self.use_interproc    = use_interproc

        # ── Sequence Encoder: CodeBERT ──────────────────────────────
        self.codebert  = AutoModel.from_pretrained(codebert_model)
        self.tokenizer = AutoTokenizer.from_pretrained(codebert_model)

        # Optional: freeze lower CodeBERT layers to reduce memory / stabilize training
        if freeze_codebert_layers > 0:
            for i, layer in enumerate(self.codebert.encoder.layer):
                if i < freeze_codebert_layers:
                    for p in layer.parameters():
                        p.requires_grad = False
            logger.info(f"Froze CodeBERT layers 0–{freeze_codebert_layers - 1}")

        # ── Graph Encoder: Type-Aware GGNN ──────────────────────────
        # Project 824-d node features → 256-d GGNN hidden dimension
        self.node_proj = nn.Linear(node_feature_dim, self.GGNN_HIDDEN)

        # 4 edge types × 3 layers = 12 GatedGraphConv modules
        # Each GatedGraphConv has its own GRU weights — types are independent
        self.edge_type_convs = nn.ModuleList([
            nn.ModuleList([
                GatedGraphConv(out_channels=self.GGNN_HIDDEN, num_layers=1)
                for _ in range(self.NUM_EDGE_TYPES)
            ])
            for _ in range(self.GGNN_LAYERS)
        ])

        # Per-layer aggregation: concat 4 type outputs (1024) → project to 256
        self.edge_agg = nn.ModuleList([
            nn.Linear(self.GGNN_HIDDEN * self.NUM_EDGE_TYPES, self.GGNN_HIDDEN)
            for _ in range(self.GGNN_LAYERS)
        ])

        # IMPORTANT: GroupNorm NOT BatchNorm — works at batch_size=1 during serving
        # 256 channels / 32 groups = 8 channels per group (stable range)
        self.ggnn_norm = nn.ModuleList([
            nn.GroupNorm(num_groups=32, num_channels=self.GGNN_HIDDEN)
            for _ in range(self.GGNN_LAYERS)
        ])
        self.ggnn_dropout = nn.Dropout(0.1)

        # Graph-level readout: mean_pool + max_pool → 512 → 256
        self.graph_readout = nn.Linear(self.GGNN_HIDDEN * 2, self.GGNN_HIDDEN)

        # ── Cross-Attention Fusion ───────────────────────────────────
        # Q = BERT [CLS], K/V = per-node GGNN embeddings
        # TRUE node-level attention: Q (B,768→256) attends to N nodes (N,256)
        self.q_proj    = nn.Linear(self.CODEBERT_DIM, self.GGNN_HIDDEN)
        self.k_proj    = nn.Linear(self.GGNN_HIDDEN,  self.GGNN_HIDDEN)
        self.v_proj    = nn.Linear(self.GGNN_HIDDEN,  self.GGNN_HIDDEN)
        self.attn_scale = self.GGNN_HIDDEN ** -0.5

        # ── Inter-Procedural Context Stub (Config E only) ───────────
        # Callee summary embeddings projected and added to call-site node features
        if use_interproc:
            self.interproc_proj = nn.Linear(self.CODEBERT_DIM, self.GGNN_HIDDEN)
            # Will be used to inject callee_embeddings (B, max_callees, 768)
            # into the relevant call-site nodes in the CPG before GGNN forward pass
            logger.info("Inter-procedural context enabled (Config E)")

        # ── Shared MLP After Fusion ──────────────────────────────────
        # Input: 1280-d (BERT_768 + Attn_256 + GGNN_mean_256)
        # Note: if inter-proc enabled, fused_dim += GGNN_HIDDEN = 1536
        fused_dim = self.FUSED_DIM + (self.GGNN_HIDDEN if use_interproc else 0)

        self.fusion_mlp = nn.Sequential(
            nn.Linear(fused_dim, self.MLP_HIDDEN),
            nn.LayerNorm(self.MLP_HIDDEN),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(self.MLP_HIDDEN, self.MLP_OUT),
            nn.GELU(),
        )

        # ── Output Heads ─────────────────────────────────────────────
        self.binary_head   = nn.Linear(self.MLP_OUT, 2)           # vuln / safe
        self.cwe_head      = nn.Linear(self.MLP_OUT, self.NUM_CWE) # 12 CWE classes
        self.severity_head = nn.Linear(self.MLP_OUT, 1)           # CVSS proxy [0,10]

    # ────────────────────────────────────────────────────────────────
    # Encoding sub-components
    # ────────────────────────────────────────────────────────────────

    def encode_sequence(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """Encode full function token sequence with CodeBERT. Returns [CLS] 768-d."""
        out = self.codebert(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state[:, 0, :]  # (B, 768)

    def encode_graph(
        self,
        x: torch.Tensor,          # (N, 824) node features
        edge_index: torch.Tensor,  # (2, E)
        edge_attr: torch.Tensor,   # (E,) edge type int {0,1,2,3}
        batch: torch.Tensor,       # (N,) batch assignment
        callee_node_mask: torch.Tensor = None,  # (N,) bool — call-site nodes
        callee_feat: torch.Tensor = None,       # (N_callees, 256) projected callee embeds
    ):
        """
        3-layer type-aware GGNN forward pass.
        Returns (h_graph, h_nodes): (B, 256), (N, 256)
        """
        h = self.node_proj(x)  # (N, 256)

        # Optional: inject inter-proc callee features at call-site nodes
        if self.use_interproc and callee_node_mask is not None and callee_feat is not None:
            h[callee_node_mask] = h[callee_node_mask] + callee_feat

        for layer_idx in range(self.GGNN_LAYERS):
            type_outputs = []

            for etype in range(self.NUM_EDGE_TYPES):
                mask = (edge_attr == etype)
                ei_type = edge_index[:, mask]

                if ei_type.size(1) == 0:
                    # No edges of this type → zero contribution (not skip)
                    type_outputs.append(torch.zeros_like(h))
                else:
                    h_type = self.edge_type_convs[layer_idx][etype](h, ei_type)
                    # Sqrt-count normalization: prevents AST edge dominance over TPG
                    count = max(mask.sum().item(), 1)
                    type_outputs.append(h_type / (count ** 0.5))

            # Aggregate: cat 4 outputs → (N, 1024) → project → (N, 256)
            h_concat = torch.cat(type_outputs, dim=-1)
            h_new    = self.edge_agg[layer_idx](h_concat)
            h_new    = self.ggnn_norm[layer_idx](h_new)  # GroupNorm
            h_new    = F.gelu(h_new)
            h_new    = self.ggnn_dropout(h_new)

            # Residual connection from layer 2 onward (not layer 1 — different dim from proj)
            h = h + h_new if layer_idx > 0 else h_new

        # Graph-level readout: mean + max pool
        h_mean  = global_mean_pool(h, batch)            # (B, 256)
        h_max   = global_max_pool(h, batch)             # (B, 256)
        h_graph = self.graph_readout(
            torch.cat([h_mean, h_max], dim=-1)          # (B, 512) → (B, 256)
        )
        return h_graph, h  # (B, 256), (N, 256)

    def cross_attention_fusion(
        self,
        bert_cls: torch.Tensor,   # (B, 768)
        h_nodes:  torch.Tensor,   # (N, 256) per-node GGNN embeddings
        batch:    torch.Tensor,   # (N,) batch assignment
    ):
        """
        TRUE node-level cross-attention.
        BERT [CLS] (B,768) queries attend over every CPG node (N, 256).
        Scatter softmax normalises attention weights within each graph.
        Returns fused (B, 1280).
        """
        Q = self.q_proj(bert_cls)  # (B, 256)
        K = self.k_proj(h_nodes)   # (N, 256)
        V = self.v_proj(h_nodes)   # (N, 256)

        # Each node gets its graph's query vector
        Q_expanded = Q[batch]      # (N, 256)

        # Raw attention scores per node
        attn_scores = (Q_expanded * K).sum(dim=-1) * self.attn_scale  # (N,)

        # Scatter softmax: sum(exp(scores)) per graph, then normalize
        exp_scores    = attn_scores.exp()
        sum_exp       = scatter(exp_scores, batch, dim=0, reduce='sum')  # (B,)
        attn_weights  = exp_scores / sum_exp[batch].clamp(min=1e-8)      # (N,)

        # Weighted value aggregation per graph
        weighted_V = V * attn_weights.unsqueeze(-1)                       # (N, 256)
        attn_out   = scatter(weighted_V, batch, dim=0, reduce='sum')      # (B, 256)

        # Graph-level mean embed for concat (not recomputed from scratch)
        graph_embed = scatter(h_nodes, batch, dim=0, reduce='mean')       # (B, 256)

        return torch.cat([bert_cls, attn_out, graph_embed], dim=-1)        # (B, 1280)

    # ────────────────────────────────────────────────────────────────
    # Full forward pass
    # ────────────────────────────────────────────────────────────────

    def forward(
        self,
        data,                       # PyG Batch (x, edge_index, edge_attr, batch)
        input_ids=None,             # (B, 512) — CodeBERT tokenized full function
        attention_mask=None,        # (B, 512)
        callee_embeddings=None,     # (B, max_callees, 768) — inter-proc context
        callee_node_indices=None,   # list of (batch_idx, node_idx) for call sites
        return_intermediates=False, # True → expose h_nodes + attn_weights for CFExplainer
    ):
        # ── Graph Encoding ───────────────────────────────────────────
        # Prepare inter-proc features if enabled
        callee_node_mask, callee_feat = None, None
        if self.use_interproc and callee_embeddings is not None:
            callee_node_mask, callee_feat = self._prepare_interproc_features(
                data, callee_embeddings, callee_node_indices
            )

        graph_embed, h_nodes = self.encode_graph(
            data.x, data.edge_index, data.edge_attr, data.batch,
            callee_node_mask, callee_feat,
        )  # (B, 256), (N, 256)

        # ── Sequence Encoding ────────────────────────────────────────
        if input_ids is not None:
            bert_cls = self.encode_sequence(input_ids, attention_mask)  # (B, 768)
        else:
            # Graph-only mode (used in ablation Config B without CodeBERT sequence)
            B = graph_embed.size(0)
            bert_cls = torch.zeros(B, self.CODEBERT_DIM, device=graph_embed.device)

        # ── Cross-Attention Fusion ───────────────────────────────────
        fused = self.cross_attention_fusion(bert_cls, h_nodes, data.batch)  # (B, 1280)

        # ── Inter-Proc Extension ─────────────────────────────────────
        if self.use_interproc and callee_embeddings is not None:
            # Pool callee embeddings → (B, 768) → project → (B, 256)
            callee_pooled = callee_embeddings.mean(dim=1)  # (B, 768)
            callee_ctx    = self.interproc_proj(callee_pooled)  # (B, 256)
            fused = torch.cat([fused, callee_ctx], dim=-1)  # (B, 1536)

        # ── Shared MLP ───────────────────────────────────────────────
        shared = self.fusion_mlp(fused)  # (B, 128)

        # ── Output Heads ─────────────────────────────────────────────
        result = {
            "logits":         self.binary_head(shared),                # (B, 2)
            "cwe_logits":     self.cwe_head(shared),                   # (B, 12)
            "severity_score": self.severity_head(shared).squeeze(-1),  # (B,)
            "embedding":      shared,                                   # (B, 128) for L_CFA
        }

        if return_intermediates:
            result["h_nodes"]         = h_nodes
            result["graph_embed"]     = graph_embed
            result["fused_embedding"] = fused
            result["bert_cls"]        = bert_cls

        return result

    def _prepare_interproc_features(self, data, callee_embeddings, callee_node_indices):
        """
        Inject callee summary embeddings into the CPG at call-site nodes.
        callee_node_indices: list of (graph_idx, local_node_idx) tuples
        Returns: (callee_node_mask, projected_callee_features)
        """
        N = data.x.size(0)
        callee_node_mask = torch.zeros(N, dtype=torch.bool, device=data.x.device)
        callee_feat      = torch.zeros(N, self.GGNN_HIDDEN, device=data.x.device)

        if callee_node_indices is None:
            return callee_node_mask, callee_feat

        # Build global node index from (graph_idx, local_node_idx)
        cumsum = torch.zeros(data.num_graphs + 1, dtype=torch.long, device=data.x.device)
        for i in range(data.num_graphs):
            cumsum[i + 1] = cumsum[i] + (data.batch == i).sum()

        for graph_idx, local_node_idx, callee_embed in callee_node_indices:
            global_idx = cumsum[graph_idx].item() + local_node_idx
            if global_idx < N:
                callee_node_mask[global_idx] = True
                callee_feat[global_idx] = self.interproc_proj(
                    callee_embed.unsqueeze(0)
                ).squeeze(0)

        return callee_node_mask, callee_feat


def save_checkpoint(model, optimizer, epoch, metrics, config, path):
    """Save full checkpoint including architecture config for serving reconstruction."""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "metrics": metrics,
        "config": {
            # Architecture config — needed by serving layer to reconstruct model
            "codebert_model":       config["base_model"],
            "node_feature_dim":     824,
            "use_interproc":        config.get("use_interproc", False),
            "freeze_codebert_layers": config.get("freeze_codebert_layers", 0),
            "num_edge_types":       4,
            "num_cwe_classes":      12,
            "ggnn_layers":          3,
            "ggnn_hidden":          256,
            "ggnn_type":            "per_edge_type_gated",  # identifies type-aware arch
            "cpg_components":       ["AST", "CFG", "DFG", "TPG"],
            "ablation_config":      config.get("ablation_config", "E_full"),
            "seed":                 config.get("seed", 42),
        }
    }
    tmp_path = path + ".tmp"
    torch.save(checkpoint, tmp_path)
    import os
    os.replace(tmp_path, path)


def load_checkpoint(path, device):
    """Load checkpoint. Returns model, optimizer_state, epoch, metrics, config."""
    checkpoint = torch.load(path, map_location=device)
    config = checkpoint["config"]
    model = StreamGuardModel(
        codebert_model=config["codebert_model"],
        node_feature_dim=config.get("node_feature_dim", 824),
        use_interproc=config.get("use_interproc", False),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, checkpoint["optimizer_state_dict"], checkpoint["epoch"], \
           checkpoint["metrics"], config
```

### 4.6 Claude Code Prompt — P4-S1

```
▶ CLAUDE CODE PROMPT — P4-S1 — Full 4-CPG Model

Read docs/MODEL.md §1 (Complete Model Code Template) completely.
Read docs/ARCHITECTURE.md §4 (Model Plane Detail) completely.
Read docs/NOVELTY.md §N3 (TPG as 4th component) and §N2 (CodeBERT+CFA).
Read training/scripts/model/model.py from Phase 1 (the CFA-GNN proof version).

CRITICAL CHANGES from Phase 1 to Phase 4:
1. node_feature_dim = 824 (was ~256 in Phase 1 proof)
2. Replace BatchNorm1d(256) with GroupNorm(num_groups=32, num_channels=256)
   REASON: BatchNorm1d fails at batch_size=1 during serving inference
3. TPG as edge type 3 (was 3 types in Phase 1)
4. Add inter-proc stub: use_interproc=False by default (Config E uses True)
5. save_checkpoint() MUST include config dict for serving reconstruction

Build model.py with EXACTLY the architecture in MODEL.md:
  - CodeBERT encoder: microsoft/codebert-base → [CLS] 768-d
  - Node projector: Linear(824, 256)
  - GGNN: 4 separate GatedGraphConv per layer × 3 layers = 12 total
  - Per-layer: [mask by etype → conv → sqrt-count normalize] × 4 → concat → Linear(1024,256)
  - GroupNorm(32, 256) — NOT BatchNorm1d — after each layer
  - Readout: mean_pool + max_pool → cat → (B,512) → Linear(512,256)
  - Cross-attention fusion: Q=BERT(768→256), K/V=h_nodes(N,256) via scatter softmax
  - Fused: cat(BERT_768, Attn_256, GGNN_mean_256) = 1280-d
  - MLP: 1280 → LayerNorm → GELU → Dropout(0.3) → 512 → GELU → 128
  - Binary head: Linear(128, 2)
  - CWE head: Linear(128, 12)
  - Severity head: Linear(128, 1)

Forward pass must return dict with keys: logits, cwe_logits, severity_score, embedding
Set return_intermediates=True to also return h_nodes, graph_embed, fused_embedding, bert_cls

MANDATORY 3 smoke tests:

Test 1 — Full forward pass:
  python3 -c "
  import torch; from torch_geometric.data import Data, Batch; from model import StreamGuardModel
  m = StreamGuardModel()
  # Create batch of 2 graphs, 10+8 nodes, 4 edge types
  x1 = torch.randn(10, 824); ei1 = torch.randint(0,10,(2,20)); ea1 = torch.randint(0,4,(20,))
  x2 = torch.randn(8,  824); ei2 = torch.randint(0,8, (2,15)); ea2 = torch.randint(0,4,(15,))
  d1 = Data(x=x1, edge_index=ei1, edge_attr=ea1, y=torch.tensor([1]))
  d2 = Data(x=x2, edge_index=ei2, edge_attr=ea2, y=torch.tensor([0]))
  batch = Batch.from_data_list([d1, d2])
  out = m(batch)
  print('logits:', out['logits'].shape)        # expected: (2, 2)
  print('cwe:',    out['cwe_logits'].shape)    # expected: (2, 12)
  print('sev:',    out['severity_score'].shape) # expected: (2,)
  print('embed:',  out['embedding'].shape)      # expected: (2, 128)
  print('ALL SHAPES CORRECT')
  "

Test 2 — Edge type isolation (proves types ARE differentiated):
  Run same 10-node graph with ALL edges as AST (type 0) vs ALL as TPG (type 3).
  Assert: outputs are DIFFERENT (if types were ignored, outputs would be identical).

Test 3 — Missing edge types (no crash on zero-edge-type graphs):
  Run a graph with ZERO DFG edges (edge_attr has no 2-values).
  Assert: no crash, no NaN in output.
  (This is common for very simple safe functions that have no data flow paths)

Write tests/test_p4s1_model.py with these 3 smoke tests plus:
  - GroupNorm check: model.ggnn_norm[0] is nn.GroupNorm (not BatchNorm1d)
  - Checkpoint round-trip: save + load + forward pass gives same logits
  - save_checkpoint includes 'ggnn_type': 'per_edge_type_gated' in config dict
  - Inter-proc stub: use_interproc=True creates interproc_proj Linear layer
  - return_intermediates=True: h_nodes shape is (N, 256)
```

### 4.7 Verification Checklist

| Check | Command | Expected |
|-------|---------|---------|
| VC-S1-01 | `pytest tests/test_p4s1_model.py` | All tests PASS |
| VC-S1-02 | Forward pass smoke test (see prompt) | Correct output shapes |
| VC-S1-03 | Edge type isolation test | TPG-only ≠ AST-only embeddings |
| VC-S1-04 | Zero-edge-type test | No crash, no NaN |
| VC-S1-05 | `type(model.ggnn_norm[0])` | `GroupNorm` (not `BatchNorm1d`) |
| VC-S1-06 | Checkpoint config has `ggnn_type` key | `"per_edge_type_gated"` |

---

## 5. P4-S2 — Inter-Procedural Context

### 5.1 What to Build

| Item | Detail |
|------|--------|
| Files | `training/scripts/model/callee_summarizer.py`, `training/scripts/model/callee_cache.py` |
| Used by | Config E ablation (`use_interproc=True`). Skip for Configs A–D. |
| Research basis | VulnSC / Inter-proc (ICSE 2024) — callee summary injection |

### 5.2 What This Does (Config E Only)

When a C function calls another function (e.g., `parse_input()`, `sanitize()`), the model needs to know what that callee does — specifically whether it sanitizes data. Without inter-proc context, the model sees a call-site node but has no information about what's being called.

Config E injects a 768-d CodeBERT summary of each called function as an additional feature on the call-site node in the CPG:

```
CPG (standard):
  CALL node for "parse_input(argv[1])" → 824-d features (code/type/taint/structural)

CPG (with inter-proc, Config E):
  CALL node for "parse_input(argv[1])" → 824-d features + projected callee summary
  where callee summary = CodeBERT([CLS]) encoding of parse_input() body
```

### 5.3 Architecture

```
CalleeSummarizer:
  1. For each CALL node in CPG: extract callee function name
  2. Look up callee source code:
     - First check: Redis cache (hash → 768-d embed)
     - Cache miss: call CodeBERT on callee body → 768-d embed
     - If callee body unavailable: zero vector fallback
  3. Return: (callee_node_indices, callee_embeddings) tuples
     → passed to model.forward() via callee_embeddings and callee_node_indices

CalleeCache (Redis):
  key:   sha256(callee_source_code)
  value: json-encoded 768-d float32 embedding
  TTL:   None (permanent cache — callee embeds don't change during training)
```

### 5.4 Code

```python
# training/scripts/model/callee_summarizer.py

import torch
import hashlib
import json
import redis
from transformers import AutoModel, AutoTokenizer
from loguru import logger
from pathlib import Path
import h5py


class CalleeCache:
    """
    Redis-backed cache of CodeBERT embeddings for callee functions.
    key: sha256(callee_code) → value: 768-d float list
    Falls back to in-memory dict if Redis not available (training mode).
    """

    def __init__(self, redis_url: str = "redis://localhost:6379/0", use_memory_fallback: bool = True):
        self._memory: dict = {}
        self._redis = None
        self._use_redis = False

        try:
            r = redis.Redis.from_url(redis_url, socket_timeout=2, socket_connect_timeout=2)
            r.ping()
            self._redis = r
            self._use_redis = True
            logger.info("CalleeCache: Redis connected")
        except Exception as e:
            if use_memory_fallback:
                logger.warning(f"CalleeCache: Redis unavailable ({e}), using memory cache")
            else:
                raise

    def _key(self, code: str) -> str:
        return "callee:" + hashlib.sha256(code.encode()).hexdigest()

    def get(self, code: str):
        k = self._key(code)
        if self._use_redis:
            v = self._redis.get(k)
            if v:
                return torch.tensor(json.loads(v), dtype=torch.float32)
        elif k in self._memory:
            return self._memory[k]
        return None

    def set(self, code: str, embed: torch.Tensor):
        k = self._key(code)
        v = json.dumps(embed.tolist())
        if self._use_redis:
            self._redis.set(k, v)
        else:
            self._memory[k] = embed

    def size(self) -> int:
        if self._use_redis:
            return self._redis.dbsize()
        return len(self._memory)


class CalleeSummarizer:
    """
    Generates CodeBERT embeddings for callee functions and injects them
    into CPG CALL nodes as inter-procedural context.

    Used only for ablation Config E (use_interproc=True).
    """

    def __init__(
        self,
        codebert_model: str = "microsoft/codebert-base",
        cache: CalleeCache = None,
        device: str = "cpu",
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(codebert_model)
        self.model     = AutoModel.from_pretrained(codebert_model).to(device).eval()
        self.cache     = cache or CalleeCache()
        self.device    = device

    @torch.no_grad()
    def embed_callee(self, callee_code: str) -> torch.Tensor:
        """
        Compute CodeBERT [CLS] embedding for a callee function.
        Returns 768-d tensor. Uses cache first.
        """
        cached = self.cache.get(callee_code)
        if cached is not None:
            return cached.to(self.device)

        inputs = self.tokenizer(
            callee_code,
            max_length=512,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        ).to(self.device)

        out   = self.model(**inputs)
        embed = out.last_hidden_state[:, 0, :].squeeze(0)  # (768,)
        self.cache.set(callee_code, embed.cpu())
        return embed

    def get_callee_context(
        self,
        cpg_json: dict,               # CPG dict with nodes + edges
        callee_sources: dict,         # {function_name: source_code}
    ):
        """
        For each CALL node in the CPG that matches a known callee:
          - Embed the callee function
          - Record the (graph_local_node_idx, callee_embed) pair

        Returns: list of (local_node_idx, embed_768d) tuples
        """
        results = []
        nodes   = cpg_json.get("nodes", [])

        for node_idx, node in enumerate(nodes):
            if node.get("_label") != "CALL":
                continue
            fname = node.get("name", "").split(".")[-1]  # strip namespace
            if fname in callee_sources:
                try:
                    embed = self.embed_callee(callee_sources[fname])
                    results.append((node_idx, embed))
                except Exception as e:
                    logger.debug(f"Callee embed failed for {fname}: {e}")

        return results
```

### 5.5 Claude Code Prompt — P4-S2

```
▶ CLAUDE CODE PROMPT — P4-S2 — Inter-Procedural Context

Read docs/ARCHITECTURE.md §2 (Data Plane) and §4 (Model Plane) for inter-proc refs.
Read docs/NOVELTY.md §N5 (Inter-Procedural Context + CFA).
Read the Phase 1 model.py's use_interproc stub you built in P4-S1.

Build:
  training/scripts/model/callee_summarizer.py
  training/scripts/model/callee_cache.py

IMPORTANT: This story is for ablation Config E only. Configs A-D do NOT use it.
The callee_summarizer provides context for call-site nodes in the CPG.

CalleeSummarizer requirements:
1. embed_callee(callee_code) → 768-d tensor (cached via CalleeCache)
2. get_callee_context(cpg_json, callee_sources) → list of (node_idx, embed) tuples
   Only CALL nodes where callee name is in callee_sources dict get embeddings.
3. Zero-vector fallback when callee source not available (don't crash)

CalleeCache requirements:
1. Redis backend with in-memory fallback if Redis unavailable
2. key = "callee:" + sha256(callee_source_code)
3. Cache hit: return cached tensor directly (skip CodeBERT call)
4. Must be initializable without Redis for training-time use

Integration into training:
For Config E training, the DataLoader must pass:
  - callee_sources: dict from sample metadata (populated if available)
  - CalleeSummarizer processes them during batch collation
  - (callee_node_indices, callee_embeddings) tensors passed to model.forward()

For Configs A-D: callee_embeddings=None, callee_node_indices=None
  model.forward() handles None gracefully (already in P4-S1 code).

Write tests/test_p4s2_callee.py with 6 tests:
1. embed_callee returns 768-d tensor
2. embed_callee uses cache on second call (not calling CodeBERT again)
3. get_callee_context: CALL node with known callee → returns (idx, embed) tuple
4. get_callee_context: non-CALL node → not included in results
5. get_callee_context: unknown callee → not included (no crash)
6. CalleeCache memory fallback: works without Redis connection
```

---

## 6. P4-S3 — CFA-Aware DataLoader

### 6.1 Why This Is Critical

The entire CFA contrastive training objective depends on pairs being in the same batch. Standard PyTorch DataLoader shuffles samples randomly — this splits CFA pairs across batches, making `L_CFA` train on random unrelated pairs.

```
WRONG (standard DataLoader):
  Batch 1: [vuln_a, safe_c, vuln_d, safe_b]  ← pair (a, a') split — a' in Batch 3
  Batch 2: [vuln_e, vuln_a', safe_d, safe_e'] ← a' here, a is in Batch 1
  Result: L_CFA(vuln_a, safe_d) — wrong pair!

CORRECT (CFAAwareBatchSampler):
  Batch 1: [vuln_a, safe_a', vuln_b, safe_b'] ← both pairs together
  Batch 2: [vuln_c, safe_c', singleton_d, ...]
  Result: L_CFA(vuln_a, safe_a') — correct pair!
```

### 6.2 HDF5 Dataset

```python
# training/scripts/model/cfa_dataloader.py

import torch
import h5py
import random
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch


class CFADataset(Dataset):
    """
    Loads PyG graph Data objects from HDF5 cache.
    Indexes all samples with their pair_id for CFA-aware batching.
    """

    def __init__(self, h5_path: str, split: str = "train"):
        self.h5_path = h5_path
        self.split   = split
        self._index  = self._build_index()

    def _build_index(self):
        """Build (sample_id, pair_id, label, cwe) index from HDF5."""
        index = []
        with h5py.File(self.h5_path, "r") as f:
            split_group = f.get(self.split)
            if split_group is None:
                # Flat HDF5 structure (all samples at root level with split attr)
                for sample_id in f.keys():
                    grp = f[sample_id]
                    if grp.attrs.get("split", "") != self.split:
                        continue
                    index.append({
                        "sample_id": sample_id,
                        "pair_id":   str(grp.attrs.get("pair_id", "")),
                        "label":     int(grp["y"][0]),
                        "cwe":       str(grp.attrs.get("cwe", "")),
                        "source":    str(grp.attrs.get("source", "")),
                    })
            else:
                for sample_id in split_group.keys():
                    grp = split_group[sample_id]
                    index.append({
                        "sample_id": sample_id,
                        "pair_id":   str(grp.attrs.get("pair_id", "")),
                        "label":     int(grp["y"][0]),
                        "cwe":       str(grp.attrs.get("cwe", "")),
                        "source":    str(grp.attrs.get("source", "")),
                    })
        return index

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        item = self._index[idx]
        graph = self._load_graph(item["sample_id"])
        return graph, item

    def _load_graph(self, sample_id: str) -> Data:
        with h5py.File(self.h5_path, "r") as f:
            # Handle both nested (split/sample_id) and flat (sample_id) HDF5 layouts
            grp = f.get(f"{self.split}/{sample_id}") or f[sample_id]

            x          = torch.from_numpy(grp["x"][:]).float()
            edge_index = torch.from_numpy(grp["edge_index"][:]).long()
            # Support both 'edge_type' (Phase 4) and 'edge_attr' (Phase 1 compatibility)
            ea_key     = "edge_type" if "edge_type" in grp else "edge_attr"
            edge_attr  = torch.from_numpy(grp[ea_key][:]).long()
            y          = torch.from_numpy(grp["y"][:]).long()

            return Data(
                x=x, edge_index=edge_index, edge_attr=edge_attr, y=y,
                sample_id=sample_id,
                pair_id=str(grp.attrs.get("pair_id", "")),
                cwe=str(grp.attrs.get("cwe", "")),
            )


class CFAAwareBatchSampler:
    """
    Groups samples by pair_id before batching.
    Ensures (vuln, cfa) pairs always appear in the same batch.

    Algorithm:
    1. Group all sample indices by pair_id
    2. Shuffle groups (not individual samples)
    3. Fill batches from groups — partial groups are carried over

    CRITICAL: If pairs are split across batches, L_CFA trains on wrong pairs.
    """

    def __init__(
        self,
        dataset: CFADataset,
        batch_size: int,
        drop_last: bool = True,
        shuffle: bool = True,
        seed: int = 42,
    ):
        self.batch_size = batch_size
        self.drop_last  = drop_last
        self.shuffle    = shuffle
        self.seed       = seed
        self._epoch     = 0

        # Group sample indices by pair_id
        pair_groups = defaultdict(list)
        for i, item in enumerate(dataset._index):
            key = item["pair_id"] if item["pair_id"] else f"singleton_{i}"
            pair_groups[key].append(i)

        self.groups = list(pair_groups.values())

    def set_epoch(self, epoch: int):
        """Call at start of each epoch for deterministic shuffling."""
        self._epoch = epoch

    def __iter__(self):
        if self.shuffle:
            rng = random.Random(self.seed + self._epoch)
            groups = list(self.groups)
            rng.shuffle(groups)
        else:
            groups = self.groups

        current_batch = []
        for group in groups:
            current_batch.extend(group)
            while len(current_batch) >= self.batch_size:
                yield current_batch[:self.batch_size]
                current_batch = current_batch[self.batch_size:]

        if current_batch and not self.drop_last:
            yield current_batch

    def __len__(self):
        total = sum(len(g) for g in self.groups)
        if self.drop_last:
            return total // self.batch_size
        return (total + self.batch_size - 1) // self.batch_size


def cfa_collate_fn(batch):
    """
    Custom collate function that separates original samples and CFA counterparts
    into two aligned sub-batches.

    Logic:
    - Samples with empty pair_id → always "original"
    - First occurrence of a pair_id → "original"
    - Second occurrence of same pair_id → "CFA counterpart"

    Returns: (orig_batch, cfa_batch, orig_metas, cfa_metas)
    """
    graphs, metas = zip(*batch)

    orig_graphs, cfa_graphs = [], []
    orig_metas,  cfa_metas  = [], []
    seen_pair_ids = {}

    for g, m in zip(graphs, metas):
        pid = m["pair_id"]

        # Empty pair_id or first time seeing this pair_id → original
        if not pid or pid not in seen_pair_ids:
            seen_pair_ids[pid] = len(orig_graphs) if pid else None
            orig_graphs.append(g)
            orig_metas.append(m)
        else:
            # This is the CFA counterpart
            cfa_graphs.append(g)
            cfa_metas.append(m)

    orig_batch = Batch.from_data_list(orig_graphs)
    cfa_batch  = Batch.from_data_list(cfa_graphs) if cfa_graphs else None

    return orig_batch, cfa_batch, orig_metas, cfa_metas


def build_dataloader(
    h5_path: str,
    split: str,
    batch_size: int,
    num_workers: int = 4,
    shuffle: bool = True,
    seed: int = 42,
) -> DataLoader:
    dataset = CFADataset(h5_path, split)
    sampler = CFAAwareBatchSampler(
        dataset, batch_size, drop_last=(split == "train"), shuffle=shuffle, seed=seed
    )
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        collate_fn=cfa_collate_fn,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
```

### 6.3 Claude Code Prompt — P4-S3

```
▶ CLAUDE CODE PROMPT — P4-S3 — CFA-Aware DataLoader

Read docs/MODEL.md §3 (CFA-Aware DataLoader) completely.
Read docs/CLAUDE.md Rule 4 (CFA Pairs Are Sacred).
Read training/scripts/model/cfa_dataloader.py from Phase 1 (the proof version).

Build training/scripts/model/cfa_dataloader.py — the FULL PRODUCTION version.

CRITICAL DIFFERENCE from Phase 1:
Phase 1 assumed simple HDF5 structure. Phase 4 must handle BOTH:
  - Nested: f["train"]["sample_id"] (Stage 7 default output)
  - Flat: f["sample_id"] with attrs["split"] (Phase 1 output)
Check which structure exists and branch accordingly.

CFADataset requirements:
1. __getitem__ returns (PyG Data, metadata dict)
2. metadata dict MUST contain: sample_id, pair_id, label, cwe, source
3. Load 'edge_type' key from HDF5 (Phase 4 format); fall back to 'edge_attr' (Phase 1)
4. x.dtype must be float32 (not float64 from numpy)
5. edge_index.dtype must be long (int64)

CFAAwareBatchSampler requirements:
1. Group by pair_id — empty pair_id → singleton group (unique key)
2. Shuffle GROUPS not individual samples (use set_epoch() for reproducible shuffling)
3. Carry over partial groups to next batch
4. drop_last=True for train, drop_last=False for val/test

cfa_collate_fn requirements:
1. Separate orig and CFA by pair_id occurrence (first = orig, second = CFA)
2. Return (orig_batch, cfa_batch, orig_metas, cfa_metas)
3. cfa_batch = None if no CFA samples in this batch (singletons batch)
4. orig_batch and cfa_batch must be Batch objects (not lists)

Verification test — run this BEFORE any training:
python3 -c "
from cfa_dataloader import build_dataloader
from collections import defaultdict

loader = build_dataloader('training/data/final/train.h5', 'train', batch_size=8)
pair_check = defaultdict(list)

for i, (orig, cfa, o_meta, c_meta) in enumerate(loader):
    # Collect all pair_ids from this batch
    for m in o_meta + (c_meta or []):
        if m['pair_id']:
            pair_check[m['pair_id']].append(i)  # i = batch number
    if i >= 4: break  # check first 5 batches only

# Every pair_id should appear in exactly ONE batch number
split_pairs = {pid: batches for pid, batches in pair_check.items() if len(set(batches)) > 1}
if split_pairs:
    print(f'FAIL: {len(split_pairs)} pairs split across batches!')
else:
    print('PASS: All pairs in same batch')
"

Write tests/test_p4s3_dataloader.py with 8 tests:
1. Dataset builds index with pair_id, label, cwe fields
2. Dataset handles nested h5 layout (split/sample_id)
3. Dataset handles flat h5 layout (sample_id with split attr)
4. Sampler: pairs (same pair_id) are in same batch
5. Sampler: set_epoch() gives different shuffling each epoch
6. Sampler: singletons distributed correctly
7. Collate: cfa_batch is None when batch has no CFA samples
8. Collate: orig and CFA counts are aligned (equal length)
```

---

## 7. P4-S4 — Loss Functions

### 7.1 The Composite Loss

```
L_total = 1.0 × L_CE
        + 0.5 × L_CFA
        + 0.1 × L_severity

L_CE:       CrossEntropyLoss (binary: vuln vs safe)
L_CFA:      cosine margin contrastive on (vuln, cfa) pairs
L_severity: HuberLoss regression for CVSS proxy
```

### 7.2 L_CFA — The Research-Critical Loss

This is the mechanistic heart of StreamGuard's novelty. It pushes vulnerable function embeddings and their counterfactual safe embeddings to **opposite sides** of the embedding space.

```
L_CFA = mean( relu( cosine_sim(emb_v, emb_v') - (-margin) ) )
       = mean( relu( cosine_sim(emb_v, emb_v') + 0.5 ) )

Goal: cosine_sim should be NEGATIVE (opposite hemispheres)
      Penalty if cosine_sim > -0.5 (not separated enough)

WRONG formula: relu( cosine_sim - 0.5 )   ← only penalizes sim > +0.5 (too weak)
CORRECT formula: relu( cosine_sim + 0.5 ) ← penalizes sim > -0.5 (strong separation)
```

**Sign matters enormously.** The wrong sign makes L_CFA a near-useless loss that only fires when embeddings are nearly identical. The correct sign enforces true antipodal separation.

### 7.3 Full Loss Code

```python
# training/scripts/model/losses.py

import torch
import torch.nn as nn
import torch.nn.functional as F


CWE_LABEL_MAP = {
    "CWE-89": 0, "CWE-78": 1, "CWE-79": 2, "CWE-119": 3,
    "CWE-120": 4, "CWE-121": 5, "CWE-122": 6, "CWE-125": 7,
    "CWE-134": 8, "CWE-190": 9, "CWE-416": 10, "CWE-476": 11,
}


class StreamGuardLoss(nn.Module):
    """
    Composite loss for CFA-paired training.

    L_total = λ1*L_CE + λ2*L_CFA + λ3*L_severity

    L_CE:       Binary CrossEntropyLoss on all samples (vuln/safe classification)
    L_CFA:      Cosine margin contrastive loss on (vuln, cfa) pairs
                relu(cosine_sim(emb_v, emb_v') + margin)
                where margin = 0.5 → penalizes cosine_sim > -0.5
                → forces pairs to OPPOSITE hemispheres of embedding space
    L_severity: HuberLoss (delta=1.0) on CVSS proxy score [0,10]

    Research basis: VISION (AIES 2025) — paired contrastive training objective
    """

    def __init__(
        self,
        lambda_ce: float   = 1.0,
        lambda_cfa: float  = 0.5,
        lambda_sev: float  = 0.1,
        cfa_margin: float  = 0.5,
        num_cwe: int       = 12,
        label_smoothing: float = 0.1,
    ):
        super().__init__()
        self.lambda_ce  = lambda_ce
        self.lambda_cfa = lambda_cfa
        self.lambda_sev = lambda_sev
        self.margin     = cfa_margin  # margin = 0.5, not -0.5 (sign in formula matters)

        self.ce_loss  = nn.CrossEntropyLoss()
        # Label smoothing on CWE head prevents over-confidence on ambiguous samples
        self.cwe_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.sev_loss = nn.HuberLoss(delta=1.0)

    def forward(
        self,
        outputs_orig,           # model output dict for original samples
        outputs_cfa=None,       # model output dict for CFA counterparts (may be None)
        labels=None,            # (B,) int64 binary labels 0/1
        cwe_labels=None,        # (B,) int64 CWE class index 0–11 (None if unknown)
        severity_labels=None,   # (B,) float severity 0–10 (-1 = unknown)
    ):
        """
        Compute composite loss. All components are optional except L_CE.

        Returns: (total_loss: Tensor, loss_dict: dict of component values)
        """
        losses = {}
        device = outputs_orig["logits"].device
        total  = torch.tensor(0.0, device=device, requires_grad=True)

        # ── L_CE: Binary classification ──────────────────────────────
        if labels is not None:
            l_ce = self.ce_loss(outputs_orig["logits"], labels)
            losses["L_CE"] = l_ce.item()
            total = total + self.lambda_ce * l_ce

        # ── L_CWE: Multi-class CWE head (auxiliary) ──────────────────
        if cwe_labels is not None:
            valid = (cwe_labels >= 0) & (cwe_labels < 12)
            if valid.any():
                l_cwe = self.cwe_loss(
                    outputs_orig["cwe_logits"][valid],
                    cwe_labels[valid]
                )
                losses["L_CWE"] = l_cwe.item()
                total = total + 0.2 * l_cwe  # small auxiliary weight

        # ── L_CFA: Cosine margin contrastive on (v, v') pairs ────────
        # CRITICAL: formula is relu(cosine_sim + margin), NOT relu(cosine_sim - margin)
        # margin = +0.5 → penalizes when cosine_sim > -0.5
        # Goal: push vuln and cfa embeddings to OPPOSITE sides of embedding space
        if outputs_cfa is not None:
            emb_v  = outputs_orig["embedding"]  # (B_pairs, 128)
            emb_vp = outputs_cfa["embedding"]   # (B_pairs, 128)

            # Ensure equal-length sub-batches (may differ if collate is unbalanced)
            min_len = min(emb_v.size(0), emb_vp.size(0))
            if min_len > 0:
                emb_v  = emb_v[:min_len]
                emb_vp = emb_vp[:min_len]

                cosine_sim = F.cosine_similarity(emb_v, emb_vp, dim=-1)  # (B_pairs,)
                # relu(sim + 0.5) penalizes when sim > -0.5
                l_cfa = F.relu(cosine_sim + self.margin).mean()
                losses["L_CFA"] = l_cfa.item()
                total = total + self.lambda_cfa * l_cfa
                # Log diagnostic: what fraction of pairs already have sim < -0.5?
                losses["frac_separated"] = (cosine_sim < -self.margin).float().mean().item()

        # ── L_severity: CVSS proxy regression ────────────────────────
        if severity_labels is not None:
            valid_mask = severity_labels >= 0  # -1 = unknown
            if valid_mask.any():
                l_sev = self.sev_loss(
                    outputs_orig["severity_score"][valid_mask],
                    severity_labels[valid_mask].float(),
                )
                losses["L_severity"] = l_sev.item()
                total = total + self.lambda_sev * l_sev

        losses["total"] = total.item()
        return total, losses
```

### 7.4 Claude Code Prompt — P4-S4

```
▶ CLAUDE CODE PROMPT — P4-S4 — Loss Functions

Read docs/MODEL.md §2 (Loss Functions) completely.
Read docs/PRD.md §7.2 (Loss Function spec) — NOTE THE SIGN of the CFA margin.

Build training/scripts/model/losses.py.

CRITICAL: The CFA loss formula sign.
  CORRECT: relu( cosine_sim(emb_v, emb_v') + margin ) where margin=0.5
    → fires when cosine_sim > -0.5 (penalizes insufficient separation)
    → Goal: push to opposite hemispheres
  WRONG: relu( cosine_sim(emb_v, emb_v') - margin ) where margin=0.5
    → only fires when cosine_sim > +0.5 (nearly no penalty in practice)
    → Fails to enforce antipodal separation

The loss class: StreamGuardLoss(nn.Module)
  __init__ params: lambda_ce=1.0, lambda_cfa=0.5, lambda_sev=0.1,
                   cfa_margin=0.5, num_cwe=12, label_smoothing=0.1
  forward returns: (total_loss: Tensor, loss_dict: dict)

L_CE: CrossEntropyLoss on binary outputs_orig["logits"]
L_CWE: CrossEntropyLoss(label_smoothing=0.1) on cwe_logits — weight 0.2
L_CFA: F.relu(F.cosine_similarity(emb_v, emb_vp) + 0.5).mean()
       Handle unequal batch sizes (orig vs cfa may differ) — use min_len
       Also log: frac_separated = fraction of pairs with cosine_sim < -0.5
L_severity: HuberLoss(delta=1.0) — skip samples where severity_label == -1

CWE_LABEL_MAP dict: maps CWE string → int index 0-11 (12 CWEs)
  Include in losses.py for use by DataLoader collate function

Write tests/test_p4s4_losses.py with 8 tests:
1. L_CE decreases on random binary task after 10 steps
2. L_CFA sign: relu(0.8 + 0.5) = 1.3, NOT relu(0.8 - 0.5) = 0.3
3. L_CFA = 0.0 when cosine_sim = -0.9 (already well-separated)
4. L_CFA > 0.0 when cosine_sim = +0.5 (not separated)
5. frac_separated: 0.0 when no pairs are separated, 1.0 when all are
6. L_severity skips samples with severity_label = -1
7. Unequal orig/cfa batch sizes handled via min_len (no crash)
8. total_loss gradient flows through all active components
```

---

## 8. P4-S5 — Training Loop

### 8.1 Training Configuration

```python
# Full Phase 4 training config
TRAINING_CONFIG_M2 = {
    # Model
    "base_model":             "microsoft/codebert-base",
    "node_feature_dim":       824,
    "use_interproc":          False,  # True only for Config E
    "freeze_codebert_layers": 0,      # Set to 9 to save ~40% GPU memory

    # Optimizer
    "optimizer":              "AdamW",
    "weight_decay":           0.01,
    "lr_codebert":            2e-5,   # Conservative for pre-trained CodeBERT
    "lr_ggnn_fusion":         1e-4,   # Faster for randomly-init components

    # Scheduler
    "warmup_ratio":           0.1,    # 10% of steps for linear warmup
    "scheduler":              "linear_warmup_cosine",

    # Batching
    "batch_size_graphs":      8,      # GPU memory limited
    "gradient_accumulation":  4,      # Effective batch = 32
    "num_workers":            4,

    # Sequence encoding
    "max_seq_len":            512,    # CodeBERT maximum
    "tokenize_during_training": True, # Tokenize batches on-the-fly

    # CPG
    "max_cpg_nodes":          200,    # After context slicing

    # Training loop
    "epochs":                 20,
    "early_stopping_patience": 5,    # On val F1
    "checkpoint_every_n_epochs": 5,  # In addition to best-model checkpoint
    "seed":                   42,

    # Loss weights
    "lambda_ce":              1.0,
    "lambda_cfa":             0.5,
    "lambda_sev":             0.1,
    "cfa_margin":             0.5,

    # MLflow
    "mlflow_experiment":      "streamguard_m2_full",
    "ablation_config":        "E_full",  # A, B, B_prime, C, D, or E_full

    # Paths
    "train_h5":               "training/data/final/train.h5",
    "val_h5":                 "training/data/final/val.h5",
    "test_h5":                "training/data/final/test.h5",
    "checkpoint_dir":         "training/checkpoints/",
}
```

### 8.2 Tokenization During Training

The HDF5 stores graph data but NOT the raw code text (too large). CodeBERT tokenization must happen during training, on each batch. This is the most important practical detail in Phase 4.

```python
def tokenize_batch(codes: list[str], tokenizer, max_length: int, device):
    """
    Tokenize a list of C function strings for CodeBERT.
    Returns input_ids and attention_mask on device.
    """
    enc = tokenizer(
        codes,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    return enc["input_ids"].to(device), enc["attention_mask"].to(device)
```

The metadata dict from the DataLoader must include the raw code string so it can be tokenized. This means the HDF5 must store raw code as an attr OR the DataLoader must have access to the original JSONL.

**Two approaches:**

**Approach A (simpler):** Store code text as HDF5 attrs during Stage 6. Adds ~5–15 KB per sample, total ~200–500 MB. Cleanest solution.

**Approach B (memory-efficient):** Keep a code lookup dict in memory: `{sample_id: code}` loaded from the Stage 3 JSONL. Larger memory footprint (~500 MB) but no HDF5 modification needed.

**Recommended:** Approach A. Store code during Stage 6 re-run (add one line: `grp.attrs["code"] = sample.get("code", "")`). This is the cleanest because everything needed for training is in one file.

### 8.3 Full Training Loop

```python
# training/scripts/model/train.py

import os
import json
import random
import time
from pathlib import Path

import torch
import mlflow
import numpy as np
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from loguru import logger

from model import StreamGuardModel, save_checkpoint
from losses import StreamGuardLoss, CWE_LABEL_MAP
from cfa_dataloader import build_dataloader
from eval import evaluate


def set_all_seeds(seed: int):
    """Global seed setting for full reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def build_optimizer(model: StreamGuardModel, config: dict):
    """Differential learning rates: lower for pre-trained CodeBERT."""
    codebert_params = list(model.codebert.parameters())
    other_params    = [p for n, p in model.named_parameters() if "codebert" not in n]

    return AdamW(
        [
            {"params": codebert_params, "lr": config["lr_codebert"]},
            {"params": other_params,    "lr": config["lr_ggnn_fusion"]},
        ],
        weight_decay=config["weight_decay"],
    )


def extract_cwe_labels(orig_metas: list, device) -> torch.Tensor:
    """Convert CWE string list to integer tensor for loss computation."""
    labels = []
    for m in orig_metas:
        cwe_str = m.get("cwe", "")
        labels.append(CWE_LABEL_MAP.get(cwe_str, -1))  # -1 if unknown CWE
    return torch.tensor(labels, dtype=torch.long, device=device)


def extract_code_strings(metas: list) -> list:
    """Extract code strings from batch metadata for tokenization."""
    return [m.get("code", "") or "" for m in metas]


def train(config: dict):
    """
    Main Phase 4 training loop.

    Produces:
    - Best checkpoint: training/checkpoints/best_model.pt
    - Epoch checkpoints (every N epochs)
    - MLflow run with all metrics
    """
    set_all_seeds(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Model ────────────────────────────────────────────────────────
    model = StreamGuardModel(
        codebert_model=config["base_model"],
        node_feature_dim=config.get("node_feature_dim", 824),
        use_interproc=config.get("use_interproc", False),
        freeze_codebert_layers=config.get("freeze_codebert_layers", 0),
    ).to(device)

    # Optional: torch.compile for 10–20% speedup (PyTorch 2.2+)
    # model = torch.compile(model, mode="reduce-overhead")

    tokenizer = AutoTokenizer.from_pretrained(config["base_model"])
    criterion = StreamGuardLoss(
        lambda_ce=config.get("lambda_ce", 1.0),
        lambda_cfa=config.get("lambda_cfa", 0.5),
        lambda_sev=config.get("lambda_sev", 0.1),
        cfa_margin=config.get("cfa_margin", 0.5),
    )

    # ── Optimizer + Scheduler ────────────────────────────────────────
    optimizer = build_optimizer(model, config)

    train_loader = build_dataloader(
        config["train_h5"], "train",
        config["batch_size_graphs"],
        num_workers=config.get("num_workers", 4),
    )
    val_loader = build_dataloader(
        config.get("val_h5", config["train_h5"]), "val",
        config["batch_size_graphs"],
        num_workers=config.get("num_workers", 4),
        shuffle=False,
    )

    total_steps  = len(train_loader) * config["epochs"] // config.get("gradient_accumulation", 4)
    warmup_steps = int(total_steps * config.get("warmup_ratio", 0.1))
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # ── AMP: Mixed-Precision Training (saves ~40% GPU memory) ────────
    use_amp = device.type == "cuda" and config.get("use_amp", True)
    scaler  = torch.cuda.amp.GradScaler() if use_amp else None
    if use_amp:
        logger.info("Using AMP (automatic mixed precision) — fp16 forward pass")

    # ── Checkpoint Resume ────────────────────────────────────────────
    checkpoint_dir = Path(config.get("checkpoint_dir", "training/checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    start_epoch  = 0
    best_val_f1  = 0.0
    patience_ctr = 0
    resume_path  = checkpoint_dir / f"latest_{config.get('ablation_config','run')}.pt"

    if resume_path.exists() and config.get("resume", True):
        logger.info(f"Resuming from {resume_path}")
        ckpt        = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_f1 = ckpt.get("best_val_f1", 0.0)
        patience_ctr = ckpt.get("patience_ctr", 0)
        logger.info(f"Resumed at epoch {start_epoch}, best val F1: {best_val_f1:.4f}")

    # ── MLflow ───────────────────────────────────────────────────────
    mlflow.set_experiment(config.get("mlflow_experiment", "streamguard_m2_full"))

    with mlflow.start_run(run_name=config.get("ablation_config", "run")):
        mlflow.log_params({k: v for k, v in config.items() if isinstance(v, (str, int, float, bool))})

        grad_accum = config.get("gradient_accumulation", 4)
        global_step = start_epoch * len(train_loader)

        for epoch in range(start_epoch, config["epochs"]):
            # ── Training epoch ───────────────────────────────────────
            model.train()
            train_loader.batch_sampler.set_epoch(epoch)

            epoch_losses = {"L_CE": 0.0, "L_CFA": 0.0, "L_severity": 0.0, "total": 0.0}
            n_batches = 0
            t_start = time.time()

            optimizer.zero_grad()

            for step, (orig_batch, cfa_batch, orig_metas, cfa_metas) in enumerate(train_loader):
                orig_batch = orig_batch.to(device)

                # Tokenize code sequences on-the-fly
                if config.get("tokenize_during_training", True):
                    orig_codes = extract_code_strings(orig_metas)
                    input_ids, attn_mask = tokenize_batch(
                        orig_codes, tokenizer, config.get("max_seq_len", 512), device
                    )
                else:
                    input_ids, attn_mask = None, None

                # ── Forward pass ─────────────────────────────────────
                with torch.cuda.amp.autocast(enabled=use_amp):
                    out_orig = model(orig_batch, input_ids=input_ids, attention_mask=attn_mask)

                    out_cfa = None
                    if cfa_batch is not None:
                        cfa_batch = cfa_batch.to(device)
                        if config.get("tokenize_during_training", True):
                            cfa_codes = extract_code_strings(cfa_metas)
                            cfa_ids, cfa_mask = tokenize_batch(
                                cfa_codes, tokenizer, config.get("max_seq_len", 512), device
                            )
                        else:
                            cfa_ids, cfa_mask = None, None
                        out_cfa = model(cfa_batch, input_ids=cfa_ids, attention_mask=cfa_mask)

                    labels     = orig_batch.y.to(device)
                    cwe_labels = extract_cwe_labels(orig_metas, device)

                    total_loss, loss_dict = criterion(
                        out_orig, out_cfa, labels,
                        cwe_labels=cwe_labels,
                    )

                # ── Backward + gradient accumulation ─────────────────
                if use_amp:
                    scaler.scale(total_loss / grad_accum).backward()
                else:
                    (total_loss / grad_accum).backward()

                if (step + 1) % grad_accum == 0:
                    if use_amp:
                        scaler.unscale_(optimizer)

                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                    if use_amp:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()

                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                # ── Logging ──────────────────────────────────────────
                for k, v in loss_dict.items():
                    epoch_losses[k] = epoch_losses.get(k, 0.0) + v
                n_batches += 1

                if step % 50 == 0:
                    mlflow.log_metrics(
                        {f"train_{k}": v for k, v in loss_dict.items()},
                        step=global_step
                    )

            # Epoch-level train metrics
            epoch_time = time.time() - t_start
            avg_losses = {k: v / max(n_batches, 1) for k, v in epoch_losses.items()}
            logger.info(
                f"Epoch {epoch:02d} | "
                f"Loss: {avg_losses.get('total',0):.4f} | "
                f"L_CE: {avg_losses.get('L_CE',0):.4f} | "
                f"L_CFA: {avg_losses.get('L_CFA',0):.4f} | "
                f"Time: {epoch_time:.0f}s"
            )
            mlflow.log_metrics({f"epoch_train_{k}": v for k, v in avg_losses.items()}, step=epoch)

            # ── Validation ───────────────────────────────────────────
            val_metrics = evaluate(model, val_loader, device, tokenizer, config)
            mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()}, step=epoch)

            logger.info(
                f"Epoch {epoch:02d} VAL | "
                f"F1: {val_metrics['f1']:.4f} | "
                f"FPR: {val_metrics.get('fpr',0):.4f} | "
                f"FNR: {val_metrics.get('fnr',0):.4f} | "
                f"Pairwise: {val_metrics.get('pairwise_accuracy',0):.4f}"
            )

            # ── Checkpoint every N epochs ────────────────────────────
            if (epoch + 1) % config.get("checkpoint_every_n_epochs", 5) == 0:
                epoch_path = checkpoint_dir / f"epoch_{epoch:02d}_{config.get('ablation_config','run')}.pt"
                save_checkpoint(model, optimizer, epoch, val_metrics, config, str(epoch_path))

            # Save latest for resume
            latest_ckpt = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "best_val_f1": best_val_f1,
                "patience_ctr": patience_ctr,
            }
            torch.save(latest_ckpt, str(resume_path) + ".tmp")
            os.replace(str(resume_path) + ".tmp", str(resume_path))

            # ── Early stopping + best model ──────────────────────────
            if val_metrics["f1"] > best_val_f1:
                best_val_f1 = val_metrics["f1"]
                patience_ctr = 0
                best_path = checkpoint_dir / f"best_{config.get('ablation_config','run')}.pt"
                save_checkpoint(model, optimizer, epoch, val_metrics, config, str(best_path))
                logger.info(f"New best val F1: {best_val_f1:.4f} → saved to {best_path}")
                mlflow.log_artifact(str(best_path))
            else:
                patience_ctr += 1
                logger.info(f"No improvement ({patience_ctr}/{config['early_stopping_patience']})")
                if patience_ctr >= config["early_stopping_patience"]:
                    logger.info(f"Early stopping at epoch {epoch}. Best val F1: {best_val_f1:.4f}")
                    break

        # ── Final test evaluation (ONCE, after all training) ─────────
        logger.info("=== FINAL TEST EVALUATION ===")
        test_loader = build_dataloader(
            config["test_h5"], "test",
            config["batch_size_graphs"],
            num_workers=config.get("num_workers", 4),
            shuffle=False,
        )
        # Reload best model
        best_model_path = checkpoint_dir / f"best_{config.get('ablation_config','run')}.pt"
        if best_model_path.exists():
            from model import load_checkpoint
            model, _, _, _, _ = load_checkpoint(str(best_model_path), device)
            model = model.to(device)

        test_metrics = evaluate(model, test_loader, device, tokenizer, config, split="test")
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

        logger.info("TEST RESULTS:")
        for k, v in test_metrics.items():
            logger.info(f"  {k}: {v:.4f}")

        # Save test predictions
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        with open(results_dir / f"test_metrics_{config.get('ablation_config','run')}.json", "w") as f:
            json.dump(test_metrics, f, indent=2)

    return best_val_f1, test_metrics
```

### 8.4 Tokenization Helper

```python
def tokenize_batch(codes: list, tokenizer, max_length: int, device) -> tuple:
    """
    Tokenize a list of C function strings for CodeBERT.
    Handles empty strings gracefully (returns padding tokens).
    """
    # Replace empty strings with a placeholder so tokenizer doesn't crash
    clean_codes = [c if c.strip() else "void placeholder(){}" for c in codes]
    enc = tokenizer(
        clean_codes,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    return enc["input_ids"].to(device), enc["attention_mask"].to(device)
```

### 8.5 Claude Code Prompt — P4-S5

```
▶ CLAUDE CODE PROMPT — P4-S5 — Training Loop

Read docs/MODEL.md §4 (Training Loop) completely.
Read docs/PRD.md §7.3 (Training Configuration) completely.
Read docs/CLAUDE.md Rule 3 (Checkpoint Everything) and Rule 7 (Test with dry-run).

Build training/scripts/model/train.py.

CRITICAL IMPLEMENTATION POINTS:

1. TOKENIZE ON-THE-FLY:
   HDF5 stores graph data. Raw code text must come from batch metadata.
   metadata dict (from DataLoader) MUST have 'code' key populated.
   If HDF5 was built WITHOUT storing code: load JSONL lookup dict at startup.
   tokenizer(codes, max_length=512, truncation=True, padding="max_length")
   Call this INSIDE the training loop for each batch — NOT pre-computed.

2. AMP (Automatic Mixed Precision):
   use_amp = True (default for GPU). Saves ~40% VRAM.
   torch.cuda.amp.autocast() wraps the forward pass.
   scaler.scale(loss).backward() and scaler.step(optimizer).
   This allows batch_size=8 on 8GB GPUs that would OOM without AMP.

3. DIFFERENTIAL LEARNING RATES:
   CodeBERT parameters: lr=2e-5 (conservative — pre-trained weights)
   All other parameters: lr=1e-4 (faster — randomly initialized)
   Build param_groups list and pass to AdamW.
   Verify: print([(g['lr'], len(g['params'])) for g in optimizer.param_groups])
   Should show: [(2e-05, N_codebert_params), (0.0001, N_other_params)]

4. GRADIENT ACCUMULATION:
   Effective batch size = batch_size_graphs × gradient_accumulation = 8 × 4 = 32
   total_loss = total_loss / gradient_accumulation BEFORE .backward()
   Only call optimizer.step() and scheduler.step() every grad_accum steps.

5. CHECKPOINT RESUME:
   Save latest checkpoint every epoch (for crash recovery).
   Atomic save: write to .tmp, then os.replace().
   On startup: if resume=True AND latest_*.pt exists, load and continue.

6. TEST SET: evaluate ONCE after training ends, using the BEST val checkpoint.
   NEVER use test set for hyperparameter decisions or early stopping.
   Log test metrics to MLflow with "test_" prefix.

build_optimizer() function must create differential LR param groups.
set_all_seeds(seed) must call: torch.manual_seed, cuda.manual_seed_all,
  np.random.seed, random.seed, cudnn.deterministic=True, cudnn.benchmark=False

CLI:
python training/scripts/model/train.py \
    --config E_full \
    --train-h5 training/data/final/train.h5 \
    --epochs 20 \
    --batch-size 8 \
    --lr-codebert 2e-5 \
    --lr-ggnn 1e-4 \
    --dry-run  (train 1 batch × 2 epochs, then exit)

--dry-run flag: run 1 batch, 2 epochs, save checkpoint, exit. Use this ALWAYS first.

Dry run verification:
  python training/scripts/model/train.py --config B_plus_ggnn --dry-run
  Expected output:
  - "Device: cuda" (or cpu)
  - "Epoch 00 | Loss: X.XXXX | L_CE: X.XXXX | L_CFA: X.XXXX"
  - "Epoch 00 VAL | F1: X.XXXX"
  - Checkpoint saved to training/checkpoints/
  - MLflow run logged

Write tests/test_p4s5_train.py with 6 tests:
1. set_all_seeds: same seed → same random numbers
2. build_optimizer: two param groups with different LRs
3. tokenize_batch: returns (B, 512) input_ids
4. tokenize_batch: empty string handled (placeholder inserted)
5. Full dry-run: 1 epoch, loss is finite, checkpoint saved
6. Resume: load checkpoint + continue from epoch 2
```

---

## 9. P4-S6 — Evaluation & Metrics

### 9.1 All Metrics to Report

| Metric | Definition | Target | Source |
|--------|-----------|--------|--------|
| F1 | 2×(P×R)/(P+R) on binary | ≥ 0.92 | Standard |
| Precision | TP/(TP+FP) | ≥ 0.90 | Standard |
| Recall | TP/(TP+FN) | ≥ 0.90 | Standard |
| FPR | FP/(FP+TN) | ≤ 0.05 | NFR-02 |
| FNR | FN/(FN+TP) | ≤ 0.08 | NFR-03 |
| Pairwise accuracy | P(vuln=1 AND cfa=0) for each pair | ≥ 0.88 | VISION paper |
| Worst-group F1 | min(F1_per_CWE) | ≥ 0.80 | VISION paper |
| Per-CWE F1 | F1 for each of 12 CWEs | ≥ 0.88 each | FR-01 |
| CWE top-1 accuracy | % of correct CWE predictions | ≥ 0.82 | FR-03 |
| Severity MAE | mean abs error on CVSS proxy | ≤ 1.8 | FR-04 |

### 9.2 Full Evaluation Code

```python
# training/scripts/model/eval.py

import torch
import numpy as np
from collections import defaultdict
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    confusion_matrix, accuracy_score, mean_absolute_error
)
from loguru import logger


def evaluate(model, dataloader, device, tokenizer=None, config=None, split="val"):
    """
    Full evaluation loop. Computes all metrics including CFA pairwise accuracy
    and per-CWE F1 (worst-group metric).

    Returns dict of all metric values.
    """
    model.eval()
    max_seq_len = (config or {}).get("max_seq_len", 512)

    all_preds,          all_labels          = [], []
    all_cwe_preds,      all_cwe_labels      = [], []
    all_severity_preds, all_severity_labels = [], []
    all_pair_results = []   # (vuln_correct, cfa_correct) tuples
    cwe_preds_dict   = defaultdict(lambda: ([], []))  # cwe → (preds, labels)

    with torch.no_grad():
        for orig_batch, cfa_batch, orig_metas, cfa_metas in dataloader:
            orig_batch = orig_batch.to(device)

            # Tokenize if tokenizer available
            input_ids, attn_mask = None, None
            if tokenizer is not None:
                from train import tokenize_batch
                orig_codes = [m.get("code", "") for m in orig_metas]
                input_ids, attn_mask = tokenize_batch(orig_codes, tokenizer, max_seq_len, device)

            out = model(orig_batch, input_ids=input_ids, attention_mask=attn_mask)

            preds  = out["logits"].argmax(dim=-1).cpu().numpy()
            labels = orig_batch.y.cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels)

            # CWE head predictions
            cwe_preds = out["cwe_logits"].argmax(dim=-1).cpu().numpy()
            all_cwe_preds.extend(cwe_preds)

            # Severity predictions (only where gt available)
            sev_preds = out["severity_score"].cpu().numpy()
            all_severity_preds.extend(sev_preds)

            # Per-meta CWE label and severity
            for i, m in enumerate(orig_metas):
                from losses import CWE_LABEL_MAP
                cwe_int = CWE_LABEL_MAP.get(m.get("cwe", ""), -1)
                all_cwe_labels.append(cwe_int)
                all_severity_labels.append(m.get("severity_score", -1.0))

                # Per-CWE tracking for worst-group metric
                if cwe_int >= 0:
                    cwe_name = m.get("cwe", "")
                    cwe_preds_dict[cwe_name][0].append(preds[i])
                    cwe_preds_dict[cwe_name][1].append(labels[i])

            # ── Pairwise accuracy ────────────────────────────────────
            if cfa_batch is not None:
                cfa_batch = cfa_batch.to(device)

                cfa_input_ids, cfa_attn_mask = None, None
                if tokenizer is not None:
                    cfa_codes = [m.get("code", "") for m in cfa_metas]
                    cfa_input_ids, cfa_attn_mask = tokenize_batch(
                        cfa_codes, tokenizer, max_seq_len, device
                    )

                out_cfa = model(cfa_batch, input_ids=cfa_input_ids, attention_mask=cfa_attn_mask)
                cfa_preds  = out_cfa["logits"].argmax(dim=-1).cpu().numpy()
                cfa_labels = cfa_batch.y.cpu().numpy()

                # Pairwise: original must be predicted vuln=1, CFA must be safe=0
                for i, (op, ol) in enumerate(zip(preds, labels)):
                    if ol == 1 and i < len(cfa_preds):  # only for true vulnerable samples
                        vuln_correct = (op == 1)
                        cfa_correct  = (cfa_preds[i] == 0)
                        all_pair_results.append(vuln_correct and cfa_correct)

    # ── Compute all metrics ─────────────────────────────────────────
    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Standard binary metrics
    metrics = {
        "f1":        f1_score(all_labels, all_preds, zero_division=0),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall":    recall_score(all_labels, all_preds, zero_division=0),
    }

    # FPR, FNR from confusion matrix
    try:
        tn, fp, fn, tp = confusion_matrix(all_labels, all_preds, labels=[0, 1]).ravel()
        metrics["fpr"] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        metrics["fnr"] = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        metrics["accuracy"] = (tp + tn) / (tp + tn + fp + fn)
    except ValueError:
        metrics["fpr"] = metrics["fnr"] = metrics["accuracy"] = 0.0

    # CFA pairwise accuracy (core VISION metric)
    if all_pair_results:
        metrics["pairwise_accuracy"] = sum(all_pair_results) / len(all_pair_results)

    # Per-CWE F1 (worst-group metric)
    per_cwe_f1 = {}
    for cwe_name, (cwe_p, cwe_l) in cwe_preds_dict.items():
        if len(cwe_l) >= 5:  # skip if too few samples for reliable F1
            per_cwe_f1[cwe_name] = f1_score(cwe_l, cwe_p, zero_division=0)
    metrics["per_cwe_f1"]    = per_cwe_f1
    metrics["worst_group_f1"] = min(per_cwe_f1.values()) if per_cwe_f1 else 0.0

    # CWE head top-1 accuracy
    cwe_labels_arr = np.array(all_cwe_labels)
    cwe_preds_arr  = np.array(all_cwe_preds)
    valid_cwe = cwe_labels_arr >= 0
    if valid_cwe.any():
        metrics["cwe_top1_accuracy"] = accuracy_score(
            cwe_labels_arr[valid_cwe], cwe_preds_arr[valid_cwe]
        )

    # Severity MAE (only where ground truth available)
    sev_labels = np.array(all_severity_labels)
    sev_preds  = np.array(all_severity_preds)
    valid_sev  = sev_labels >= 0
    if valid_sev.sum() > 0:
        metrics["severity_mae"] = float(mean_absolute_error(
            sev_labels[valid_sev], sev_preds[valid_sev]
        ))

    # Log per-CWE F1 table
    if per_cwe_f1:
        logger.info(f"\nPer-CWE F1 ({split}):")
        for cwe, f1_val in sorted(per_cwe_f1.items()):
            target = 0.88  # from PRD.md §5.3
            status = "✓" if f1_val >= target else "✗"
            logger.info(f"  {cwe:<12}: {f1_val:.4f} {status}")

    return metrics
```

### 9.3 Claude Code Prompt — P4-S6

```
▶ CLAUDE CODE PROMPT — P4-S6 — Evaluation & Metrics

Read docs/MODEL.md §5 (Evaluation Metrics) completely.
Read docs/PRD.md §9.3 (Novel Metrics) for pairwise_contrast_accuracy.
Read docs/NOVELTY.md §N1 and §N4 for what metrics to prioritize.

Build training/scripts/model/eval.py.

The evaluate() function must compute ALL of these:

Standard:
  f1, precision, recall, fpr (FP rate), fnr (FN rate), accuracy

Novel (VISION paper metrics, required for paper):
  pairwise_accuracy: P(predict vuln=1 for orig AND predict safe=0 for cfa_counterpart)
    Only count pairs where the ORIGINAL is truly vulnerable (label=1)
  worst_group_f1: minimum F1 across all CWE subgroups
    Only include CWEs with >= 5 test samples (too few = unreliable F1)
  per_cwe_f1: dict {CWE-89: 0.xx, CWE-120: 0.xx, ...}

CWE head:
  cwe_top1_accuracy: accuracy of CWE classification head

Severity:
  severity_mae: mean absolute error on CVSS proxy (skip samples with severity_label=-1)

Per-CWE F1 log format:
  Print table after evaluation with ✓/✗ vs 0.88 target from PRD.md §5.3

IMPORTANT: evaluate() must accept tokenizer=None for graph-only ablation (Config A uses
  sequence encoding but some ablations may not)

Write tests/test_p4s6_eval.py with 8 tests:
1. F1 computed correctly on synthetic preds/labels
2. FPR/FNR computed correctly
3. pairwise_accuracy: 1.0 when all pairs correctly classified
4. pairwise_accuracy: 0.0 when all pairs wrong
5. worst_group_f1: returns min across CWEs
6. per_cwe_f1 skips CWEs with < 5 samples
7. severity_mae skips samples with label=-1
8. cwe_top1_accuracy computed correctly
```

---

## 10. P4-S7 — Ablation Study Runner

### 10.1 The 5 Ablation Configurations

These are the configurations that produce the core paper table. Every config must train on the **exact same** `train.h5 / val.h5 / test.h5` with `seed=42`.

| Config | Description | Novel Component Tested | Expected F1 |
|--------|-------------|----------------------|-------------|
| **A: Baseline** | CodeBERT sequence only, no graph, BCE only | None (LineVul-style) | ~0.79 |
| **B: +GGNN** | CodeBERT + type-blind GGNN, 3-CPG, BCE only | Graph encoder | ~0.83 |
| **B': +Type-Aware** | CodeBERT + per-edge-type GGNN, 3-CPG, BCE only | N3 type awareness | ~0.85 |
| **C: +CFA** | Config B' + L_CFA contrastive training | **N1+N4: CFA works** | ~0.89 |
| **D: +TPG** | Config C + Taint Propagation Graph (4-CPG) | N3: TPG component | ~0.91 |
| **E: Full** | Config D + inter-procedural callee context | N5: inter-proc | ~0.93 |

### 10.2 How Configs Differ in Model Setup

```python
# Config A: No graph at all — CodeBERT sequence only
# model gets empty graph data, input_ids provided
# In forward(): encode_graph skipped, bert_cls used alone
# Special model subclass or config flag needed

# Config B: CodeBERT + type-BLIND GGNN (single GatedGraphConv — all edges combined)
# Requires a TypeBlindStreamGuardModel variant

# Config B': CodeBERT + type-AWARE GGNN (4 separate GatedGraphConv)
# This is the standard StreamGuardModel (no change)

# Config C: Config B' + L_CFA (just enable CFA loss — same model, different training)
# use_cfa=True in training config

# Config D: Config C + TPG edges included
# train.h5 already has TPG edges — just enable all 4 types in model

# Config E: Config D + inter-proc callee summaries
# use_interproc=True + CalleeSummarizer active
```

### 10.3 Ablation Runner Code

```python
# training/scripts/model/run_ablations.py

import json
from pathlib import Path
from loguru import logger
import mlflow

from train import train


# ── Config definitions ───────────────────────────────────────────────

BASE_CONFIG = {
    "base_model":              "microsoft/codebert-base",
    "node_feature_dim":        824,
    "batch_size_graphs":       8,
    "gradient_accumulation":   4,
    "max_seq_len":             512,
    "epochs":                  20,
    "early_stopping_patience": 5,
    "checkpoint_every_n_epochs": 5,
    "seed":                    42,
    "weight_decay":            0.01,
    "lr_codebert":             2e-5,
    "lr_ggnn_fusion":          1e-4,
    "warmup_ratio":            0.1,
    "num_workers":             4,
    "use_amp":                 True,
    "resume":                  True,
    "train_h5":                "training/data/final/train.h5",
    "val_h5":                  "training/data/final/val.h5",
    "test_h5":                 "training/data/final/test.h5",
    "checkpoint_dir":          "training/checkpoints/",
    "mlflow_experiment":       "streamguard_m2_ablation",
    # CFA loss weights
    "lambda_ce":               1.0,
    "lambda_cfa":              0.5,
    "lambda_sev":              0.1,
    "cfa_margin":              0.5,
    "tokenize_during_training": True,
}

ABLATION_CONFIGS = {

    # Config A: CodeBERT sequence only — no graph
    # This ablation tests whether the graph component adds value at all
    "A_baseline": {
        **BASE_CONFIG,
        "ablation_config":     "A_baseline",
        "use_graph":           False,     # graph encoding disabled in forward()
        "lambda_cfa":          0.0,       # no CFA
        "lambda_sev":          0.0,
        "use_interproc":       False,
    },

    # Config B: CodeBERT + type-BLIND GGNN, 3-CPG, no CFA
    # Reproduces Vul-LMGNN baseline
    "B_plus_ggnn": {
        **BASE_CONFIG,
        "ablation_config":     "B_plus_ggnn",
        "use_graph":           True,
        "type_aware_edges":    False,     # single GatedGraphConv for all edge types
        "cpg_edge_types":      [0, 1, 2], # AST, CFG, DFG only (no TPG)
        "lambda_cfa":          0.0,       # no CFA
        "lambda_sev":          0.0,
        "use_interproc":       False,
    },

    # Config B': CodeBERT + type-AWARE GGNN, 3-CPG, no CFA
    # Tests per-edge-type GNN architecture value
    "B_prime_type_aware": {
        **BASE_CONFIG,
        "ablation_config":     "B_prime_type_aware",
        "use_graph":           True,
        "type_aware_edges":    True,      # 4 separate GatedGraphConv
        "cpg_edge_types":      [0, 1, 2], # AST, CFG, DFG only (no TPG)
        "lambda_cfa":          0.0,       # no CFA
        "lambda_sev":          0.0,
        "use_interproc":       False,
    },

    # Config C: B' + CFA contrastive training
    # PRIMARY NOVELTY TEST: does CFA improve over baseline?
    "C_plus_cfa": {
        **BASE_CONFIG,
        "ablation_config":     "C_plus_cfa",
        "use_graph":           True,
        "type_aware_edges":    True,
        "cpg_edge_types":      [0, 1, 2], # Still 3-CPG
        "lambda_cfa":          0.5,       # CFA ENABLED
        "lambda_sev":          0.1,
        "use_interproc":       False,
    },

    # Config D: C + TPG (4-CPG)
    # N3 NOVELTY TEST: does TPG improve over 3-CPG+CFA?
    "D_plus_tpg": {
        **BASE_CONFIG,
        "ablation_config":     "D_plus_tpg",
        "use_graph":           True,
        "type_aware_edges":    True,
        "cpg_edge_types":      [0, 1, 2, 3],  # AST, CFG, DFG, TPG — all 4
        "lambda_cfa":          0.5,
        "lambda_sev":          0.1,
        "use_interproc":       False,
    },

    # Config E: Full StreamGuard (D + inter-proc)
    # N5 NOVELTY TEST: does callee context add value?
    "E_full": {
        **BASE_CONFIG,
        "ablation_config":     "E_full",
        "use_graph":           True,
        "type_aware_edges":    True,
        "cpg_edge_types":      [0, 1, 2, 3],
        "lambda_cfa":          0.5,
        "lambda_sev":          0.1,
        "use_interproc":       True,      # INTER-PROC ENABLED
    },
}


def run_ablations(
    configs_to_run: list = None,
    dry_run: bool = False,
):
    """
    Run all 5 ablation configs in sequence.
    Each config trains independently with same seed and data.
    Results saved to results/ablation_table.json.

    MANDATORY: same train.h5/val.h5/test.h5 for ALL configs.
    MANDATORY: seed=42 for ALL configs.
    """
    if configs_to_run is None:
        configs_to_run = list(ABLATION_CONFIGS.keys())

    results = {}
    mlflow.set_experiment(BASE_CONFIG["mlflow_experiment"])

    for config_name in configs_to_run:
        if config_name not in ABLATION_CONFIGS:
            logger.warning(f"Unknown config: {config_name}. Skipping.")
            continue

        config = dict(ABLATION_CONFIGS[config_name])

        if dry_run:
            config["epochs"] = 2
            config["early_stopping_patience"] = 2
            logger.info(f"DRY RUN mode: max 2 epochs for {config_name}")

        logger.info(f"\n{'='*60}")
        logger.info(f"RUNNING ABLATION: {config_name}")
        logger.info(f"{'='*60}")

        try:
            best_val_f1, test_metrics = train(config)
            results[config_name] = {
                "best_val_f1": best_val_f1,
                **{f"test_{k}": v for k, v in test_metrics.items()
                   if isinstance(v, (int, float))}  # skip dicts like per_cwe_f1
            }
            results[config_name]["per_cwe_f1"] = test_metrics.get("per_cwe_f1", {})
            logger.info(f"Config {config_name} complete. Test F1: {test_metrics.get('f1', 0):.4f}")
        except Exception as e:
            logger.error(f"Config {config_name} FAILED: {e}")
            results[config_name] = {"error": str(e)}

    # ── Save ablation table ──────────────────────────────────────────
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    table_path = results_dir / "ablation_table.json"

    with open(table_path, "w") as f:
        json.dump(results, f, indent=2)

    # ── Print summary table ──────────────────────────────────────────
    print_ablation_table(results)
    logger.info(f"Results saved to {table_path}")

    return results


def print_ablation_table(results: dict):
    """Print ablation results as a formatted table (paper Table 2)."""
    print("\n" + "="*80)
    print("ABLATION STUDY RESULTS (Paper Table 2)")
    print("="*80)
    print(f"{'Config':<24} {'F1':>6} {'FPR':>6} {'FNR':>6} {'Pairwise':>10} {'Worst-CWE':>10}")
    print("-"*80)

    config_order = ["A_baseline", "B_plus_ggnn", "B_prime_type_aware",
                    "C_plus_cfa", "D_plus_tpg", "E_full"]

    for config_name in config_order:
        if config_name not in results:
            continue
        r = results[config_name]
        if "error" in r:
            print(f"{config_name:<24} ERROR: {r['error']}")
            continue

        f1    = r.get("test_f1",               0.0)
        fpr   = r.get("test_fpr",              0.0)
        fnr   = r.get("test_fnr",              0.0)
        pair  = r.get("test_pairwise_accuracy", 0.0)
        worst = r.get("test_worst_group_f1",   0.0)

        print(f"{config_name:<24} {f1:>6.4f} {fpr:>6.4f} {fnr:>6.4f} {pair:>10.4f} {worst:>10.4f}")

    print("="*80)

    # Check CFA works: Config C should beat Config B
    if "C_plus_cfa" in results and "B_prime_type_aware" in results:
        c_f1 = results["C_plus_cfa"].get("test_f1", 0)
        b_f1 = results["B_prime_type_aware"].get("test_f1", 0)
        if c_f1 > b_f1:
            print(f"✓ CFA PROOF POSITIVE: Config C ({c_f1:.4f}) > Config B' ({b_f1:.4f})")
        else:
            print(f"✗ CFA PROOF NEGATIVE: Config C ({c_f1:.4f}) <= Config B' ({b_f1:.4f})")

    # Check TPG works: Config D should beat Config C
    if "D_plus_tpg" in results and "C_plus_cfa" in results:
        d_f1 = results["D_plus_tpg"].get("test_f1", 0)
        c_f1 = results["C_plus_cfa"].get("test_f1", 0)
        if d_f1 > c_f1:
            print(f"✓ TPG PROOF POSITIVE: Config D ({d_f1:.4f}) > Config C ({c_f1:.4f})")
        else:
            print(f"✗ TPG ADDS LITTLE: Config D ({d_f1:.4f}) ≤ Config C ({c_f1:.4f})")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", default=None,
                        help="Which configs to run (default: all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Train 2 epochs per config for smoke test")
    args = parser.parse_args()

    run_ablations(args.configs, dry_run=args.dry_run)
```

### 10.4 Type-Blind GGNN Variant (Config B)

Config B requires a type-blind GGNN (single GatedGraphConv, ignores edge types). Add this as a flag in `StreamGuardModel`:

```python
# In model.py — add type_aware_edges parameter

def __init__(
    self,
    codebert_model: str = "microsoft/codebert-base",
    node_feature_dim: int = 824,
    type_aware_edges: bool = True,   # False = Config B type-blind
    cpg_edge_types: list = None,     # None = all 4; [0,1,2] = 3-CPG without TPG
    use_interproc: bool = False,
    ...
):
    self.type_aware_edges = type_aware_edges
    self.cpg_edge_types   = cpg_edge_types or [0, 1, 2, 3]

    if type_aware_edges:
        # 12 convolutions (4 types × 3 layers)
        self.edge_type_convs = nn.ModuleList([
            nn.ModuleList([GatedGraphConv(...) for _ in range(4)])
            for _ in range(3)
        ])
    else:
        # Single type-blind GatedGraphConv per layer (Config B)
        self.single_conv = nn.ModuleList([
            GatedGraphConv(out_channels=self.GGNN_HIDDEN, num_layers=1)
            for _ in range(self.GGNN_LAYERS)
        ])
```

### 10.5 Claude Code Prompt — P4-S7

```
▶ CLAUDE CODE PROMPT — P4-S7 — Ablation Study Runner

Read docs/EXPERIMENTS.md completely.
Read docs/PRD.md §9 (Evaluation Design) completely.
Read the existing model.py and train.py from P4-S1 and P4-S5.

Build training/scripts/model/run_ablations.py.

The 5 ablation configs are (see PRD.md §9.1):
  A: CodeBERT sequence only, no graph, no CFA
  B: CodeBERT + type-BLIND GGNN, 3-CPG, no CFA
  B': CodeBERT + type-AWARE GGNN, 3-CPG, no CFA
  C: B' + CFA (L_CFA enabled, lambda_cfa=0.5)
  D: C + TPG (4-CPG: include edge_type=3)
  E: D + inter-proc callee context (use_interproc=True)

MANDATORY RULES (breaking any invalidates the paper):
  1. Same seed=42 for ALL configs (already in BASE_CONFIG)
  2. Same train.h5/val.h5/test.h5 for ALL configs (don't re-split)
  3. Test.h5 evaluated ONCE per config AFTER training (never for hyperparameter tuning)
  4. All runs logged to same MLflow experiment "streamguard_m2_ablation"

For Config A (no graph):
  - model.encode_graph() skipped — bert_cls used directly in fusion
  - Add use_graph=False flag to StreamGuardModel.forward()
  - When use_graph=False: fused = cat(bert_cls, bert_cls_proj, bert_cls_proj2) 
    (use bert_cls twice to maintain 1280-d shape, or add a separate A-config MLP)

For Config B (type-blind):
  - All edges combined, single GatedGraphConv per layer
  - Add type_aware_edges=False flag to StreamGuardModel
  - When False: ignore edge_attr, treat all edges as same type

For Configs C and D:
  - Config C: cpg_edge_types=[0,1,2] — TPG edges present in train.h5 but IGNORED
    Set edge_attr=0 for all TPG edges when cpg_edge_types=[0,1,2]
  - Config D: cpg_edge_types=[0,1,2,3] — all 4 types processed

For Config E:
  - use_interproc=True
  - CalleeSummarizer initialized and passed to DataLoader collate
  - callee_embeddings passed to model.forward()
  - If callee source not available in metadata: zeros (don't crash)

print_ablation_table() output format (paper Table 2):
  Config | F1 | FPR | FNR | Pairwise | Worst-CWE
  A      | ...
  B      | ... (Vul-LMGNN baseline)
  B'     | ... (type-aware edge baseline)
  C      | ... (+ CFA — core N1 result)
  D      | ... (+ TPG — core N3 result)
  E      | ... (Full StreamGuard)

After printing table:
  Check: Config C F1 > Config B' F1 → "CFA PROOF POSITIVE" or "NEGATIVE"
  Check: Config D F1 > Config C F1 → "TPG PROOF POSITIVE" or negative

Write tests/test_p4s7_ablations.py with 6 tests:
1. BASE_CONFIG has seed=42
2. All configs share same train_h5/val_h5/test_h5 paths
3. Config B has lambda_cfa=0.0 (no contrastive loss)
4. Config C has lambda_cfa=0.5 (CFA enabled)
5. Dry-run: run A and B configs for 2 epochs each, no crash
6. Results saved to results/ablation_table.json after run
```

---

## 11. Production Risks & Mitigations

### 11.1 Critical Risks (Project-Killing)

| Risk | Description | Probability | Recovery | Mitigation |
|------|-------------|------------|----------|------------|
| **PR-01** | GPU OOM mid-training epoch | HIGH | Re-run (1hr) | Enable AMP; use freeze_codebert_layers=9; reduce batch_size to 4+accum=8 |
| **PR-02** | BatchNorm1d used instead of GroupNorm → wrong F1 at serving | MEDIUM | Re-train + re-serve | Always use GroupNorm; add assertion in tests |
| **PR-03** | Wrong L_CFA sign: relu(sim − 0.5) instead of relu(sim + 0.5) | MEDIUM | Re-train (24hrs) | Numerical test: verify L_CFA decreases when pairs are pushed apart |
| **PR-04** | CodeBERT tokenization skipped → model trains without sequence signal | LOW | Re-train | Always verify input_ids is not None in first batch |
| **PR-05** | pair_id mismatch in HDF5 → L_CFA trains on random pairs | LOW | Re-run Stage 6+7 | Verify pair_id attrs in HDF5 before training (P3 Gate G-08) |
| **PR-06** | Test set used for hyperparameter search → inflated F1 | MEDIUM | Must re-run | Only evaluate test set ONCE at the very end |
| **PR-07** | Ablation configs use different datasets → results incomparable | LOW | Re-run ablations | Assert same HDF5 paths across all configs at run start |

### 11.2 High Risks (Week of Rework)

| Risk | Description | Probability | Mitigation |
|------|-------------|------------|------------|
| **HR-01** | NaN loss during training | MEDIUM | Gradient clipping (max_norm=1.0); check for NaN features in HDF5 before training |
| **HR-02** | Catastrophic forgetting in CodeBERT | MEDIUM | Differential LR (2e-5 for CodeBERT); optionally freeze layers 0-8 for first 5 epochs |
| **HR-03** | CFA pairs split across batches (sampler bug) | MEDIUM | Run CFAAwareBatchSampler verification test before full run |
| **HR-04** | MLflow not logging → can't reproduce paper results | LOW | Verify MLflow server running before train; add CSV fallback logging |
| **HR-05** | Checkpoint corruption on kill -9 | MEDIUM | Atomic checkpoint (write .tmp → os.replace); always save latest each epoch |
| **HR-06** | `torch.compile` incompatible with scatter ops | LOW | Disable torch.compile if training crashes in first batch |
| **HR-07** | Epoch takes 6+ hours → budget exceeded | MEDIUM | Start with smaller dataset (SARD only) to verify timing; enable AMP |

### 11.3 Medium Risks (Days of Debugging)

| Risk | Mitigation |
|------|------------|
| CWE labels missing in HDF5 | Fall back to CWE='UNKNOWN', skip CWE loss; add -1 handling in CWE_LABEL_MAP |
| Code strings not in HDF5 metadata | Fall back to code_lookup dict loaded from JSONL |
| Wrong `frac_separated` (diagnostic) | This is informational only — doesn't affect loss computation |
| Pairwise accuracy stuck at 0 | Verify cfa_batch is not None in eval loop; print pair counts |
| All val_F1 < 0.5 | Check label balance; verify model not predicting all-safe |

### 11.4 Detailed Mitigation for GPU OOM (Most Common Issue)

```
GPU OOM typically happens when processing a very large graph in the batch
(graphs up to 200 nodes × 824-d features × batch_size=8).

Step 1: Enable AMP
  use_amp = True  → saves ~40% VRAM, allows larger effective batch

Step 2: Reduce batch_size + increase gradient accumulation
  batch_size=4, gradient_accumulation=8  → same effective batch (32), half VRAM

Step 3: Freeze CodeBERT lower layers
  freeze_codebert_layers=9  → reduces trainable parameters by ~70%
  Still fine-tunes top 3 layers (layers 9-11), which is typically sufficient

Step 4: Reduce max_cpg_nodes
  max_cpg_nodes=150  → smaller graphs, less memory per batch
  Stage 4 context slicing parameter — requires re-running Stage 4/5/6

Step 5: Gradient checkpointing in CodeBERT
  model.codebert.gradient_checkpointing_enable()
  Trades compute for memory — ~25% slower but ~30% less VRAM

Step 6: As last resort — train on CPU
  Not recommended for full training, but can debug the training loop
```

### 11.5 Diagnostic Checklist Before Full Training Run

Run these checks BEFORE starting the full 20-epoch training:

```bash
# 1. Verify GPU memory is sufficient
python3 -c "
import torch
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f'GPU: {props.name}')
    print(f'VRAM: {props.total_memory/1e9:.1f} GB')
    print(f'Min needed: 8GB (with AMP) or 16GB (without AMP)')
else:
    print('No CUDA GPU — training will be VERY slow on CPU')
"

# 2. Verify HDF5 structure is correct
python3 -c "
import h5py
with h5py.File('training/data/final/train.h5', 'r') as f:
    samples = list(f.keys())[:5]
    print(f'Total train samples: {len(list(f.keys()))}')
    for sid in samples:
        grp = f[sid]
        pair_id = grp.attrs.get('pair_id', '')
        cwe = grp.attrs.get('cwe', '')
        print(f'  {sid[:8]}: x={grp[\"x\"].shape}, pair_id={pair_id[:8] if pair_id else \"none\"}, cwe={cwe}')
"

# 3. Verify CFA pairs present in HDF5
python3 -c "
import h5py
with h5py.File('training/data/final/train.h5', 'r') as f:
    pairs = {}
    for sid in f.keys():
        pid = f[sid].attrs.get('pair_id','')
        if pid:
            pairs.setdefault(pid, []).append(int(f[sid]['y'][0]))
    valid_pairs = sum(1 for v in pairs.values() if 0 in v and 1 in v)
    print(f'Valid CFA pairs: {valid_pairs}')
    assert valid_pairs > 0, 'NO CFA PAIRS IN HDF5!'
"

# 4. Verify DataLoader keeps pairs together (CRITICAL)
python3 -c "
from cfa_dataloader import build_dataloader
from collections import defaultdict
loader = build_dataloader('training/data/final/train.h5', 'train', 8)
pair_batches = defaultdict(set)
for i, (orig, cfa, o_meta, c_meta) in enumerate(loader):
    for m in o_meta + (c_meta or []):
        if m['pair_id']:
            pair_batches[m['pair_id']].add(i)
    if i >= 9: break
split = {p:b for p,b in pair_batches.items() if len(b)>1}
print(f'Split pairs: {len(split)} (must be 0)')
assert len(split) == 0, 'PAIRS ARE SPLIT ACROSS BATCHES!'
print('DataLoader: PASS')
"

# 5. Dry-run: verify training loop runs without error
python3 training/scripts/model/train.py --config C_plus_cfa --dry-run
```

---

## 12. GPU Infrastructure & Performance

### 12.1 Minimum GPU Requirements

| Config | Min GPU | Recommended | Notes |
|--------|---------|-------------|-------|
| Configs A-D | 8 GB VRAM (with AMP) | 16 GB | A100/V100 on Colab Pro |
| Config E (inter-proc) | 12 GB VRAM (with AMP) | 24 GB | Extra memory for callee embeds |
| Ablation (all 5) | 8 GB (sequential) | 16 GB | Run one at a time |

### 12.2 Expected Training Times

| Config | A100 (40 GB) | V100 (16 GB) | T4 (16 GB, Colab) |
|--------|-------------|--------------|-------------------|
| Per epoch (train) | ~15 min | ~25 min | ~45 min |
| Per epoch (val) | ~3 min | ~5 min | ~8 min |
| Full 20 epochs | ~6 hours | ~10 hours | ~18 hours |
| All 5 ablation configs | ~30 hours | ~50 hours | ~90 hours |

**Colab strategy:** Run configs C, D, E (the most important) first. A and B are faster (less memory, potentially smaller models).

### 12.3 Memory Budget Breakdown (batch_size=8, AMP enabled)

```
Model parameters:
  CodeBERT (125M params):        ~250 MB (fp16)
  GGNN (12 convolutions):        ~48 MB
  Cross-attn + MLP + heads:      ~8 MB
  Total model:                   ~306 MB

Per-batch runtime:
  Input graphs (8 × 200 nodes × 824-d):  ~42 MB (fp32) → ~21 MB (fp16)
  CodeBERT activations (8 × 512 tokens): ~200 MB (fp16)
  GGNN activations (N_total × 256-d):    ~32 MB
  Optimizer states (AdamW × 2):          ~612 MB (fp32)

Total peak:
  ~1.2 GB forward + ~1.2 GB backward + optimizer = ~2.5 GB active
  Model parameters (fp16 in AMP) = ~300 MB
  GRAND TOTAL: ~3 GB working memory

On 8 GB GPU: comfortable with AMP
On 16 GB GPU: can use batch_size=16 without AMP
```

---

## 13. Go / No-Go Gates

Every gate must be GREEN before starting the next story.

| Gate | Check | Story | Status |
|------|-------|-------|--------|
| **G-01** | `pytest tests/test_p4s1_model.py` — all pass | P4-S1 | ☐ |
| **G-02** | GroupNorm confirmed: `type(model.ggnn_norm[0]) == GroupNorm` | P4-S1 | ☐ |
| **G-03** | Edge type isolation test: TPG ≠ AST embeddings | P4-S1 | ☐ |
| **G-04** | `pytest tests/test_p4s3_dataloader.py` — all pass | P4-S3 | ☐ |
| **G-05** | CFA pairs in same batch: split_pairs count == 0 | P4-S3 | ☐ |
| **G-06** | L_CFA sign verified: `relu(0.8 + 0.5) = 1.3` | P4-S4 | ☐ |
| **G-07** | `pytest tests/test_p4s4_losses.py` — all pass | P4-S4 | ☐ |
| **G-08** | Dry-run completes: 1 epoch, loss finite, MLflow logged | P4-S5 | ☐ |
| **G-09** | Differential LR verified: CodeBERT lr=2e-5, GGNN lr=1e-4 | P4-S5 | ☐ |
| **G-10** | Pairwise accuracy appears in val metrics (not always 0) | P4-S6 | ☐ |
| **G-11** | Per-CWE F1 table printed after evaluation | P4-S6 | ☐ |
| **G-12** | Dry-run ablations: all 5 configs train 2 epochs without error | P4-S7 | ☐ |
| **G-13** | Config C F1 > Config B F1 (CFA works) → ablation table | P4-S7 | ☐ |
| **G-14** | test.h5 evaluated ONCE (check MLflow: only one test_f1 per run) | P4-S5 | ☐ |

### Critical Pre-Training Checks

Before starting ANY full training run (not just dry run):

```bash
# Run ALL pre-training checks in sequence
python3 -c "
# Check 1: Phase 3 audit
import subprocess
r = subprocess.run(['python', 'training/scripts/preprocessing/pre_training_audit.py', '--m2'])
assert r.returncode == 0, 'PRE-TRAINING AUDIT FAILED'

# Check 2: GPU available
import torch
assert torch.cuda.is_available(), 'No GPU — will take 90+ hours'
print(f'GPU: {torch.cuda.get_device_name(0)}')

# Check 3: HDF5 pair_id present
import h5py
with h5py.File('training/data/final/train.h5','r') as f:
    sample = list(f.keys())[0]
    assert 'pair_id' in f[sample].attrs, 'pair_id missing from HDF5!'

# Check 4: MLflow
import mlflow
mlflow.set_experiment('streamguard_preflight')
with mlflow.start_run():
    mlflow.log_param('test', 'ok')
print('ALL PRE-TRAINING CHECKS PASSED')
"
```

---

## 14. Day-by-Day Execution Timeline

| Day | Story | Task | Gate |
|-----|-------|------|------|
| **Day 1** | P4-S1 | Build full 4-CPG model with GroupNorm | G-01, G-02, G-03 |
| **Day 1** | P4-S2 | Build callee summarizer (Config E prep) | P4-S2 tests pass |
| **Day 2** | P4-S3 | Build CFA-aware DataLoader | G-04, G-05 |
| **Day 2** | P4-S4 | Build loss functions, verify L_CFA sign | G-06, G-07 |
| **Day 3** | P4-S5 | Build training loop, run dry-run | G-08, G-09 |
| **Day 3** | P4-S6 | Build evaluation, verify pairwise accuracy | G-10, G-11 |
| **Day 4** | P4-S7 | Build ablation runner, dry-run all 5 configs | G-12 |
| **Day 4 (evening)** | — | **Start Config C (CFA proof) full training run** | Background |
| **Day 5** | — | Config C finishes (~10 hrs); start Config D | G-13 |
| **Day 5** | — | While D trains: run Configs A and B | Sequential |
| **Day 6** | — | Config D + B' + E complete | All configs done |
| **Day 6** | — | Print ablation table; evaluate on test.h5 | Paper Table 2 |

### Recommended Training Order

Run in this priority order to get the most important results first:

1. **Config C (CFA proof)** — Start first. This is the core paper claim.
2. **Config D (+TPG)** — Proves Novel N3.
3. **Config B' (type-aware baseline)** — Control for Config C.
4. **Config E (full)** — Proves Novel N5. Needs callee summarizer.
5. **Config B (type-blind)** — Reproduces Vul-LMGNN baseline.
6. **Config A (no graph)** — Fastest (smallest model), run last.

---

## Appendix A: Key File Summary

| File | Purpose | Priority |
|------|---------|---------|
| `training/scripts/model/model.py` | StreamGuardModel — full 4-CPG architecture | P0 |
| `training/scripts/model/losses.py` | L_CE + L_CFA + L_severity composite loss | P0 |
| `training/scripts/model/cfa_dataloader.py` | CFA-aware batch sampler + collate | P0 |
| `training/scripts/model/train.py` | Full training loop with AMP + MLflow | P0 |
| `training/scripts/model/eval.py` | All metrics including pairwise + per-CWE | P0 |
| `training/scripts/model/run_ablations.py` | 5-config ablation runner | P0 |
| `training/scripts/model/callee_summarizer.py` | Inter-proc context (Config E) | P1 |
| `training/scripts/model/callee_cache.py` | Redis-backed callee embed cache | P1 |
| `tests/test_p4s1_model.py` | Model architecture tests | P0 |
| `tests/test_p4s3_dataloader.py` | DataLoader pair integrity tests | P0 |
| `tests/test_p4s4_losses.py` | L_CFA sign and correctness tests | P0 |
| `tests/test_p4s5_train.py` | Training loop dry-run tests | P0 |
| `tests/test_p4s6_eval.py` | Evaluation metrics tests | P0 |
| `tests/test_p4s7_ablations.py` | Ablation runner tests | P0 |
| `results/ablation_table.json` | Final paper Table 2 results | OUTPUT |
| `results/per_cwe_f1.json` | Final paper Table 3 (per-CWE) | OUTPUT |
| `training/checkpoints/best_C_plus_cfa.pt` | Best Config C model | OUTPUT |

---

## Appendix B: Common Error Messages and Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `CUDA error: device-side assert triggered` | edge_attr has values ≥ 4 (CDG edges leaked) | Re-run Stage 4 with CDG filter |
| `RuntimeError: Expected all tensors to be on the same device` | orig_batch on GPU, cfa_batch not moved | Add `cfa_batch = cfa_batch.to(device)` |
| `ValueError: Expected input batch_size to match target batch_size` | orig and cfa batch sizes differ | Use `min_len = min(...)` in loss |
| `torch.nn.modules.module.ModuleAttributeError: 'DataParallel' object has no attribute 'forward'` | torch.compile or DataParallel wrapping | Access `model.module.forward` or disable DataParallel |
| `NaN in loss after epoch 1` | Learning rate too high for CodeBERT | Lower `lr_codebert` to 1e-5; add gradient clipping |
| `RuntimeError: Expected more than 1 value per channel when training, got input size torch.Size([1, 256])` | BatchNorm1d at batch_size=1 during serving test | Replace with GroupNorm (P4-S1 critical fix) |
| `AssertionError: pairs are split across batches` | CFAAwareBatchSampler bug | Check that `group → individual shuffle` path is not taken |
| `IndexError: index X is out of bounds for dimension 0 with size Y` | edge_index not remapped to 0..N-1 | Re-run Stage 6 with node_id_to_idx remapping |

---

*StreamGuard Phase 4 — Full Model + Training | v1.0 | March 2026*
