# training/scripts/model/model.py
#
# StreamGuard Phase 4: CodeBERT + 3-layer Type-Aware GGNN + Node-level Cross-Attention
#
# Architecture (see docs/New Docs/StreamGuard_Phase4_Full_Model_Training.md §4):
#   - CodeBERT encoder:   microsoft/codebert-base → [CLS] 768-d
#   - Node projector:     Linear(824, 256)   — 824-d from Stage 5
#   - Type-Aware GGNN:    4 GatedGraphConv per layer × 3 layers = 12 convolutions
#   - Cross-Attn Fusion:  Q=BERT (B,768→256), K/V=per-node (N,256) → scatter softmax
#   - Fused repr:         concat(BERT_768, Attn_256, GGNN_mean_256) = 1280-d
#   - Shared MLP:         1280 → LayerNorm → GELU → Dropout(0.3) → 512 → GELU → 128
#   - Output heads:       Binary(2) + CWE(12) + Severity(1)
#
# Phase 4 changes from Phase 1 proof:
#   - node_feature_dim = 824 (was ~256)
#   - GroupNorm(32, 256) replaces BatchNorm1d (works at batch_size=1 for serving)
#   - TPG as edge type 3 (4 types total)
#   - Inter-procedural callee injection (Config E)
#   - freeze_codebert_layers param
#   - save_checkpoint() includes config dict for serving reconstruction

import os
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GatedGraphConv, global_mean_pool, global_max_pool
from torch_geometric.utils import scatter
from transformers import AutoModel, AutoTokenizer


class StreamGuardModel(nn.Module):
    """
    StreamGuard: CodeBERT + 3-layer Type-Aware GGNN + Cross-Attention Fusion

    Novel contributions vs Devign/VISION:
      1. CodeBERT contextual node embeddings (vs Word2Vec in VISION)
      2. True node-level cross-attention (vs graph-embed concatenation)
      3. Per-edge-type GatedGraphConv (each type has own GRU weights)
      4. 4-component CPG including TPG as edge type 3
      5. Inter-procedural callee summary injection (stub for Config E)
    """

    CODEBERT_DIM   = 768
    NODE_FEAT_DIM  = 824   # from Stage 5 preprocessing
    GGNN_HIDDEN    = 256
    GGNN_LAYERS    = 3
    NUM_EDGE_TYPES = 4     # AST=0, CFG=1, DFG=2, TPG=3
    FUSED_DIM      = CODEBERT_DIM + GGNN_HIDDEN + GGNN_HIDDEN  # 1280
    MLP_HIDDEN     = 512
    MLP_OUT        = 128
    NUM_CWE        = 12

    def __init__(
        self,
        codebert_model: str = "microsoft/codebert-base",
        node_feature_dim: int = 824,
        use_interproc: bool = False,
        freeze_codebert_layers: int = 0,
        use_graph: bool = True,
        type_aware_edges: bool = True,
    ):
        super().__init__()

        self.node_feature_dim   = node_feature_dim
        self.use_interproc      = use_interproc
        self.use_graph          = use_graph
        self.type_aware_edges   = type_aware_edges

        # ── Sequence Encoder: CodeBERT ──────────────────────────────────
        self.codebert  = AutoModel.from_pretrained(codebert_model)
        self.tokenizer = AutoTokenizer.from_pretrained(codebert_model)

        if freeze_codebert_layers > 0:
            for i, layer in enumerate(self.codebert.encoder.layer):
                if i < freeze_codebert_layers:
                    for p in layer.parameters():
                        p.requires_grad = False

        # ── Node projector: parameterized, NOT hardcoded 824 ────────────
        self.node_proj = nn.Linear(node_feature_dim, self.GGNN_HIDDEN)

        # ── Type-Aware GGNN (Option B) ──────────────────────────────────
        # 4 edge types × 3 layers = 12 GatedGraphConv modules.
        # Each GatedGraphConv uses num_layers=1 (1 GRU timestep) to keep total
        # propagation depth at 3 and allow per-layer residual connections.
        self.edge_type_convs = nn.ModuleList([
            nn.ModuleList([
                GatedGraphConv(out_channels=self.GGNN_HIDDEN, num_layers=1)
                for _ in range(self.NUM_EDGE_TYPES)
            ])
            for _ in range(self.GGNN_LAYERS)
        ])

        # Config B: Type-blind single GatedGraphConv per layer (R-15 fix).
        # Uses dedicated weights separate from the type-aware convs.
        # All edges processed equally regardless of edge_attr type.
        if not type_aware_edges:
            self.single_conv = nn.ModuleList([
                GatedGraphConv(out_channels=self.GGNN_HIDDEN, num_layers=1)
                for _ in range(self.GGNN_LAYERS)
            ])

        # Per-layer aggregation: concat 4 type outputs (1024-d) → 256-d
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

        # ── Node-level Cross-Attention ──────────────────────────────────
        # Q = BERT [CLS] (768→256); K,V = per-node GGNN embeddings (256→256)
        # TRUE node-level attention: Q (B,768→256) attends to N nodes (N,256)
        self.q_proj    = nn.Linear(self.CODEBERT_DIM, self.GGNN_HIDDEN)
        self.k_proj    = nn.Linear(self.GGNN_HIDDEN,  self.GGNN_HIDDEN)
        self.v_proj    = nn.Linear(self.GGNN_HIDDEN,  self.GGNN_HIDDEN)
        self.attn_scale = self.GGNN_HIDDEN ** -0.5

        # ── Inter-Procedural Context Stub (Config E only) ───────────────
        if use_interproc:
            self.interproc_proj = nn.Linear(self.CODEBERT_DIM, self.GGNN_HIDDEN)

        # ── Shared MLP After Fusion ─────────────────────────────────────
        # Input: 1280-d (BERT_768 + Attn_256 + GGNN_mean_256)
        # If inter-proc enabled, fused_dim += GGNN_HIDDEN = 1536
        fused_dim = self.FUSED_DIM + (self.GGNN_HIDDEN if use_interproc else 0)

        self.fusion_mlp = nn.Sequential(
            nn.Linear(fused_dim, self.MLP_HIDDEN),
            nn.LayerNorm(self.MLP_HIDDEN),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(self.MLP_HIDDEN, self.MLP_OUT),
            nn.GELU(),
        )

        # ── Output Heads ────────────────────────────────────────────────
        self.binary_head   = nn.Linear(self.MLP_OUT, 2)             # vuln / safe
        self.cwe_head      = nn.Linear(self.MLP_OUT, self.NUM_CWE)  # 12 CWE classes
        self.severity_head = nn.Linear(self.MLP_OUT, 1)             # CVSS proxy [0,10]

    # ────────────────────────────────────────────────────────────────────
    # Encoding sub-components
    # ────────────────────────────────────────────────────────────────────

    def encode_sequence(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Encode full function token sequence with CodeBERT. Returns [CLS] 768-d."""
        out = self.codebert(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state[:, 0, :]  # (B, 768)

    def encode_graph(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
        callee_node_mask: torch.Tensor = None,
        callee_feat: torch.Tensor = None,
    ):
        """
        3-layer type-aware GGNN forward pass.

        Args:
            x:                (N, node_feature_dim) node features
            edge_index:       (2, E)
            edge_attr:        (E,) edge type integers {0=AST, 1=CFG, 2=DFG, 3=TPG}
            batch:            (N,) batch assignment vector
            callee_node_mask: (N,) bool — call-site nodes (Config E)
            callee_feat:      (N_callees, 256) projected callee embeds (Config E)

        Returns:
            h_graph: (B, 256)  graph-level pooled embedding
            h_nodes: (N, 256)  per-node embeddings (for cross-attention)
        """
        # Validate edge types
        if edge_attr.numel() > 0:
            bad_mask = edge_attr >= self.NUM_EDGE_TYPES
            if bad_mask.any():
                bad_vals = edge_attr[bad_mask].unique().tolist()
                raise ValueError(
                    f"Invalid edge_attr values {bad_vals} found. "
                    f"Expected only {{0,1,2,3}}. Check Stage 4 CDG filtering."
                )

        h = self.node_proj(x)  # (N, 256)

        # Inject inter-proc callee features at call-site nodes
        if self.use_interproc and callee_node_mask is not None and callee_feat is not None:
            h[callee_node_mask] = h[callee_node_mask] + callee_feat

        for layer_idx in range(self.GGNN_LAYERS):
            if not self.type_aware_edges:
                # Config B: type-blind GGNN — dedicated single conv on ALL edges (R-15)
                if edge_index.size(1) == 0:
                    h_new_raw = torch.zeros_like(h)
                else:
                    h_new_raw = self.single_conv[layer_idx](h, edge_index)
                # Pad to 4-type concat width so edge_agg input shape matches
                type_outputs = [h_new_raw] + [torch.zeros_like(h)] * (self.NUM_EDGE_TYPES - 1)
            else:
                type_outputs = []
                for etype in range(self.NUM_EDGE_TYPES):
                    mask    = (edge_attr == etype)
                    ei_type = edge_index[:, mask]   # (2, E_type)

                    if ei_type.size(1) == 0:
                        # No edges of this type → zero contribution (not skip)
                        type_outputs.append(torch.zeros_like(h))
                    else:
                        h_type = self.edge_type_convs[layer_idx][etype](h, ei_type)
                        # Sqrt-count normalization: prevents AST edge dominance over TPG
                        count = mask.sum().float().clamp(min=1.0)
                        type_outputs.append(h_type / count.sqrt())

            # Aggregate: cat 4 outputs → (N, 1024) → project → (N, 256)
            h_concat = torch.cat(type_outputs, dim=-1)     # (N, 1024)
            h_new    = self.edge_agg[layer_idx](h_concat)  # (N, 256)
            h_new    = self.ggnn_norm[layer_idx](h_new)    # GroupNorm
            h_new    = F.gelu(h_new)
            h_new    = self.ggnn_dropout(h_new)

            # Residual connection from layer 2 onward (not layer 1 — different dim from proj)
            h = h + h_new if layer_idx > 0 else h_new

        # NaN guard
        if torch.isnan(h).any():
            raise RuntimeError(
                "NaN in GGNN node embeddings. Check input node features x for NaN."
            )

        # Graph-level readout: mean + max pool
        h_mean  = global_mean_pool(h, batch)               # (B, 256)
        h_max   = global_max_pool(h, batch)                # (B, 256)
        h_graph = self.graph_readout(
            torch.cat([h_mean, h_max], dim=-1)             # (B, 512) → (B, 256)
        )
        return h_graph, h  # (B, 256), (N, 256)

    def cross_attention_fusion(
        self,
        bert_cls: torch.Tensor,
        h_nodes:  torch.Tensor,
        batch:    torch.Tensor,
    ) -> torch.Tensor:
        """
        TRUE node-level cross-attention.
        BERT [CLS] (B,768) queries attend over every CPG node (N, 256).
        Scatter softmax normalises attention weights within each graph.
        Returns fused (B, 1280).
        """
        Q = self.q_proj(bert_cls)   # (B, 256)
        K = self.k_proj(h_nodes)    # (N, 256)
        V = self.v_proj(h_nodes)    # (N, 256)

        # Each node gets its graph's query vector
        Q_expanded = Q[batch]       # (N, 256)

        # Raw attention scores per node
        attn_scores = (Q_expanded * K).sum(dim=-1) * self.attn_scale  # (N,)

        # Numerically stable scatter softmax:
        # 1. Subtract per-graph max to prevent exp() overflow → NaN
        # 2. Then exp → sum → normalize
        attn_max     = scatter(attn_scores, batch, dim=0, reduce='max')   # (B,)
        attn_scores  = attn_scores - attn_max[batch]                      # (N,) shifted
        exp_scores   = attn_scores.exp()                                  # (N,) safe
        sum_exp      = scatter(exp_scores, batch, dim=0, reduce='sum')    # (B,)
        attn_weights = exp_scores / sum_exp[batch].clamp(min=1e-8)        # (N,)

        # Weighted value aggregation per graph
        weighted_V = V * attn_weights.unsqueeze(-1)                       # (N, 256)
        attn_out   = scatter(weighted_V, batch, dim=0, reduce='sum')      # (B, 256)

        # Graph-level mean embed for concat
        graph_embed = scatter(h_nodes, batch, dim=0, reduce='mean')       # (B, 256)

        return torch.cat([bert_cls, attn_out, graph_embed], dim=-1)       # (B, 1280)

    # ────────────────────────────────────────────────────────────────────
    # Full forward pass
    # ────────────────────────────────────────────────────────────────────

    def forward(
        self,
        data,
        input_ids=None,
        attention_mask=None,
        callee_embeddings=None,
        callee_node_indices=None,
        return_intermediates=False,
    ) -> dict:
        """
        Args:
            data:               PyG Batch (x, edge_index, edge_attr, batch)
            input_ids:          (B, 512) — CodeBERT tokenized full function
            attention_mask:     (B, 512)
            callee_embeddings:  (B, max_callees, 768) — inter-proc context (Config E)
            callee_node_indices: list of (graph_idx, local_node_idx, callee_embed) tuples
            return_intermediates: True → expose h_nodes + graph_embed for explainability

        Returns dict with keys:
            logits          (B, 2)   — main classification head
            cwe_logits      (B, 12)  — CWE type head
            severity_score  (B,)     — CVSS proxy (scalar per sample)
            embedding       (B, 128) — shared 128-d representation (contrastive)
        """
        # ── Sequence Encoding ───────────────────────────────────────────
        if input_ids is not None:
            bert_cls = self.encode_sequence(input_ids, attention_mask)  # (B, 768)
        else:
            B = data.batch.max().item() + 1 if data is not None else 1
            device = data.x.device if data is not None else torch.device("cpu")
            bert_cls = torch.zeros(B, self.CODEBERT_DIM, device=device)
            if self.training:
                warnings.warn(
                    "input_ids is None during training — CodeBERT running on zeros. "
                    "F1 will plateau ~0.72. Pass tokenized code to unlock full accuracy.",
                    RuntimeWarning,
                    stacklevel=2,
                )

        # ── Config A: CodeBERT-only (no graph) ─────────────────────────
        if not self.use_graph:
            # No GGNN, no cross-attention — pad with zeros to match fused_dim
            B = bert_cls.size(0)
            zero_attn  = torch.zeros(B, self.GGNN_HIDDEN, device=bert_cls.device)
            zero_graph = torch.zeros(B, self.GGNN_HIDDEN, device=bert_cls.device)
            fused = torch.cat([bert_cls, zero_attn, zero_graph], dim=-1)  # (B, 1280)
        else:
            # ── Graph Encoding ──────────────────────────────────────────
            callee_node_mask, callee_feat = None, None
            if self.use_interproc and callee_embeddings is not None:
                callee_node_mask, callee_feat = self._prepare_interproc_features(
                    data, callee_embeddings, callee_node_indices
                )

            graph_embed, h_nodes = self.encode_graph(
                data.x, data.edge_index, data.edge_attr, data.batch,
                callee_node_mask, callee_feat,
            )  # (B, 256), (N, 256)

            # ── Cross-Attention Fusion ──────────────────────────────────
            fused = self.cross_attention_fusion(bert_cls, h_nodes, data.batch)  # (B, 1280)

        # ── Inter-Proc Extension ────────────────────────────────────────
        if self.use_interproc:
            if callee_embeddings is not None:
                callee_pooled = callee_embeddings.mean(dim=1)       # (B, 768)
                callee_ctx    = self.interproc_proj(callee_pooled)  # (B, 256)
            else:
                B = bert_cls.size(0)
                callee_ctx = torch.zeros(B, self.GGNN_HIDDEN, device=bert_cls.device)
            fused = torch.cat([fused, callee_ctx], dim=-1)          # (B, 1536)

        # ── Shared MLP ──────────────────────────────────────────────────
        shared = self.fusion_mlp(fused)  # (B, 128)

        # ── Output Heads ────────────────────────────────────────────────
        result = {
            "logits":         self.binary_head(shared),                # (B, 2)
            "cwe_logits":     self.cwe_head(shared),                   # (B, 12)
            "severity_score": self.severity_head(shared).squeeze(-1),  # (B,)
            "embedding":      shared,                                   # (B, 128)
        }

        if return_intermediates:
            result["h_nodes"]         = h_nodes       # (N, 256)
            result["graph_embed"]     = graph_embed   # (B, 256)
            result["fused_embedding"] = fused         # (B, 1280) or (B, 1536) with interproc
            result["bert_cls"]        = bert_cls      # (B, 768)

        return result

    def _prepare_interproc_features(self, data, callee_embeddings, callee_node_indices):
        """
        Inject callee summary embeddings into the CPG at call-site nodes.

        Args:
            data: PyG Batch
            callee_embeddings: (B, max_callees, 768) or unused if callee_node_indices provided
            callee_node_indices: list of (graph_idx, local_node_idx, callee_embed_768d) tuples

        Returns:
            (callee_node_mask, projected_callee_features)
        """
        N = data.x.size(0)
        callee_node_mask = torch.zeros(N, dtype=torch.bool, device=data.x.device)
        callee_feat      = torch.zeros(N, self.GGNN_HIDDEN, device=data.x.device)

        if callee_node_indices is None:
            return callee_node_mask, callee_feat

        # Build cumulative node offsets per graph
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


# ────────────────────────────────────────────────────────────────────────
# Checkpoint save / load
# ────────────────────────────────────────────────────────────────────────

def save_checkpoint(model, optimizer, epoch, metrics, config, path):
    """Save full checkpoint including architecture config for serving reconstruction."""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "metrics": metrics,
        "config": {
            "codebert_model":         config.get("base_model", "microsoft/codebert-base"),
            "node_feature_dim":       824,
            "use_interproc":          config.get("use_interproc", False),
            "freeze_codebert_layers": config.get("freeze_codebert_layers", 0),
            "use_graph":              config.get("use_graph", True),
            "type_aware_edges":       config.get("type_aware_edges", True),
            "num_edge_types":         4,
            "num_cwe_classes":        12,
            "ggnn_layers":            3,
            "ggnn_hidden":            256,
            "ggnn_type":              "per_edge_type_gated",
            "cpg_components":         ["AST", "CFG", "DFG", "TPG"],
            "ablation_config":        config.get("ablation_config", "E_full"),
            "seed":                   config.get("seed", 42),
        },
    }
    tmp_path = path + ".tmp"
    torch.save(checkpoint, tmp_path)
    os.replace(tmp_path, path)


def load_checkpoint(path, device="cpu"):
    """Load checkpoint. Returns (model, optimizer_state_dict, epoch, metrics, config)."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = StreamGuardModel(
        codebert_model=config.get("codebert_model", "microsoft/codebert-base"),
        node_feature_dim=config.get("node_feature_dim", 824),
        use_interproc=config.get("use_interproc", False),
        freeze_codebert_layers=config.get("freeze_codebert_layers", 0),
        use_graph=config.get("use_graph", True),
        type_aware_edges=config.get("type_aware_edges", True),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return (
        model,
        checkpoint["optimizer_state_dict"],
        checkpoint["epoch"],
        checkpoint["metrics"],
        config,
    )
