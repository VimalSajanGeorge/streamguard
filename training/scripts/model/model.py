# training/scripts/model/model.py
#
# StreamGuardModel: CodeBERT + Type-Aware 3-layer GGNN + Node-level Cross-Attention
#
# Architecture notes:
#   - GGNN uses Option B: 3 GatedGraphConv(num_layers=1) modules stacked with residuals.
#     GatedGraphConv(num_layers=N) runs N GRU timesteps internally. Stacking three
#     modules with num_layers=3 each would be 9 GRU steps — over-smoothing territory.
#     num_layers=1 per module gives 3 total timesteps with residual connections.
#   - Cross-attention is node-level (BERT queries attend to N per-node embeddings via
#     scatter softmax). A graph-embed level K/V would be a scalar attention weight
#     (always 1.0 after softmax) — mathematically identical to concatenation.
#   - 4 separate GatedGraphConv per layer (one per edge type AST/CFG/DFG/TPG) so each
#     edge type has dedicated GRU weights. Must mask edge_index per type.
#   - GroupNorm instead of BatchNorm1d: works at batch_size=1 (required for serving).

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GatedGraphConv, global_mean_pool, global_max_pool
from torch_geometric.utils import scatter
from transformers import AutoModel, AutoTokenizer


class StreamGuardModel(nn.Module):
    """
    StreamGuard vulnerability detector.

    Novel contributions vs Devign/VISION:
      1. CodeBERT (contextual) vs Word2Vec (static) node embeddings
      2. Node-level cross-attention fusion (not simple concatenation)
      3. 4-type edge handling in GGNN (AST/CFG/DFG/TPG)
      4. Multi-task heads (binary + CWE + severity)
    """

    CODEBERT_DIM  = 768
    GGNN_HIDDEN   = 256
    GGNN_LAYERS   = 3
    NUM_EDGE_TYPES = 4   # AST=0, CFG=1, DFG=2, TPG=3
    FUSED_DIM     = CODEBERT_DIM + GGNN_HIDDEN + GGNN_HIDDEN  # 1280
    MLP_HIDDEN    = 512
    MLP_OUT       = 128
    NUM_CWE       = 12

    def __init__(
        self,
        node_feature_dim: int = 824,
        codebert_model: str = "microsoft/codebert-base",
        use_interproc: bool = False,
    ):
        super().__init__()
        self.use_interproc = use_interproc

        # ── Sequence encoder (CodeBERT) ────────────────────────────────────────
        self.codebert  = AutoModel.from_pretrained(codebert_model)
        self.tokenizer = AutoTokenizer.from_pretrained(codebert_model)

        # ── Node projector: parameterized, NOT hardcoded 824 ──────────────────
        self.node_proj = nn.Linear(node_feature_dim, self.GGNN_HIDDEN)

        # ── Type-Aware GGNN (Option B) ─────────────────────────────────────────
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

        # Per-layer aggregation: concat 4 type outputs (1024-d) → 256-d
        self.edge_agg = nn.ModuleList([
            nn.Linear(self.GGNN_HIDDEN * self.NUM_EDGE_TYPES, self.GGNN_HIDDEN)
            for _ in range(self.GGNN_LAYERS)
        ])

        # GroupNorm: 256 channels / 32 groups = 8 ch/group (stable; works at batch=1)
        self.ggnn_norm = nn.ModuleList([
            nn.GroupNorm(num_groups=32, num_channels=self.GGNN_HIDDEN)
            for _ in range(self.GGNN_LAYERS)
        ])
        self.ggnn_dropout = nn.Dropout(0.1)

        # Graph-level readout: mean‖max (512) → 256
        self.graph_readout = nn.Linear(self.GGNN_HIDDEN * 2, self.GGNN_HIDDEN)

        # ── Node-level Cross-Attention ─────────────────────────────────────────
        # Q from BERT [CLS] (768→256); K,V from per-node GGNN embeddings (256→256).
        # Using scatter softmax: each batch item attends over its own nodes only.
        self.q_proj    = nn.Linear(self.CODEBERT_DIM, self.GGNN_HIDDEN)
        self.k_proj    = nn.Linear(self.GGNN_HIDDEN,  self.GGNN_HIDDEN)
        self.v_proj    = nn.Linear(self.GGNN_HIDDEN,  self.GGNN_HIDDEN)
        self.attn_scale = self.GGNN_HIDDEN ** -0.5

        # ── Shared MLP after fusion (1280 → 512 → 128) ───────────────────────
        self.fusion_mlp = nn.Sequential(
            nn.Linear(self.FUSED_DIM, self.MLP_HIDDEN),
            nn.LayerNorm(self.MLP_HIDDEN),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(self.MLP_HIDDEN, self.MLP_OUT),
            nn.GELU(),
        )

        # ── Output heads ──────────────────────────────────────────────────────
        self.binary_head   = nn.Linear(self.MLP_OUT, 2)           # vuln / safe
        self.cwe_head      = nn.Linear(self.MLP_OUT, self.NUM_CWE)  # 12 CWE types
        self.severity_head = nn.Linear(self.MLP_OUT, 1)           # CVSS proxy

        # ── M2 forward-compat stubs ───────────────────────────────────────────
        if use_interproc:
            # Callee summary injection (no-op in M1, wired in M2)
            self.interproc_proj = nn.Linear(self.CODEBERT_DIM, self.GGNN_HIDDEN)

    # ──────────────────────────────────────────────────────────────────────────
    # Encoder sub-modules
    # ──────────────────────────────────────────────────────────────────────────

    def encode_sequence(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Encode token sequence with CodeBERT. Returns [CLS] vector (B, 768)."""
        out = self.codebert(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state[:, 0, :]  # (B, 768)

    def encode_graph(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
    ):
        """
        3-layer type-aware GGNN.

        Args:
            x:          (N, node_feature_dim) node features
            edge_index: (2, E)
            edge_attr:  (E,) edge type integers {0=AST, 1=CFG, 2=DFG, 3=TPG}
            batch:      (N,) batch assignment vector

        Returns:
            h_graph: (B, 256)  graph-level pooled embedding
            h_nodes: (N, 256)  per-node embeddings (for cross-attention)
        """
        # Validate edge types — catch CDG/DOMINATE edges that slipped through Stage 4
        if edge_attr.numel() > 0:
            bad_mask = edge_attr >= self.NUM_EDGE_TYPES
            if bad_mask.any():
                bad_vals = edge_attr[bad_mask].unique().tolist()
                raise ValueError(
                    f"Invalid edge_attr values {bad_vals} found. "
                    f"Expected only {{0,1,2,3}}. Check Stage 4 CDG filtering."
                )

        h = self.node_proj(x)  # (N, 256)

        for layer_idx in range(self.GGNN_LAYERS):
            type_outputs = []
            for etype in range(self.NUM_EDGE_TYPES):
                mask     = edge_attr == etype
                ei_type  = edge_index[:, mask]   # (2, E_type)
                if ei_type.size(1) == 0:
                    # No edges of this type in this batch → zero contribution
                    type_outputs.append(torch.zeros_like(h))
                else:
                    h_type = self.edge_type_convs[layer_idx][etype](h, ei_type)
                    # GPU-side normalization — no CPU sync from .item()
                    count  = mask.sum().float().clamp(min=1.0)
                    type_outputs.append(h_type / count.sqrt())

            # Concat 4 type outputs → aggregate → norm → activation → residual
            h_concat = torch.cat(type_outputs, dim=-1)    # (N, 1024)
            h_new    = self.edge_agg[layer_idx](h_concat) # (N, 256)
            h_new    = self.ggnn_norm[layer_idx](h_new)
            h_new    = F.gelu(h_new)
            h_new    = self.ggnn_dropout(h_new)
            # Residual from layer 2 onward (layer 0 is the initial projection)
            h = h + h_new if layer_idx > 0 else h_new

        # NaN guard — catch explosion before it propagates to cross-attention
        if torch.isnan(h).any():
            raise RuntimeError(
                "NaN in GGNN node embeddings. Check input node features x for NaN."
            )

        # Graph-level readout: mean‖max → project to 256
        h_mean  = global_mean_pool(h, batch)              # (B, 256)
        h_max   = global_max_pool(h, batch)               # (B, 256)
        h_graph = self.graph_readout(
            torch.cat([h_mean, h_max], dim=-1)            # (B, 512)
        )                                                  # (B, 256)
        return h_graph, h

    def cross_attention_fusion(
        self,
        bert_cls: torch.Tensor,
        h_nodes:  torch.Tensor,
        batch:    torch.Tensor,
    ) -> torch.Tensor:
        """
        Node-level cross-attention: BERT [CLS] queries attend over per-node GGNN
        embeddings using scatter softmax (handles variable graph sizes).

        This is TRUE cross-attention — the attention matrix is (B, N_i) per graph,
        not (B, 1, 1) which would be a scalar and mathematically identical to
        simple concatenation.

        Args:
            bert_cls: (B, 768)
            h_nodes:  (N, 256) all nodes in the batch
            batch:    (N,) batch index for each node

        Returns:
            fused: (B, 1280) = bert_cls ‖ attn_out ‖ graph_mean
        """
        Q = self.q_proj(bert_cls)   # (B, 256)
        K = self.k_proj(h_nodes)    # (N, 256)
        V = self.v_proj(h_nodes)    # (N, 256)

        # Expand Q so each node gets its batch item's query vector
        Q_expanded = Q[batch]                                        # (N, 256)

        # Dot-product attention score per node
        attn_scores = (Q_expanded * K).sum(dim=-1) * self.attn_scale  # (N,)

        # Numerically stable scatter softmax:
        # 1. Subtract per-graph max to prevent exp() overflow → NaN
        # 2. Then exp → sum → normalize
        attn_max     = scatter(attn_scores, batch, dim=0, reduce='max')  # (B,)
        attn_scores  = attn_scores - attn_max[batch]                     # (N,) shifted
        exp_scores   = attn_scores.exp()                                 # (N,) safe
        sum_exp      = scatter(exp_scores, batch, dim=0, reduce='sum')   # (B,)
        attn_weights = exp_scores / sum_exp[batch].clamp(min=1e-8)       # (N,)

        # Weighted sum of value vectors per graph
        weighted_V = V * attn_weights.unsqueeze(-1)                    # (N, 256)
        attn_out   = scatter(weighted_V, batch, dim=0, reduce='sum')   # (B, 256)

        # Mean-pool nodes as the "graph summary" in the fused vector
        graph_mean = scatter(h_nodes, batch, dim=0, reduce='mean')     # (B, 256)

        return torch.cat([bert_cls, attn_out, graph_mean], dim=-1)     # (B, 1280)

    # ──────────────────────────────────────────────────────────────────────────
    # Forward
    # ──────────────────────────────────────────────────────────────────────────

    def forward(
        self,
        data,
        input_ids:          torch.Tensor | None = None,
        attention_mask:     torch.Tensor | None = None,
        callee_embeddings:  torch.Tensor | None = None,   # M2 stub (ignored in M1)
        return_intermediates: bool = False,
    ) -> dict:
        """
        Args:
            data:               PyG Batch (x, edge_index, edge_attr, batch)
            input_ids:          (B, seq_len) tokenised function code — optional
            attention_mask:     (B, seq_len)
            callee_embeddings:  reserved for M2 inter-procedural context
            return_intermediates: expose internal tensors for explainability

        Returns dict with keys:
            binary_logits   (B, 2)   — main classification head
            cwe_logits      (B, 12)  — CWE type head
            severity        (B,)     — CVSS proxy (scalar per sample)
            embedding       (B, 128) — shared 128-d representation (contrastive)
        """
        # 1. Graph encoding (per-node and graph-level)
        h_graph, h_nodes = self.encode_graph(
            data.x, data.edge_index, data.edge_attr, data.batch
        )  # (B, 256), (N, 256)

        # 2. Sequence encoding — zero vector if tokens not provided
        if input_ids is not None:
            bert_cls = self.encode_sequence(input_ids, attention_mask)  # (B, 768)
        else:
            B = h_graph.size(0)
            bert_cls = torch.zeros(B, self.CODEBERT_DIM, device=h_graph.device)
            if self.training:
                warnings.warn(
                    "input_ids is None during training — CodeBERT running on zeros. "
                    "F1 will plateau ~0.72. Pass tokenized code to unlock full accuracy.",
                    RuntimeWarning,
                    stacklevel=2,
                )

        # 3. M2 stub: callee context injection (no-op in M1)
        if self.use_interproc and callee_embeddings is not None:
            # In M2: project callee summary and add to bert_cls or h_nodes
            # Left intentionally as no-op in M1 to keep the interface stable
            pass

        # 4. Node-level cross-attention fusion
        fused = self.cross_attention_fusion(bert_cls, h_nodes, data.batch)  # (B, 1280)

        # 5. Shared MLP
        shared = self.fusion_mlp(fused)  # (B, 128)

        # 6. Output heads
        outputs = {
            "binary_logits": self.binary_head(shared),               # (B, 2)
            "cwe_logits":    self.cwe_head(shared),                   # (B, 12)
            "severity":      self.severity_head(shared).squeeze(-1),  # (B,)
            "embedding":     shared,                                   # (B, 128)
        }

        if return_intermediates:
            outputs["fused_embedding"]      = fused       # (B, 1280)
            outputs["ggnn_node_embeddings"] = h_nodes     # (N, 256)
            outputs["bert_cls"]             = bert_cls    # (B, 768)
            outputs["graph_embed"]          = h_graph     # (B, 256)

        return outputs
