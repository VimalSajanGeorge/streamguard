# docs/MODEL.md — Neural Architecture Specification

> Read this before implementing: model.py, losses.py, cfa_dataloader.py, train.py, eval.py

---

## 1. Complete Model Code Template

```python
# training/scripts/model/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GatedGraphConv, global_mean_pool, global_max_pool
from transformers import AutoModel, AutoTokenizer

class StreamGuardModel(nn.Module):
    """
    StreamGuard: CodeBERT + 3-layer GGNN + Cross-Attention Fusion
    
    Novel contributions vs Devign/VISION:
    1. CodeBERT (contextual) vs Word2Vec (static) node embeddings
    2. Cross-attention fusion (not simple concatenation)
    3. 4-type edge handling in GGNN (AST/CFG/DFG/TPG)
    4. Multi-task heads (binary + CWE + severity)
    """
    
    CODEBERT_DIM = 768
    GGNN_HIDDEN = 256
    GGNN_LAYERS = 3
    FUSED_DIM = CODEBERT_DIM + GGNN_HIDDEN + GGNN_HIDDEN  # 1280
    MLP_HIDDEN = 512
    MLP_OUT = 128
    NUM_CWE = 12
    NUM_EDGE_TYPES = 4  # AST=0, CFG=1, DFG=2, TPG=3
    
    def __init__(self, codebert_model: str = "microsoft/codebert-base"):
        super().__init__()
        
        # ── Sequence encoder (CodeBERT) ─────────────────────────────
        self.codebert = AutoModel.from_pretrained(codebert_model)
        self.tokenizer = AutoTokenizer.from_pretrained(codebert_model)
        
        # ── Graph encoder (Type-Aware GGNN — Option B) ────────────────
        # 4 separate GatedGraphConv per layer, one per edge type
        # Each edge type (AST/CFG/DFG/TPG) has dedicated GRU weights
        # NODE_FEAT_DIM = 824 (from preprocessing stage 5)
        self.node_proj = nn.Linear(824, self.GGNN_HIDDEN)  # project to GGNN hidden dim
        
        # 4 types × 3 layers = 12 GatedGraphConv modules total
        self.edge_type_convs = nn.ModuleList([
            nn.ModuleList([
                GatedGraphConv(out_channels=self.GGNN_HIDDEN, num_layers=1)
                for _ in range(self.NUM_EDGE_TYPES)
            ])
            for _ in range(self.GGNN_LAYERS)
        ])
        
        # Per-layer aggregation: concat 4 type outputs → 256
        self.edge_agg = nn.ModuleList([
            nn.Linear(self.GGNN_HIDDEN * self.NUM_EDGE_TYPES, self.GGNN_HIDDEN)
            for _ in range(self.GGNN_LAYERS)
        ])
        
        # GroupNorm instead of BatchNorm1d — works at batch_size=1 (required for serving)
        # 256 channels / 32 groups = 8 channels per group (stable range)
        self.ggnn_bn = nn.ModuleList([
            nn.GroupNorm(num_groups=32, num_channels=self.GGNN_HIDDEN)
            for _ in range(self.GGNN_LAYERS)
        ])
        self.ggnn_dropout = nn.Dropout(0.1)
        
        # Readout: mean + max pool → concat → project to 256-d
        self.graph_readout = nn.Linear(self.GGNN_HIDDEN * 2, self.GGNN_HIDDEN)
        
        # ── Cross-Attention Fusion ───────────────────────────────────
        self.q_proj = nn.Linear(self.CODEBERT_DIM, self.GGNN_HIDDEN)
        self.k_proj = nn.Linear(self.GGNN_HIDDEN, self.GGNN_HIDDEN)
        self.v_proj = nn.Linear(self.GGNN_HIDDEN, self.GGNN_HIDDEN)
        self.attn_scale = self.GGNN_HIDDEN ** -0.5
        
        # ── Shared MLP after fusion ──────────────────────────────────
        self.fusion_mlp = nn.Sequential(
            nn.Linear(self.FUSED_DIM, self.MLP_HIDDEN),
            nn.LayerNorm(self.MLP_HIDDEN),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(self.MLP_HIDDEN, self.MLP_OUT),
            nn.GELU(),
        )
        
        # ── Output heads ─────────────────────────────────────────────
        self.binary_head   = nn.Linear(self.MLP_OUT, 2)       # vuln/safe
        self.cwe_head      = nn.Linear(self.MLP_OUT, self.NUM_CWE)  # 12 CWEs
        self.severity_head = nn.Linear(self.MLP_OUT, 1)       # CVSS proxy
    
    def encode_sequence(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """Encode token sequence with CodeBERT. Returns [CLS] 768-d."""
        outputs = self.codebert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0, :]  # [CLS] token, shape: (B, 768)
    
    def encode_graph(self, x, edge_index, edge_attr, batch):
        """
        Encode CPG with 3-layer type-aware GGNN (Option B).
        Each edge type (AST/CFG/DFG/TPG) has its own GatedGraphConv.
        x: (N, 824) node features
        edge_index: (2, E)
        edge_attr: (E,) edge type integers {AST=0, CFG=1, DFG=2, TPG=3}
        batch: (N,) batch assignment vector
        Returns: (h_graph, h_nodes) where h_graph=(B, 256), h_nodes=(N, 256)
        """
        h = self.node_proj(x)  # (N, 256)
        
        for layer_idx in range(self.GGNN_LAYERS):
            type_outputs = []
            for etype in range(self.NUM_EDGE_TYPES):
                mask = (edge_attr == etype)
                ei_type = edge_index[:, mask]   # (2, E_type)
                if ei_type.size(1) == 0:
                    # No edges of this type in batch → zero contribution
                    type_outputs.append(torch.zeros_like(h))
                else:
                    h_type = self.edge_type_convs[layer_idx][etype](h, ei_type)
                    # Normalize by edge count so rare types (TPG) aren't drowned out
                    count = max(mask.sum().item(), 1)
                    type_outputs.append(h_type / (count ** 0.5))
            
            # Aggregate: concat 4 type outputs → project back to 256
            h_concat = torch.cat(type_outputs, dim=-1)   # (N, 1024)
            h_new = self.edge_agg[layer_idx](h_concat)   # (N, 256)
            h_new = self.ggnn_bn[layer_idx](h_new)
            h_new = F.gelu(h_new)
            h_new = self.ggnn_dropout(h_new)
            h = h + h_new if layer_idx > 0 else h_new    # residual from layer 2+
        
        # Graph-level readout: mean + max pool
        h_mean = global_mean_pool(h, batch)  # (B, 256)
        h_max  = global_max_pool(h, batch)   # (B, 256)
        h_graph = torch.cat([h_mean, h_max], dim=-1)  # (B, 512)
        return self.graph_readout(h_graph), h  # (B, 256), (N, 256)
    
    def cross_attention_fusion(self, bert_cls, h_nodes, batch):
        """
        Node-level cross-attention: BERT attends over per-node GGNN embeddings.
        bert_cls:  (B, 768)
        h_nodes:   (N, 256) per-node GGNN embeddings
        batch:     (N,) batch assignment vector
        Returns fused: (B, 1280)
        
        Uses scatter softmax to handle variable-size graphs per batch item.
        This is TRUE cross-attention (BERT queries attend to N graph nodes),
        not the degenerate case where K/V are a single graph-level vector
        (which mathematically reduces to simple concatenation).
        """
        from torch_geometric.utils import scatter
        
        Q = self.q_proj(bert_cls)              # (B, 256)
        K = self.k_proj(h_nodes)               # (N, 256)
        V = self.v_proj(h_nodes)               # (N, 256)
        
        # Expand Q to per-node: each node gets its batch's query
        Q_expanded = Q[batch]                  # (N, 256)
        
        # Attention scores per node
        attn_scores = (Q_expanded * K).sum(dim=-1) * self.attn_scale  # (N,)
        
        # Scatter softmax: normalize within each graph in the batch
        attn_weights = scatter(attn_scores.exp(), batch, dim=0, reduce='sum')
        attn_weights = attn_scores.exp() / attn_weights[batch].clamp(min=1e-8)  # (N,)
        
        # Weighted sum of values per graph
        weighted_V = V * attn_weights.unsqueeze(-1)  # (N, 256)
        attn_out = scatter(weighted_V, batch, dim=0, reduce='sum')  # (B, 256)
        
        # Graph-level embedding for concat
        graph_embed = scatter(h_nodes, batch, dim=0, reduce='mean')  # (B, 256)
        
        # Concatenate all representations
        return torch.cat([bert_cls, attn_out, graph_embed], dim=-1)  # (B, 1280)
    
    def forward(self, data, input_ids=None, attention_mask=None,
                callee_embeddings=None, return_intermediates=False):
        """
        data: PyG Batch object (x, edge_index, edge_attr, batch)
        input_ids: (B, 512) CodeBERT tokens of the full function
        attention_mask: (B, 512)
        callee_embeddings: reserved for M2 inter-procedural (ignored for now)
        return_intermediates: if True, expose h_nodes + attn for explainability
        """
        # 1. Graph encoding (returns both graph-level and node-level embeddings)
        graph_embed, h_nodes = self.encode_graph(
            data.x, data.edge_index, data.edge_attr, data.batch
        )  # (B, 256), (N, 256)
        
        # 2. Sequence encoding (if provided; else use zero vector)
        if input_ids is not None:
            bert_cls = self.encode_sequence(input_ids, attention_mask)  # (B, 768)
        else:
            B = graph_embed.size(0)
            bert_cls = torch.zeros(B, self.CODEBERT_DIM, device=graph_embed.device)
        
        # 3. Node-level cross-attention fusion
        fused = self.cross_attention_fusion(bert_cls, h_nodes, data.batch)  # (B, 1280)
        
        # 4. Shared MLP
        shared = self.fusion_mlp(fused)  # (B, 128)
        
        # 5. Output heads
        result = {
            "logits":          self.binary_head(shared),    # (B, 2)
            "cwe_logits":      self.cwe_head(shared),       # (B, 12)
            "severity_score":  self.severity_head(shared).squeeze(-1),  # (B,)
            "embedding":       shared,                       # (B, 128) for contrastive
        }
        
        if return_intermediates:
            result["h_nodes"] = h_nodes
            result["graph_embed"] = graph_embed
            result["fused_embedding"] = fused
        
        return result
```

---

## 2. Loss Functions

```python
# training/scripts/model/losses.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class StreamGuardLoss(nn.Module):
    """
    Composite loss for CFA-paired training.
    
    L_total = λ1*L_CE + λ2*L_CFA + λ3*L_severity
    
    L_CE:       Binary cross-entropy on all samples
    L_CFA:      Cosine margin contrastive loss on (vuln, cfa) pairs
                Forces embeddings of paired samples to opposite sides of decision boundary
    L_severity: Huber regression loss on CVSS-proxy score
    
    Research basis: VISION (AIES 2025) - paired training with contrastive objective
    """
    
    def __init__(self, lambda_ce=1.0, lambda_cfa=0.5, lambda_sev=0.1,
                 cfa_margin=0.5, num_cwe=12):
        super().__init__()
        self.lambda_ce  = lambda_ce
        self.lambda_cfa = lambda_cfa
        self.lambda_sev = lambda_sev
        self.margin     = cfa_margin
        
        self.ce_loss  = nn.CrossEntropyLoss()
        self.cwe_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.sev_loss = nn.HuberLoss(delta=1.0)
    
    def forward(self, outputs_orig, outputs_cfa=None, labels=None,
                cwe_labels=None, severity_labels=None):
        """
        outputs_orig: model output dict for original samples
        outputs_cfa:  model output dict for CFA pairs (may be None if no pairs in batch)
        labels:       (B,) binary labels 0/1
        cwe_labels:   (B,) CWE class 0-11
        severity_labels: (B,) float severity scores [0,10]
        """
        losses = {}
        total = torch.tensor(0.0, device=outputs_orig["logits"].device)
        
        # ── L_CE: binary classification ─────────────────────────────
        if labels is not None:
            l_ce = self.ce_loss(outputs_orig["logits"], labels)
            losses["L_CE"] = l_ce.item()
            total = total + self.lambda_ce * l_ce
        
        # ── L_CFA: cosine margin contrastive on (v, v') pairs ───────
        if outputs_cfa is not None and outputs_orig is not None:
            emb_v  = outputs_orig["embedding"]   # (B_pairs, 128)
            emb_vp = outputs_cfa["embedding"]    # (B_pairs, 128)
            
            # Cosine similarity between paired embeddings
            cosine_sim = F.cosine_similarity(emb_v, emb_vp, dim=-1)  # (B_pairs,)
            
            # Margin loss: push paired embeddings apart
            # We WANT cosine_sim to be NEGATIVE (opposite sides of sphere)
            # Loss penalises if similarity > -margin
            l_cfa = F.relu(cosine_sim - (-self.margin)).mean()
            losses["L_CFA"] = l_cfa.item()
            total = total + self.lambda_cfa * l_cfa
        
        # ── L_severity: regression ───────────────────────────────────
        if severity_labels is not None:
            valid_mask = severity_labels >= 0  # -1 means unknown
            if valid_mask.any():
                l_sev = self.sev_loss(
                    outputs_orig["severity_score"][valid_mask],
                    severity_labels[valid_mask]
                )
                losses["L_severity"] = l_sev.item()
                total = total + self.lambda_sev * l_sev
        
        losses["total"] = total.item()
        return total, losses
```

---

### Optional: Tier-Weighted Contrastive Loss

When `cfa_tier` metadata is available in the batch (populated by `CFAAwareBatchSampler`
from HDF5 attrs), you can optionally weight the L_CFA term by generation confidence.
Tier 1 (deterministic AST) pairs are the most reliable; Tier 5 (critique-refine) pairs
carry slight uncertainty. This is an **optional enhancement** — the default `StreamGuardLoss`
above works correctly without it and should be used for all ablation configs A–E.

```python
# Optional tier-weighted variant — only use in Config E (Full StreamGuard)
# Enable with: loss = StreamGuardLoss(use_tier_weighting=True)

TIER_CONFIDENCE = {
    1: 1.00,   # AST rule — deterministic, structurally guaranteed
    2: 0.90,   # zero-shot LLM — compile-validated
    3: 0.82,   # CoT LLM — taint-path validated for injection CWEs
    4: 0.75,   # few-shot LLM — exemplar-guided but still LLM
    5: 0.65,   # critique-refine — accepted after failure, lower confidence
    0: 1.00,   # native SARD pair — NIST-validated ground truth
}

class StreamGuardLossTierWeighted(StreamGuardLoss):
    """
    Extends StreamGuardLoss with CFA tier confidence weighting on L_CFA.
    All other losses (L_CE, L_severity) are unchanged.

    Use only when cfa_tier is available in batch metadata.
    Falls back to standard L_CFA when tier info is absent.
    """

    def forward(self, outputs_orig, outputs_cfa=None, labels=None,
                cwe_labels=None, severity_labels=None, cfa_tiers=None):
        """
        cfa_tiers: optional (B_pairs,) int tensor of tier values 0-5.
                   If None: falls back to standard unweighted L_CFA.
        """
        # Compute standard losses (L_CE, L_severity) from parent
        total, losses = super().forward(
            outputs_orig, outputs_cfa=None,  # skip CFA in parent
            labels=labels, cwe_labels=cwe_labels, severity_labels=severity_labels
        )

        # Override L_CFA with tier-weighted version
        if outputs_cfa is not None and outputs_orig is not None:
            emb_v  = outputs_orig["embedding"]
            emb_vp = outputs_cfa["embedding"]
            cosine_sim = F.cosine_similarity(emb_v, emb_vp, dim=-1)
            margin_loss = F.relu(cosine_sim - (-self.margin))

            if cfa_tiers is not None:
                weights = torch.tensor(
                    [TIER_CONFIDENCE.get(int(t), 0.80) for t in cfa_tiers],
                    dtype=torch.float32, device=emb_v.device
                )
                l_cfa = (margin_loss * weights).mean()
            else:
                l_cfa = margin_loss.mean()

            losses["L_CFA"] = l_cfa.item()
            total = total + self.lambda_cfa * l_cfa

        losses["total"] = total.item()
        return total, losses
```

> **When to use:** Only in Config E (Full StreamGuard) when `cfa_tier` is populated in
> HDF5 by Stage 6 and forwarded by `CFAAwareBatchSampler`. For ablation configs A–D,
> use the standard `StreamGuardLoss` to keep comparisons clean.

---

## 3. CFA-Aware DataLoader

```python
# training/scripts/model/cfa_dataloader.py

import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch
import h5py
from collections import defaultdict
import random

class CFADataset(Dataset):
    """
    Loads PyG graph Data objects from HDF5 cache.
    Groups samples by pair_id for CFA-aware batching.
    """
    
    def __init__(self, h5_path: str, split: str = "train"):
        self.h5_path = h5_path
        self.split   = split
        self._index  = self._build_index()
    
    def _build_index(self):
        """Build (sample_id, pair_id, label) index from HDF5."""
        index = []
        with h5py.File(self.h5_path, 'r') as f:
            for sample_id in f[self.split].keys():
                grp = f[self.split][sample_id]
                index.append({
                    "sample_id": sample_id,
                    "pair_id":   str(grp.attrs.get("pair_id", "")),
                    "label":     int(grp["y"][0]),
                    "cwe":       str(grp.attrs.get("cwe", "")),
                })
        return index
    
    def __len__(self):
        return len(self._index)
    
    def __getitem__(self, idx):
        item = self._index[idx]
        return self._load_graph(item["sample_id"]), item
    
    def _load_graph(self, sample_id: str):
        from torch_geometric.data import Data
        with h5py.File(self.h5_path, 'r') as f:
            g = f[self.split][sample_id]
            return Data(
                x=torch.from_numpy(g["x"][:]),
                edge_index=torch.from_numpy(g["edge_index"][:]),
                edge_attr=torch.from_numpy(g["edge_attr"][:]),
                y=torch.from_numpy(g["y"][:]),
                sample_id=sample_id,
                pair_id=str(g.attrs.get("pair_id", "")),
                cwe=str(g.attrs.get("cwe", "")),
            )


class CFAAwareBatchSampler:
    """
    CRITICAL: Keeps CFA pairs in the same batch.
    Standard random sampling splits pairs across batches,
    making the contrastive loss train on random non-paired samples.
    
    Strategy:
    1. Group all samples by pair_id
    2. Shuffle groups
    3. Fill batches from groups (pairs stay together)
    """
    
    def __init__(self, dataset: CFADataset, batch_size: int, drop_last: bool = True):
        self.batch_size = batch_size
        self.drop_last  = drop_last
        
        # Group by pair_id
        pair_groups = defaultdict(list)
        for i, item in enumerate(dataset._index):
            key = item["pair_id"] or f"singleton_{i}"
            pair_groups[key].append(i)
        self.groups = list(pair_groups.values())
    
    def __iter__(self):
        random.shuffle(self.groups)
        
        current_batch = []
        for group in self.groups:
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


def build_dataloader(h5_path: str, split: str, batch_size: int,
                     num_workers: int = 4) -> DataLoader:
    dataset = CFADataset(h5_path, split)
    sampler = CFAAwareBatchSampler(dataset, batch_size)
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        collate_fn=cfa_collate_fn,
    )


def cfa_collate_fn(batch):
    """
    Custom collate: separate original and CFA samples into two sub-batches.
    Returns: (orig_batch, cfa_batch, orig_meta, cfa_meta)
    """
    graphs, metas = zip(*batch)
    
    orig_graphs, cfa_graphs = [], []
    orig_metas, cfa_metas   = [], []
    pair_map = {}  # pair_id → orig index
    
    for g, m in zip(graphs, metas):
        pid = m["pair_id"]
        if not pid or pid.startswith("singleton_"):
            orig_graphs.append(g)
            orig_metas.append(m)
        elif pid not in pair_map:
            pair_map[pid] = len(orig_graphs)
            orig_graphs.append(g)
            orig_metas.append(m)
        else:
            # This is the CFA counterpart
            cfa_graphs.append(g)
            cfa_metas.append(m)
    
    orig_batch = Batch.from_data_list(orig_graphs)
    cfa_batch  = Batch.from_data_list(cfa_graphs) if cfa_graphs else None
    return orig_batch, cfa_batch, orig_metas, cfa_metas
```

---

## 4. Training Loop

```python
# training/scripts/model/train.py

import torch
import mlflow
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import get_linear_schedule_with_warmup

def train(config: dict):
    """
    Main training loop.
    config: see TRAINING_CONFIG in PRD.md
    """
    torch.manual_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = StreamGuardModel(config["base_model"]).to(device)
    criterion = StreamGuardLoss()
    
    # Differential learning rates: lower for pre-trained CodeBERT
    param_groups = [
        {"params": model.codebert.parameters(), "lr": config["lr_codebert"]},
        {"params": [p for n,p in model.named_parameters()
                    if "codebert" not in n],    "lr": config["lr_ggnn_fusion"]},
    ]
    optimizer = AdamW(param_groups, weight_decay=config["weight_decay"])
    
    train_loader = build_dataloader(config["train_h5"], "train", config["batch_size_graphs"])
    val_loader   = build_dataloader(config["train_h5"], "val",   config["batch_size_graphs"])
    
    total_steps  = len(train_loader) * config["epochs"]
    warmup_steps = int(total_steps * config["warmup_ratio"])
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    
    best_val_f1  = 0.0
    patience_ctr = 0
    
    with mlflow.start_run():
        mlflow.log_params(config)
        
        for epoch in range(config["epochs"]):
            # ── Training ──────────────────────────────────────────
            model.train()
            for step, (orig_batch, cfa_batch, orig_metas, _) in enumerate(train_loader):
                orig_batch = orig_batch.to(device)
                
                # Forward pass on original samples
                # NOTE: for sequence encoding, tokenize orig_metas[].code here
                out_orig = model(orig_batch)
                
                # Forward pass on CFA counterparts (if present in batch)
                out_cfa = None
                if cfa_batch is not None:
                    cfa_batch = cfa_batch.to(device)
                    out_cfa = model(cfa_batch)
                
                # Labels from batch
                labels = orig_batch.y.to(device)
                
                total_loss, loss_dict = criterion(out_orig, out_cfa, labels)
                
                # Gradient accumulation
                total_loss = total_loss / config["gradient_accumulation"]
                total_loss.backward()
                
                if (step + 1) % config["gradient_accumulation"] == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                
                if step % 50 == 0:
                    mlflow.log_metrics(loss_dict, step=epoch*len(train_loader)+step)
            
            # ── Validation ────────────────────────────────────────
            val_metrics = evaluate(model, val_loader, device)
            mlflow.log_metrics({f"val_{k}": v for k,v in val_metrics.items()}, step=epoch)
            
            print(f"Epoch {epoch}: val_F1={val_metrics['f1']:.4f} | "
                  f"val_pairwise_acc={val_metrics.get('pairwise_accuracy',0):.4f}")
            
            # ── Checkpoint + Early stopping ───────────────────────
            if val_metrics["f1"] > best_val_f1:
                best_val_f1 = val_metrics["f1"]
                patience_ctr = 0
                save_checkpoint(model, optimizer, epoch, val_metrics, config)
            else:
                patience_ctr += 1
                if patience_ctr >= config["early_stopping_patience"]:
                    print(f"Early stopping at epoch {epoch}. Best val F1: {best_val_f1:.4f}")
                    break
```

---

## 5. Evaluation Metrics

```python
# training/scripts/model/eval.py

from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import numpy as np

def evaluate(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    all_pair_results = []  # for pairwise accuracy
    
    with torch.no_grad():
        for orig_batch, cfa_batch, orig_metas, cfa_metas in dataloader:
            orig_batch = orig_batch.to(device)
            out = model(orig_batch)
            
            preds  = out["logits"].argmax(dim=-1).cpu().numpy()
            labels = orig_batch.y.cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels)
            
            # Pairwise accuracy: both v and v' correctly classified
            if cfa_batch is not None:
                cfa_batch = cfa_batch.to(device)
                out_cfa = model(cfa_batch)
                cfa_preds  = out_cfa["logits"].argmax(dim=-1).cpu().numpy()
                cfa_labels = cfa_batch.y.cpu().numpy()
                
                # Pair: orig must be vuln (1), cfa must be safe (0)
                for op, ol, cp, cl in zip(preds, labels, cfa_preds, cfa_labels):
                    if ol == 1:  # only check pairs where original is vulnerable
                        all_pair_results.append(op == 1 and cp == 0)
    
    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    metrics = {
        "f1":        f1_score(all_labels, all_preds, zero_division=0),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall":    recall_score(all_labels, all_preds, zero_division=0),
    }
    
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds, labels=[0,1]).ravel()
    metrics["fpr"] = fp / (fp + tn) if (fp + tn) > 0 else 0
    metrics["fnr"] = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    if all_pair_results:
        metrics["pairwise_accuracy"] = sum(all_pair_results) / len(all_pair_results)
    
    # Per-CWE F1 (worst-group metric)
    # Requires cwe labels in batch metadata
    # ... (compute per-CWE separately)
    
    return metrics
```

---

*docs/MODEL.md | StreamGuard v1.0 | March 2026*
