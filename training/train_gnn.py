"""
Enhanced Taint-Flow GNN Training with Memory-Aware Batching

Features:
- Memory profiling for safe batch sizing
- Graph statistics-based batching
- S3 checkpointing for Spot instances
- Reproducibility tracking
- Early stopping on binary F1
- PyTorch Geometric integration
"""

import os
import json
import argparse
import random
import hashlib
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, WeightedRandomSampler
from torch.amp import GradScaler, autocast

try:
    import torch_geometric
    from torch_geometric.data import Data, DataLoader
    from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    print("[!] PyTorch Geometric not available. Install: pip install torch-geometric")

from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report
)

# Optional: LR Finder and Cache utilities
try:
    from training.utils.lr_finder import LRFinder, analyze_lr_loss_curve, validate_and_cap_lr
    from training.utils.lr_cache import compute_cache_key, save_lr_cache, load_lr_cache
    LR_FINDER_AVAILABLE = True
except ImportError:
    try:
        from utils.lr_finder import LRFinder, analyze_lr_loss_curve, validate_and_cap_lr
        from utils.lr_cache import compute_cache_key, save_lr_cache, load_lr_cache
        LR_FINDER_AVAILABLE = True
    except ImportError:
        LR_FINDER_AVAILABLE = False
        print("[!] LR Finder utilities not available.")

# Optional: Focal Loss
try:
    from training.losses.focal_loss import FocalLoss
    FOCAL_LOSS_AVAILABLE = True
except ImportError:
    try:
        from losses.focal_loss import FocalLoss
        FOCAL_LOSS_AVAILABLE = True
    except ImportError:
        FOCAL_LOSS_AVAILABLE = False
        print("[!] Focal Loss not available.")

# Optional: boto3 for S3
try:
    import boto3
    S3_AVAILABLE = True
except ImportError:
    S3_AVAILABLE = False
    print("[!] boto3 not available. S3 checkpointing disabled.")


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[+] Random seed set to {seed}")


def get_git_commit() -> str:
    """Get current git commit hash."""
    try:
        commit = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            stderr=subprocess.DEVNULL
        ).decode('utf-8').strip()
        return commit
    except:
        return 'unknown'


def compute_file_checksum(file_path: Path) -> str:
    """Compute SHA256 checksum."""
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


class GraphDataset(Dataset):
    """Dataset for graph-based vulnerability detection."""

    def __init__(
        self,
        data_path: Path,
        use_weights: bool = False,
        encoder_features: str = 'none',
        encoder_feature_dim: int = 768
    ):
        """
        Initialize graph dataset.

        Args:
            data_path: Path to preprocessed JSONL
            use_weights: Whether to use sample weights
        """
        if not TORCH_GEOMETRIC_AVAILABLE:
            raise ImportError("PyTorch Geometric required. Install: pip install torch-geometric")

        self.use_weights = use_weights
        self.encoder_features = encoder_features
        self.encoder_feature_dim = encoder_feature_dim
        self.graphs = []
        self.node_counts = []
        self.edge_counts = []

        print(f"[*] Loading graph dataset from {data_path}")

        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                sample = json.loads(line)

                # Extract graph components or fallback
                ast_nodes = sample.get('ast_nodes', [])
                edge_index = sample.get('edge_index', [])
                label = int(sample.get('label', sample.get('target', 0)))
                weight = float(sample.get('weight', 1.0)) if use_weights else 1.0

                if len(ast_nodes) == 0:
                    # Fallback: simple sequential graph from code tokens hashed to ids
                    code = sample.get('code') or sample.get('func') or ''
                    tokens = [t for t in code.replace('\t', ' ').split() if t]
                    if not tokens:
                        continue
                    ids = [abs(hash(t)) % 1000 for t in tokens]
                    ast_nodes = ids
                    edge_index = [[i, i + 1] for i in range(len(ids) - 1)] + [[i + 1, i] for i in range(len(ids) - 1)]

                x = torch.tensor(ast_nodes, dtype=torch.long).unsqueeze(1)
                if len(edge_index) > 0:
                    edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
                else:
                    edge_index_tensor = torch.tensor([[0], [0]], dtype=torch.long)
                y = torch.tensor([label], dtype=torch.long)

                graph_data = Data(x=x, edge_index=edge_index_tensor, y=y)

                # Optional graph-level encoder features (CLS)
                if self.encoder_features == 'cls' and 'graph_cls' in sample:
                    try:
                        gf = torch.tensor(sample['graph_cls'], dtype=torch.float32)
                        if gf.ndim == 1:
                            graph_data.graph_features = gf
                    except Exception:
                        pass

                if use_weights:
                    graph_data.weight = torch.tensor([weight], dtype=torch.float)

                self.graphs.append(graph_data)
                self.node_counts.append(len(ast_nodes))
                self.edge_counts.append(len(edge_index))

        print(f"[+] Loaded {len(self.graphs)} graphs")

        # Statistics
        labels = [g.y.item() for g in self.graphs]
        vuln_count = sum(labels)
        safe_count = len(labels) - vuln_count

        print(f"    Vulnerable: {vuln_count} ({vuln_count/len(labels):.1%})")
        print(f"    Safe: {safe_count} ({safe_count/len(labels):.1%})")

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]

    def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics."""
        return {
            'total_graphs': len(self.graphs),
            'avg_nodes': np.mean(self.node_counts),
            'median_nodes': np.median(self.node_counts),
            'p95_nodes': np.percentile(self.node_counts, 95),
            'max_nodes': np.max(self.node_counts),
            'avg_edges': np.mean(self.edge_counts),
            'p95_edges': np.percentile(self.edge_counts, 95)
        }


def recommend_batch_size(
    dataset: GraphDataset,
    gpu_mem_gb: float = 16.0,
    hidden_dim: int = 256,
    safety_margin: float = 0.5
) -> int:
    """
    Recommend safe batch size based on graph statistics.

    Args:
        dataset: Graph dataset
        gpu_mem_gb: GPU memory in GB
        hidden_dim: Hidden dimension
        safety_margin: Safety margin (0.5 = use 50% of memory)

    Returns:
        Recommended batch size
    """
    stats = dataset.get_statistics()
    p95_nodes = stats['p95_nodes']

    # Estimate memory per node
    bytes_per_param = 4  # float32
    mem_per_node = hidden_dim * bytes_per_param

    # Available memory
    available_mem = gpu_mem_gb * 1e9 * safety_margin

    # Recommended batch
    recommended = int(available_mem / (p95_nodes * mem_per_node))
    recommended = max(1, min(recommended, 64))  # Clamp to [1, 64]

    print(f"\n[*] Graph Statistics:")
    print(f"    P95 nodes: {p95_nodes:.0f}")
    print(f"    Recommended batch size: {recommended}")

    return recommended


class EnhancedTaintFlowGNN(nn.Module):
    """Enhanced Taint-Flow GNN with attention."""

    def __init__(
        self,
        node_vocab_size: int = 1000,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_labels: int = 2,
        dropout: float = 0.3,
        precomputed_feature_dim: Optional[int] = None
    ):
        """
        Initialize GNN model.

        Args:
            node_vocab_size: Vocabulary size for node types
            embedding_dim: Embedding dimension
            hidden_dim: Hidden dimension
            num_layers: Number of GNN layers
            num_labels: Number of output labels
            dropout: Dropout rate
        """
        super().__init__()

        if not TORCH_GEOMETRIC_AVAILABLE:
            raise ImportError("PyTorch Geometric required")

        # Input handling
        self.using_precomputed = precomputed_feature_dim is not None
        if self.using_precomputed:
            self.input_proj = nn.Linear(precomputed_feature_dim, embedding_dim)
        else:
            self.embedding = nn.Embedding(node_vocab_size, embedding_dim)

        # GNN layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(embedding_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        self.dropout = nn.Dropout(dropout)

        self.graph_feature_dim = 0
        self.graph_feature_dropout = nn.Dropout(dropout)

        # Classification head (input size may expand when graph_features are present)
        self.classifier_in_dim = hidden_dim * 2
        self.classifier = nn.Sequential(
            nn.Linear(self.classifier_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels)
        )

    def forward(self, data):
        """
        Forward pass.

        Args:
            data: PyG Data batch

        Returns:
            Logits [batch, num_labels]
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Embed or project nodes
        if self.using_precomputed:
            x = self.input_proj(x.float())  # [num_nodes, embedding_dim]
        else:
            x = self.embedding(x.squeeze(1))  # [num_nodes, embedding_dim]

        # GNN layers
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)

        # Global pooling (mean + max)
        mean_pool = global_mean_pool(x, batch)
        max_pool = global_max_pool(x, batch)
        graph_embedding = torch.cat([mean_pool, max_pool], dim=1)
        # Concat optional graph-level features
        if hasattr(data, 'graph_features'):
            gf = data.graph_features
            if gf.ndim == 1:
                gf = gf.unsqueeze(0).repeat(graph_embedding.size(0), 1) if graph_embedding.size(0) == 1 else gf
            gf = self.graph_feature_dropout(gf)
            if gf.shape[0] == graph_embedding.shape[0]:
                # Expand classifier if not already expanded
                if self.graph_feature_dim == 0:
                    self.graph_feature_dim = gf.shape[1]
                    new_in = self.classifier_in_dim + self.graph_feature_dim
                    # Rebuild classifier to match new input dim
                    self.classifier = nn.Sequential(
                        nn.Linear(new_in, graph_embedding.shape[1] // 2),
                        nn.ReLU(),
                        nn.Dropout(self.graph_feature_dropout.p),
                        nn.Linear(graph_embedding.shape[1] // 2, 2)
                    )
                graph_embedding = torch.cat([graph_embedding, gf], dim=1)

        # Classification
        logits = self.classifier(graph_embedding)

        return logits


class S3CheckpointManager:
    """Manage checkpoints with S3 backup."""

    def __init__(
        self,
        local_dir: Path,
        s3_bucket: Optional[str] = None,
        s3_prefix: Optional[str] = None
    ):
        """Initialize checkpoint manager."""
        self.local_dir = Path(local_dir)
        self.local_dir.mkdir(parents=True, exist_ok=True)

        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix
        self.s3_client = None

        if s3_bucket and S3_AVAILABLE:
            self.s3_client = boto3.client('s3')
            print(f"[+] S3 checkpointing enabled: s3://{s3_bucket}/{s3_prefix}")

    def save_checkpoint(
        self,
        epoch: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        metrics: Dict[str, float],
        is_best: bool = False,
        extra_metadata: Optional[Dict[str, Any]] = None
    ):
        """Save checkpoint locally and to S3 (with enhanced metadata and atomic writes)."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'metrics': metrics
        }

        # PHASE 1: Add enhanced metadata
        if extra_metadata:
            checkpoint.update(extra_metadata)

        # PHASE 1: Atomic writes (tmp file → rename)
        checkpoint_path = self.local_dir / f'checkpoint_epoch_{epoch}.pt'
        tmp_checkpoint_path = checkpoint_path.with_suffix('.pt.tmp')
        torch.save(checkpoint, tmp_checkpoint_path)
        tmp_checkpoint_path.replace(checkpoint_path)

        if is_best:
            best_path = self.local_dir / 'best_model.pt'
            tmp_best_path = best_path.with_suffix('.pt.tmp')
            torch.save(checkpoint, tmp_best_path)
            tmp_best_path.replace(best_path)
            print(f"[+] Saved best model: {best_path}")

            # PHASE 1: Save metadata.json alongside checkpoint
            metadata_path = self.local_dir / 'best_model_metadata.json'
            if extra_metadata:
                with open(metadata_path, 'w') as f:
                    json.dump(extra_metadata, f, indent=2, default=str)

            if self.s3_client:
                s3_key = f"{self.s3_prefix}/best_model.pt" if self.s3_prefix else "best_model.pt"
                try:
                    self.s3_client.upload_file(str(best_path), self.s3_bucket, s3_key)
                except Exception as e:
                    print(f"[!] S3 upload failed: {e}")

        # Latest for Spot resilience
        latest_path = self.local_dir / 'latest.pt'
        tmp_latest_path = latest_path.with_suffix('.pt.tmp')
        torch.save(checkpoint, tmp_latest_path)
        tmp_latest_path.replace(latest_path)

        if self.s3_client:
            s3_key = f"{self.s3_prefix}/latest.pt" if self.s3_prefix else "latest.pt"
            try:
                self.s3_client.upload_file(str(latest_path), self.s3_bucket, s3_key)
            except:
                pass

    def load_checkpoint(self, checkpoint_name: str = 'latest.pt') -> Optional[Dict]:
        """Load checkpoint from local or S3."""
        local_path = self.local_dir / checkpoint_name

        if local_path.exists():
            return torch.load(local_path)

        if self.s3_client:
            s3_key = f"{self.s3_prefix}/{checkpoint_name}" if self.s3_prefix else checkpoint_name
            try:
                self.s3_client.download_file(self.s3_bucket, s3_key, str(local_path))
                return torch.load(local_path)
            except:
                pass

        return None


def compute_binary_f1(y_true: np.ndarray, y_pred: np.ndarray, positive_class: int = 1) -> float:
    """Compute F1 for vulnerable class."""
    _, _, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[positive_class], average=None
    )
    return f1[0]


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    criterion: Optional[nn.Module] = None
) -> Dict[str, float]:
    """Evaluate model (with prediction distribution tracking for collapse detection)."""
    model.eval()

    all_preds = []
    all_labels = []
    total_loss = 0.0

    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)
            logits = model(data)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())

            if criterion:
                loss = criterion(logits, data.y)
                total_loss += loss.item()

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', pos_label=1
    )
    binary_f1 = compute_binary_f1(all_labels, all_preds, positive_class=1)

    # PHASE 1: Track prediction distribution for collapse detection
    pred_distribution = {
        'predicted_vulnerable': int((all_preds == 1).sum()),
        'predicted_safe': int((all_preds == 0).sum()),
        'actual_vulnerable': int((all_labels == 1).sum()),
        'actual_safe': int((all_labels == 0).sum())
    }

    return {
        'loss': total_loss / len(dataloader) if criterion else 0.0,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'binary_f1_vulnerable': binary_f1,
        'prediction_distribution': pred_distribution
    }


def collect_validation_outputs(model: nn.Module, dataloader: DataLoader, device: torch.device) -> Dict[str, List[float]]:
    """Collect probabilities, predictions and labels for threshold analysis."""
    model.eval()
    probs: List[float] = []
    preds: List[int] = []
    labels: List[int] = []
    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)
            logits = model(data)
            p = torch.softmax(logits, dim=1)[:, 1]
            probs.extend(p.cpu().tolist())
            preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
            labels.extend(data.y.cpu().tolist())
    return {'probs': probs, 'preds': preds, 'labels': labels}


def run_threshold_sweep(probs: List[float], labels: List[int], thresholds: List[float]) -> Tuple[List[Dict[str, float]], Optional[Dict[str, float]]]:
    if not probs or not labels:
        return [], None
    probs_arr = np.array(probs)
    labels_arr = np.array(labels)
    results: List[Dict[str, float]] = []
    best: Optional[Dict[str, float]] = None
    for t in thresholds:
        pred = (probs_arr >= t).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(labels_arr, pred, average='binary', pos_label=1, zero_division=0)
        entry = {'threshold': float(t), 'precision': float(precision), 'recall': float(recall), 'f1': float(f1)}
        results.append(entry)
        if best is None or entry['f1'] > best['f1']:
            best = entry
    return results, best


def compute_confusion_counts(labels: List[int], preds: List[int]) -> Dict[str, int]:
    if not labels:
        return {'tn': 0, 'fp': 0, 'fn': 0, 'tp': 0}
    from sklearn.metrics import confusion_matrix
    matrix = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    return {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}


def compute_balanced_accuracy_from_counts(counts: Dict[str, int]) -> float:
    tp, fp, tn, fn = counts.get('tp', 0), counts.get('fp', 0), counts.get('tn', 0), counts.get('fn', 0)
    def _safe(num, den):
        return num / den if den > 0 else 0.0
    tpr = _safe(tp, tp + fn)
    tnr = _safe(tn, tn + fp)
    return (tpr + tnr) / 2.0


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: Optional[GradScaler] = None,
    grad_clip_norm: float = 1.0,
    amp_dtype: Optional[torch.dtype] = None,
    scheduler: Optional[Any] = None
) -> float:
    """Train for one epoch (with gradient clipping)."""
    model.train()
    total_loss = 0.0

    for data in dataloader:
        data = data.to(device)

        optimizer.zero_grad()

        if scaler is not None and amp_dtype is not None and device.type == 'cuda':
            with autocast(device_type='cuda', dtype=amp_dtype):
                logits = model(data)
                loss = criterion(logits, data.y)
        else:
            logits = model(data)
            loss = criterion(logits, data.y)

        # Apply sample weights if available
        if hasattr(data, 'weight'):
            weights = data.weight.to(device)
            loss = (loss * weights).mean()

        loss.backward()

        # PHASE 1: Gradient clipping (AMP-compatible)
        if scaler is not None:
            # If AMP is enabled, unscale before clipping
            scaler.unscale_(optimizer)
        
        if grad_clip_norm and grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)

        if scaler is not None:
            # If AMP is enabled, use scaler for optimizer step
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        total_loss += loss.item()
        # Step per-batch for step schedulers
        if scheduler is not None and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            try:
                scheduler.step()
            except Exception:
                pass

    return total_loss / len(dataloader)


def save_experiment_config(args: argparse.Namespace, output_dir: Path, data_paths: Dict[str, Path]):
    """Save experiment configuration."""
    config = {
        'timestamp': datetime.now().isoformat(),
        'git_commit': get_git_commit(),
        'seed': args.seed,
        'hyperparameters': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.lr,
            'weight_decay': args.weight_decay,
            'hidden_dim': args.hidden_dim,
            'num_layers': args.num_layers,
            'dropout': args.dropout
        },
        'dataset_checksums': {
            split: compute_file_checksum(path)
            for split, path in data_paths.items()
        }
    }

    config_path = output_dir / 'exp_config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"[+] Experiment config saved: {config_path}")


def main():
    parser = argparse.ArgumentParser(description="Train Enhanced Taint-Flow GNN")

    # Data
    parser.add_argument('--train-data', type=Path, required=True)
    parser.add_argument('--val-data', type=Path, required=True)
    parser.add_argument('--test-data', type=Path, default=None)

    # Model
    parser.add_argument('--node-vocab-size', type=int, default=1000)
    parser.add_argument('--embedding-dim', type=int, default=128)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--num-layers', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.3)

    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--early-stopping-patience', type=int, default=10)
    parser.add_argument('--use-weights', action='store_true')

    # Infrastructure
    parser.add_argument('--output-dir', type=Path, default=Path('models/gnn'))
    parser.add_argument('--s3-bucket', type=str, default=None)
    parser.add_argument('--s3-prefix', type=str, default='checkpoints/gnn')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--quick-test', action='store_true')
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--prefetch-factor', type=int, default=2)
    parser.add_argument('--persistent-workers', action='store_true')
    parser.add_argument('--drop-last', action='store_true')
    parser.add_argument('--mixed-precision', action='store_true')
    parser.add_argument('--amp-dtype', choices=['fp16', 'bf16'], default='fp16')
    parser.add_argument('--grad-clip-norm', type=float, default=1.0)
    parser.add_argument('--scheduler', choices=['plateau', 'cosine', 'none'], default='plateau')
    parser.add_argument('--auto-batch-size', action='store_true', help='Auto-detect batch size from graph stats')

    # Phase 1: LR Finder arguments
    parser.add_argument('--find-lr', action='store_true', help='Run LR Finder before training')
    parser.add_argument('--force-find-lr', action='store_true', help='Force LR Finder to run even if cached')
    parser.add_argument('--lr-override', type=float, default=None, help='Override suggested LR')
    parser.add_argument('--lr-fallback', type=float, default=1e-4, help='Fallback LR for GNN (default: 1e-4)')
    parser.add_argument('--lr-cap', type=float, default=1e-3, help='Max LR cap for GNN (default: 1e-3)')
    parser.add_argument('--lr-range-start', type=float, default=1e-6, help='LR Finder start (default: 1e-6)')
    parser.add_argument('--lr-range-end', type=float, default=1e-1, help='LR Finder end (default: 1e-1)')
    parser.add_argument('--lr-finder-max-iter', type=int, default=100, help='Max LR Finder iterations')
    parser.add_argument('--lr-finder-subsample', type=int, default=None, help='Subsample dataset for LR Finder')
    parser.add_argument('--lr-cache-max-age', type=int, default=168, help='LR cache expiry (hours, default: 168)')
    parser.add_argument('--lr-cache-key-salt', type=str, default='', help='Salt for LR cache key isolation')

    # Phase 1: Class balancing arguments
    parser.add_argument('--use-weighted-sampler', action='store_true', help='Use weighted sampler for class balance')
    parser.add_argument('--weight-multiplier', type=float, default=1.0, help='Class weight multiplier (default: 1.0)')
    parser.add_argument('--focal-loss', action='store_true', help='Use focal loss instead of CrossEntropyLoss')
    parser.add_argument('--focal-gamma', type=float, default=2.0, help='Focal loss gamma (default: 2.0)')

    # Phase 1: Safety arguments
    parser.add_argument('--min-epochs-before-collapse-check', type=int, default=2, help='Min epochs before collapse detection')

    # Encoder features (Stage 3B, opt-in)
    parser.add_argument('--encoder-features', choices=['none', 'cls', 'token'], default='none',
                        help='Use encoder-derived features (default: none). token mode not implemented here.')
    parser.add_argument('--encoder-feature-dim', type=int, default=768, help='Encoder feature dim (CLS/token)')

    args = parser.parse_args()

    if not TORCH_GEOMETRIC_AVAILABLE:
        print("[!] PyTorch Geometric required. Install: pip install torch-geometric")
        return 1

    # Set seed
    set_seed(args.seed)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[+] Using device: {device}")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load datasets
    train_dataset = GraphDataset(
        args.train_data,
        use_weights=args.use_weights,
        encoder_features=args.encoder_features,
        encoder_feature_dim=args.encoder_feature_dim
    )
    val_dataset = GraphDataset(
        args.val_data,
        use_weights=False,
        encoder_features=args.encoder_features,
        encoder_feature_dim=args.encoder_feature_dim
    )

    # Quick test
    if args.quick_test:
        print("[*] Quick test mode: using 100 samples")
        train_dataset.graphs = train_dataset.graphs[:100]
        val_dataset.graphs = val_dataset.graphs[:50]

    # Auto batch size
    if args.auto_batch_size:
        args.batch_size = recommend_batch_size(train_dataset, hidden_dim=args.hidden_dim)

    # Save experiment config
    data_paths = {'train': args.train_data, 'val': args.val_data}
    if args.test_data:
        data_paths['test'] = args.test_data
    save_experiment_config(args, args.output_dir, data_paths)

    # ========================================================================
    # PHASE 1: CLASS BALANCING (Day 2)
    # ========================================================================

    # Extract graph-level labels
    train_labels = [g.y.item() for g in train_dataset.graphs]
    class_counts = torch.bincount(torch.tensor(train_labels))

    print(f"\n[*] Calculating class weights for balanced training...")
    print(f"    Class distribution: Safe={class_counts[0]}, Vulnerable={class_counts[1]}")

    # Calculate inverse-frequency class weights
    total = len(train_labels)
    weight_safe = total / (2.0 * class_counts[0])
    weight_vulnerable = total / (2.0 * class_counts[1])

    # Apply multiplier
    if args.weight_multiplier != 1.0:
        weight_vulnerable *= args.weight_multiplier
        print(f"[*] Applying weight_multiplier={args.weight_multiplier} to vulnerable class")

    # Safety cap
    max_weight_cap = 5.0
    if weight_vulnerable > max_weight_cap:
        print(f"[!] Capping vulnerable class weight: {weight_vulnerable:.2f} → {max_weight_cap}")
        weight_vulnerable = max_weight_cap

    # Triple weighting auto-adjustment
    triple_weighting = (
        args.use_weighted_sampler and
        args.weight_multiplier > 1.0 and
        args.focal_loss
    )

    if triple_weighting:
        original_weight_multiplier = args.weight_multiplier
        args.weight_multiplier = max(1.0, args.weight_multiplier * 0.8)
        # Recalculate with adjusted multiplier
        weight_vulnerable = (total / (2.0 * class_counts[1])) * args.weight_multiplier
        weight_vulnerable = min(weight_vulnerable, max_weight_cap)
        print(f"[!] Triple weighting detected (sampler + weights + focal)")
        print(f"    Auto-adjusting weight_multiplier: {original_weight_multiplier} → {args.weight_multiplier}")
        print(f"    Adjusted vulnerable weight: {weight_vulnerable:.4f}")

    class_weights = torch.tensor([weight_safe, weight_vulnerable], dtype=torch.float32).to(device)
    print(f"[+] Class weights: Safe={weight_safe:.4f}, Vulnerable={weight_vulnerable:.4f}")

    # Create weighted sampler if requested
    sampler = None
    if args.use_weighted_sampler:
        print(f"[*] Creating weighted sampler for class balance...")
        class_weights_sampling = 1.0 / class_counts.float()
        sample_weights = torch.tensor([class_weights_sampling[label] for label in train_labels])
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_dataset),
            replacement=True
        )
        print(f"[+] Weighted sampler created (resamples minority class)")

    # Data loaders
    pin_memory = torch.cuda.is_available()
    def _loader_kwargs(drop_last: bool) -> Dict[str, Any]:
        kw: Dict[str, Any] = {
            'batch_size': args.batch_size,
            'num_workers': args.num_workers,
            'pin_memory': pin_memory,
            'drop_last': drop_last
        }
        if args.num_workers > 0:
            kw['prefetch_factor'] = args.prefetch_factor
            kw['persistent_workers'] = args.persistent_workers
        return kw

    if sampler is not None:
        # PyG DataLoader with sampler (shuffle MUST be False)
        sampler_kwargs = _loader_kwargs(drop_last=args.drop_last)
        train_loader = DataLoader(
            train_dataset,
            sampler=sampler,
            shuffle=False,  # Required when using sampler
            **sampler_kwargs
        )
        print(f"[+] Train loader: using weighted sampler (shuffle=False)")
    else:
        shuffle_kwargs = _loader_kwargs(drop_last=args.drop_last)
        train_loader = DataLoader(
            train_dataset,
            shuffle=True,
            **shuffle_kwargs
        )
        print(f"[+] Train loader: random shuffle (no sampler)")

    val_kwargs = _loader_kwargs(drop_last=False)
    val_kwargs.pop('drop_last', None)
    val_loader = DataLoader(val_dataset, shuffle=False, **val_kwargs)

    # Model
    print(f"[*] Initializing GNN model")
    model = EnhancedTaintFlowGNN(
        node_vocab_size=args.node_vocab_size,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        precomputed_feature_dim=(args.encoder_feature_dim if args.encoder_features == 'token' else None)
    )
    model.to(device)

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Scheduler
    if args.scheduler == 'cosine':
        total_steps = max(1, len(train_loader) * args.epochs)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    elif args.scheduler == 'none':
        scheduler = None
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5
        )

    # Loss function (with focal loss support)
    if args.focal_loss:
        if not FOCAL_LOSS_AVAILABLE:
            print("[!] WARNING: Focal loss not available, falling back to CrossEntropyLoss")
            criterion = nn.CrossEntropyLoss(weight=class_weights, reduction='mean')
        else:
            criterion = FocalLoss(alpha=class_weights, gamma=args.focal_gamma, reduction='mean')
            print(f"[+] Loss: FocalLoss (gamma={args.focal_gamma}, alpha={class_weights.tolist()})")
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights, reduction='mean')
        print(f"[+] Loss: CrossEntropyLoss (weights={class_weights.tolist()})")


    # Checkpoint manager
    checkpoint_mgr = S3CheckpointManager(
        args.output_dir / 'checkpoints',
        args.s3_bucket,
        args.s3_prefix
    )

    # ========================================================================
    # PHASE 1: LR FINDER (Day 1)
    # ========================================================================

    lr_finder_analysis = None  # Store for checkpoint metadata

    if args.find_lr and LR_FINDER_AVAILABLE:
        print("\n" + "="*70)
        print("RUNNING LR FINDER (GNN)")
        print("="*70)

        # Compute cache key
        model_hash = hashlib.sha256(
            json.dumps({
                'node_vocab_size': args.node_vocab_size,
                'embedding_dim': args.embedding_dim,
                'hidden_dim': args.hidden_dim,
                'num_layers': args.num_layers,
                'dropout': args.dropout
            }, sort_keys=True).encode()
        ).hexdigest()[:16]

        cache_key = compute_cache_key(
            dataset_path=args.train_data,
            model_name=f'gnn_{model_hash}',
            batch_size=args.batch_size,
            extra={
                'hidden_dim': args.hidden_dim,
                'num_layers': args.num_layers,
                'salt': args.lr_cache_key_salt
            }
        )

        # Try to load from cache
        cached_result = None
        if not args.force_find_lr:
            cached_result = load_lr_cache(cache_key, max_age_hours=args.lr_cache_max_age)

        if cached_result and not args.force_find_lr:
            print(f"[+] Loaded LR from cache (key: {cache_key[:12]}...)")
            suggested_lr = cached_result['suggested_lr']
            lr_finder_analysis = cached_result.get('metadata', {}).get('analysis', {})
            print(f"[*] Cached LR: {suggested_lr:.2e}")
        else:
            if args.force_find_lr:
                print(f"[*] Forcing LR Finder to run (--force-find-lr)")
            elif cached_result:
                print(f"[!] Cache expired, re-running LR Finder")

            # Subsample dataset if requested
            if args.lr_finder_subsample:
                print(f"[*] Using subsample: {args.lr_finder_subsample} graphs (faster LR Finder)")
                subsample_indices = list(range(min(args.lr_finder_subsample, len(train_dataset))))
                subsample_graphs = [train_dataset.graphs[i] for i in subsample_indices]
                from torch_geometric.data import InMemoryDataset
                # Create temporary dataset
                class TempDataset(Dataset):
                    def __init__(self, graphs):
                        self.graphs = graphs
                    def __len__(self):
                        return len(self.graphs)
                    def __getitem__(self, idx):
                        return self.graphs[idx]
                finder_dataset = TempDataset(subsample_graphs)
                finder_loader = DataLoader(finder_dataset, batch_size=args.batch_size, shuffle=True)
            else:
                finder_loader = train_loader
                if args.lr_finder_subsample is None and len(train_dataset) > 1000:
                    print(f"[!] WARNING: Running LR Finder on large dataset ({len(train_dataset)} graphs)")
                    print(f"    Consider using --lr-finder-subsample 256 for faster results")

            # Create fresh model and optimizer for LR Finder
            finder_model = EnhancedTaintFlowGNN(
                node_vocab_size=args.node_vocab_size,
                embedding_dim=args.embedding_dim,
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                dropout=args.dropout
            ).to(device)

            finder_optimizer = torch.optim.Adam(
                finder_model.parameters(),
                lr=args.lr_range_start,
                weight_decay=args.weight_decay
            )

            # Run LR Finder (PyG adapted)
            lr_finder = LRFinder(finder_model, finder_optimizer, criterion, device=device)
            lr_history, loss_history = lr_finder.range_test(
                finder_loader,
                start_lr=args.lr_range_start,
                end_lr=args.lr_range_end,
                num_iter=args.lr_finder_max_iter,
                step_mode='exp'
            )

            print(f"[+] LR Finder complete. Tested {len(lr_history)} learning rates")

            # Analyze curve
            lr_finder_analysis = analyze_lr_loss_curve(lr_history, loss_history)
            raw_suggested_lr = lr_finder_analysis.get('suggested_lr', args.lr_fallback)

            # Validate and cap
            validation_result = validate_and_cap_lr(
                raw_suggested_lr,
                lr_finder_analysis,
                cap=args.lr_cap,
                conservative_fallback=args.lr_fallback
            )

            suggested_lr = validation_result['final_lr']
            lr_finder_analysis['reason'] = validation_result.get('reason', [])

            print(f"\n[*] LR Finder Results:")
            print(f"    Raw suggestion: {raw_suggested_lr:.2e}")
            print(f"    Confidence: {lr_finder_analysis.get('confidence', 'unknown')}")
            print(f"    Final LR: {suggested_lr:.2e} ({','.join(validation_result.get('reason', []))})")

            if validation_result.get('used_fallback'):
                print(f"[!] WARNING: Used conservative fallback ({args.lr_fallback:.2e}) due to poor curve quality")
                reasons = ', '.join(lr_finder_analysis.get('reason', ['unknown']))
                print(f"    Reasons: {reasons}")

            # Save to cache
            save_lr_cache(
                cache_key,
                suggested_lr,
                lr_history={
                    'min_loss': float(min(loss_history)),
                    'max_loss': float(max(loss_history)),
                    'num_points': len(lr_history)
                },
                metadata={
                    'analysis': lr_finder_analysis,
                    'validation': validation_result,
                    'timestamp': datetime.now().isoformat(),
                    'model_hash': model_hash,
                    'batch_size': args.batch_size
                }
            )
            print(f"[+] LR Finder results cached (key: {cache_key[:12]}...)")

        # Apply suggested LR (unless overridden)
        if args.lr_override is not None:
            final_lr = args.lr_override
            print(f"[*] Using overridden LR: {final_lr:.2e} (--lr-override)")
        else:
            final_lr = suggested_lr
            print(f"[*] Applying suggested LR: {final_lr:.2e}")

        # Rebuild optimizer with new LR
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=final_lr,
            weight_decay=args.weight_decay
        )

        # Rebuild scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5
        )

        print(f"[+] Optimizer and scheduler rebuilt with LR={final_lr:.2e}")
        print("="*70 + "\n")

    elif args.find_lr and not LR_FINDER_AVAILABLE:
        print("[!] WARNING: --find-lr specified but LR Finder not available")
        print("    Continuing with default LR")

    # Training loop
    print("\n" + "="*70)
    print("TRAINING START")
    print("="*70)

    best_val_f1 = 0.0
    patience_counter = 0

    # PHASE 1: Collapse detection tracking
    consecutive_collapses = 0
    collapse_history = []

    # PHASE 1: CSV metrics logging
    csv_path = args.output_dir / 'metrics_history.csv'
    with open(csv_path, 'w') as f:
        f.write('epoch,train_loss,val_loss,val_acc,val_precision,val_recall,val_f1,pred_vulnerable,pred_safe\n')
    print(f"[+] CSV metrics will be saved to: {csv_path}")

    # AMP scaler/dtype
    scaler = None
    amp_dtype = None
    if args.mixed_precision and torch.cuda.is_available():
        scaler = GradScaler()
        if args.amp_dtype == 'bf16':
            try:
                major, _ = torch.cuda.get_device_capability(torch.cuda.current_device())
                amp_dtype = torch.bfloat16 if major >= 8 else torch.float16
                if major < 8:
                    print('[!] bf16 requested but not supported; falling back to fp16')
            except Exception:
                amp_dtype = torch.float16
        else:
            amp_dtype = torch.float16

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 70)

        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device,
            scaler=scaler, grad_clip_norm=args.grad_clip_norm, amp_dtype=amp_dtype,
            scheduler=scheduler if isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR) else None
        )

        # Evaluate
        val_metrics = evaluate(model, val_loader, device, criterion)

        # Extract prediction distribution
        dist = val_metrics.get('prediction_distribution', {})

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f}")
        print(f"Val Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"Val Precision: {val_metrics['precision']:.4f}")
        print(f"Val Recall: {val_metrics['recall']:.4f}")
        print(f"Val F1 (vulnerable): {val_metrics['binary_f1_vulnerable']:.4f}")
        print(f"Predictions: Vulnerable={dist.get('predicted_vulnerable', 0)}/{dist.get('actual_vulnerable', 0)}, "
              f"Safe={dist.get('predicted_safe', 0)}/{dist.get('actual_safe', 0)}")

        # PHASE 1: Collapse detection
        if epoch >= args.min_epochs_before_collapse_check:
            val_collapsed = (dist.get('predicted_vulnerable', 0) == 0 or dist.get('predicted_safe', 0) == 0)

            if val_collapsed:
                consecutive_collapses += 1
                collapse_history.append(epoch + 1)
                if dist.get('predicted_vulnerable', 0) == 0:
                    print(f"[!] COLLAPSE SIGNAL: Zero vulnerable predictions")
                else:
                    print(f"[!] COLLAPSE SIGNAL: Zero safe predictions")
            else:
                consecutive_collapses = 0

            # Stop training if 2 consecutive collapses
            if consecutive_collapses >= 2:
                print(f"\n{'='*70}")
                print(f"[!] CRITICAL: Collapse detected for {consecutive_collapses} consecutive epochs")
                print(f"[!] Collapse history: {collapse_history}")
                print(f"\n[!] STOPPING TRAINING. Recommended fixes:")
                print(f"    1. Add: --use-weighted-sampler")
                print(f"    2. Try: --lr-override {args.lr * 2:.2e}")
                print(f"    3. Increase: --weight-multiplier 2.0")
                print(f"    4. Try: --focal-loss")
                print(f"{'='*70}\n")

                # Save diagnostic report
                diagnostic_report = {
                    'collapse_epochs': collapse_history,
                    'final_epoch': epoch + 1,
                    'train_loss': train_loss,
                    'val_metrics': {k: v for k, v in val_metrics.items() if k != 'prediction_distribution'},
                    'prediction_distribution': dist,
                    'hyperparameters': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    'recommendations': [
                        'use_weighted_sampler=True',
                        f'lr_override={args.lr * 2:.2e}',
                        'weight_multiplier=2.0',
                        'focal_loss=True'
                    ]
                }

                report_path = args.output_dir / 'collapse_report.json'
                with open(report_path, 'w') as f:
                    json.dump(diagnostic_report, f, indent=2, default=str)
                print(f"[+] Saved diagnostic report: {report_path}")

                # Also save human-readable report
                txt_report_path = args.output_dir / 'collapse_report.txt'
                with open(txt_report_path, 'w') as f:
                    f.write(f"COLLAPSE DETECTION REPORT\n")
                    f.write(f"=" * 70 + "\n\n")
                    f.write(f"Collapse detected at epochs: {collapse_history}\n")
                    f.write(f"Final epoch: {epoch + 1}\n")
                    f.write(f"Train loss: {train_loss:.4f}\n")
                    f.write(f"Val F1: {val_metrics['binary_f1_vulnerable']:.4f}\n\n")
                    f.write(f"Prediction distribution:\n")
                    for k, v in dist.items():
                        f.write(f"  {k}: {v}\n")
                    f.write(f"\nRecommended fixes:\n")
                    for rec in diagnostic_report['recommendations']:
                        f.write(f"  - {rec}\n")
                print(f"[+] Saved human-readable report: {txt_report_path}")

                sys.exit(2)  # Exit code 2 = collapse failure

        # PHASE 1: CSV metrics logging
        with open(csv_path, 'a') as f:
            f.write(f"{epoch+1},{train_loss:.4f},{val_metrics['loss']:.4f},"
                    f"{val_metrics['accuracy']:.4f},{val_metrics['precision']:.4f},"
                    f"{val_metrics['recall']:.4f},{val_metrics['binary_f1_vulnerable']:.4f},"
                    f"{dist.get('predicted_vulnerable', 0)},{dist.get('predicted_safe', 0)}\n")
            f.flush()

        # Step scheduler
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_metrics['binary_f1_vulnerable'])

        # PHASE 1: Prepare enhanced metadata for checkpoint
        enhanced_metadata = {
            'seed': args.seed,
            'git_commit': get_git_commit(),
            'timestamp': datetime.now().isoformat(),
            'lr_finder_used': bool(args.find_lr and LR_FINDER_AVAILABLE),
            'suggested_lr': float(optimizer.param_groups[0]['lr']),
            'triple_weighting': triple_weighting,
            'weighted_sampler': args.use_weighted_sampler,
            'focal_loss': args.focal_loss,
            'weight_multiplier': args.weight_multiplier,
            'class_weights': class_weights.cpu().tolist(),
            'prediction_distribution': dist,
            'collapse_history': collapse_history,
            'consecutive_collapses': consecutive_collapses
        }

        if lr_finder_analysis:
            enhanced_metadata['lr_finder_analysis'] = lr_finder_analysis

        # Save checkpoint
        is_best = val_metrics['binary_f1_vulnerable'] > best_val_f1
        checkpoint_mgr.save_checkpoint(
            epoch + 1, model, optimizer, scheduler, val_metrics, is_best,
            extra_metadata=enhanced_metadata
        )

        if is_best:
            best_val_f1 = val_metrics['binary_f1_vulnerable']
            patience_counter = 0
            print(f"[+] New best model! F1: {best_val_f1:.4f}")
        else:
            patience_counter += 1
            print(f"[*] No improvement. Patience: {patience_counter}/{args.early_stopping_patience}")

        # Early stopping
        if patience_counter >= args.early_stopping_patience:
            print(f"\n[!] Early stopping at epoch {epoch + 1}")
            break

    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Best validation F1 (vulnerable): {best_val_f1:.4f}")

    # Save extended validation metrics JSON
    best_checkpoint = checkpoint_mgr.load_checkpoint('best_model.pt')
    if best_checkpoint:
        model.load_state_dict(best_checkpoint['model_state_dict'])

    val_outputs = collect_validation_outputs(model, val_loader, device)
    default_threshold = 0.5
    sweep, best = run_threshold_sweep(val_outputs['probs'], val_outputs['labels'], thresholds=np.linspace(0.30, 0.70, 41).tolist())
    best_threshold = best['threshold'] if best else default_threshold

    default_preds = (np.array(val_outputs['probs']) >= default_threshold).astype(int).tolist()
    cm_default = compute_confusion_counts(val_outputs['labels'], default_preds)
    cm_best = None
    if best:
        best_preds = (np.array(val_outputs['probs']) >= best_threshold).astype(int).tolist()
        cm_best = compute_confusion_counts(val_outputs['labels'], best_preds)
    balanced_at_best = compute_balanced_accuracy_from_counts(cm_best) if cm_best else None

    from sklearn.metrics import f1_score
    metrics_payload = {
        'best_f1_vulnerable': float(best_val_f1),
        'f1_at_0_5': float(f1_score(val_outputs['labels'], default_preds)),
        'f1_at_best_threshold': (float(best['f1']) if best else None),
        'best_threshold_by_f1': float(best_threshold) if best else None,
        'balanced_accuracy_at_best_threshold': float(balanced_at_best) if balanced_at_best is not None else None,
        'threshold_sweep': sweep,
        'confusion_matrix_threshold_0_5': cm_default,
        'confusion_matrix_best_threshold': cm_best,
        'timestamp': datetime.now().isoformat()
    }
    metrics_path = args.output_dir / 'metrics.json'
    with open(metrics_path, 'w', encoding='utf-8') as mf:
        json.dump(metrics_payload, mf, indent=2)
    print(f"[ok] Metrics saved to: {metrics_path}")

    # Test evaluation
    if args.test_data:
        print("\n" + "="*70)
        print("TEST EVALUATION")
        print("="*70)

        test_dataset = GraphDataset(args.test_data, use_weights=False)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        # Load best model
        best_checkpoint = checkpoint_mgr.load_checkpoint('best_model.pt')
        if best_checkpoint:
            model.load_state_dict(best_checkpoint['model_state_dict'])

        test_metrics = evaluate(model, test_loader, device, criterion)

        print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"Test Precision: {test_metrics['precision']:.4f}")
        print(f"Test Recall: {test_metrics['recall']:.4f}")
        print(f"Test F1: {test_metrics['f1']:.4f}")
        print(f"Test F1 (vulnerable): {test_metrics['binary_f1_vulnerable']:.4f}")

    print(f"\n[+] Model saved to: {args.output_dir}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
