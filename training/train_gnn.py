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
from torch.utils.data import Dataset

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
        use_weights: bool = False
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
        self.graphs = []
        self.node_counts = []
        self.edge_counts = []

        print(f"[*] Loading graph dataset from {data_path}")

        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                sample = json.loads(line.strip())

                # Extract graph components
                ast_nodes = sample.get('ast_nodes', [])
                edge_index = sample.get('edge_index', [])
                label = sample.get('label', 0)
                weight = sample.get('weight', 1.0) if use_weights else 1.0

                if len(ast_nodes) == 0:
                    continue  # Skip empty graphs

                # Convert to tensors
                # ast_nodes are node type IDs
                x = torch.tensor(ast_nodes, dtype=torch.long).unsqueeze(1)  # [num_nodes, 1]

                # edge_index
                if len(edge_index) > 0:
                    edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
                else:
                    # Empty graph - create self-loop
                    edge_index_tensor = torch.tensor([[0], [0]], dtype=torch.long)

                y = torch.tensor([label], dtype=torch.long)

                # Create PyG Data object
                graph_data = Data(
                    x=x,
                    edge_index=edge_index_tensor,
                    y=y
                )

                # Store weight as attribute
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
        dropout: float = 0.3
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

        # Node embedding
        self.embedding = nn.Embedding(node_vocab_size, embedding_dim)

        # GNN layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(embedding_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        self.dropout = nn.Dropout(dropout)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # *2 for mean+max pooling
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

        # Embed nodes
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
        is_best: bool = False
    ):
        """Save checkpoint locally and to S3."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'metrics': metrics
        }

        # Save locally
        checkpoint_path = self.local_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)

        if is_best:
            best_path = self.local_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            print(f"[+] Saved best model: {best_path}")

            if self.s3_client:
                s3_key = f"{self.s3_prefix}/best_model.pt" if self.s3_prefix else "best_model.pt"
                try:
                    self.s3_client.upload_file(str(best_path), self.s3_bucket, s3_key)
                except Exception as e:
                    print(f"[!] S3 upload failed: {e}")

        # Latest for Spot resilience
        latest_path = self.local_dir / 'latest.pt'
        torch.save(checkpoint, latest_path)

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
    """Evaluate model."""
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

    return {
        'loss': total_loss / len(dataloader) if criterion else 0.0,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'binary_f1_vulnerable': binary_f1
    }


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0

    for data in dataloader:
        data = data.to(device)

        optimizer.zero_grad()

        logits = model(data)
        loss = criterion(logits, data.y)

        # Apply sample weights if available
        if hasattr(data, 'weight'):
            weights = data.weight.to(device)
            loss = (loss * weights).mean()

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

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
    parser.add_argument('--auto-batch-size', action='store_true', help='Auto-detect batch size from graph stats')

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
    train_dataset = GraphDataset(args.train_data, args.use_weights)
    val_dataset = GraphDataset(args.val_data, use_weights=False)

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

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Model
    print(f"[*] Initializing GNN model")
    model = EnhancedTaintFlowGNN(
        node_vocab_size=args.node_vocab_size,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout
    )
    model.to(device)

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )

    # Loss
    criterion = nn.CrossEntropyLoss(reduction='none')

    # Checkpoint manager
    checkpoint_mgr = S3CheckpointManager(
        args.output_dir / 'checkpoints',
        args.s3_bucket,
        args.s3_prefix
    )

    # Training loop
    print("\n" + "="*70)
    print("TRAINING START")
    print("="*70)

    best_val_f1 = 0.0
    patience_counter = 0

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 70)

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)

        # Evaluate
        val_metrics = evaluate(model, val_loader, device, criterion)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f}")
        print(f"Val Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"Val F1 (vulnerable): {val_metrics['binary_f1_vulnerable']:.4f}")

        # Step scheduler
        scheduler.step(val_metrics['binary_f1_vulnerable'])

        # Save checkpoint
        is_best = val_metrics['binary_f1_vulnerable'] > best_val_f1
        checkpoint_mgr.save_checkpoint(
            epoch + 1, model, optimizer, scheduler, val_metrics, is_best
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
