"""
Cell 52: GNN v1.7 Phase-1 Production Training

Features:
- PyTorch Geometric data loading
- Weighted graph sampler for class imbalance
- Focal Loss (γ=1.5) for hard negatives
- Mixed precision (AMP)
- AMP-safe gradient clipping
- Collapse detection
- Graph statistics logging
- 3-seed reproducibility (42, 2025, 7)

Story Points: 8
Status: Production-Ready
"""

import sys
import os
from pathlib import Path
import warnings

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Core imports
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from datetime import datetime
from tqdm.auto import tqdm

# PyTorch Geometric imports
try:
    from torch_geometric.data import Data, DataLoader
    from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    warnings.warn("PyTorch Geometric not available. Install: pip install torch-geometric")

# StreamGuard imports
from training.train_gnn import (
    EnhancedTaintFlowGNN,
    set_seed
)
from training.utils.adaptive_config import load_adaptive_config
from training.utils.json_safety import atomic_write_json
from training.utils.collapse_detector import CollapseDetector
from training.utils.amp_utils import clip_gradients_amp_safe
from training.utils.lr_finder import LRFinder
from training.utils.lr_cache import compute_cache_key, save_lr_cache, load_lr_cache

from sklearn.metrics import precision_recall_fscore_support, accuracy_score


# Check PyG availability
if not TORCH_GEOMETRIC_AVAILABLE:
    print("[!] PyTorch Geometric not available. Exiting.")
    sys.exit(1)


# ==================== CONFIGURATION ====================

print("=" * 80)
print("CELL 52: GNN v1.7 PHASE-1 PRODUCTION TRAINING")
print("=" * 80)

# Production seeds
SEEDS = [42, 2025, 7]

# Output directory
OUTPUT_DIR = Path("training/outputs/gnn_v17_production")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Model configuration
NUM_LABELS = 2
NODE_FEATURE_DIM = 768  # Assuming CodeBERT embeddings
HIDDEN_DIM = 256

# Training hyperparameters
BASE_CONFIG = {
    "batch_size": 64,  # Number of graphs per batch
    "num_epochs": 15,
    "learning_rate": None,  # Will be found by LR Finder
    "weight_decay": 1e-4,
    "warmup_ratio": 0.10,
    "max_grad_norm": 1.0,
    "mixed_precision": True,
    "focal_loss_gamma": 1.5  # Focal Loss parameter
}

# Dataset paths
TRAIN_DATA_PATH = Path("data/processed/graphs/train")
VAL_DATA_PATH = Path("data/processed/graphs/val")


# ==================== FOCAL LOSS ====================

class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance and hard negatives.

    Reference: Lin et al., "Focal Loss for Dense Object Detection" (2017)

    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    Args:
        gamma: Focusing parameter (default: 1.5)
        alpha: Class weights (optional)
    """

    def __init__(self, gamma=1.5, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, labels):
        """
        Compute focal loss.

        Args:
            logits: Model predictions (batch_size, num_classes)
            labels: Ground truth (batch_size,)

        Returns:
            Focal loss value
        """
        # Cross entropy loss
        ce_loss = nn.functional.cross_entropy(logits, labels, reduction='none')

        # Get probabilities
        probs = torch.softmax(logits, dim=1)
        p_t = probs.gather(1, labels.unsqueeze(1)).squeeze(1)

        # Focal term: (1 - p_t)^gamma
        focal_term = (1 - p_t) ** self.gamma

        # Focal loss
        loss = focal_term * ce_loss

        # Apply alpha if provided
        if self.alpha is not None:
            alpha_t = self.alpha[labels]
            loss = alpha_t * loss

        return loss.mean()


# ==================== GRAPH DATA LOADING ====================

def load_graph_dataset(data_path: Path):
    """Load graph dataset from disk."""
    if not data_path.exists():
        warnings.warn(f"Data path not found: {data_path}. Using dummy data.")
        return []

    # Load graph files
    graph_files = list(data_path.glob("*.pt"))
    graphs = []

    for file in tqdm(graph_files, desc=f"Loading {data_path.name}"):
        try:
            graph = torch.load(file)
            graphs.append(graph)
        except Exception as e:
            warnings.warn(f"Failed to load {file}: {e}")

    return graphs


def create_weighted_graph_sampler(dataset):
    """Create weighted sampler for graph class imbalance."""
    # Extract labels
    labels = np.array([graph.y.item() for graph in dataset])

    # Count classes
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts

    # Sample weights
    sample_weights = class_weights[labels]

    from torch.utils.data import WeightedRandomSampler
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )


def compute_graph_statistics(dataset):
    """Compute statistics of graph dataset."""
    num_nodes = [graph.num_nodes for graph in dataset]
    num_edges = [graph.num_edges for graph in dataset]

    stats = {
        "num_graphs": len(dataset),
        "avg_nodes": float(np.mean(num_nodes)),
        "std_nodes": float(np.std(num_nodes)),
        "max_nodes": int(np.max(num_nodes)),
        "min_nodes": int(np.min(num_nodes)),
        "avg_edges": float(np.mean(num_edges)),
        "std_edges": float(np.std(num_edges)),
        "max_edges": int(np.max(num_edges)),
        "min_edges": int(np.min(num_edges))
    }

    return stats


# ==================== LR FINDER FOR GRAPHS ====================

def run_lr_finder_pyg(model, train_loader, device, cache_key):
    """Run LR Finder for PyG graphs with caching."""
    print("\n[*] Checking LR Finder cache...")

    # Check cache
    cached = load_lr_cache(cache_key, max_age_hours=168)
    if cached:
        print(f"[+] Using cached LR: {cached['suggested_lr']:.2e}")
        return cached['suggested_lr']

    print("[*] Running LR Finder for GNN (quick mode)...")

    # Optimizer and criterion
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-7)
    criterion = FocalLoss(gamma=BASE_CONFIG['focal_loss_gamma'])

    # LR Finder
    lr_history = []
    loss_history = []

    model.train()
    lr = 1e-7
    end_lr = 1e-2
    num_iter = 100
    lr_mult = (end_lr / lr) ** (1 / num_iter)

    train_iter = iter(train_loader)

    for iteration in range(num_iter):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        # Update LR
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Forward pass
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch)
        loss = criterion(logits, batch.y)

        # Backward
        loss.backward()
        optimizer.step()

        # Track
        lr_history.append(lr)
        loss_history.append(loss.item())

        # Increase LR
        lr *= lr_mult

    # Find best LR (steepest descent)
    gradients = np.gradient(loss_history)
    best_idx = np.argmin(gradients)
    suggested_lr = lr_history[best_idx]

    # Cap to safe range (1e-4 to 1e-3 for GNNs)
    suggested_lr = max(1e-4, min(suggested_lr, 1e-3))

    print(f"[+] Suggested LR: {suggested_lr:.2e}")

    # Cache
    save_lr_cache(
        cache_key,
        suggested_lr,
        {
            "min_loss": float(min(loss_history)),
            "max_loss": float(max(loss_history)),
            "num_points": len(lr_history)
        },
        {"confidence": "high", "mode": "quick_pyg"}
    )

    return suggested_lr


# ==================== TRAINING FUNCTIONS ====================

def train_one_epoch(
    model, train_loader, optimizer, criterion, scaler,
    scheduler, collapse_detector, device, epoch
):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")

    for batch_idx, batch in enumerate(pbar):
        batch = batch.to(device)

        # Forward pass with AMP
        optimizer.zero_grad()

        with autocast(enabled=scaler is not None):
            logits = model(batch)
            loss = criterion(logits, batch.y)

        # Backward
        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Gradient clipping
        grad_stats = clip_gradients_amp_safe(
            model,
            max_grad_norm=BASE_CONFIG['max_grad_norm'],
            scaler=scaler,
            verbose=False
        )

        # Optimizer step
        if scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        scheduler.step()

        # Collapse detection
        collapse_status = collapse_detector.step(
            model,
            loss.item(),
            logits,
            batch.y,
            step=epoch * len(train_loader) + batch_idx
        )

        if collapse_status['should_stop']:
            print("\n[!] Model collapse detected!")
            return None

        total_loss += loss.item()
        num_batches += 1

        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'grad_norm': f"{grad_stats['total_norm']:.4f}"
        })

    return total_loss / num_batches


def evaluate(model, val_loader, criterion, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            batch = batch.to(device)

            logits = model(batch)
            loss = criterion(logits, batch.y)

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(batch.y.detach().cpu().numpy())

            total_loss += loss.item()

    # Metrics
    avg_loss = total_loss / len(val_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0
    )

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# ==================== MAIN TRAINING ====================

def main():
    """Main GNN production training."""

    # Load config
    config = load_adaptive_config(model_type="gnn", override=BASE_CONFIG)

    print(f"\n[+] Configuration:")
    print(f"    GPU: {config['gpu_info']['full_name']}")
    print(f"    Batch size: {config['batch_size']}")
    print(f"    Focal Loss γ: {config['focal_loss_gamma']}")

    device = torch.device(config['gpu_info']['device'])

    # Load datasets
    print(f"[+] Loading graph datasets...")
    train_dataset = load_graph_dataset(TRAIN_DATA_PATH)
    val_dataset = load_graph_dataset(VAL_DATA_PATH)

    if len(train_dataset) == 0:
        print("[!] No training data found. Exiting.")
        return

    print(f"    Train graphs: {len(train_dataset)}")
    print(f"    Val graphs: {len(val_dataset)}")

    # Graph statistics
    train_stats = compute_graph_statistics(train_dataset)
    print(f"\n[+] Graph statistics:")
    print(f"    Avg nodes: {train_stats['avg_nodes']:.1f} ± {train_stats['std_nodes']:.1f}")
    print(f"    Avg edges: {train_stats['avg_edges']:.1f} ± {train_stats['std_edges']:.1f}")

    # Train for each seed
    results_all_seeds = []

    for seed in SEEDS:
        print(f"\n{'=' * 80}")
        print(f"TRAINING WITH SEED: {seed}")
        print(f"{'=' * 80}")

        set_seed(seed)

        seed_output_dir = OUTPUT_DIR / f"seed_{seed}"
        seed_output_dir.mkdir(parents=True, exist_ok=True)

        # Save graph stats
        atomic_write_json(train_stats, seed_output_dir / "graph_statistics.json")

        # Data loaders
        train_sampler = create_weighted_graph_sampler(train_dataset)
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            sampler=train_sampler
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False
        )

        # Initialize model
        print(f"\n[+] Initializing GNN...")
        model = EnhancedTaintFlowGNN(
            node_feature_dim=NODE_FEATURE_DIM,
            hidden_dim=HIDDEN_DIM,
            num_labels=NUM_LABELS,
            dropout=0.1
        ).to(device)

        # LR Finder
        cache_key = compute_cache_key(
            TRAIN_DATA_PATH,
            "gnn_taint_flow",
            config['batch_size'],
            {'seed': seed}
        )
        learning_rate = run_lr_finder_pyg(model, train_loader, device, cache_key)
        config['learning_rate'] = learning_rate

        # Reinitialize model
        model = EnhancedTaintFlowGNN(
            node_feature_dim=NODE_FEATURE_DIM,
            hidden_dim=HIDDEN_DIM,
            num_labels=NUM_LABELS,
            dropout=0.1
        ).to(device)

        # Optimizer & scheduler
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=config['weight_decay']
        )

        num_training_steps = len(train_loader) * config['num_epochs']
        num_warmup_steps = int(num_training_steps * config['warmup_ratio'])

        from transformers import get_linear_schedule_with_warmup
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )

        # Loss (Focal Loss)
        criterion = FocalLoss(gamma=config['focal_loss_gamma'])

        # GradScaler
        scaler = GradScaler() if config['mixed_precision'] else None

        # Collapse detector
        collapse_detector = CollapseDetector(
            window_size=5,
            collapse_threshold=3,
            enable_auto_stop=True,
            report_path=seed_output_dir / "collapse_report.json"
        )

        # Training loop
        print(f"\n[+] Starting training...")
        best_f1 = 0.0
        metrics_history = []

        for epoch in range(config['num_epochs']):
            print(f"\n--- Epoch {epoch+1}/{config['num_epochs']} ---")

            train_loss = train_one_epoch(
                model, train_loader, optimizer, criterion, scaler,
                scheduler, collapse_detector, device, epoch
            )

            if train_loss is None:
                break

            val_metrics = evaluate(model, val_loader, criterion, device)

            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val F1: {val_metrics['f1']:.4f}")

            metrics_history.append({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                **val_metrics
            })

            # Save best
            if val_metrics['f1'] > best_f1:
                best_f1 = val_metrics['f1']
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_state_dict': scaler.state_dict() if scaler else None,
                    'best_f1': best_f1,
                    'config': config
                }, seed_output_dir / "model_checkpoint.pt")

        # Save metadata
        metadata = {
            "seed": seed,
            "model": "gnn_v17",
            "config": config,
            "graph_stats": train_stats,
            "best_f1": best_f1,
            "metrics_history": metrics_history,
            "timestamp": datetime.now().isoformat()
        }
        atomic_write_json(metadata, seed_output_dir / "training_metadata.json")

        results_all_seeds.append({"seed": seed, "best_f1": best_f1})

        print(f"\n[+] Seed {seed} complete. Best F1: {best_f1:.4f}")

    # Summary
    print(f"\n{'=' * 80}")
    print("GNN TRAINING COMPLETE")
    print(f"{'=' * 80}")

    for result in results_all_seeds:
        print(f"Seed {result['seed']}: F1 = {result['best_f1']:.4f}")

    mean_f1 = np.mean([r['best_f1'] for r in results_all_seeds])
    std_f1 = np.std([r['best_f1'] for r in results_all_seeds])
    print(f"\nMean F1: {mean_f1:.4f} ± {std_f1:.4f}")

    summary = {
        "model": "gnn_v17",
        "seeds": SEEDS,
        "results": results_all_seeds,
        "mean_f1": mean_f1,
        "std_f1": std_f1,
        "timestamp": datetime.now().isoformat()
    }
    atomic_write_json(summary, OUTPUT_DIR / "production_summary.json")

    print(f"\n[+] Outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
