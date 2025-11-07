"""
Fusion Layer Training with Out-of-Fold Predictions

Combines Enhanced SQL Intent Transformer + Enhanced Taint-Flow GNN predictions
with proper data leakage prevention via out-of-fold (OOF) predictions.

Features:
- 5-fold CV for OOF prediction generation
- Weighted averaging fusion strategy
- S3 checkpointing
- Reproducibility tracking
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
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import KFold
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report
)

# Import training modules
try:
    from train_transformer import (
        EnhancedSQLIntentTransformer, CodeDataset as TransformerDataset,
        S3CheckpointManager, set_seed, get_git_commit, compute_file_checksum
    )
    from train_gnn import (
        EnhancedTaintFlowGNN, GraphDataset as GNNDataset
    )
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"[!] Import error: {e}")
    print("    Ensure train_transformer.py and train_gnn.py are in the same directory")
    IMPORTS_AVAILABLE = False


class FusionLayer(nn.Module):
    """
    Fusion layer for combining Transformer + GNN predictions.

    Uses weighted averaging with learned weights.
    """

    def __init__(
        self,
        num_labels: int = 2,
        fusion_strategy: str = "learned_weights"
    ):
        """
        Initialize fusion layer.

        Args:
            num_labels: Number of output labels
            fusion_strategy: 'learned_weights' or 'average'
        """
        super().__init__()

        self.fusion_strategy = fusion_strategy
        self.num_labels = num_labels

        if fusion_strategy == "learned_weights":
            # Learnable weights for transformer and GNN
            self.transformer_weight = nn.Parameter(torch.tensor(0.5))
            self.gnn_weight = nn.Parameter(torch.tensor(0.5))

            # Optional: Small MLP for combining features
            self.fusion_mlp = nn.Sequential(
                nn.Linear(num_labels * 2, num_labels * 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(num_labels * 2, num_labels)
            )
        else:
            # Simple averaging
            pass

    def forward(
        self,
        transformer_logits: torch.Tensor,
        gnn_logits: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse predictions.

        Args:
            transformer_logits: [batch, num_labels]
            gnn_logits: [batch, num_labels]

        Returns:
            Fused logits [batch, num_labels]
        """
        if self.fusion_strategy == "learned_weights":
            # Normalize weights to sum to 1
            total_weight = torch.abs(self.transformer_weight) + torch.abs(self.gnn_weight)
            w_transformer = torch.abs(self.transformer_weight) / total_weight
            w_gnn = torch.abs(self.gnn_weight) / total_weight

            # Weighted combination
            weighted = w_transformer * transformer_logits + w_gnn * gnn_logits

            # Optional: Pass through MLP
            combined = torch.cat([transformer_logits, gnn_logits], dim=1)
            mlp_output = self.fusion_mlp(combined)

            # Final fusion (50/50 weighted vs MLP)
            return 0.5 * weighted + 0.5 * mlp_output

        else:
            # Simple average
            return 0.5 * (transformer_logits + gnn_logits)


class OOFPredictionGenerator:
    """Generate out-of-fold predictions to prevent data leakage."""

    def __init__(
        self,
        transformer_checkpoint: Path,
        gnn_checkpoint: Path,
        device: torch.device,
        n_folds: int = 5,
        seed: int = 42
    ):
        """
        Initialize OOF generator.

        Args:
            transformer_checkpoint: Path to trained transformer checkpoint
            gnn_checkpoint: Path to trained GNN checkpoint
            device: Device
            n_folds: Number of CV folds
            seed: Random seed
        """
        self.transformer_checkpoint = transformer_checkpoint
        self.gnn_checkpoint = gnn_checkpoint
        self.device = device
        self.n_folds = n_folds
        self.seed = seed

        # Load base models (will be re-initialized for each fold)
        print(f"[*] Loading base model checkpoints...")

        # Transformer
        transformer_ckpt = torch.load(transformer_checkpoint, map_location=device)
        self.transformer_config = {
            'model_name': 'microsoft/codebert-base',
            'hidden_dim': 768,
            'dropout': 0.1
        }

        # GNN
        gnn_ckpt = torch.load(gnn_checkpoint, map_location=device)
        self.gnn_config = {
            'node_vocab_size': 1000,
            'embedding_dim': 128,
            'hidden_dim': 256,
            'num_layers': 4,
            'dropout': 0.3
        }

        print(f"[+] Base model configurations loaded")

    def generate_oof_predictions(
        self,
        train_data_path: Path,
        tokenizer: Any
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate OOF predictions for training data.

        Args:
            train_data_path: Path to training data
            tokenizer: Tokenizer for transformer

        Returns:
            Tuple of (oof_transformer_logits, oof_gnn_logits, true_labels)
        """
        print(f"\n[*] Generating OOF predictions with {self.n_folds}-fold CV")

        # Load datasets
        transformer_dataset = TransformerDataset(
            train_data_path, tokenizer, max_seq_len=512, use_weights=False
        )
        gnn_dataset = GNNDataset(train_data_path, use_weights=False)

        n_samples = len(transformer_dataset)
        print(f"    Total samples: {n_samples}")

        # Initialize OOF arrays
        oof_transformer_logits = np.zeros((n_samples, 2))
        oof_gnn_logits = np.zeros((n_samples, 2))
        true_labels = np.zeros(n_samples)

        # K-Fold split
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)

        for fold, (train_idx, val_idx) in enumerate(kfold.split(range(n_samples))):
            print(f"\n[*] Fold {fold + 1}/{self.n_folds}")
            print(f"    Train: {len(train_idx)}, Val: {len(val_idx)}")

            # Create fold datasets
            fold_train_transformer = Subset(transformer_dataset, train_idx)
            fold_val_transformer = Subset(transformer_dataset, val_idx)
            fold_train_gnn = Subset(gnn_dataset, train_idx)
            fold_val_gnn = Subset(gnn_dataset, val_idx)

            # Train Transformer on this fold (fine-tune from base)
            print(f"    Training Transformer...")
            transformer_model = self._train_transformer_fold(
                fold_train_transformer, fold_val_transformer
            )

            # Train GNN on this fold
            print(f"    Training GNN...")
            gnn_model = self._train_gnn_fold(
                fold_train_gnn, fold_val_gnn
            )

            # Generate predictions for validation fold
            print(f"    Generating predictions for fold {fold + 1}...")

            # Transformer predictions
            val_transformer_loader = DataLoader(
                fold_val_transformer, batch_size=32, shuffle=False
            )
            fold_transformer_logits = self._predict_transformer(
                transformer_model, val_transformer_loader
            )

            # GNN predictions
            from torch_geometric.data import DataLoader as PyGDataLoader
            val_gnn_loader = PyGDataLoader(
                fold_val_gnn, batch_size=32, shuffle=False
            )
            fold_gnn_logits = self._predict_gnn(
                gnn_model, val_gnn_loader
            )

            # Store OOF predictions
            oof_transformer_logits[val_idx] = fold_transformer_logits
            oof_gnn_logits[val_idx] = fold_gnn_logits

            # Store labels
            for i, idx in enumerate(val_idx):
                true_labels[idx] = transformer_dataset[idx]['label'].item()

        print(f"\n[+] OOF prediction generation complete")

        return oof_transformer_logits, oof_gnn_logits, true_labels

    def _train_transformer_fold(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
        epochs: int = 3
    ) -> nn.Module:
        """Train transformer on one fold (quick fine-tune)."""
        model = EnhancedSQLIntentTransformer(**self.transformer_config)
        model.to(self.device)

        # Load pre-trained weights
        checkpoint = torch.load(self.transformer_checkpoint, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])

        # Quick fine-tune (3 epochs)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        criterion = nn.CrossEntropyLoss()

        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

        model.train()
        for epoch in range(epochs):
            for batch in train_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                optimizer.zero_grad()
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

        return model

    def _train_gnn_fold(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
        epochs: int = 10
    ) -> nn.Module:
        """Train GNN on one fold (quick fine-tune)."""
        model = EnhancedTaintFlowGNN(**self.gnn_config)
        model.to(self.device)

        # Load pre-trained weights
        checkpoint = torch.load(self.gnn_checkpoint, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])

        # Quick fine-tune
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss()

        from torch_geometric.data import DataLoader as PyGDataLoader
        train_loader = PyGDataLoader(train_dataset, batch_size=32, shuffle=True)

        model.train()
        for epoch in range(epochs):
            for data in train_loader:
                data = data.to(self.device)

                optimizer.zero_grad()
                logits = model(data)
                loss = criterion(logits, data.y)
                loss.backward()
                optimizer.step()

        return model

    def _predict_transformer(
        self,
        model: nn.Module,
        dataloader: DataLoader
    ) -> np.ndarray:
        """Get transformer predictions."""
        model.eval()
        all_logits = []

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                logits = model(input_ids, attention_mask)
                all_logits.append(logits.cpu().numpy())

        return np.vstack(all_logits)

    def _predict_gnn(
        self,
        model: nn.Module,
        dataloader
    ) -> np.ndarray:
        """Get GNN predictions."""
        model.eval()
        all_logits = []

        with torch.no_grad():
            for data in dataloader:
                data = data.to(self.device)

                logits = model(data)
                all_logits.append(logits.cpu().numpy())

        return np.vstack(all_logits)


def train_fusion_model(
    oof_transformer_logits: np.ndarray,
    oof_gnn_logits: np.ndarray,
    true_labels: np.ndarray,
    val_transformer_logits: np.ndarray,
    val_gnn_logits: np.ndarray,
    val_labels: np.ndarray,
    output_dir: Path,
    epochs: int = 20,
    lr: float = 1e-3,
    device: torch.device = None
) -> FusionLayer:
    """
    Train fusion layer on OOF predictions.

    Args:
        oof_transformer_logits: OOF transformer logits [N, 2]
        oof_gnn_logits: OOF GNN logits [N, 2]
        true_labels: True labels [N]
        val_transformer_logits: Validation transformer logits
        val_gnn_logits: Validation GNN logits
        val_labels: Validation labels
        output_dir: Output directory
        epochs: Training epochs
        lr: Learning rate
        device: Device

    Returns:
        Trained fusion layer
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n[*] Training fusion layer")
    print(f"    Training samples: {len(true_labels)}")
    print(f"    Validation samples: {len(val_labels)}")

    # Create model
    fusion_model = FusionLayer(num_labels=2, fusion_strategy="learned_weights")
    fusion_model.to(device)

    # Optimizer
    optimizer = torch.optim.Adam(fusion_model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Convert to tensors
    train_trans = torch.tensor(oof_transformer_logits, dtype=torch.float32)
    train_gnn = torch.tensor(oof_gnn_logits, dtype=torch.float32)
    train_labels = torch.tensor(true_labels, dtype=torch.long)

    val_trans = torch.tensor(val_transformer_logits, dtype=torch.float32)
    val_gnn = torch.tensor(val_gnn_logits, dtype=torch.float32)
    val_labels_tensor = torch.tensor(val_labels, dtype=torch.long)

    # Training loop
    best_val_f1 = 0.0

    for epoch in range(epochs):
        # Train
        fusion_model.train()

        train_trans_batch = train_trans.to(device)
        train_gnn_batch = train_gnn.to(device)
        train_labels_batch = train_labels.to(device)

        optimizer.zero_grad()
        logits = fusion_model(train_trans_batch, train_gnn_batch)
        loss = criterion(logits, train_labels_batch)
        loss.backward()
        optimizer.step()

        # Validate
        fusion_model.eval()
        with torch.no_grad():
            val_trans_batch = val_trans.to(device)
            val_gnn_batch = val_gnn.to(device)
            val_logits = fusion_model(val_trans_batch, val_gnn_batch)
            val_preds = torch.argmax(val_logits, dim=1).cpu().numpy()

        # Metrics
        _, _, val_f1, _ = precision_recall_fscore_support(
            val_labels, val_preds, average='binary', pos_label=1
        )

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {loss.item():.4f}, Val F1: {val_f1:.4f}")

        # Save best
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(fusion_model.state_dict(), output_dir / 'best_fusion.pt')

    print(f"[+] Fusion training complete. Best Val F1: {best_val_f1:.4f}")

    # Load best model
    fusion_model.load_state_dict(torch.load(output_dir / 'best_fusion.pt'))

    return fusion_model


def main():
    parser = argparse.ArgumentParser(description="Train Fusion Layer with OOF Predictions")

    # Data
    parser.add_argument('--train-data', type=Path, required=True)
    parser.add_argument('--val-data', type=Path, required=True)
    parser.add_argument('--test-data', type=Path, default=None)

    # Base models
    parser.add_argument('--transformer-checkpoint', type=Path, required=True)
    parser.add_argument('--gnn-checkpoint', type=Path, required=True)

    # Training
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--n-folds', type=int, default=5)

    # Infrastructure
    parser.add_argument('--output-dir', type=Path, default=Path('models/fusion'))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--skip-oof', action='store_true', help='Skip OOF generation (use cached)')

    args = parser.parse_args()

    if not IMPORTS_AVAILABLE:
        print("[!] Required imports not available. Exiting.")
        return 1

    # Set seed
    set_seed(args.seed)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[+] Using device: {device}")

    # Create output dir
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('microsoft/codebert-base')

    # Generate or load OOF predictions
    oof_cache = args.output_dir / 'oof_predictions.npz'

    if args.skip_oof and oof_cache.exists():
        print(f"[*] Loading cached OOF predictions from {oof_cache}")
        oof_data = np.load(oof_cache)
        oof_trans = oof_data['transformer_logits']
        oof_gnn = oof_data['gnn_logits']
        train_labels = oof_data['labels']
    else:
        print(f"[*] Generating OOF predictions...")
        oof_generator = OOFPredictionGenerator(
            args.transformer_checkpoint,
            args.gnn_checkpoint,
            device,
            n_folds=args.n_folds,
            seed=args.seed
        )

        oof_trans, oof_gnn, train_labels = oof_generator.generate_oof_predictions(
            args.train_data, tokenizer
        )

        # Cache OOF predictions
        np.savez(
            oof_cache,
            transformer_logits=oof_trans,
            gnn_logits=oof_gnn,
            labels=train_labels
        )
        print(f"[+] OOF predictions cached to {oof_cache}")

    # Get validation predictions
    print(f"\n[*] Generating validation predictions...")

    # Load models
    transformer_model = EnhancedSQLIntentTransformer()
    transformer_model.load_state_dict(
        torch.load(args.transformer_checkpoint, map_location=device)['model_state_dict']
    )
    transformer_model.to(device)
    transformer_model.eval()

    gnn_model = EnhancedTaintFlowGNN()
    gnn_model.load_state_dict(
        torch.load(args.gnn_checkpoint, map_location=device)['model_state_dict']
    )
    gnn_model.to(device)
    gnn_model.eval()

    # Validation data
    val_transformer_dataset = TransformerDataset(args.val_data, tokenizer, 512, False)
    val_gnn_dataset = GNNDataset(args.val_data, False)

    val_trans_loader = DataLoader(val_transformer_dataset, batch_size=32, shuffle=False)
    from torch_geometric.data import DataLoader as PyGDataLoader
    val_gnn_loader = PyGDataLoader(val_gnn_dataset, batch_size=32, shuffle=False)

    # Get predictions
    val_trans_logits = []
    val_labels = []
    with torch.no_grad():
        for batch in val_trans_loader:
            logits = transformer_model(
                batch['input_ids'].to(device),
                batch['attention_mask'].to(device)
            )
            val_trans_logits.append(logits.cpu().numpy())
            val_labels.extend(batch['label'].detach().cpu().numpy())

    val_trans_logits = np.vstack(val_trans_logits)
    val_labels = np.array(val_labels)

    val_gnn_logits = []
    with torch.no_grad():
        for data in val_gnn_loader:
            data = data.to(device)
            logits = gnn_model(data)
            val_gnn_logits.append(logits.cpu().numpy())

    val_gnn_logits = np.vstack(val_gnn_logits)

    # Train fusion
    fusion_model = train_fusion_model(
        oof_trans, oof_gnn, train_labels,
        val_trans_logits, val_gnn_logits, val_labels,
        args.output_dir, args.epochs, args.lr, device
    )

    print(f"\n[+] Fusion model saved to: {args.output_dir}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
