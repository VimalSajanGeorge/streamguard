"""
Enhanced SQL Intent Transformer Training with Production Safety

Features:
- S3 checkpointing for Spot instance resilience
- Reproducibility (seeds, checksums, git tracking)
- Weighted sampling support for Phase 2
- CloudWatch metrics integration
- Early stopping on binary F1
- Mixed precision training
- Experiment config generation
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
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.amp import autocast, GradScaler
from transformers import (
    AutoModel, AutoTokenizer, AutoConfig,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix
)

# Optional: boto3 for S3 checkpointing
try:
    import boto3
    S3_AVAILABLE = True
except ImportError:
    S3_AVAILABLE = False
    print("[!] boto3 not available. S3 checkpointing disabled.")


def set_seed(seed: int):
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[+] Random seed set to {seed}")


def get_git_commit() -> str:
    """
    Get current git commit hash.

    Returns:
        Git commit hash or 'unknown'
    """
    try:
        commit = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            stderr=subprocess.DEVNULL
        ).decode('utf-8').strip()
        return commit
    except:
        return 'unknown'


def compute_file_checksum(file_path: Path) -> str:
    """
    Compute SHA256 checksum of file.

    Args:
        file_path: Path to file

    Returns:
        Hex digest
    """
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


class CodeDataset(Dataset):
    """Dataset for code vulnerability detection."""

    def __init__(
        self,
        data_path: Path,
        tokenizer,
        max_seq_len: int = 512,
        use_weights: bool = False
    ):
        """
        Initialize dataset.

        Args:
            data_path: Path to preprocessed JSONL file
            tokenizer: Tokenizer instance
            max_seq_len: Maximum sequence length
            use_weights: Whether to use sample weights
        """
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.use_weights = use_weights

        # Load samples
        self.samples = []
        self.weights = []

        print(f"[*] Loading dataset from {data_path}")

        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                sample = json.loads(line.strip())
                self.samples.append(sample)

                # Extract weight (default 1.0)
                weight = sample.get('weight', 1.0) if use_weights else 1.0
                self.weights.append(weight)

        print(f"[+] Loaded {len(self.samples)} samples")

        # Label distribution
        labels = [s['label'] for s in self.samples]
        vuln_count = sum(labels)
        safe_count = len(labels) - vuln_count

        print(f"    Vulnerable: {vuln_count} ({vuln_count/len(labels):.1%})")
        print(f"    Safe: {safe_count} ({safe_count/len(labels):.1%})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Use pre-tokenized tokens if available, else tokenize
        if 'tokens' in sample and len(sample['tokens']) > 0:
            tokens = sample['tokens'][:self.max_seq_len]

            # Pad if needed
            if len(tokens) < self.max_seq_len:
                tokens = tokens + [self.tokenizer.pad_token_id] * (self.max_seq_len - len(tokens))

            input_ids = torch.tensor(tokens, dtype=torch.long)
            attention_mask = torch.ones(len(sample['tokens'][:self.max_seq_len]), dtype=torch.long)

            if len(attention_mask) < self.max_seq_len:
                padding = torch.zeros(self.max_seq_len - len(attention_mask), dtype=torch.long)
                attention_mask = torch.cat([attention_mask, padding])

        else:
            # Fallback: tokenize on the fly
            encoding = self.tokenizer(
                sample['code'],
                max_length=self.max_seq_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            input_ids = encoding['input_ids'].squeeze(0)
            attention_mask = encoding['attention_mask'].squeeze(0)

        label = torch.tensor(sample['label'], dtype=torch.long)
        weight = torch.tensor(self.weights[idx], dtype=torch.float)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': label,
            'weight': weight
        }


class EnhancedSQLIntentTransformer(nn.Module):
    """Enhanced SQL Intent Transformer with attention."""

    def __init__(
        self,
        model_name: str = "microsoft/codebert-base",
        hidden_dim: int = 768,
        num_labels: int = 2,
        dropout: float = 0.1
    ):
        """
        Initialize transformer model.

        Args:
            model_name: Pre-trained model name
            hidden_dim: Hidden dimension
            num_labels: Number of output labels
            dropout: Dropout rate
        """
        super().__init__()

        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_labels)
        )

    def forward(self, input_ids, attention_mask):
        """
        Forward pass.

        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]

        Returns:
            Logits [batch, num_labels]
        """
        # Get encoder outputs
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Use CLS token representation
        pooled = outputs.last_hidden_state[:, 0]  # [batch, hidden]
        pooled = self.dropout(pooled)

        # Classification
        logits = self.classifier(pooled)  # [batch, num_labels]

        return logits


class S3CheckpointManager:
    """Manage checkpoints with S3 backup."""

    def __init__(
        self,
        local_dir: Path,
        s3_bucket: Optional[str] = None,
        s3_prefix: Optional[str] = None
    ):
        """
        Initialize checkpoint manager.

        Args:
            local_dir: Local directory for checkpoints
            s3_bucket: S3 bucket name (optional)
            s3_prefix: S3 prefix (optional)
        """
        self.local_dir = Path(local_dir)
        self.local_dir.mkdir(parents=True, exist_ok=True)

        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix
        self.s3_client = None

        if s3_bucket and S3_AVAILABLE:
            self.s3_client = boto3.client('s3')
            print(f"[+] S3 checkpointing enabled: s3://{s3_bucket}/{s3_prefix}")
        else:
            print("[!] S3 checkpointing disabled")

    def save_checkpoint(
        self,
        epoch: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        metrics: Dict[str, float],
        is_best: bool = False
    ):
        """
        Save checkpoint locally and to S3.

        Args:
            epoch: Current epoch
            model: Model instance
            optimizer: Optimizer instance
            scheduler: LR scheduler
            metrics: Metrics dictionary
            is_best: Whether this is the best checkpoint
        """
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
        print(f"[+] Saved checkpoint: {checkpoint_path}")

        if is_best:
            best_path = self.local_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            print(f"[+] Saved best model: {best_path}")

            # Upload to S3 if available
            if self.s3_client:
                s3_key = f"{self.s3_prefix}/best_model.pt" if self.s3_prefix else "best_model.pt"
                try:
                    self.s3_client.upload_file(
                        str(best_path),
                        self.s3_bucket,
                        s3_key
                    )
                    print(f"[+] Uploaded to S3: s3://{self.s3_bucket}/{s3_key}")
                except Exception as e:
                    print(f"[!] S3 upload failed: {e}")

        # Also save latest for Spot resilience
        latest_path = self.local_dir / 'latest.pt'
        torch.save(checkpoint, latest_path)

        if self.s3_client:
            s3_key = f"{self.s3_prefix}/latest.pt" if self.s3_prefix else "latest.pt"
            try:
                self.s3_client.upload_file(
                    str(latest_path),
                    self.s3_bucket,
                    s3_key
                )
            except Exception as e:
                print(f"[!] S3 upload failed for latest: {e}")

    def load_checkpoint(self, checkpoint_name: str = 'latest.pt') -> Optional[Dict]:
        """
        Load checkpoint from local or S3.

        Args:
            checkpoint_name: Checkpoint filename

        Returns:
            Checkpoint dictionary or None
        """
        local_path = self.local_dir / checkpoint_name

        # Try local first
        if local_path.exists():
            print(f"[*] Loading checkpoint from {local_path}")
            return torch.load(local_path)

        # Try S3
        if self.s3_client:
            s3_key = f"{self.s3_prefix}/{checkpoint_name}" if self.s3_prefix else checkpoint_name
            try:
                print(f"[*] Downloading checkpoint from S3: {s3_key}")
                self.s3_client.download_file(
                    self.s3_bucket,
                    s3_key,
                    str(local_path)
                )
                return torch.load(local_path)
            except Exception as e:
                print(f"[!] S3 download failed: {e}")

        return None


def compute_binary_f1(y_true: np.ndarray, y_pred: np.ndarray, positive_class: int = 1) -> float:
    """
    Compute F1 score for specific class (vulnerable).

    Args:
        y_true: True labels
        y_pred: Predicted labels
        positive_class: Positive class label

    Returns:
        F1 score for positive class
    """
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
    """
    Evaluate model on dataset.

    Args:
        model: Model instance
        dataloader: Data loader
        device: Device
        criterion: Loss criterion (optional, not used - we create eval-specific one)

    Returns:
        Metrics dictionary
    """
    model.eval()

    all_preds = []
    all_labels = []
    total_loss = 0.0

    # Create evaluation-specific criterion with mean reduction
    eval_criterion = nn.CrossEntropyLoss(reduction='mean') if criterion is not None else None

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            logits = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            if eval_criterion:
                loss = eval_criterion(logits, labels)  # Already reduced to scalar
                total_loss += loss.item()

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', pos_label=1
    )

    # Binary F1 for vulnerable class (early stopping metric)
    binary_f1 = compute_binary_f1(all_labels, all_preds, positive_class=1)

    metrics = {
        'loss': total_loss / len(dataloader) if criterion else 0.0,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'binary_f1_vulnerable': binary_f1
    }

    return metrics


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: Optional[torch.amp.GradScaler] = None,
    accumulation_steps: int = 1
) -> float:
    """
    Train for one epoch.

    Args:
        model: Model instance
        dataloader: Data loader
        optimizer: Optimizer
        criterion: Loss criterion
        device: Device
        scaler: GradScaler for mixed precision (optional)
        accumulation_steps: Gradient accumulation steps

    Returns:
        Average loss
    """
    model.train()
    total_loss = 0.0

    optimizer.zero_grad()

    for step, batch in enumerate(dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        weights = batch['weight'].to(device)

        # Mixed precision training
        if scaler:
            with autocast(device_type='cuda'):
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)  # Shape: [batch_size]

                # Apply sample weights if available
                if weights is not None and len(weights) > 0:
                    # Ensure weights are float and on same device
                    weights = weights.to(device=loss.device, dtype=torch.float32)
                    # Validate shapes match
                    if weights.shape[0] == loss.shape[0]:
                        loss = (loss * weights).mean()
                    else:
                        print(f"⚠️  Weight shape mismatch: {weights.shape} vs {loss.shape}, using unweighted")
                        loss = loss.mean()
                else:
                    loss = loss.mean()

                loss = loss / accumulation_steps

            scaler.scale(loss).backward()

            if (step + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        else:
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

            if weights is not None and len(weights) > 0:
                weights = weights.to(device=loss.device, dtype=torch.float32)
                if weights.shape[0] == loss.shape[0]:
                    loss = (loss * weights).mean()
                else:
                    loss = loss.mean()
            else:
                loss = loss.mean()

            loss = loss / accumulation_steps
            loss.backward()

            if (step + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps

    return total_loss / len(dataloader)


def save_experiment_config(args: argparse.Namespace, output_dir: Path, data_paths: Dict[str, Path]):
    """
    Save experiment configuration for reproducibility.

    Args:
        args: Command line arguments
        output_dir: Output directory
        data_paths: Paths to data files
    """
    config = {
        'timestamp': datetime.now().isoformat(),
        'git_commit': get_git_commit(),
        'seed': args.seed,
        'hyperparameters': {
            'model_name': args.model_name,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.lr,
            'weight_decay': args.weight_decay,
            'warmup_ratio': args.warmup_ratio,
            'max_seq_len': args.max_seq_len,
            'dropout': args.dropout,
            'accumulation_steps': args.accumulation_steps
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
    parser = argparse.ArgumentParser(description="Train Enhanced SQL Intent Transformer")

    # Data
    parser.add_argument('--train-data', type=Path, required=True, help='Training data path')
    parser.add_argument('--val-data', type=Path, required=True, help='Validation data path')
    parser.add_argument('--test-data', type=Path, default=None, help='Test data path')

    # Model
    parser.add_argument('--model-name', type=str, default='microsoft/codebert-base', help='Base model')
    parser.add_argument('--hidden-dim', type=int, default=768, help='Hidden dimension')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')

    # Training
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--warmup-ratio', type=float, default=0.1, help='Warmup ratio')
    parser.add_argument('--max-seq-len', type=int, default=512, help='Max sequence length')
    parser.add_argument('--accumulation-steps', type=int, default=1, help='Gradient accumulation')

    # Regularization
    parser.add_argument('--early-stopping-patience', type=int, default=2, help='Early stopping patience')
    parser.add_argument('--use-weights', action='store_true', help='Use sample weights')

    # Infrastructure
    parser.add_argument('--output-dir', type=Path, default=Path('models/transformer'), help='Output directory')
    parser.add_argument('--s3-bucket', type=str, default=None, help='S3 bucket for checkpoints')
    parser.add_argument('--s3-prefix', type=str, default='checkpoints/transformer', help='S3 prefix')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--mixed-precision', action='store_true', help='Use mixed precision')
    parser.add_argument('--quick-test', action='store_true', help='Quick test with 100 samples')

    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[+] Using device: {device}")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Save experiment config
    data_paths = {'train': args.train_data, 'val': args.val_data}
    if args.test_data:
        data_paths['test'] = args.test_data
    save_experiment_config(args, args.output_dir, data_paths)

    # Load tokenizer
    print(f"[*] Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # CRITICAL: Validate max_seq_len against model's position embeddings limit
    # CodeBERT/RoBERTa has max position embeddings of 514 (512 + 2 special tokens)
    model_config = AutoConfig.from_pretrained(args.model_name)
    max_position_embeddings = getattr(model_config, 'max_position_embeddings', 512)

    if args.max_seq_len > max_position_embeddings:
        print(f"\n[!] WARNING: max_seq_len ({args.max_seq_len}) exceeds model limit ({max_position_embeddings})")
        print(f"    This will cause tensor size mismatch errors during training!")
        print(f"    Automatically reducing max_seq_len to {max_position_embeddings}")
        args.max_seq_len = max_position_embeddings
        print(f"[+] Updated max_seq_len: {args.max_seq_len}\n")

    # Load datasets
    train_dataset = CodeDataset(
        args.train_data, tokenizer, args.max_seq_len, args.use_weights
    )
    val_dataset = CodeDataset(
        args.val_data, tokenizer, args.max_seq_len, use_weights=False
    )

    # Quick test mode
    if args.quick_test:
        print("[*] Quick test mode: using 100 samples")
        train_dataset.samples = train_dataset.samples[:100]
        val_dataset.samples = val_dataset.samples[:50]

    # Data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    # Model
    print(f"[*] Initializing model: {args.model_name}")
    model = EnhancedSQLIntentTransformer(
        model_name=args.model_name,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout
    )
    model.to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Scheduler
    total_steps = len(train_loader) * args.epochs // args.accumulation_steps
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, warmup_steps, total_steps
    )

    # Loss
    criterion = nn.CrossEntropyLoss(reduction='none')  # For weighted loss

    # Mixed precision
    scaler = GradScaler('cuda') if args.mixed_precision and torch.cuda.is_available() else None

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
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion,
            device, scaler, args.accumulation_steps
        )

        # Evaluate
        val_metrics = evaluate(model, val_loader, device, criterion)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f}")
        print(f"Val Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"Val F1 (vulnerable): {val_metrics['binary_f1_vulnerable']:.4f}")

        # Step scheduler
        scheduler.step()

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
            print(f"\n[!] Early stopping triggered at epoch {epoch + 1}")
            break

    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Best validation F1 (vulnerable): {best_val_f1:.4f}")

    # Test evaluation if provided
    if args.test_data:
        print("\n" + "="*70)
        print("TEST EVALUATION")
        print("="*70)

        test_dataset = CodeDataset(
            args.test_data, tokenizer, args.max_seq_len, use_weights=False
        )
        test_loader = DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False
        )

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
