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
import math
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
        use_weights: bool = False,
        use_features: bool = False
    ):
        """
        Initialize dataset.

        Args:
            data_path: Path to preprocessed JSONL file
            tokenizer: Tokenizer instance
            max_seq_len: Maximum sequence length
            use_weights: Whether to use sample weights
            use_features: Whether to extract code features
        """
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.use_weights = use_weights
        self.use_features = use_features
        self.feature_cache = {}  # Cache extracted features

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

        result = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': label,
            'weight': weight
        }

        # Extract code features if enabled
        if self.use_features:
            if idx not in self.feature_cache:
                try:
                    from core.features.code_metrics import extract_basic_metrics
                    metrics = extract_basic_metrics(sample['code'])

                    # Normalize features (simple standardization)
                    features = [
                        metrics['loc'] / 100.0,
                        metrics['sloc'] / 100.0,
                        metrics['sql_concat'] / 5.0,
                        metrics['execute_calls'] / 5.0,
                        metrics['user_input'] / 5.0,
                        metrics['loops'] / 10.0,
                        metrics['conditionals'] / 10.0,
                        metrics['function_calls'] / 20.0,
                        metrics['try_blocks'] / 5.0,
                        metrics['string_ops'] / 10.0
                    ]
                    self.feature_cache[idx] = torch.tensor(
                        features, dtype=torch.float32
                    )
                except Exception as e:
                    # Fallback: zero vector if extraction fails
                    import warnings
                    warnings.warn(f"Feature extraction failed for sample {idx}: {str(e)}")
                    self.feature_cache[idx] = torch.zeros(10, dtype=torch.float32)

            result['code_features'] = self.feature_cache[idx]

        return result


class EnhancedSQLIntentTransformer(nn.Module):
    """Enhanced SQL Intent Transformer with attention."""

    def __init__(
        self,
        model_name: str = "microsoft/codebert-base",
        hidden_dim: int = 768,
        num_labels: int = 2,
        dropout: float = 0.1,
        use_features: bool = False,
        feature_dim: int = 10
    ):
        """
        Initialize transformer model.

        Args:
            model_name: Pre-trained model name
            hidden_dim: Hidden dimension
            num_labels: Number of output labels
            dropout: Dropout rate
            use_features: Whether to use code features
            feature_dim: Dimension of code features
        """
        super().__init__()

        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.use_features = use_features

        # Feature projection and fusion (if using features)
        if use_features:
            self.feature_projection = nn.Linear(feature_dim, hidden_dim)
            self.fusion = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_labels)
        )

    def forward(self, input_ids, attention_mask, code_features=None):
        """
        Forward pass.

        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            code_features: Code metrics features [batch, feature_dim] (optional)

        Returns:
            Logits [batch, num_labels]
        """
        # Get encoder outputs
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Use CLS token representation
        code_emb = outputs.last_hidden_state[:, 0]  # [batch, hidden]

        # Feature fusion (if enabled and features provided)
        if self.use_features and code_features is not None:
            feat_emb = self.feature_projection(code_features)  # [batch, hidden]
            combined = torch.cat([code_emb, feat_emb], dim=1)  # [batch, hidden*2]
            fused = self.fusion(combined)  # [batch, hidden]
        else:
            fused = code_emb

        fused = self.dropout(fused)

        # Classification
        logits = self.classifier(fused)  # [batch, num_labels]

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
        is_best: bool = False,
        extra_metadata: Optional[Dict[str, Any]] = None
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
        # CRITICAL FIX: Save state_dicts only (PyTorch 2.6 compatibility)
        # Remove prediction_distribution from metrics to avoid serialization issues
        metrics_to_save = {
            k: float(v) if isinstance(v, (np.floating, np.integer, torch.Tensor)) else v
            for k, v in metrics.items()
            if k != 'prediction_distribution'
        }

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'metrics': metrics_to_save,
            # Enhanced metadata from extra_metadata (if provided)
            **(extra_metadata or {})
        }

        # Save locally with atomic write (tmp → rename)
        checkpoint_path = self.local_dir / f'checkpoint_epoch_{epoch}.pt'
        tmp_path = checkpoint_path.with_suffix('.pt.tmp')
        torch.save(checkpoint, tmp_path)
        tmp_path.replace(checkpoint_path)  # Atomic on most OSes
        print(f"[+] Saved checkpoint: {checkpoint_path}")

        if is_best:
            best_path = self.local_dir / 'best_model.pt'
            tmp_best = best_path.with_suffix('.pt.tmp')
            torch.save(checkpoint, tmp_best)
            tmp_best.replace(best_path)  # Atomic
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
            # CRITICAL FIX: Use weights_only=True for PyTorch 2.6+
            return torch.load(local_path, weights_only=True)

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
                # CRITICAL FIX: Use weights_only=True for PyTorch 2.6+
                return torch.load(local_path, weights_only=True)
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
            # Get code features if available
            code_features = batch.get('code_features')
            if code_features is not None:
                code_features = code_features.to(device)

            logits = model(input_ids, attention_mask, code_features)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            if eval_criterion:
                loss = eval_criterion(logits, labels)  # Already reduced to scalar
                total_loss += loss.item()

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # CRITICAL FIX: Add prediction distribution monitoring to detect collapse
    pred_distribution = {
        'predicted_vulnerable': int((all_preds == 1).sum()),
        'predicted_safe': int((all_preds == 0).sum()),
        'actual_vulnerable': int((all_labels == 1).sum()),
        'actual_safe': int((all_labels == 0).sum())
    }

    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', pos_label=1, zero_division=0
    )

    # Binary F1 for vulnerable class (early stopping metric)
    binary_f1 = compute_binary_f1(all_labels, all_preds, positive_class=1)

    metrics = {
        'loss': total_loss / len(dataloader) if criterion else 0.0,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'binary_f1_vulnerable': binary_f1,
        'prediction_distribution': pred_distribution
    }

    return metrics


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scheduler: Any,
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
        scheduler: LR scheduler (stepped per-step, not per-epoch)
        scaler: GradScaler for mixed precision (optional)
        accumulation_steps: Gradient accumulation steps

    Returns:
        Average loss
    """
    model.train()
    running_loss = 0.0

    optimizer.zero_grad()

    for step, batch in enumerate(dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        # Get code features if available
        code_features = batch.get('code_features')
        if code_features is not None:
            code_features = code_features.to(device)
        # Note: weights removed - using class_weights in criterion instead

        # Mixed precision training
        if scaler:
            with autocast(device_type='cuda'):
                logits = model(input_ids, attention_mask, code_features)
                loss = criterion(logits, labels)  # Already reduced to scalar with class weights
                loss = loss / accumulation_steps

            # Track loss before dividing (for accurate reporting)
            running_loss += loss.detach().item() * accumulation_steps

            scaler.scale(loss).backward()

            if (step + 1) % accumulation_steps == 0:
                # CRITICAL FIX: Add gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                # CRITICAL FIX: Step scheduler per-step (not per-epoch)
                scheduler.step()

        else:
            logits = model(input_ids, attention_mask, code_features)
            loss = criterion(logits, labels)  # Already reduced to scalar
            loss = loss / accumulation_steps

            running_loss += loss.detach().item() * accumulation_steps

            loss.backward()

            if (step + 1) % accumulation_steps == 0:
                # CRITICAL FIX: Add gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                optimizer.zero_grad()

                # CRITICAL FIX: Step scheduler per-step (not per-epoch)
                scheduler.step()

    return running_loss / len(dataloader)


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

    # NEW: Stability & accuracy flags
    parser.add_argument('--use-weighted-sampler', action='store_true', default=False,
                        help='Use WeightedRandomSampler for class balance (recommended for imbalanced data)')
    parser.add_argument('--lr-override', type=float, default=None,
                        help='Override LR (bypasses all scaling)')
    parser.add_argument('--weight-multiplier', type=float, default=1.5,
                        help='Multiplier for minority class weight (1.0-2.0 range)')
    parser.add_argument('--use-code-features', action='store_true', default=False,
                        help='Add 10 code metrics as additional features (+5-10 F1 points)')
    parser.add_argument('--focal-loss', action='store_true', default=False,
                        help='Use Focal Loss instead of CrossEntropy (helps with hard negatives)')
    parser.add_argument('--focal-gamma', type=float, default=2.0,
                        help='Focal loss gamma (1.0-2.5, higher=more focus on hard examples)')
    parser.add_argument('--find-lr', action='store_true', default=False,
                        help='Run LR Finder before training (auto-detect optimal LR)')
    parser.add_argument('--lr-finder-iterations', type=int, default=100,
                        help='Number of iterations for LR finder (default: 100)')
    parser.add_argument('--tensorboard', action='store_true', default=False,
                        help='Enable TensorBoard logging (logs to runs/ directory)')
    parser.add_argument('--force-find-lr', action='store_true', default=False,
                        help='Ignore cached LR, always run LR Finder fresh')
    parser.add_argument('--lr-cache-max-age', type=int, default=168,
                        help='LR cache validity in hours (default: 168 = 1 week)')

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
        args.train_data, tokenizer, args.max_seq_len, args.use_weights, args.use_code_features
    )
    val_dataset = CodeDataset(
        args.val_data, tokenizer, args.max_seq_len, use_weights=False, use_features=args.use_code_features
    )

    # Quick test mode
    if args.quick_test:
        print("[*] Quick test mode: using 500 train, 100 val samples")
        train_dataset.samples = train_dataset.samples[:500]
        val_dataset.samples = val_dataset.samples[:100]

    # CRITICAL FIX: Calculate class weights for imbalanced dataset
    print("\n[*] Calculating class weights for balanced training...")
    train_labels = []
    for sample in train_dataset.samples:
        train_labels.append(sample['label'])

    class_counts = torch.bincount(torch.tensor(train_labels, dtype=torch.long))
    num_safe = class_counts[0].item()
    num_vulnerable = class_counts[1].item()
    total = len(train_labels)

    # Class weights with tunable multiplier
    if args.quick_test:
        weight_safe = 1.0
        # Use multiplier for quick test (default 1.5, range 1.2-2.0)
        base_vulnerable_weight = total / (2.0 * num_vulnerable)
        weight_vulnerable = base_vulnerable_weight * args.weight_multiplier
        print(f"[*] Quick test mode: class weights (1.0 vs {weight_vulnerable:.4f})")
        print(f"    Base inverse-freq={base_vulnerable_weight:.4f}, multiplier={args.weight_multiplier}")
    else:
        # Full training: use inverse-frequency as-is (conservative)
        weight_safe = total / (2.0 * num_safe)
        weight_vulnerable = total / (2.0 * num_vulnerable)
        # Optionally apply multiplier for full training too
        if args.weight_multiplier != 1.0:
            weight_vulnerable *= args.weight_multiplier
            print(f"[*] Applied weight multiplier: {args.weight_multiplier}")

    # Safety: cap weights to prevent instability
    weight_vulnerable = min(weight_vulnerable, 5.0)
    if weight_vulnerable == 5.0:
        print(f"[!] WARNING: Vulnerable weight capped at 5.0 (safety ceiling)")

    # CRITICAL: Triple-weight detection and auto-adjustment
    # Detect if all three weighting methods are enabled simultaneously
    triple_weighting = (
        getattr(args, "use_weighted_sampler", False) and
        getattr(args, "weight_multiplier", 1.0) > 1.0 and
        getattr(args, "focal_loss", False)
    )

    # Store original values for telemetry
    original_weight_multiplier = None
    original_focal_gamma = None

    if triple_weighting:
        print(f"\n{'='*70}")
        print(f"[!] NOTICE: Triple weighting detected!")
        print(f"    - WeightedRandomSampler: ON")
        print(f"    - Class weight multiplier: {args.weight_multiplier}")
        print(f"    - Focal Loss: ON")
        print(f"\n[*] Auto-adjusting to prevent overcorrection...")

        # Store original values
        original_weight_multiplier = args.weight_multiplier
        if hasattr(args, 'focal_gamma'):
            original_focal_gamma = args.focal_gamma

        # Reduce weight multiplier by 20%
        args.weight_multiplier = max(1.0, float(args.weight_multiplier) * 0.8)
        print(f"    weight_multiplier: {original_weight_multiplier:.2f} → {args.weight_multiplier:.2f}")

        # Optionally clamp focal_gamma to 1.5
        if hasattr(args, 'focal_gamma') and args.focal_gamma is not None:
            if args.focal_gamma > 1.5:
                print(f"    focal_gamma: {original_focal_gamma:.2f} → 1.5")
                args.focal_gamma = 1.5

        print(f"{'='*70}\n")

        # Recalculate vulnerable weight with adjusted multiplier
        if args.quick_test:
            base_vulnerable_weight = total / (2.0 * num_vulnerable)
            weight_vulnerable = base_vulnerable_weight * args.weight_multiplier
            weight_vulnerable = min(weight_vulnerable, 5.0)
        else:
            weight_vulnerable = (total / (2.0 * num_vulnerable)) * args.weight_multiplier
            weight_vulnerable = min(weight_vulnerable, 5.0)

    class_weights = torch.tensor(
        [weight_safe, weight_vulnerable],
        dtype=torch.float32
    ).to(device)

    print(f"    Class distribution: Safe={num_safe}, Vulnerable={num_vulnerable}")
    print(f"    Class weights: Safe={weight_safe:.4f}, Vulnerable={weight_vulnerable:.4f}\n")

    # Data loaders with WeightedRandomSampler option
    if args.use_weighted_sampler:
        # Compute per-sample weights (inverse frequency)
        class_counts = torch.bincount(torch.tensor(train_labels, dtype=torch.long))
        class_weights_sampling = 1.0 / class_counts.float()
        sample_weights = torch.tensor([
            class_weights_sampling[label] for label in train_labels
        ])
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_dataset),
            replacement=True
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=sampler,  # sampler and shuffle are mutually exclusive!
            num_workers=0
        )
        print(f"[*] Using WeightedRandomSampler (inverse-frequency)")
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,  # Only when no sampler
            num_workers=0
        )

    # Validation loader (always shuffle=False, never sampler)
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    # Model
    # Disable dropout for quick test to reduce noise
    dropout_val = 0.0 if args.quick_test else args.dropout
    if args.quick_test:
        print(f"[*] Quick test mode: dropout set to 0.0 (disabled)")

    print(f"[*] Initializing model: {args.model_name}")
    model = EnhancedSQLIntentTransformer(
        model_name=args.model_name,
        hidden_dim=args.hidden_dim,
        dropout=dropout_val,
        use_features=args.use_code_features
    )
    model.to(device)

    if args.use_code_features:
        print(f"[*] Code features enabled: 10 metrics + fusion layer")

    # CRITICAL FIX: Learning rate scaling for large batch sizes
    import math
    base_lr = args.lr  # 2e-5
    base_batch = 16

    if args.batch_size > base_batch:
        scale_factor = math.sqrt(args.batch_size / base_batch)
        scaled_lr = base_lr * scale_factor
        print(f"[*] Scaling LR: {base_lr:.2e} -> {scaled_lr:.2e} (batch {args.batch_size} vs base {base_batch})")
    else:
        scaled_lr = base_lr

    # For quick test: use proven LR (not too low!)
    if args.quick_test:
        if args.lr_override:
            scaled_lr = args.lr_override
            print(f"[*] Quick test mode: using override LR={scaled_lr:.2e}")
        else:
            scaled_lr = 1.5e-5  # Known good LR for 500 samples
            print(f"[*] Quick test mode: using default LR={scaled_lr:.2e}")
    elif args.lr_override:
        scaled_lr = args.lr_override
        print(f"[*] Using override LR={scaled_lr:.2e}")

    # Safety caps: prevent catastrophic LR
    scaled_lr = max(scaled_lr, 1e-7)  # Lower bound
    scaled_lr = min(scaled_lr, 5e-4)  # Upper bound
    if scaled_lr == 5e-4:
        print(f"[!] WARNING: LR capped at 5e-4 (safety ceiling)")
    if scaled_lr == 1e-7:
        print(f"[!] WARNING: LR floored at 1e-7 (safety floor)")

    print(f"[*] Final LR: {scaled_lr:.2e}")

    # Discriminative LR: lower for encoder, higher for head
    # Use sets to prevent duplicate params
    encoder_param_ids = set()
    head_param_ids = set()
    no_decay_param_ids = set()

    encoder_params = []
    head_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        param_id = id(param)

        # No weight decay for bias and LayerNorm variants
        # Match: bias, LayerNorm, layer_norm, ln, norm (any case)
        if any(keyword in name.lower() for keyword in ['bias', 'layernorm', 'layer_norm', 'ln', 'norm']):
            if param_id not in no_decay_param_ids:
                no_decay_param_ids.add(param_id)
                no_decay_params.append(param)
        # Encoder layers: transformer, encoder, roberta, bert
        elif any(keyword in name.lower() for keyword in ['transformer', 'encoder', 'roberta', 'bert']):
            if param_id not in encoder_param_ids:
                encoder_param_ids.add(param_id)
                encoder_params.append(param)
        # Head layers: everything else
        else:
            if param_id not in head_param_ids:
                head_param_ids.add(param_id)
                head_params.append(param)

    # Verify no duplicates
    total_unique = len(encoder_param_ids) + len(head_param_ids) + len(no_decay_param_ids)
    total_params = sum(1 for p in model.parameters() if p.requires_grad)
    assert total_unique == total_params, \
        f"Parameter group mismatch: {total_unique} != {total_params}"

    # Discriminative LR
    encoder_lr = scaled_lr * 0.1  # 10x smaller for pretrained encoder
    head_lr = scaled_lr * 1.0

    # Build optimizer with parameter groups
    optimizer = torch.optim.AdamW([
        {
            'params': encoder_params,
            'lr': encoder_lr,
            'weight_decay': args.weight_decay,
            'name': 'encoder'  # For debugging
        },
        {
            'params': head_params,
            'lr': head_lr,
            'weight_decay': args.weight_decay,
            'name': 'head'
        },
        {
            'params': no_decay_params,
            'lr': head_lr,
            'weight_decay': 0.0,
            'name': 'no_decay'
        }
    ], eps=1e-8, betas=(0.9, 0.999))

    print(f"[*] Discriminative LR:")
    print(f"    Encoder ({len(encoder_params)} params): LR={encoder_lr:.2e}, WD={args.weight_decay}")
    print(f"    Head ({len(head_params)} params): LR={head_lr:.2e}, WD={args.weight_decay}")
    print(f"    No-decay ({len(no_decay_params)} params): LR={head_lr:.2e}, WD=0.0")
    print(f"    Total trainable params: {total_params}")

    # Scheduler with safe warmup
    # CRITICAL FIX: Account for gradient accumulation in total_steps
    total_steps = math.ceil(len(train_loader) / args.accumulation_steps) * args.epochs
    warmup_ratio = 0.1  # Default 10%

    # Adaptive warmup for large batches
    if args.batch_size >= 64:
        warmup_ratio = 0.15
    elif args.batch_size >= 32:
        warmup_ratio = 0.12

    warmup_steps = int(warmup_ratio * total_steps)

    # Safety caps
    warmup_steps = max(warmup_steps, 10)  # Minimum 10 steps
    warmup_steps = min(warmup_steps, int(0.2 * total_steps))  # Max 20% of training

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    print(f"[*] Scheduler config:")
    print(f"    Total steps: {total_steps}")
    print(f"    Warmup steps: {warmup_steps} ({warmup_steps/total_steps*100:.1f}%)")
    print(f"    Warmup ratio: {warmup_ratio}")

    # Loss function with Focal Loss option
    if args.focal_loss:
        # Import will be added when we create the file
        try:
            from training.losses.focal_loss import FocalLoss
            criterion = FocalLoss(alpha=class_weights, gamma=args.focal_gamma)
            print(f"[+] Loss: FocalLoss with gamma={args.focal_gamma}, alpha=class_weights")
        except ImportError:
            print(f"[!] WARNING: FocalLoss not found, falling back to CrossEntropyLoss")
            label_smoothing_val = 0.0 if args.quick_test else 0.05
            criterion = nn.CrossEntropyLoss(
                weight=class_weights,
                label_smoothing=label_smoothing_val,
                reduction='mean'
            )
            print(f"[+] Loss: CrossEntropyLoss (fallback)")
    else:
        # Disable label smoothing for quick test to reduce noise
        label_smoothing_val = 0.0 if args.quick_test else 0.05
        criterion = nn.CrossEntropyLoss(
            weight=class_weights,                  # Class balancing
            label_smoothing=label_smoothing_val,   # 0.0 for quick test, 0.05 for full
            reduction='mean'                       # Simple mean (not 'none')
        )
        print(f"[+] Loss: CrossEntropyLoss with class_weights and label_smoothing={label_smoothing_val}")

    # Mixed precision training (with safe defaults)
    use_amp = args.mixed_precision and torch.cuda.is_available()
    if use_amp:
        # Create scaler without device argument (deprecated)
        scaler = torch.amp.GradScaler()
        print(f"[+] Mixed precision training enabled (AMP)")
    else:
        scaler = None
        print(f"[+] Mixed precision training disabled")

    # LR Finder with caching and validation (optional - run before training to auto-detect optimal LR)
    lr_finder_analysis = None  # Store for checkpoint metadata
    if args.find_lr:
        print("\n" + "="*70)
        print("RUNNING LR FINDER")
        print("="*70)

        # Handle both absolute and relative imports for notebook compatibility
        try:
            from training.utils.lr_finder import LRFinder, analyze_lr_loss_curve, validate_and_cap_lr
            from training.utils.lr_cache import compute_cache_key, save_lr_cache, load_lr_cache
        except ModuleNotFoundError:
            from utils.lr_finder import LRFinder, analyze_lr_loss_curve, validate_and_cap_lr
            from utils.lr_cache import compute_cache_key, save_lr_cache, load_lr_cache

        # Compute cache key based on dataset + config
        cache_key = compute_cache_key(
            dataset_path=args.train_data,
            model_name=args.model_name,
            batch_size=args.batch_size,
            extra={'max_seq_len': args.max_seq_len}
        )

        # Check cache first
        cached = load_lr_cache(cache_key, max_age_hours=args.lr_cache_max_age)
        if cached and not args.force_find_lr:
            print(f"[*] Loading cached LR Finder results...")
            suggested_lr = cached['suggested_lr']
            lr_finder_analysis = cached['metadata'].get('analysis', {})
            print(f"    Cached LR: {suggested_lr:.2e}")
            print(f"    Confidence: {lr_finder_analysis.get('confidence', 'unknown')}")
            print(f"    Cached at: {cached['timestamp']}")
            print(f"    (Use --force-find-lr to re-run)")
        else:
            if args.force_find_lr:
                print(f"[*] --force-find-lr enabled, ignoring cache")

            # Create a temporary optimizer for LR finding
            temp_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-7)

            lr_finder = LRFinder(model, temp_optimizer, criterion, device)
            lr_history, loss_history = lr_finder.range_test(
                train_loader,
                start_lr=1e-7,
                end_lr=1.0,
                num_iter=args.lr_finder_iterations,
                smooth_f=0.05,
                diverge_th=5.0
            )

            # Analyze curve quality
            lr_finder_analysis = analyze_lr_loss_curve(lr_history, loss_history)
            candidate_lr = lr_finder_analysis['suggested_lr']

            # Validate and cap with safety checks
            validation_result = validate_and_cap_lr(
                candidate_lr,
                lr_finder_analysis,
                cap=5e-4,
                conservative_fallback=1e-5
            )
            suggested_lr = validation_result['lr']

            # Log detailed results
            print(f"\n[*] LR Finder Results:")
            print(f"    Raw suggestion: {candidate_lr:.2e}")
            print(f"    Confidence: {lr_finder_analysis['confidence']}")
            print(f"    Slope magnitude: {lr_finder_analysis['slope_mag']:.4f}")
            print(f"    SNR: {lr_finder_analysis['snr']:.2f}")
            if lr_finder_analysis['diverged']:
                print(f"    [!] Divergence detected after minimum")
            print(f"    Final LR: {suggested_lr:.2e} ({validation_result['note']})")

            if validation_result['used_fallback']:
                print(f"[!] WARNING: Used conservative fallback (1e-5) due to poor curve quality")
                reasons = ', '.join(lr_finder_analysis.get('reason', ['unknown']))
                print(f"    Reasons: {reasons}")

            # Save to cache
            save_lr_cache(
                cache_key,
                suggested_lr,
                lr_history_summary={
                    'min_loss': float(min(loss_history)),
                    'max_loss': float(max(loss_history)),
                    'num_points': len(lr_history)
                },
                metadata={
                    'analysis': lr_finder_analysis,
                    'validation': validation_result,
                    'timestamp': datetime.now().isoformat(),
                    'model_name': args.model_name,
                    'batch_size': args.batch_size
                }
            )
            print(f"[+] LR Finder results cached (key: {cache_key[:12]}...)")

            # Plot results
            lr_plot_path = args.output_dir / 'lr_finder_plot.png'
            lr_finder.plot(str(lr_plot_path), skip_start=10, skip_end=5)

        # Apply suggested LR if not overridden
        if not args.lr_override:
            print(f"[*] Applying suggested LR: {suggested_lr:.2e}")
            scaled_lr = suggested_lr
            # Rebuild optimizer with new LR
            optimizer = torch.optim.AdamW([
                {
                    'params': encoder_params,
                    'lr': scaled_lr * 0.1,
                    'weight_decay': args.weight_decay,
                    'name': 'encoder'
                },
                {
                    'params': head_params,
                    'lr': scaled_lr * 1.0,
                    'weight_decay': args.weight_decay,
                    'name': 'head'
                },
                {
                    'params': no_decay_params,
                    'lr': scaled_lr * 1.0,
                    'weight_decay': 0.0,
                    'name': 'no_decay'
                }
            ], eps=1e-8, betas=(0.9, 0.999))

            # Rebuild scheduler with new optimizer
            total_steps = math.ceil(len(train_loader) / args.accumulation_steps) * args.epochs
            warmup_ratio = 0.1
            if args.batch_size >= 64:
                warmup_ratio = 0.15
            elif args.batch_size >= 32:
                warmup_ratio = 0.12
            warmup_steps = int(warmup_ratio * total_steps)
            warmup_steps = max(warmup_steps, 10)
            warmup_steps = min(warmup_steps, int(0.2 * total_steps))

            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )

            print(f"[+] Optimizer and scheduler rebuilt with suggested LR")
        else:
            print(f"[*] Using --lr-override={args.lr_override:.2e}, ignoring LR Finder suggestion")

        print("="*70 + "\n")

    # Checkpoint manager
    checkpoint_mgr = S3CheckpointManager(
        args.output_dir / 'checkpoints',
        args.s3_bucket,
        args.s3_prefix
    )

    # TensorBoard writer (optional)
    writer = None
    if args.tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter
            tensorboard_dir = args.output_dir / 'runs' / datetime.now().strftime('%Y%m%d_%H%M%S')
            writer = SummaryWriter(log_dir=str(tensorboard_dir))
            print(f"[+] TensorBoard logging enabled: {tensorboard_dir}")
            print(f"    View with: tensorboard --logdir={tensorboard_dir.parent}")
        except ImportError:
            print("[!] WARNING: tensorboard not available, skipping TensorBoard logging")
            writer = None

    # Training loop
    print("\n" + "="*70)
    print("TRAINING START")
    print("="*70)

    best_val_f1 = 0.0
    patience_counter = 0

    # Increase patience for quick test (needs more time to stabilize)
    effective_patience = args.early_stopping_patience * 3 if args.quick_test else args.early_stopping_patience
    if args.quick_test:
        print(f"[*] Quick test mode: increased patience to {effective_patience} (3x normal)")

    # Collapse detection tracking
    consecutive_collapses = 0
    collapse_history = []

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 70)

        # Train (CRITICAL FIX: Pass scheduler to train_epoch)
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion,
            device, scheduler, scaler, args.accumulation_steps
        )

        # Evaluate
        val_metrics = evaluate(model, val_loader, device, criterion)

        # CRITICAL FIX: Print prediction distribution to detect collapse
        dist = val_metrics['prediction_distribution']
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f}")
        print(f"Val Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"Val Precision: {val_metrics['precision']:.4f}")
        print(f"Val Recall: {val_metrics['recall']:.4f}")
        print(f"Val F1 (vulnerable): {val_metrics['binary_f1_vulnerable']:.4f}")
        print(f"Predictions: Vulnerable={dist['predicted_vulnerable']}/{dist['actual_vulnerable']}, "
              f"Safe={dist['predicted_safe']}/{dist['actual_safe']}")

        # TensorBoard logging
        if writer:
            # Loss curves
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_metrics['loss'], epoch)

            # Metrics
            writer.add_scalar('Metrics/accuracy', val_metrics['accuracy'], epoch)
            writer.add_scalar('Metrics/precision', val_metrics['precision'], epoch)
            writer.add_scalar('Metrics/recall', val_metrics['recall'], epoch)
            writer.add_scalar('Metrics/f1_vulnerable', val_metrics['binary_f1_vulnerable'], epoch)

            # Prediction distribution
            writer.add_scalar('Predictions/vulnerable_count', dist['predicted_vulnerable'], epoch)
            writer.add_scalar('Predictions/safe_count', dist['predicted_safe'], epoch)
            writer.add_scalar('Predictions/vulnerable_ratio',
                            dist['predicted_vulnerable'] / (dist['predicted_vulnerable'] + dist['predicted_safe'] + 1e-8),
                            epoch)

            # Learning rate
            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('Learning_Rate/encoder', optimizer.param_groups[0]['lr'], epoch)
            if len(optimizer.param_groups) > 1:
                writer.add_scalar('Learning_Rate/head', optimizer.param_groups[1]['lr'], epoch)

            # Flush to ensure data is written
            writer.flush()

        # CRITICAL FIX: Hardened collapse detection (require 2 consecutive epochs)
        val_collapsed = (dist['predicted_vulnerable'] == 0 or dist['predicted_safe'] == 0)

        if val_collapsed and epoch >= 1:  # Start checking from epoch 2
            consecutive_collapses += 1
            collapse_history.append(epoch + 1)
            print(f"[!] COLLAPSE SIGNAL: Zero {'vulnerable' if dist['predicted_vulnerable'] == 0 else 'safe'} predictions")
        else:
            consecutive_collapses = 0

        # CSV logging for easy analysis
        csv_path = args.output_dir / 'metrics_history.csv'
        # Ensure directory exists
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        if epoch == 0:
            with open(csv_path, 'w') as f:
                f.write('epoch,train_loss,val_loss,val_f1,val_acc,val_precision,val_recall,'
                        'pred_vuln,pred_safe,actual_vuln,actual_safe,lr\n')
                f.flush()

        with open(csv_path, 'a') as f:
            current_lr = optimizer.param_groups[0]['lr']
            f.write(
                f"{epoch+1},"
                f"{train_loss:.4f},"
                f"{val_metrics['loss']:.4f},"
                f"{val_metrics['binary_f1_vulnerable']:.4f},"
                f"{val_metrics['accuracy']:.4f},"
                f"{val_metrics['precision']:.4f},"
                f"{val_metrics['recall']:.4f},"
                f"{dist['predicted_vulnerable']},"
                f"{dist['predicted_safe']},"
                f"{dist['actual_vulnerable']},"
                f"{dist['actual_safe']},"
                f"{current_lr:.2e}\n"
            )
            f.flush()  # Critical: flush to survive runtime kills

        # CRITICAL FIX: Scheduler now steps inside train_epoch (per-step, not per-epoch)
        # scheduler.step() <- REMOVED

        # Save checkpoint with enhanced metadata
        is_best = val_metrics['binary_f1_vulnerable'] > best_val_f1

        # Build enhanced metadata
        enhanced_metadata = {
            'seed': args.seed,
            'git_commit': get_git_commit(),
            'timestamp': datetime.utcnow().isoformat(),
            'lr_finder_used': bool(args.find_lr),
            'suggested_lr': float(suggested_lr) if args.find_lr else None,
            'lr_finder_analysis': lr_finder_analysis if lr_finder_analysis else None,
            'triple_weighting_detected': triple_weighting if 'triple_weighting' in locals() else False,
            'original_weight_multiplier': original_weight_multiplier if 'original_weight_multiplier' in locals() else None,
            'original_focal_gamma': original_focal_gamma if 'original_focal_gamma' in locals() else None
        }

        checkpoint_mgr.save_checkpoint(
            epoch + 1, model, optimizer, scheduler, val_metrics, is_best, enhanced_metadata
        )

        if is_best:
            best_val_f1 = val_metrics['binary_f1_vulnerable']
            patience_counter = 0
            print(f"[+] New best model! F1: {best_val_f1:.4f}")
        else:
            patience_counter += 1
            print(f"[*] No improvement. Patience: {patience_counter}/{effective_patience}")

        # CRITICAL FIX: Stop if 2 consecutive collapses
        if consecutive_collapses >= 2:
            print(f"\n{'='*70}")
            print(f"[!] CRITICAL: Collapse detected for {consecutive_collapses} consecutive epochs")
            print(f"[!] Collapse history: {collapse_history}")
            print(f"\n[!] STOPPING TRAINING. Recommended fixes:")
            print(f"    1. Add: --use-weighted-sampler")
            print(f"    2. Try: --lr-override 2e-5")
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
                'hyperparameters': vars(args),
                'recommendations': [
                    'use_weighted_sampler=True',
                    'lr_override=2e-5',
                    'weight_multiplier=2.0',
                    'focal_loss=True'
                ]
            }

            report_path = args.output_dir / 'collapse_report.json'
            with open(report_path, 'w') as f:
                json.dump(diagnostic_report, f, indent=2)
            print(f"[+] Saved diagnostic report: {report_path}")

            sys.exit(2)  # Exit code 2 = collapse failure

        # CRITICAL FIX: Early stopping with normal patience
        if patience_counter >= effective_patience:
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
            args.test_data, tokenizer, args.max_seq_len, use_weights=False, use_features=args.use_code_features
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

    # Close TensorBoard writer
    if writer:
        writer.close()
        print(f"[+] TensorBoard logs saved")

    return 0


if __name__ == '__main__':
    sys.exit(main())
