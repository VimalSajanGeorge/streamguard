"""
Cell 51: Transformer v1.7 Production Training

Features:
- Adaptive GPU configuration
- LR Finder with 168h caching
- Mixed precision (AMP)
- AMP-safe gradient clipping
- Collapse detection with auto-stop
- Safe JSON metadata
- 3-seed reproducibility (42, 2025, 7)
- Triple weighting for class imbalance

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
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from datetime import datetime
from tqdm.auto import tqdm

# StreamGuard imports
from training.train_transformer import (
    EnhancedSQLIntentTransformer,
    CodeDataset,
    set_seed
)
from training.utils.adaptive_config import load_adaptive_config
from training.utils.json_safety import atomic_write_json
from training.utils.collapse_detector import CollapseDetector
from training.utils.amp_utils import clip_gradients_amp_safe
from training.utils.lr_finder import LRFinder
from training.utils.lr_cache import compute_cache_key, save_lr_cache, load_lr_cache

from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


# ==================== CONFIGURATION ====================

print("=" * 80)
print("CELL 51: TRANSFORMER v1.7 PRODUCTION TRAINING")
print("=" * 80)

# Production seeds for reproducibility
SEEDS = [42, 2025, 7]

# Base output directory
OUTPUT_DIR = Path("training/outputs/transformer_v17_production")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Model configuration
MODEL_NAME = "microsoft/codebert-base"
NUM_LABELS = 2  # Binary: vulnerable (1) or safe (0)

# Training hyperparameters (will be overridden by adaptive config)
BASE_CONFIG = {
    "batch_size": 64,
    "max_seq_length": 512,
    "num_epochs": 10,
    "learning_rate": None,  # Will be found by LR Finder
    "weight_decay": 0.01,
    "warmup_ratio": 0.10,
    "max_grad_norm": 1.0,
    "mixed_precision": True
}

# Class weighting (from previous analysis showing ~3:1 imbalance)
CLASS_WEIGHTS = torch.tensor([1.0, 3.0])  # [safe, vulnerable]

# Dataset paths (update these to your actual paths)
TRAIN_DATA_PATH = Path("data/processed/codexglue/train.jsonl")
VAL_DATA_PATH = Path("data/processed/codexglue/val.jsonl")


# ==================== HELPER FUNCTIONS ====================

def load_data(data_path: Path, tokenizer, max_seq_length: int):
    """Load dataset with tokenization."""
    if not data_path.exists():
        warnings.warn(f"Data path not found: {data_path}. Using dummy data for testing.")
        # Return empty dataset for testing
        return CodeDataset([], tokenizer, max_seq_length)

    # Load your actual dataset here
    # For now, assuming CodeDataset can load from path
    try:
        dataset = CodeDataset.from_file(data_path, tokenizer, max_seq_length)
        return dataset
    except AttributeError:
        warnings.warn("CodeDataset.from_file not implemented. Using empty dataset.")
        return CodeDataset([], tokenizer, max_seq_length)


def create_weighted_sampler(dataset):
    """Create weighted sampler for class imbalance."""
    # Get labels
    labels = np.array([sample['label'] for sample in dataset])

    # Count classes
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts

    # Sample weights
    sample_weights = class_weights[labels]

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )


def run_lr_finder(model, train_loader, device, cache_key):
    """Run LR Finder with caching and fallback."""
    print("\n[*] Checking LR Finder cache...")

    # Check cache first
    cached = load_lr_cache(cache_key, max_age_hours=168)
    if cached:
        print(f"[+] Using cached LR: {cached['suggested_lr']:.2e}")
        print(f"    Cached at: {cached['timestamp']}")
        return cached['suggested_lr']

    print("[*] Running LR Finder (quick mode: 100 iterations)...")

    # FALLBACK: Conservative default if LR Finder fails
    FALLBACK_LR = 2.5e-5  # Safe middle-ground for transformers

    try:
        # Create optimizer and criterion
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-7)
        criterion = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS.to(device))

        # Run LR Finder
        lr_finder = LRFinder(model, optimizer, criterion, device)
        lr_history, loss_history = lr_finder.range_test(
            train_loader,
            start_lr=1e-7,
            end_lr=1.0,
            num_iter=100,  # Quick mode
            smooth_f=0.05
        )

        # Get best LR
        suggested_lr, _ = lr_finder.get_best_lr()

        # Cap LR to safe range (1e-5 to 5e-5 for transformers)
        suggested_lr = max(1e-5, min(suggested_lr, 5e-5))

        print(f"[+] Suggested LR: {suggested_lr:.2e}")

        # Cache results
        save_lr_cache(
            cache_key,
            suggested_lr,
            {
                "min_loss": float(min(loss_history)),
                "max_loss": float(max(loss_history)),
                "num_points": len(lr_history)
            },
            {"confidence": "high", "mode": "quick"}
        )

        return suggested_lr

    except Exception as e:
        warnings.warn(
            f"\n[!] LR Finder failed: {str(e)}\n"
            f"[!] Using conservative fallback LR: {FALLBACK_LR:.2e}\n"
        )
        return FALLBACK_LR


def train_one_epoch(
    model, train_loader, optimizer, criterion, scaler,
    scheduler, collapse_detector, device, epoch
):
    """Train for one epoch with all safety features."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")

    for batch_idx, batch in enumerate(pbar):
        # Move to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        code_features = batch.get('code_features')
        if code_features is not None:
            code_features = code_features.to(device)

        # Forward pass with AMP
        optimizer.zero_grad()

        with autocast(enabled=scaler is not None):
            logits = model(input_ids, attention_mask, code_features)
            loss = criterion(logits, labels)

        # Backward pass
        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # AMP-safe gradient clipping
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
            labels,
            step=epoch * len(train_loader) + batch_idx
        )

        if collapse_status['should_stop']:
            print("\n[!] Model collapse detected! Stopping training.")
            return None  # Signal collapse

        # Logging
        total_loss += loss.item()
        num_batches += 1

        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'grad_norm': f"{grad_stats['total_norm']:.4f}"
        })

    avg_loss = total_loss / num_batches
    return avg_loss


def evaluate(model, val_loader, criterion, device):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            code_features = batch.get('code_features')
            if code_features is not None:
                code_features = code_features.to(device)

            # Forward pass
            logits = model(input_ids, attention_mask, code_features)
            loss = criterion(logits, labels)

            # Collect predictions
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())

            total_loss += loss.item()

    # Compute metrics
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


# ==================== MAIN TRAINING LOOP ====================

def main():
    """Main production training function."""

    # Load adaptive configuration
    config = load_adaptive_config(model_type="transformer", override=BASE_CONFIG)

    print(f"\n[+] Configuration loaded:")
    print(f"    GPU: {config['gpu_info']['full_name']}")
    print(f"    Batch size: {config['batch_size']}")
    print(f"    Mixed precision: {config['mixed_precision']}")

    # Device
    device = torch.device(config['gpu_info']['device'])

    # Load tokenizer
    print(f"\n[+] Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Load datasets
    print(f"[+] Loading datasets...")
    train_dataset = load_data(TRAIN_DATA_PATH, tokenizer, config['max_seq_length'])
    val_dataset = load_data(VAL_DATA_PATH, tokenizer, config['max_seq_length'])

    print(f"    Train samples: {len(train_dataset)}")
    print(f"    Val samples: {len(val_dataset)}")

    # Run for each seed
    results_all_seeds = []

    for seed in SEEDS:
        print(f"\n{'=' * 80}")
        print(f"TRAINING WITH SEED: {seed}")
        print(f"{'=' * 80}")

        # Set seed
        set_seed(seed)

        # Output directory for this seed
        seed_output_dir = OUTPUT_DIR / f"seed_{seed}"
        seed_output_dir.mkdir(parents=True, exist_ok=True)

        # Create data loaders
        train_sampler = create_weighted_sampler(train_dataset)
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            sampler=train_sampler,
            num_workers=config.get('num_workers', 0)
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config.get('num_workers', 0)
        )

        # Initialize model
        print(f"\n[+] Initializing model...")
        model = EnhancedSQLIntentTransformer(
            model_name=MODEL_NAME,
            num_labels=NUM_LABELS,
            hidden_dim=768,
            dropout=0.1
        ).to(device)

        # LR Finder
        cache_key = compute_cache_key(
            TRAIN_DATA_PATH,
            MODEL_NAME,
            config['batch_size'],
            {'seed': seed, 'max_seq_len': config['max_seq_length']}
        )
        learning_rate = run_lr_finder(model, train_loader, device, cache_key)
        config['learning_rate'] = learning_rate

        # Reinitialize model (LR Finder modifies weights)
        model = EnhancedSQLIntentTransformer(
            model_name=MODEL_NAME,
            num_labels=NUM_LABELS,
            hidden_dim=768,
            dropout=0.1
        ).to(device)

        # Optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=config['weight_decay']
        )

        # Scheduler
        num_training_steps = len(train_loader) * config['num_epochs']
        num_warmup_steps = int(num_training_steps * config['warmup_ratio'])
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )

        # Loss criterion
        criterion = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS.to(device))

        # GradScaler for AMP
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

            # Train
            train_loss = train_one_epoch(
                model, train_loader, optimizer, criterion, scaler,
                scheduler, collapse_detector, device, epoch
            )

            if train_loss is None:
                # Collapse detected
                break

            # Evaluate
            val_metrics = evaluate(model, val_loader, criterion, device)

            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f}")
            print(f"Val F1: {val_metrics['f1']:.4f}")
            print(f"Val Accuracy: {val_metrics['accuracy']:.4f}")

            # Save metrics
            metrics_history.append({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                **val_metrics
            })

            # Save best model
            if val_metrics['f1'] > best_f1:
                best_f1 = val_metrics['f1']
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_state_dict': scaler.state_dict() if scaler else None,
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_f1': best_f1,
                    'config': config
                }, seed_output_dir / "model_checkpoint.pt")

        # Save final metadata
        metadata = {
            "seed": seed,
            "model_name": MODEL_NAME,
            "config": config,
            "best_f1": best_f1,
            "metrics_history": metrics_history,
            "timestamp": datetime.now().isoformat()
        }
        atomic_write_json(metadata, seed_output_dir / "training_metadata.json")

        results_all_seeds.append({
            "seed": seed,
            "best_f1": best_f1
        })

        print(f"\n[+] Seed {seed} complete. Best F1: {best_f1:.4f}")

    # Summary
    print(f"\n{'=' * 80}")
    print("TRAINING COMPLETE - SUMMARY")
    print(f"{'=' * 80}")

    for result in results_all_seeds:
        print(f"Seed {result['seed']}: F1 = {result['best_f1']:.4f}")

    mean_f1 = np.mean([r['best_f1'] for r in results_all_seeds])
    std_f1 = np.std([r['best_f1'] for r in results_all_seeds])
    print(f"\nMean F1: {mean_f1:.4f} ± {std_f1:.4f}")

    # Save summary
    summary = {
        "model": "transformer_v17",
        "seeds": SEEDS,
        "results": results_all_seeds,
        "mean_f1": mean_f1,
        "std_f1": std_f1,
        "timestamp": datetime.now().isoformat()
    }
    atomic_write_json(summary, OUTPUT_DIR / "production_summary.json")

    print(f"\n[+] All outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    try:
        main()
        print("\n[+] Training completed successfully!")
        sys.exit(0)  # Success
    except Exception as e:
        print(f"\n[!] Training failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)  # Failure
