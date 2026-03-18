# training/scripts/model/train.py
#
# StreamGuard M1 Training Loop.
#
# Usage:
#   python training/scripts/model/train.py \
#       --config C_plus_cfa \
#       --train-h5 training/data/final/train.h5 \
#       --val-h5 training/data/final/val.h5 \
#       --epochs 20 --use-cfa true
#
#   Dry-run (100 samples, 1 epoch):
#   python training/scripts/model/train.py \
#       --config C_plus_cfa \
#       --train-h5 training/data/final/train.h5 \
#       --val-h5 training/data/final/val.h5 \
#       --use-cfa true --epochs 1 --max-samples 100
#
# Carried-forward production notes:
#   1. Checkpoint config dict: saved alongside state_dict (Appendix Conflict 6)
#   2. Severity: lambda_sev=0.0 for M1 (SARD has no real CVSS scores)
#   3. LR scheduler: uses count_batches() for true step count (not len(loader))
#   4. Dataset reuse: CFADataset built once, sampler rebuilt per epoch
#   5. CodeBERT input_ids: not available in HDF5 — model runs graph-only mode.
#      This is correct for M1 SARD POC. input_ids support wired for M2.

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
try:
    from torch.amp import GradScaler  # PyTorch >= 2.1
except ImportError:
    from torch.cuda.amp import GradScaler  # PyTorch < 2.1
from transformers import get_linear_schedule_with_warmup

# Suppress known harmless warnings
warnings.filterwarnings("ignore", message=".*clean_up_tokenization_spaces.*")
warnings.filterwarnings("ignore", message=".*RequestsDependencyWarning.*")

from training.scripts.model.model import StreamGuardModel
from training.scripts.model.losses import StreamGuardLoss
from training.scripts.model.cfa_dataloader import (
    CFADataset,
    CFAAwareBatchSampler,
    cfa_collate_fn,
    count_batches,
)
from torch.utils.data import DataLoader


# ─────────────────────────────────────────────────────────────────────────────
# Ablation configs (from EXPERIMENTS.md)
# ─────────────────────────────────────────────────────────────────────────────

ABLATION_CONFIGS = {
    "B_plus_ggnn": {
        "use_cfa": False,
        "cpg_components": ["AST", "CFG", "DFG"],
    },
    "C_plus_cfa": {
        "use_cfa": True,
        "cpg_components": ["AST", "CFG", "DFG"],
    },
    "D_plus_tpg": {
        "use_cfa": True,
        "cpg_components": ["AST", "CFG", "DFG", "TPG"],
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# Training defaults (PRD.md §7.3)
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "base_model":            "microsoft/codebert-base",
    "node_feature_dim":      824,
    "optimizer":             "AdamW",
    "lr_codebert":           2e-5,
    "lr_ggnn_fusion":        1e-4,
    "weight_decay":          0.01,
    "warmup_ratio":          0.1,
    "batch_size_graphs":     8,
    "gradient_accumulation": 4,
    "max_seq_len":           512,
    "max_cpg_nodes":         200,
    "epochs":                20,
    "early_stopping_patience": 5,
    "seed":                  42,
    "grad_clip":             1.0,
    "lambda_ce":             1.0,
    "lambda_cfa":            0.5,
    "lambda_sev":            0.0,    # M1: no real CVSS scores in SARD data
    "cfa_margin":            0.5,
    "freeze_codebert_layers": 9,     # freeze layers 0-8
    "use_interproc":         False,
    "num_cwe_classes":       12,
}


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, dataloader, device, use_cfa=True):
    """
    Evaluate on val/test set. Returns metrics dict.

    Computes: F1, precision, recall, FPR, FNR, pairwise_accuracy.
    Pairwise accuracy: fraction of CFA pair_id groups where the model
    correctly predicts vuln=1 AND safe=0 for both members.
    """
    model.eval()
    all_preds, all_labels = [], []
    all_cwes = []
    pair_results = {}  # pair_id -> {vuln_correct, safe_correct}

    for batch in dataloader:
        batch = batch.to(device)

        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            out = model(batch)

        preds  = out["binary_logits"].argmax(dim=-1).cpu()
        labels = batch.y.cpu()
        all_preds.append(preds)
        all_labels.append(labels)

        # Collect CWE labels for per-CWE metrics
        if hasattr(batch, "cwe"):
            all_cwes.extend(batch.cwe)

        # Track pairwise accuracy
        if use_cfa and hasattr(batch, "pair_id"):
            pair_ids = batch.pair_id
            for idx, pid in enumerate(pair_ids):
                if pid in ("", "None", "null", "0"):
                    continue
                if pid not in pair_results:
                    pair_results[pid] = {"vuln_correct": None, "safe_correct": None}
                label = labels[idx].item()
                pred  = preds[idx].item()
                if label == 1:
                    pair_results[pid]["vuln_correct"] = (pred == 1)
                elif label == 0:
                    pair_results[pid]["safe_correct"] = (pred == 0)

    all_preds  = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    metrics = {
        "f1":        float(f1_score(all_labels, all_preds, zero_division=0)),
        "precision": float(precision_score(all_labels, all_preds, zero_division=0)),
        "recall":    float(recall_score(all_labels, all_preds, zero_division=0)),
    }

    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics["fpr"] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        metrics["fnr"] = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
    else:
        metrics["fpr"] = 0.0
        metrics["fnr"] = 0.0

    # Pairwise accuracy: both vuln and safe correctly classified
    valid_pairs = [
        v for v in pair_results.values()
        if v["vuln_correct"] is not None and v["safe_correct"] is not None
    ]
    if valid_pairs:
        metrics["pairwise_accuracy"] = float(
            sum(1 for v in valid_pairs if v["vuln_correct"] and v["safe_correct"])
            / len(valid_pairs)
        )
    else:
        metrics["pairwise_accuracy"] = 0.0

    # Per-CWE F1 (worst-group metric)
    if all_cwes and len(set(all_cwes)) > 1:
        cwe_f1s = {}
        for cwe in set(all_cwes):
            mask = np.array([c == cwe for c in all_cwes])
            if mask.sum() < 2:
                continue
            cwe_f1s[cwe] = float(f1_score(all_labels[mask], all_preds[mask], zero_division=0))
        if cwe_f1s:
            metrics["worst_cwe_f1"] = min(cwe_f1s.values())
            metrics["best_cwe_f1"]  = max(cwe_f1s.values())

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(model, optimizer, scheduler, epoch, best_f1, config, path):
    """
    Save checkpoint with full config for serving reconstruction.

    MUST include config dict (Appendix Conflict 6). Without it, the serving
    layer cannot reconstruct the architecture when loading best_model.pt.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    torch.save({
        "epoch":                epoch,
        "best_f1":              best_f1,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "config": {
            "node_feature_dim":  config["node_feature_dim"],
            "use_ggnn":          True,
            "ggnn_type":         "per_edge_type_gated",
            "cpg_components":    config["cpg_components"],
            "num_edge_types":    len(config["cpg_components"]),
            "use_cfa":           config["use_cfa"],
            "use_interproc":     config["use_interproc"],
            "num_cwe_classes":   config["num_cwe_classes"],
            "ablation_config":   config["config_name"],
            "base_model":        config["base_model"],
            "freeze_layers":     config["freeze_codebert_layers"],
        },
    }, tmp_path)
    os.replace(tmp_path, path)  # atomic write


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _select_with_complete_groups(index: list[dict], target_n: int) -> list[dict]:
    """
    Select approximately `target_n` samples from index, keeping complete
    pair_id groups together. Without this, --max-samples takes the first N
    sequential entries which scatter across pair groups, and CFA loss gets
    zero signal in dry-runs.
    """
    from collections import defaultdict
    groups = defaultdict(list)
    for i, item in enumerate(index):
        pid = item["pair_id"]
        if pid in ("", "None", "null", "0"):
            groups[f"__singleton_{i}"] = [i]
        else:
            groups[pid].append(i)

    selected = set()
    for group_indices in groups.values():
        if len(selected) >= target_n:
            break
        selected.update(group_indices)

    # Return items in original order
    selected_sorted = sorted(selected)
    return [index[i] for i in selected_sorted]


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(config: dict):
    """Main training loop."""
    # ── Reproducibility ──────────────────────────────────────────────
    seed = config["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")

    # ── Data ─────────────────────────────────────────────────────────
    # Build datasets ONCE (avoids re-scanning HDF5 metadata each epoch)
    max_samples = config.get("max_samples")

    train_dataset = CFADataset(config["train_h5"])
    val_dataset   = CFADataset(config["val_h5"])

    if max_samples:
        train_dataset.index = _select_with_complete_groups(
            train_dataset.index, max_samples
        )
        val_dataset.index = _select_with_complete_groups(
            val_dataset.index, max(max_samples // 4, 20)
        )
        print(f"Dry-run: {len(train_dataset)} train, {len(val_dataset)} val samples")

    # Build sampler once to compute true step count for scheduler
    batch_size = config["batch_size_graphs"]
    train_sampler = CFAAwareBatchSampler(train_dataset, batch_size, drop_last=True)
    batches_per_epoch = count_batches(train_sampler)

    val_sampler = CFAAwareBatchSampler(val_dataset, batch_size, drop_last=False)
    val_loader  = DataLoader(
        val_dataset, batch_sampler=val_sampler,
        num_workers=0, collate_fn=cfa_collate_fn,
    )

    print(f"Train: {len(train_dataset)} samples, {train_sampler._n_groups} groups, "
          f"{batches_per_epoch} batches/epoch")
    print(f"Val:   {len(val_dataset)} samples")

    # ── Model ────────────────────────────────────────────────────────
    model = StreamGuardModel(
        node_feature_dim=config["node_feature_dim"],
        codebert_model=config["base_model"],
        use_interproc=config["use_interproc"],
    ).to(device)

    # Freeze CodeBERT layers 0-8 (reduces trainable params by ~70%)
    freeze_n = config["freeze_codebert_layers"]
    frozen_params = 0
    for i, layer in enumerate(model.codebert.encoder.layer):
        if i < freeze_n:
            for p in layer.parameters():
                p.requires_grad = False
                frozen_params += p.numel()
    # Also freeze embeddings
    for p in model.codebert.embeddings.parameters():
        p.requires_grad = False
        frozen_params += p.numel()

    total_params    = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total_params:,} total, {trainable_params:,} trainable "
          f"({frozen_params:,} frozen)")

    # ── Loss ─────────────────────────────────────────────────────────
    use_cfa = config["use_cfa"]
    criterion = StreamGuardLoss(
        lambda_ce=config["lambda_ce"],
        lambda_cfa=config["lambda_cfa"] if use_cfa else 0.0,
        lambda_sev=config["lambda_sev"],
        cfa_margin=config["cfa_margin"],
    )

    # ── Optimizer with differential LR ───────────────────────────────
    # Only pass parameters that require grad to avoid AdamW warning
    codebert_params = [
        p for n, p in model.named_parameters()
        if "codebert" in n and p.requires_grad
    ]
    other_params = [
        p for n, p in model.named_parameters()
        if "codebert" not in n and p.requires_grad
    ]

    param_groups = []
    if codebert_params:
        param_groups.append({"params": codebert_params, "lr": config["lr_codebert"]})
    param_groups.append({"params": other_params, "lr": config["lr_ggnn_fusion"]})

    optimizer = torch.optim.AdamW(param_groups, weight_decay=config["weight_decay"])

    # ── Scheduler: linear warmup → cosine decay ─────────────────────
    # Use true batch count, not estimated len(sampler)
    total_steps  = batches_per_epoch * config["epochs"]
    warmup_steps = int(total_steps * config["warmup_ratio"])
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    print(f"Scheduler: {total_steps} total steps, {warmup_steps} warmup steps")

    # ── AMP ──────────────────────────────────────────────────────────
    use_amp = device.type == "cuda"
    scaler  = GradScaler(enabled=use_amp)

    # ── MLflow ───────────────────────────────────────────────────────
    try:
        import mlflow
        # Explicit tracking URI avoids WinError 5 on paths with spaces
        mlruns_dir = os.path.join(os.getcwd(), "mlruns")
        mlflow.set_tracking_uri(f"file:///{mlruns_dir}")
        mlflow.set_experiment("streamguard_m1_sard_proof")
        mlflow_available = True
    except Exception:
        mlflow_available = False
        print("MLflow not available — metrics logged to stdout only")

    # ── Checkpoint path ──────────────────────────────────────────────
    ckpt_dir = Path(config.get("checkpoint_dir", "training/checkpoints"))
    config_name = config["config_name"]
    best_ckpt_path = str(ckpt_dir / f"best_model_{config_name}.pt")

    # ── Training ─────────────────────────────────────────────────────
    best_val_f1  = -1.0  # not 0.0: ensures epoch 0 checkpoint always saves
    patience_ctr = 0
    grad_accum   = config["gradient_accumulation"]
    global_step  = 0

    # Suppress the zero-BERT warning during training — we know input_ids
    # is None for M1 and it's intentional.
    warnings.filterwarnings("ignore", message=".*input_ids is None.*")

    if mlflow_available:
        import mlflow as _mlflow
        run_context = _mlflow.start_run()
    else:
        run_context = _nullcontext()

    with run_context:
        if mlflow_available:
            # Log all config params (flatten nested lists for mlflow)
            log_config = {k: str(v) if isinstance(v, list) else v
                          for k, v in config.items()}
            mlflow.log_params(log_config)

        for epoch in range(config["epochs"]):
            epoch_start = time.time()

            # ── Build fresh sampler each epoch (reshuffles groups) ────
            train_sampler = CFAAwareBatchSampler(
                train_dataset, batch_size, drop_last=True
            )
            train_loader = DataLoader(
                train_dataset, batch_sampler=train_sampler,
                num_workers=0, collate_fn=cfa_collate_fn,
            )

            # ── Train epoch ──────────────────────────────────────────
            model.train()
            optimizer.zero_grad()
            epoch_losses = {"L_CE": 0.0, "L_CFA": 0.0, "L_severity": 0.0, "L_total": 0.0}
            n_steps = 0
            cfa_batches = 0

            for step, batch in enumerate(train_loader):
                batch = batch.to(device)

                with torch.autocast(device_type=device.type, enabled=use_amp):
                    out = model(batch)
                    total_loss, loss_dict = criterion(out, batch, use_cfa=use_cfa)

                # Track CFA activity
                if loss_dict["L_CFA"] > 0:
                    cfa_batches += 1

                # Gradient accumulation: scale loss
                scaled_loss = total_loss / grad_accum
                scaler.scale(scaled_loss).backward()

                if (step + 1) % grad_accum == 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                # Accumulate epoch losses
                for k in epoch_losses:
                    epoch_losses[k] += loss_dict.get(k, 0.0)
                n_steps += 1

                # Log every 50 steps
                if mlflow_available and step % 50 == 0 and step > 0:
                    mlflow.log_metrics(
                        {f"train_{k}": v / n_steps for k, v in epoch_losses.items()},
                        step=global_step,
                    )

            # Average epoch losses
            avg_losses = {k: v / max(n_steps, 1) for k, v in epoch_losses.items()}

            # ── Validation ───────────────────────────────────────────
            val_metrics = evaluate(model, val_loader, device, use_cfa=use_cfa)
            epoch_time = time.time() - epoch_start

            # ── Logging ──────────────────────────────────────────────
            val_f1 = val_metrics["f1"]
            print(
                f"Epoch {epoch:2d}/{config['epochs']} | "
                f"train_loss={avg_losses['L_total']:.4f} "
                f"(CE={avg_losses['L_CE']:.4f} CFA={avg_losses['L_CFA']:.4f}) | "
                f"val_F1={val_f1:.4f} prec={val_metrics['precision']:.4f} "
                f"rec={val_metrics['recall']:.4f} | "
                f"pairwise_acc={val_metrics['pairwise_accuracy']:.4f} | "
                f"CFA_batches={cfa_batches}/{n_steps} | "
                f"{epoch_time:.1f}s"
            )

            if mlflow_available:
                mlflow.log_metrics({
                    "train_loss":      avg_losses["L_total"],
                    "train_L_CE":      avg_losses["L_CE"],
                    "train_L_CFA":     avg_losses["L_CFA"],
                    "train_L_severity": avg_losses["L_severity"],
                    "val_f1":          val_f1,
                    "val_precision":   val_metrics["precision"],
                    "val_recall":      val_metrics["recall"],
                    "val_fpr":         val_metrics.get("fpr", 0),
                    "val_fnr":         val_metrics.get("fnr", 0),
                    "val_pairwise_accuracy": val_metrics["pairwise_accuracy"],
                    "cfa_batch_ratio": cfa_batches / max(n_steps, 1),
                    "epoch_time_s":    epoch_time,
                    "lr":              scheduler.get_last_lr()[-1],
                }, step=epoch)
                if "worst_cwe_f1" in val_metrics:
                    mlflow.log_metrics({
                        "val_worst_cwe_f1": val_metrics["worst_cwe_f1"],
                        "val_best_cwe_f1":  val_metrics["best_cwe_f1"],
                    }, step=epoch)

            # ── Checkpoint + Early stopping ──────────────────────────
            if val_f1 > best_val_f1:
                best_val_f1  = val_f1
                patience_ctr = 0
                save_checkpoint(
                    model, optimizer, scheduler, epoch,
                    best_val_f1, config, best_ckpt_path,
                )
                print(f"  -> Saved best model (F1={best_val_f1:.4f}) to {best_ckpt_path}")
            else:
                patience_ctr += 1
                if patience_ctr >= config["early_stopping_patience"]:
                    print(f"Early stopping at epoch {epoch}. "
                          f"Best val F1: {best_val_f1:.4f}")
                    break

    print(f"\nTraining complete. Best val F1: {best_val_f1:.4f}")
    print(f"Checkpoint: {best_ckpt_path}")
    return best_val_f1


class _nullcontext:
    """Minimal context manager for when mlflow is unavailable."""
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="StreamGuard M1 training")
    p.add_argument("--config", choices=list(ABLATION_CONFIGS.keys()),
                    default="C_plus_cfa",
                    help="Ablation config name")
    p.add_argument("--train-h5", required=True, help="Path to train.h5")
    p.add_argument("--val-h5",   required=True, help="Path to val.h5")
    p.add_argument("--use-cfa",  default=None,
                    help="Override CFA setting (true/false)")
    p.add_argument("--cpg-components", nargs="+", default=None,
                    help="Override CPG components (AST CFG DFG TPG)")
    p.add_argument("--epochs",       type=int, default=None)
    p.add_argument("--max-samples",  type=int, default=None,
                    help="Limit samples for dry-run")
    p.add_argument("--batch-size",   type=int, default=None)
    p.add_argument("--checkpoint-dir", default=None)
    p.add_argument("--seed",         type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()

    # Start from defaults
    config = dict(DEFAULT_CONFIG)

    # Overlay ablation config
    ablation = ABLATION_CONFIGS[args.config]
    config["use_cfa"]         = ablation["use_cfa"]
    config["cpg_components"]  = ablation["cpg_components"]
    config["config_name"]     = args.config

    # CLI overrides
    config["train_h5"] = args.train_h5
    config["val_h5"]   = args.val_h5

    if args.use_cfa is not None:
        config["use_cfa"] = args.use_cfa.lower() in ("true", "1", "yes")
    if args.cpg_components is not None:
        config["cpg_components"] = args.cpg_components
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.max_samples is not None:
        config["max_samples"] = args.max_samples
    if args.batch_size is not None:
        config["batch_size_graphs"] = args.batch_size
    if args.checkpoint_dir is not None:
        config["checkpoint_dir"] = args.checkpoint_dir
    if args.seed is not None:
        config["seed"] = args.seed

    print(f"Config: {config['config_name']}")
    print(f"  use_cfa={config['use_cfa']}, "
          f"cpg_components={config['cpg_components']}, "
          f"epochs={config['epochs']}, "
          f"max_samples={config.get('max_samples', 'all')}")
    print()

    train(config)


if __name__ == "__main__":
    main()
