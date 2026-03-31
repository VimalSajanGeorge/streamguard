# training/scripts/model/train.py
#
# Phase 4 StreamGuard Training Loop (M2).
#
# Key changes from M1:
#   1. Tokenize on-the-fly: JSONL lookup dict → CodeBERT tokenizer per batch
#   2. CFA-aware forward: orig_batch + cfa_batch from cfa_collate_fn
#   3. AMP with GradScaler (saves ~40% VRAM)
#   4. Differential LR: CodeBERT=2e-5, GGNN/fusion=1e-4
#   5. Gradient accumulation: effective batch = batch_size × grad_accum
#   6. Checkpoint resume: latest checkpoint per epoch, resume on startup
#   7. --dry-run: 1 batch × 2 epochs, save checkpoint, exit
#   8. 6 ablation configs: A, B, B', C, D, E
#   9. Test set evaluated ONCE at end with best val checkpoint (R-01)
#
# Risk mitigations:
#   R-01: test set never used during training
#   R-05: all configs share same HDF5 paths
#   R-06: all configs share seed=42
#   R-07: tokenize_batch asserts input_ids is not None
#   R-08: latest checkpoint saved every epoch
#   R-09: AMP enabled by default on GPU
#   R-10: gradient clipping at max_norm=1.0
#   R-11: differential LR (CodeBERT=2e-5)
#   R-12: build_optimizer creates two param groups
#   R-14: use_amp = True on CUDA
#   R-19: JSONL lookup + placeholder for missing code
#   R-20: scheduler steps only on optimizer steps
#
# Usage:
#   python -m training.scripts.model.train \
#       --config D_plus_tpg \
#       --train-h5 training/data/final/train.h5 \
#       --val-h5 training/data/final/val.h5 \
#       --test-h5 training/data/final/test.h5 \
#       --dry-run
#
#   python -m training.scripts.model.train \
#       --config E_full \
#       --train-h5 training/data/final/train.h5 \
#       --val-h5 training/data/final/val.h5 \
#       --test-h5 training/data/final/test.h5 \
#       --epochs 20 --batch-size 8

import argparse
import json
import os
import random
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

try:
    from torch.amp import GradScaler, autocast  # PyTorch >= 2.4
except ImportError:
    from torch.cuda.amp import GradScaler, autocast  # older PyTorch

from transformers import AutoTokenizer, get_linear_schedule_with_warmup

warnings.filterwarnings("ignore", message=".*clean_up_tokenization_spaces.*")
warnings.filterwarnings("ignore", message=".*RequestsDependencyWarning.*")

from training.scripts.model.model import StreamGuardModel
from training.scripts.model.losses import StreamGuardLoss, CWE_LABEL_MAP
from training.scripts.model.cfa_dataloader import (
    CFADataset,
    CFAAwareBatchSampler,
    cfa_collate_fn,
    count_batches,
    build_dataloader,
    EMPTY_PAIR_SENTINELS,
)
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Ablation configs (§8.5 of Phase 4 doc)
# ---------------------------------------------------------------------------
# All share the same BASE_CONFIG, only these flags differ.

ABLATION_CONFIGS = {
    "A_baseline": {
        "use_graph": False,
        "type_aware_edges": True,
        "use_cfa": False,
        "cpg_components": ["AST", "CFG", "DFG"],
        "use_interproc": False,
    },
    "B_plus_ggnn": {
        "use_graph": True,
        "type_aware_edges": False,
        "use_cfa": False,
        "cpg_components": ["AST", "CFG", "DFG"],
        "use_interproc": False,
    },
    "B_prime_type_aware": {
        "use_graph": True,
        "type_aware_edges": True,
        "use_cfa": False,
        "cpg_components": ["AST", "CFG", "DFG"],
        "use_interproc": False,
    },
    "C_plus_cfa": {
        "use_graph": True,
        "type_aware_edges": True,
        "use_cfa": True,
        "cpg_components": ["AST", "CFG", "DFG"],
        "use_interproc": False,
    },
    "D_plus_tpg": {
        "use_graph": True,
        "type_aware_edges": True,
        "use_cfa": True,
        "cpg_components": ["AST", "CFG", "DFG", "TPG"],
        "use_interproc": False,
    },
    "E_full": {
        "use_graph": True,
        "type_aware_edges": True,
        "use_cfa": True,
        "cpg_components": ["AST", "CFG", "DFG", "TPG"],
        "use_interproc": True,
    },
}

# ---------------------------------------------------------------------------
# Training defaults
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "base_model":              "microsoft/codebert-base",
    "node_feature_dim":        824,
    "optimizer":               "AdamW",
    "lr_codebert":             2e-5,
    "lr_ggnn_fusion":          1e-4,
    "weight_decay":            0.01,
    "warmup_ratio":            0.1,
    "batch_size_graphs":       8,
    "gradient_accumulation":   4,
    "max_seq_len":             512,
    "epochs":                  20,
    "early_stopping_patience": 5,
    "seed":                    42,
    "grad_clip":               1.0,
    "lambda_ce":               1.0,
    "lambda_cfa":              0.5,
    "lambda_sev":              0.1,
    "cfa_margin":              0.5,
    "freeze_codebert_layers":  9,
    "num_cwe_classes":         12,
    "use_graph":               True,
    "type_aware_edges":        True,
    "use_cfa":                 True,
    "use_interproc":           False,
    "cpg_components":          ["AST", "CFG", "DFG", "TPG"],
}


# ---------------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------------

def set_all_seeds(seed: int):
    """Full reproducibility: torch, CUDA, numpy, random, cuDNN."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Optimizer with differential LR (R-11, R-12)
# ---------------------------------------------------------------------------

def build_optimizer(model: nn.Module, config: dict) -> torch.optim.AdamW:
    """
    Two param groups: CodeBERT params at lr_codebert, everything else at lr_ggnn_fusion.
    Only includes parameters with requires_grad=True.
    """
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

    return torch.optim.AdamW(param_groups, weight_decay=config["weight_decay"])


# ---------------------------------------------------------------------------
# Tokenization on-the-fly (R-07, R-19)
# ---------------------------------------------------------------------------

def load_code_lookup(jsonl_paths: list[str]) -> dict[str, str]:
    """
    Load sample_id → code mapping from ALL existing JSONL files.

    Merges all files; earlier files take priority for duplicate IDs.
    This is critical because with_cfa/ has ~49K IDs but HDF5 has ~42K
    samples — the gap (~8.8%) only exists in cleaned/ or deduped/.
    """
    lookup = {}
    for path in jsonl_paths:
        if os.path.exists(path):
            added = 0
            with open(path, encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    sid = rec["id"]
                    if sid not in lookup:  # earlier files take priority
                        lookup[sid] = rec.get("code", "")
                        added += 1
            print(f"  Code lookup: {path} → {added} new IDs")
    return lookup


def tokenize_batch(
    codes: list[str],
    tokenizer,
    max_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Tokenize a list of code strings for CodeBERT.

    Empty/whitespace-only strings get a C placeholder to avoid degenerate
    tokenization (R-19).

    Returns: (input_ids: (B, max_length), attention_mask: (B, max_length))
    """
    PLACEHOLDER = "void placeholder(void){}"
    clean = [c if c.strip() else PLACEHOLDER for c in codes]
    enc = tokenizer(
        clean,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    return enc["input_ids"].to(device), enc["attention_mask"].to(device)


# ---------------------------------------------------------------------------
# Evaluation (R-01: test set evaluated ONCE at end)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model,
    dataloader,
    device,
    tokenizer,
    code_lookup,
    config,
):
    """
    Evaluate on val or test set.

    Returns dict with: f1, precision, recall, fpr, fnr, pairwise_accuracy,
    worst_cwe_f1, per_cwe_f1.
    """
    model.eval()
    all_preds, all_labels = [], []
    pair_results = {}
    cwe_groups = defaultdict(lambda: {"preds": [], "labels": []})
    use_graph = config.get("use_graph", True)
    max_seq_len = config.get("max_seq_len", 512)

    for orig_batch, cfa_batch, orig_metas, cfa_metas in dataloader:
        orig_batch = orig_batch.to(device)

        # Tokenize orig
        orig_codes = [code_lookup.get(m["sample_id"], "") for m in orig_metas]
        input_ids, attn_mask = tokenize_batch(orig_codes, tokenizer, max_seq_len, device)

        with autocast(device_type=device.type, enabled=(device.type == "cuda")):
            out = model(
                orig_batch,
                input_ids=input_ids,
                attention_mask=attn_mask,
            )

        preds = out["logits"].argmax(dim=-1).cpu()
        labels = orig_batch.y.cpu()
        all_preds.append(preds)
        all_labels.append(labels)

        # Per-CWE tracking
        for i, m in enumerate(orig_metas):
            cwe = m.get("cwe", "")
            if cwe in CWE_LABEL_MAP:
                cwe_groups[cwe]["preds"].append(preds[i].item())
                cwe_groups[cwe]["labels"].append(labels[i].item())

        # Pairwise accuracy (CFA pairs)
        if cfa_batch is not None:
            cfa_batch_dev = cfa_batch.to(device)
            cfa_codes = [code_lookup.get(m["sample_id"], "") for m in cfa_metas]
            cfa_ids, cfa_mask = tokenize_batch(cfa_codes, tokenizer, max_seq_len, device)

            with autocast(device_type=device.type, enabled=(device.type == "cuda")):
                out_cfa = model(cfa_batch_dev, input_ids=cfa_ids, attention_mask=cfa_mask)

            cfa_preds = out_cfa["logits"].argmax(dim=-1).cpu()

            for i, m in enumerate(orig_metas):
                pid = m.get("pair_id", "")
                if pid in EMPTY_PAIR_SENTINELS:
                    continue
                if pid not in pair_results:
                    pair_results[pid] = {"vuln_correct": None, "safe_correct": None}
                label_val = labels[i].item()
                pred_val = preds[i].item()
                if label_val == 1:
                    pair_results[pid]["vuln_correct"] = (pred_val == 1)
                elif label_val == 0:
                    pair_results[pid]["safe_correct"] = (pred_val == 0)

            for i, m in enumerate(cfa_metas):
                pid = m.get("pair_id", "")
                if pid in EMPTY_PAIR_SENTINELS:
                    continue
                if pid not in pair_results:
                    pair_results[pid] = {"vuln_correct": None, "safe_correct": None}
                if i < len(cfa_batch_dev.y):
                    cfa_label = cfa_batch_dev.y[i].item()
                    cfa_pred = cfa_preds[i].item()
                    if cfa_label == 1:
                        pair_results[pid]["vuln_correct"] = (cfa_pred == 1)
                    elif cfa_label == 0:
                        pair_results[pid]["safe_correct"] = (cfa_pred == 0)

    all_preds = torch.cat(all_preds).numpy()
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

    # Pairwise accuracy
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

    # Per-CWE F1
    per_cwe_f1 = {}
    for cwe_name, data in cwe_groups.items():
        if len(data["labels"]) >= 5:
            per_cwe_f1[cwe_name] = float(f1_score(
                data["labels"], data["preds"], zero_division=0,
            ))
    metrics["per_cwe_f1"] = per_cwe_f1
    metrics["worst_cwe_f1"] = min(per_cwe_f1.values()) if per_cwe_f1 else 0.0
    metrics["best_cwe_f1"] = max(per_cwe_f1.values()) if per_cwe_f1 else 0.0

    return metrics


# ---------------------------------------------------------------------------
# Checkpoint save/load (R-08, R-25)
# ---------------------------------------------------------------------------

def save_checkpoint(
    model, optimizer, scheduler, scaler, epoch, best_f1,
    patience_ctr, config, path,
):
    """Atomic checkpoint: write .tmp then os.replace()."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = str(path) + ".tmp"
    torch.save({
        "epoch":                epoch,
        "best_f1":              best_f1,
        "patience_ctr":         patience_ctr,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict":    scaler.state_dict(),
        "config": {
            "node_feature_dim":    config["node_feature_dim"],
            "use_graph":           config["use_graph"],
            "type_aware_edges":    config["type_aware_edges"],
            "ggnn_type":           "per_edge_type_gated",
            "cpg_components":      config["cpg_components"],
            "num_edge_types":      len(config["cpg_components"]),
            "use_cfa":             config["use_cfa"],
            "use_interproc":       config["use_interproc"],
            "num_cwe_classes":     config["num_cwe_classes"],
            "ablation_config":     config["config_name"],
            "base_model":          config["base_model"],
            "freeze_layers":       config["freeze_codebert_layers"],
            "seed":                config["seed"],
        },
    }, tmp_path)
    os.replace(tmp_path, str(path))


def load_checkpoint(path, model, optimizer, scheduler, scaler, device):
    """Load checkpoint and return (start_epoch, best_f1, patience_ctr)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if "scaler_state_dict" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    start_epoch = ckpt["epoch"] + 1
    best_f1 = ckpt["best_f1"]
    patience_ctr = ckpt.get("patience_ctr", 0)
    return start_epoch, best_f1, patience_ctr


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _select_with_complete_groups(index: list[dict], target_n: int) -> list[dict]:
    """Select ~target_n samples keeping pair_id groups intact."""
    groups = defaultdict(list)
    for i, item in enumerate(index):
        pid = item["pair_id"]
        if pid in EMPTY_PAIR_SENTINELS:
            groups[f"__singleton_{i}"] = [i]
        else:
            groups[pid].append(i)

    selected = set()
    for group_indices in groups.values():
        if len(selected) >= target_n:
            break
        selected.update(group_indices)

    return [index[i] for i in sorted(selected)]


class _NullContext:
    """Context manager fallback when MLflow unavailable."""
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(config: dict):
    """Main training loop. Returns best_val_f1."""
    # ── R-29: Guard against val == test path ──────────────────────
    test_h5 = config.get("test_h5")
    if test_h5 and os.path.abspath(config["val_h5"]) == os.path.abspath(test_h5):
        raise ValueError(
            f"val_h5 and test_h5 point to the same file: {config['val_h5']}. "
            "This would contaminate the test set during early stopping."
        )

    # ── Reproducibility ───────────────────────────────────────────
    set_all_seeds(config["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        vram = getattr(props, 'total_memory', None) or getattr(props, 'total_mem', 0)
        print(f"VRAM: {vram / 1024**3:.1f} GB")

    # ── Tokenizer ─────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(config["base_model"])

    # ── Code lookup (R-19) ────────────────────────────────────────
    code_lookup = load_code_lookup([
        "training/data/processed/with_cfa/samples.jsonl",
        "training/data/processed/deduped/samples.jsonl",
        "training/data/processed/cleaned/samples.jsonl",
    ])
    print(f"Code lookup: {len(code_lookup)} samples loaded")

    # ── Data ──────────────────────────────────────────────────────
    dry_run = config.get("dry_run", False)
    max_samples = config.get("max_samples")

    train_dataset = CFADataset(config["train_h5"])
    val_dataset = CFADataset(config["val_h5"])

    if dry_run or max_samples:
        n = max_samples or 32
        train_dataset.index = _select_with_complete_groups(train_dataset.index, n)
        val_dataset.index = _select_with_complete_groups(
            val_dataset.index, max(n // 4, 8),
        )
        print(f"Dry-run: {len(train_dataset)} train, {len(val_dataset)} val samples")

    batch_size = config["batch_size_graphs"]
    train_sampler = CFAAwareBatchSampler(
        train_dataset, batch_size, drop_last=True, shuffle=True, seed=config["seed"],
    )
    batches_per_epoch = count_batches(train_sampler)

    val_sampler = CFAAwareBatchSampler(
        val_dataset, batch_size, drop_last=False, shuffle=False,
    )
    val_loader = DataLoader(
        val_dataset, batch_sampler=val_sampler,
        num_workers=0, collate_fn=cfa_collate_fn,
    )

    print(f"Train: {len(train_dataset)} samples, {train_sampler._n_groups} groups, "
          f"{batches_per_epoch} batches/epoch")
    print(f"Val:   {len(val_dataset)} samples")

    # ── Model ─────────────────────────────────────────────────────
    model = StreamGuardModel(
        node_feature_dim=config["node_feature_dim"],
        codebert_model=config["base_model"],
        use_interproc=config["use_interproc"],
        freeze_codebert_layers=config["freeze_codebert_layers"],
        use_graph=config.get("use_graph", True),
        type_aware_edges=config.get("type_aware_edges", True),
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total_params:,} total, {trainable_params:,} trainable")

    # ── Loss ──────────────────────────────────────────────────────
    use_cfa = config["use_cfa"]
    criterion = StreamGuardLoss(
        lambda_ce=config["lambda_ce"],
        lambda_cfa=config["lambda_cfa"] if use_cfa else 0.0,
        lambda_sev=config["lambda_sev"],
        cfa_margin=config["cfa_margin"],
    )

    # ── Optimizer (R-11, R-12) ────────────────────────────────────
    optimizer = build_optimizer(model, config)

    # Verify differential LR (G-P4-05)
    lr_info = [(g["lr"], len(g["params"])) for g in optimizer.param_groups]
    print(f"Optimizer param groups: {lr_info}")

    # ── Scheduler ─────────────────────────────────────────────────
    epochs = 2 if dry_run else config["epochs"]
    total_steps = batches_per_epoch * epochs
    warmup_steps = int(total_steps * config["warmup_ratio"])
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    print(f"Scheduler: {total_steps} total steps, {warmup_steps} warmup steps")

    # ── AMP (R-09, R-14) ─────────────────────────────────────────
    use_amp = device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    # ── Resume checkpoint (R-08) ──────────────────────────────────
    ckpt_dir = Path(config.get("checkpoint_dir", "training/checkpoints"))
    config_name = config["config_name"]
    latest_ckpt_path = ckpt_dir / f"latest_{config_name}.pt"
    best_ckpt_path = ckpt_dir / f"best_model_{config_name}.pt"

    start_epoch = 0
    best_val_f1 = -1.0
    patience_ctr = 0

    if config.get("resume", False) and latest_ckpt_path.exists():
        print(f"Resuming from {latest_ckpt_path}")
        start_epoch, best_val_f1, patience_ctr = load_checkpoint(
            str(latest_ckpt_path), model, optimizer, scheduler, scaler, device,
        )
        print(f"  Resumed at epoch {start_epoch}, best_f1={best_val_f1:.4f}")

    # ── MLflow ────────────────────────────────────────────────────
    mlflow_available = False
    try:
        import mlflow
        mlruns_dir = os.path.join(os.getcwd(), "mlruns")
        mlflow.set_tracking_uri(f"file:///{mlruns_dir}")
        mlflow.set_experiment(config.get("mlflow_experiment", "streamguard_m2_ablation"))
        mlflow_available = True
    except Exception:
        print("MLflow not available — metrics logged to stdout only")

    # ── Training ──────────────────────────────────────────────────
    grad_accum = config["gradient_accumulation"]
    max_seq_len = config["max_seq_len"]
    global_step = 0

    if mlflow_available:
        run_context = mlflow.start_run(run_name=f"{config_name}_train")
    else:
        run_context = _NullContext()

    with run_context:
        if mlflow_available:
            log_config = {
                k: str(v) if isinstance(v, (list, dict)) else v
                for k, v in config.items()
            }
            mlflow.log_params(log_config)

        for epoch in range(start_epoch, epochs):
            epoch_start = time.time()

            # Rebuild sampler each epoch (re-shuffles groups)
            train_sampler = CFAAwareBatchSampler(
                train_dataset, batch_size, drop_last=True,
                shuffle=True, seed=config["seed"],
            )
            train_sampler.set_epoch(epoch)
            train_loader = DataLoader(
                train_dataset, batch_sampler=train_sampler,
                num_workers=0, collate_fn=cfa_collate_fn,
            )

            # ── Train epoch ───────────────────────────────────────
            model.train()
            optimizer.zero_grad()
            epoch_losses = defaultdict(float)
            n_steps = 0
            cfa_batches = 0

            for step, (orig_batch, cfa_batch, orig_metas, cfa_metas) in enumerate(train_loader):
                # Dry-run: only 1 batch
                if dry_run and step >= 1:
                    break

                orig_batch = orig_batch.to(device)

                # Tokenize on-the-fly (R-07)
                orig_codes = [code_lookup.get(m["sample_id"], "") for m in orig_metas]
                # Issue 4: warn on first code lookup miss per epoch
                if step == 0:
                    missing = [m["sample_id"] for m, c in zip(orig_metas, orig_codes) if not c.strip()]
                    if missing:
                        print(f"  WARN: {len(missing)} samples with empty code in first batch "
                              f"(e.g. {missing[0]})")
                input_ids, attn_mask = tokenize_batch(
                    orig_codes, tokenizer, max_seq_len, device,
                )

                # Labels
                labels = orig_batch.y.to(device)
                cwe_labels = torch.tensor(
                    [CWE_LABEL_MAP.get(m.get("cwe", ""), -1) for m in orig_metas],
                    dtype=torch.long, device=device,
                )
                severity_labels = torch.full(
                    (len(orig_metas),), -1.0, device=device,
                )

                # Forward orig
                with autocast(device_type=device.type, enabled=use_amp):
                    out_orig = model(
                        orig_batch,
                        input_ids=input_ids,
                        attention_mask=attn_mask,
                    )

                    # Forward CFA (if pairs present and CFA enabled)
                    out_cfa = None
                    if use_cfa and cfa_batch is not None:
                        cfa_batch_dev = cfa_batch.to(device)
                        cfa_codes = [code_lookup.get(m["sample_id"], "") for m in cfa_metas]
                        cfa_ids, cfa_mask = tokenize_batch(
                            cfa_codes, tokenizer, max_seq_len, device,
                        )
                        out_cfa = model(
                            cfa_batch_dev,
                            input_ids=cfa_ids,
                            attention_mask=cfa_mask,
                        )
                        cfa_batches += 1

                    # Composite loss
                    total_loss, loss_dict = criterion(
                        out_orig,
                        outputs_cfa=out_cfa,
                        labels=labels,
                        cwe_labels=cwe_labels,
                        severity_labels=severity_labels,
                    )

                # Issue 3: NaN guard — skip batch if loss is NaN
                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    print(f"  WARN: NaN/Inf loss at epoch {epoch} step {step}, skipping batch")
                    optimizer.zero_grad()
                    continue

                # Gradient accumulation (R-20)
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

                # Accumulate losses
                for k, v in loss_dict.items():
                    epoch_losses[k] += v
                n_steps += 1

                # Log every 50 steps
                if mlflow_available and step > 0 and step % 50 == 0:
                    mlflow.log_metrics(
                        {f"train_{k}": v / n_steps for k, v in epoch_losses.items()},
                        step=global_step,
                    )

            # Flush remaining gradients (handle partial accumulation at end)
            if n_steps % grad_accum != 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            avg_losses = {k: v / max(n_steps, 1) for k, v in epoch_losses.items()}

            # ── Validation ────────────────────────────────────────
            val_metrics = evaluate(
                model, val_loader, device, tokenizer, code_lookup, config,
            )
            epoch_time = time.time() - epoch_start

            # ── Logging ───────────────────────────────────────────
            val_f1 = val_metrics["f1"]
            cfa_str = (
                f"CFA={avg_losses.get('L_CFA', 0):.4f} "
                f"sep={avg_losses.get('frac_separated', 0):.3f} "
            ) if use_cfa else ""
            print(
                f"Epoch {epoch:02d} | "
                f"Loss: {avg_losses.get('total', 0):.4f} | "
                f"L_CE: {avg_losses.get('L_CE', 0):.4f} | "
                f"{cfa_str}"
                f"val_F1: {val_f1:.4f} | "
                f"prec: {val_metrics['precision']:.4f} rec: {val_metrics['recall']:.4f} | "
                f"pairwise: {val_metrics['pairwise_accuracy']:.4f} | "
                f"CFA_batches: {cfa_batches}/{n_steps} | "
                f"{epoch_time:.1f}s"
            )

            if mlflow_available:
                mlflow.log_metrics({
                    "train_loss":       avg_losses.get("total", 0),
                    "train_L_CE":       avg_losses.get("L_CE", 0),
                    "train_L_CFA":      avg_losses.get("L_CFA", 0),
                    "train_frac_separated": avg_losses.get("frac_separated", 0),
                    "val_f1":           val_f1,
                    "val_precision":    val_metrics["precision"],
                    "val_recall":       val_metrics["recall"],
                    "val_fpr":          val_metrics.get("fpr", 0),
                    "val_fnr":          val_metrics.get("fnr", 0),
                    "val_pairwise_accuracy": val_metrics["pairwise_accuracy"],
                    "val_worst_cwe_f1": val_metrics.get("worst_cwe_f1", 0),
                    "cfa_batch_ratio":  cfa_batches / max(n_steps, 1),
                    "epoch_time_s":     epoch_time,
                    "lr":               scheduler.get_last_lr()[-1],
                }, step=epoch)

            # ── Latest checkpoint every epoch (R-08) ──────────────
            save_checkpoint(
                model, optimizer, scheduler, scaler, epoch,
                best_val_f1, patience_ctr, config, str(latest_ckpt_path),
            )

            # ── Best checkpoint + early stopping ──────────────────
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_ctr = 0
                save_checkpoint(
                    model, optimizer, scheduler, scaler, epoch,
                    best_val_f1, patience_ctr, config, str(best_ckpt_path),
                )
                print(f"  -> Best model saved (F1={best_val_f1:.4f}) to {best_ckpt_path}")
            else:
                patience_ctr += 1
                if patience_ctr >= config["early_stopping_patience"]:
                    print(f"Early stopping at epoch {epoch}. Best val F1: {best_val_f1:.4f}")
                    break

        # ── Test set evaluation ONCE (R-01) ───────────────────────
        test_metrics = {}
        test_h5 = config.get("test_h5")
        if test_h5 and os.path.exists(test_h5) and best_ckpt_path.exists():
            print(f"\n--- Test set evaluation (ONCE, best checkpoint) ---")
            # Load best model
            best_ckpt = torch.load(str(best_ckpt_path), map_location=device, weights_only=False)
            model.load_state_dict(best_ckpt["model_state_dict"])

            test_dataset = CFADataset(test_h5)
            if dry_run:
                test_dataset.index = _select_with_complete_groups(
                    test_dataset.index, max(32, 8),
                )
            test_sampler = CFAAwareBatchSampler(
                test_dataset, batch_size, drop_last=False, shuffle=False,
            )
            test_loader = DataLoader(
                test_dataset, batch_sampler=test_sampler,
                num_workers=0, collate_fn=cfa_collate_fn,
            )
            test_metrics = evaluate(
                model, test_loader, device, tokenizer, code_lookup, config,
            )
            print(
                f"TEST | F1: {test_metrics['f1']:.4f} | "
                f"Precision: {test_metrics['precision']:.4f} | "
                f"Recall: {test_metrics['recall']:.4f} | "
                f"FPR: {test_metrics['fpr']:.4f} | "
                f"FNR: {test_metrics['fnr']:.4f} | "
                f"Pairwise: {test_metrics['pairwise_accuracy']:.4f} | "
                f"Worst-CWE: {test_metrics['worst_cwe_f1']:.4f}"
            )

            if mlflow_available:
                mlflow.log_metrics(
                    {f"test_{k}": v for k, v in test_metrics.items()
                     if not isinstance(v, dict)},
                )

    print(f"\nTraining complete. Best val F1: {best_val_f1:.4f}")
    print(f"Checkpoints: {ckpt_dir}")
    return best_val_f1, test_metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="StreamGuard Phase 4 Training")
    p.add_argument("--config", choices=list(ABLATION_CONFIGS.keys()),
                    default="D_plus_tpg",
                    help="Ablation config name")
    p.add_argument("--train-h5", required=True, help="Path to train.h5")
    p.add_argument("--val-h5", required=True, help="Path to val.h5")
    p.add_argument("--test-h5", default=None, help="Path to test.h5 (evaluated ONCE at end)")
    p.add_argument("--code-jsonl", default=None,
                    help="JSONL file for code lookup (overrides default search)")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr-codebert", type=float, default=None)
    p.add_argument("--lr-ggnn", type=float, default=None)
    p.add_argument("--dry-run", action="store_true",
                    help="Train 1 batch x 2 epochs, save checkpoint, exit")
    p.add_argument("--resume", action="store_true",
                    help="Resume from latest checkpoint if it exists")
    p.add_argument("--checkpoint-dir", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--gradient-accumulation", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()

    # Build config: defaults → ablation overlay → CLI overrides
    config = dict(DEFAULT_CONFIG)

    ablation = ABLATION_CONFIGS[args.config]
    config.update(ablation)
    config["config_name"] = args.config

    config["train_h5"] = args.train_h5
    config["val_h5"] = args.val_h5
    if args.test_h5:
        config["test_h5"] = args.test_h5
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.batch_size is not None:
        config["batch_size_graphs"] = args.batch_size
    if args.lr_codebert is not None:
        config["lr_codebert"] = args.lr_codebert
    if args.lr_ggnn is not None:
        config["lr_ggnn_fusion"] = args.lr_ggnn
    if args.dry_run:
        config["dry_run"] = True
    if args.resume:
        config["resume"] = True
    if args.checkpoint_dir:
        config["checkpoint_dir"] = args.checkpoint_dir
    if args.seed is not None:
        config["seed"] = args.seed
    if args.gradient_accumulation is not None:
        config["gradient_accumulation"] = args.gradient_accumulation

    print(f"Config: {config['config_name']}")
    print(f"  use_graph={config['use_graph']}, use_cfa={config['use_cfa']}, "
          f"use_interproc={config['use_interproc']}")
    print(f"  cpg_components={config['cpg_components']}, "
          f"epochs={config['epochs']}, dry_run={config.get('dry_run', False)}")
    print()

    best_val_f1, test_metrics = train(config)


if __name__ == "__main__":
    main()
