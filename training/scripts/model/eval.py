# training/scripts/model/eval.py
#
# Phase 4 StreamGuard Evaluation & Metrics (P4-S6).
#
# Computes all metrics required for the paper:
#   Standard:  f1, precision, recall, fpr, fnr, accuracy
#   Novel:     pairwise_accuracy, worst_group_f1, per_cwe_f1
#   CWE head:  cwe_top1_accuracy
#   Severity:  severity_mae
#
# Risk mitigations:
#   R-21: pairwise_accuracy gracefully absent when no CFA pairs
#   R-22: per-CWE F1 threshold = 5 minimum samples
#   R-26: CWE_LABEL_MAP imported from losses.py
#
# Usage (library):
#   from training.scripts.model.eval import evaluate
#   metrics = evaluate(model, val_loader, device, tokenizer=tok, code_lookup=lookup)
#
# Usage (CLI):
#   python -m training.scripts.model.eval \
#       --checkpoint training/checkpoints/best_model.pt \
#       --test-h5 training/data/final/test.h5

import argparse
import json
import os
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    accuracy_score,
    mean_absolute_error,
)
from loguru import logger
from torch.utils.data import DataLoader

from training.scripts.model.losses import CWE_LABEL_MAP
from training.scripts.model.model import StreamGuardModel
from training.scripts.model.cfa_dataloader import (
    CFADataset,
    CFAAwareBatchSampler,
    cfa_collate_fn,
)

warnings.filterwarnings("ignore", message=".*clean_up_tokenization_spaces.*")
warnings.filterwarnings("ignore", message=".*RequestsDependencyWarning.*")


# ─────────────────────────────────────────────────────────────────────────────
# Pure computation — no model dependency, fully testable
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(
    all_preds: np.ndarray,
    all_labels: np.ndarray,
    all_cwe_preds: np.ndarray,
    all_cwe_labels: np.ndarray,
    all_severity_preds: np.ndarray,
    all_severity_labels: np.ndarray,
    all_pair_results: list[bool],
    cwe_group_preds: dict[str, list[int]],
    cwe_group_labels: dict[str, list[int]],
    min_cwe_samples: int = 5,
) -> dict:
    """
    Compute all evaluation metrics from raw collected arrays.

    Args:
        all_preds:            Binary predictions (0/1) for all samples.
        all_labels:           Ground-truth binary labels for all samples.
        all_cwe_preds:        CWE head predictions (int class indices).
        all_cwe_labels:       Ground-truth CWE labels (-1 = unknown).
        all_severity_preds:   Severity head predictions (float).
        all_severity_labels:  Ground-truth severity (-1 = skip).
        all_pair_results:     Per-pair bool (True = both orig and CFA correct).
        cwe_group_preds:      {cwe_name: [binary_pred, ...]} for per-CWE F1.
        cwe_group_labels:     {cwe_name: [binary_label, ...]} for per-CWE F1.
        min_cwe_samples:      Minimum samples per CWE for per-CWE F1 (R-22).

    Returns:
        Dict with all metric values.
    """
    metrics = {}

    # ── Standard binary metrics ─────────────────────────────────────
    metrics["f1"] = float(f1_score(all_labels, all_preds, zero_division=0))
    metrics["precision"] = float(precision_score(all_labels, all_preds, zero_division=0))
    metrics["recall"] = float(recall_score(all_labels, all_preds, zero_division=0))

    try:
        tn, fp, fn, tp = confusion_matrix(all_labels, all_preds, labels=[0, 1]).ravel()
        metrics["fpr"] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        metrics["fnr"] = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
        metrics["accuracy"] = float((tp + tn) / (tp + tn + fp + fn))
    except ValueError:
        metrics["fpr"] = 0.0
        metrics["fnr"] = 0.0
        metrics["accuracy"] = 0.0

    # ── Pairwise accuracy (VISION paper core metric) ────────────────
    if all_pair_results:
        metrics["pairwise_accuracy"] = sum(all_pair_results) / len(all_pair_results)

    # ── Per-CWE F1 and worst-group F1 ──────────────────────────────
    per_cwe_f1 = {}
    for cwe_name in cwe_group_labels:
        labels = cwe_group_labels[cwe_name]
        preds = cwe_group_preds[cwe_name]
        if len(labels) >= min_cwe_samples:
            per_cwe_f1[cwe_name] = float(f1_score(labels, preds, zero_division=0))
    metrics["per_cwe_f1"] = per_cwe_f1
    metrics["worst_group_f1"] = min(per_cwe_f1.values()) if per_cwe_f1 else 0.0

    # ── CWE head top-1 accuracy ────────────────────────────────────
    cwe_labels_arr = np.array(all_cwe_labels)
    cwe_preds_arr = np.array(all_cwe_preds)
    valid_cwe = cwe_labels_arr >= 0
    if valid_cwe.any():
        metrics["cwe_top1_accuracy"] = float(
            accuracy_score(cwe_labels_arr[valid_cwe], cwe_preds_arr[valid_cwe])
        )
    else:
        metrics["cwe_top1_accuracy"] = 0.0

    # ── Severity MAE (skip -1 labels) ──────────────────────────────
    sev_labels = np.array(all_severity_labels, dtype=np.float64)
    sev_preds = np.array(all_severity_preds, dtype=np.float64)
    valid_sev = sev_labels >= 0
    if valid_sev.sum() > 0:
        metrics["severity_mae"] = float(
            mean_absolute_error(sev_labels[valid_sev], sev_preds[valid_sev])
        )
    else:
        metrics["severity_mae"] = 0.0

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Full evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(model, dataloader, device, tokenizer=None, code_lookup=None,
             config=None, split="val"):
    """
    Full evaluation loop over a dataloader split.

    Runs model inference, collects predictions, and computes all metrics
    including CFA pairwise accuracy and per-CWE F1.

    Args:
        model:       StreamGuardModel instance.
        dataloader:  DataLoader yielding (orig_batch, cfa_batch, orig_metas, cfa_metas).
        device:      torch.device.
        tokenizer:   CodeBERT tokenizer (None for graph-only ablation).
        code_lookup: dict sample_id → code string (needed when tokenizer is not None).
        config:      dict with optional keys (max_seq_len, ...).
        split:       Name of split for logging ("val" or "test").

    Returns:
        Dict with all metric values.
    """
    from training.scripts.model.train import tokenize_batch

    model.eval()
    max_seq_len = (config or {}).get("max_seq_len", 512)

    all_preds = []
    all_labels = []
    all_cwe_preds = []
    all_cwe_labels = []
    all_severity_preds = []
    all_severity_labels = []
    all_pair_results = []
    cwe_group_preds = defaultdict(list)
    cwe_group_labels = defaultdict(list)

    with torch.no_grad():
        for orig_batch, cfa_batch, orig_metas, cfa_metas in dataloader:
            orig_batch = orig_batch.to(device)

            # Tokenize if tokenizer available
            input_ids, attn_mask = None, None
            if tokenizer is not None:
                lookup = code_lookup or {}
                orig_codes = [lookup.get(m["sample_id"], "") for m in orig_metas]
                input_ids, attn_mask = tokenize_batch(
                    orig_codes, tokenizer, max_seq_len, device,
                )

            out = model(orig_batch, input_ids=input_ids, attention_mask=attn_mask)

            preds = out["logits"].argmax(dim=-1).cpu().numpy()
            labels = orig_batch.y.cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels)

            # CWE head predictions
            cwe_preds = out["cwe_logits"].argmax(dim=-1).cpu().numpy()
            all_cwe_preds.extend(cwe_preds)

            # Severity predictions
            sev_preds = out["severity_score"].cpu().numpy()
            all_severity_preds.extend(sev_preds)

            # Per-sample metadata extraction
            for i, m in enumerate(orig_metas):
                cwe_int = CWE_LABEL_MAP.get(m.get("cwe", ""), -1)
                all_cwe_labels.append(cwe_int)
                all_severity_labels.append(m.get("severity_score", -1.0))

                # Per-CWE binary tracking for worst-group metric
                cwe_name = m.get("cwe", "")
                if cwe_name and cwe_int >= 0:
                    cwe_group_preds[cwe_name].append(int(preds[i]))
                    cwe_group_labels[cwe_name].append(int(labels[i]))

            # ── Pairwise accuracy ────────────────────────────────────
            if cfa_batch is not None:
                cfa_batch = cfa_batch.to(device)

                cfa_input_ids, cfa_attn_mask = None, None
                if tokenizer is not None:
                    lookup = code_lookup or {}
                    cfa_codes = [lookup.get(m["sample_id"], "") for m in cfa_metas]
                    cfa_input_ids, cfa_attn_mask = tokenize_batch(
                        cfa_codes, tokenizer, max_seq_len, device,
                    )

                out_cfa = model(
                    cfa_batch,
                    input_ids=cfa_input_ids,
                    attention_mask=cfa_attn_mask,
                )
                cfa_preds = out_cfa["logits"].argmax(dim=-1).cpu().numpy()

                # Pair: original must be predicted vuln=1, CFA must be safe=0
                for i, (op, ol) in enumerate(zip(preds, labels)):
                    if ol == 1 and i < len(cfa_preds):
                        vuln_correct = (op == 1)
                        cfa_correct = (cfa_preds[i] == 0)
                        all_pair_results.append(vuln_correct and cfa_correct)

    # ── Compute all metrics ─────────────────────────────────────────
    metrics = compute_metrics(
        all_preds=np.array(all_preds),
        all_labels=np.array(all_labels),
        all_cwe_preds=np.array(all_cwe_preds),
        all_cwe_labels=np.array(all_cwe_labels),
        all_severity_preds=np.array(all_severity_preds),
        all_severity_labels=np.array(all_severity_labels),
        all_pair_results=all_pair_results,
        cwe_group_preds=dict(cwe_group_preds),
        cwe_group_labels=dict(cwe_group_labels),
    )

    # ── Log per-CWE F1 table ───────────────────────────────────────
    per_cwe_f1 = metrics.get("per_cwe_f1", {})
    if per_cwe_f1:
        logger.info(f"\nPer-CWE F1 ({split}):")
        for cwe, f1_val in sorted(per_cwe_f1.items()):
            target = 0.88
            status = "\u2713" if f1_val >= target else "\u2717"
            logger.info(f"  {cwe:<12}: {f1_val:.4f} {status}")

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# CLI: Standalone checkpoint evaluation
# ─────────────────────────────────────────────────────────────────────────────

def load_model_from_checkpoint(checkpoint_path: str, device: torch.device):
    """Load StreamGuardModel from a saved checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]

    model = StreamGuardModel(
        node_feature_dim=cfg["node_feature_dim"],
        codebert_model=cfg["base_model"],
        use_interproc=cfg.get("use_interproc", False),
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg, ckpt


def build_test_loader(test_h5: str, batch_size: int = 8):
    """Build a CFA-aware DataLoader for the test set."""
    dataset = CFADataset(test_h5)
    sampler = CFAAwareBatchSampler(dataset, batch_size, drop_last=False)
    loader = DataLoader(
        dataset, batch_sampler=sampler,
        num_workers=0, collate_fn=cfa_collate_fn,
    )
    return loader, dataset


def print_metrics(metrics: dict, config_name: str = ""):
    """Pretty-print evaluation metrics."""
    header = f"Evaluation Results: {config_name}" if config_name else "Evaluation Results"
    print(f"\n{'=' * 60}")
    print(f"  {header}")
    print(f"{'=' * 60}")
    print(f"  F1:             {metrics['f1']:.4f}")
    print(f"  Precision:      {metrics['precision']:.4f}")
    print(f"  Recall:         {metrics['recall']:.4f}")
    print(f"  FPR:            {metrics['fpr']:.4f}")
    print(f"  FNR:            {metrics['fnr']:.4f}")
    print(f"  Accuracy:       {metrics['accuracy']:.4f}")
    pa = metrics.get("pairwise_accuracy")
    print(f"  Pairwise Acc:   {pa:.4f}" if pa is not None else "  Pairwise Acc:   N/A (no pairs)")
    print(f"  Worst CWE F1:   {metrics['worst_group_f1']:.4f}")
    print(f"  CWE Top-1 Acc:  {metrics['cwe_top1_accuracy']:.4f}")
    print(f"  Severity MAE:   {metrics['severity_mae']:.4f}")

    per_cwe_f1 = metrics.get("per_cwe_f1", {})
    if per_cwe_f1:
        print(f"\n  Per-CWE F1:")
        for cwe, f1 in sorted(per_cwe_f1.items()):
            target = 0.88
            status = "\u2713" if f1 >= target else "\u2717"
            print(f"    {cwe:<12}: {f1:.4f} {status}")
    print(f"{'=' * 60}\n")


def main():
    p = argparse.ArgumentParser(description="StreamGuard checkpoint evaluation")
    p.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    p.add_argument("--test-h5", required=True, help="Path to test.h5")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--output-json", default=None,
                   help="Path to write results JSON (optional)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    model, cfg, ckpt = load_model_from_checkpoint(args.checkpoint, device)
    config_name = cfg.get("ablation_config", "unknown")
    print(f"Config: {config_name}, epoch={ckpt['epoch']}, best_f1={ckpt['best_f1']:.4f}")

    # Build test loader
    print(f"Loading test data: {args.test_h5}")
    loader, dataset = build_test_loader(args.test_h5, args.batch_size)
    print(f"Test set: {len(dataset)} samples")

    # Evaluate
    print("Evaluating...")
    metrics = evaluate(model, loader, device, tokenizer=None, split="test")
    print_metrics(metrics, config_name)

    # Save JSON if requested
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        serializable = {}
        for k, v in metrics.items():
            if isinstance(v, (np.floating, np.integer)):
                serializable[k] = float(v)
            elif isinstance(v, dict):
                serializable[k] = {
                    kk: float(vv) if isinstance(vv, (np.floating, np.integer)) else vv
                    for kk, vv in v.items()
                }
            else:
                serializable[k] = v
        serializable["config_name"] = config_name
        serializable["checkpoint"] = args.checkpoint
        Path(args.output_json).write_text(json.dumps(serializable, indent=2))
        print(f"Results saved to {args.output_json}")

    return metrics


if __name__ == "__main__":
    main()
