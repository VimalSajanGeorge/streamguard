# training/scripts/model/eval.py
#
# Standalone evaluation script for StreamGuard checkpoints.
#
# Usage:
#   python -m training.scripts.model.eval \
#       --checkpoint training/checkpoints/best_model_C_plus_cfa.pt \
#       --test-h5 training/data/final/test.h5
#
# Metrics computed:
#   Standard:  F1, precision, recall, FPR, FNR
#   per_cwe_f1:  {CWE-120: 0.xx, ...} for every CWE in the test set
#   pairwise_contrast_accuracy:  P(both vuln AND safe correct in CFA pair)
#   worst_group_f1:  min F1 across CWE subgroups

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
)
from torch.utils.data import DataLoader

from training.scripts.model.model import StreamGuardModel
from training.scripts.model.cfa_dataloader import (
    CFADataset,
    CFAAwareBatchSampler,
    cfa_collate_fn,
)

warnings.filterwarnings("ignore", message=".*clean_up_tokenization_spaces.*")
warnings.filterwarnings("ignore", message=".*RequestsDependencyWarning.*")
warnings.filterwarnings("ignore", message=".*input_ids is None.*")


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


@torch.no_grad()
def evaluate_checkpoint(
    model,
    dataloader,
    device: torch.device,
    use_cfa: bool = True,
) -> dict:
    """
    Full evaluation on a test set.

    Returns dict with:
        f1, precision, recall, fpr, fnr,
        per_cwe_f1, worst_group_f1, best_group_f1,
        pairwise_contrast_accuracy, n_valid_pairs,
        n_samples
    """
    model.eval()
    all_preds, all_labels = [], []
    all_cwes = []
    pair_results = {}  # pair_id -> {vuln_correct, safe_correct}

    for batch in dataloader:
        batch = batch.to(device)

        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            out = model(batch)

        preds = out["binary_logits"].argmax(dim=-1).cpu()
        labels = batch.y.cpu()
        all_preds.append(preds)
        all_labels.append(labels)

        # CWE labels
        if hasattr(batch, "cwe"):
            all_cwes.extend(batch.cwe)

        # Pairwise accuracy
        if use_cfa and hasattr(batch, "pair_id"):
            pair_ids = batch.pair_id
            for idx, pid in enumerate(pair_ids):
                if pid in ("", "None", "null", "0"):
                    continue
                if pid not in pair_results:
                    pair_results[pid] = {"vuln_correct": None, "safe_correct": None}
                label = labels[idx].item()
                pred = preds[idx].item()
                if label == 1:
                    pair_results[pid]["vuln_correct"] = (pred == 1)
                elif label == 0:
                    pair_results[pid]["safe_correct"] = (pred == 0)

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    # ── Standard metrics ──────────────────────────────────────────
    metrics = {
        "f1": float(f1_score(all_labels, all_preds, zero_division=0)),
        "precision": float(precision_score(all_labels, all_preds, zero_division=0)),
        "recall": float(recall_score(all_labels, all_preds, zero_division=0)),
        "n_samples": int(len(all_labels)),
    }

    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics["fpr"] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        metrics["fnr"] = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
    else:
        metrics["fpr"] = 0.0
        metrics["fnr"] = 0.0

    # ── Per-CWE F1 ───────────────────────────────────────────────
    per_cwe_f1 = {}
    if all_cwes and len(set(all_cwes)) > 1:
        for cwe in sorted(set(all_cwes)):
            mask = np.array([c == cwe for c in all_cwes])
            if mask.sum() < 2:
                continue
            per_cwe_f1[cwe] = float(
                f1_score(all_labels[mask], all_preds[mask], zero_division=0)
            )

    metrics["per_cwe_f1"] = per_cwe_f1
    if per_cwe_f1:
        metrics["worst_group_f1"] = min(per_cwe_f1.values())
        metrics["best_group_f1"] = max(per_cwe_f1.values())
        metrics["avg_cwe_f1"] = float(np.mean(list(per_cwe_f1.values())))
    else:
        metrics["worst_group_f1"] = 0.0
        metrics["best_group_f1"] = 0.0
        metrics["avg_cwe_f1"] = 0.0

    # ── Pairwise contrast accuracy ───────────────────────────────
    valid_pairs = [
        v for v in pair_results.values()
        if v["vuln_correct"] is not None and v["safe_correct"] is not None
    ]
    if valid_pairs:
        metrics["pairwise_contrast_accuracy"] = float(
            sum(1 for v in valid_pairs if v["vuln_correct"] and v["safe_correct"])
            / len(valid_pairs)
        )
    else:
        metrics["pairwise_contrast_accuracy"] = 0.0
    metrics["n_valid_pairs"] = len(valid_pairs)

    return metrics


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
    print(f"  Samples:    {metrics['n_samples']}")
    print(f"  F1:         {metrics['f1']:.4f}")
    print(f"  Precision:  {metrics['precision']:.4f}")
    print(f"  Recall:     {metrics['recall']:.4f}")
    print(f"  FPR:        {metrics['fpr']:.4f}")
    print(f"  FNR:        {metrics['fnr']:.4f}")
    print(f"  Pairwise:   {metrics['pairwise_contrast_accuracy']:.4f} "
          f"({metrics['n_valid_pairs']} pairs)")
    print(f"  Worst CWE:  {metrics['worst_group_f1']:.4f}")
    print(f"  Best CWE:   {metrics['best_group_f1']:.4f}")
    print(f"  Avg CWE F1: {metrics['avg_cwe_f1']:.4f}")

    if metrics["per_cwe_f1"]:
        print(f"\n  Per-CWE F1:")
        for cwe, f1 in sorted(metrics["per_cwe_f1"].items()):
            print(f"    {cwe:>10s}: {f1:.4f}")
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
    use_cfa = cfg.get("use_cfa", True)
    print(f"Config: {config_name}, use_cfa={use_cfa}, "
          f"epoch={ckpt['epoch']}, best_f1={ckpt['best_f1']:.4f}")

    # Build test loader
    print(f"Loading test data: {args.test_h5}")
    loader, dataset = build_test_loader(args.test_h5, args.batch_size)
    print(f"Test set: {len(dataset)} samples")

    # Evaluate
    print("Evaluating...")
    metrics = evaluate_checkpoint(model, loader, device, use_cfa=use_cfa)
    print_metrics(metrics, config_name)

    # Save JSON if requested
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        # Convert numpy types for JSON serialization
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
