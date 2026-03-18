# training/scripts/model/ablation_cfa_vs_baseline.py
#
# CFA Proof: Ablation comparison across Config B / C / D.
#
# Usage:
#   python -m training.scripts.model.ablation_cfa_vs_baseline \
#       --test-h5 training/data/final/test.h5 \
#       --checkpoint-dir training/checkpoints/
#
# Outputs:
#   - Comparison table to stdout
#   - proof/cfa_proof_result.json with proof_positive flag

import argparse
import json
import os
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from training.scripts.model.eval import (
    load_model_from_checkpoint,
    build_test_loader,
    evaluate_checkpoint,
    print_metrics,
)

warnings.filterwarnings("ignore", message=".*clean_up_tokenization_spaces.*")
warnings.filterwarnings("ignore", message=".*RequestsDependencyWarning.*")
warnings.filterwarnings("ignore", message=".*input_ids is None.*")

# Ablation configs to evaluate, in order
CONFIGS = [
    {
        "name": "B_plus_ggnn",
        "label": "B: No CFA",
        "checkpoint_file": "best_model_B_plus_ggnn.pt",
    },
    {
        "name": "C_plus_cfa",
        "label": "C: +CFA",
        "checkpoint_file": "best_model_C_plus_cfa.pt",
    },
    {
        "name": "D_plus_tpg",
        "label": "D: +TPG+CFA",
        "checkpoint_file": "best_model_D_plus_tpg.pt",
    },
]


def run_ablation(test_h5: str, checkpoint_dir: str, batch_size: int = 8):
    """Evaluate all 3 configs and produce comparison."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Test data: {test_h5}")
    print(f"Checkpoint dir: {checkpoint_dir}\n")

    # Build test loader once — reused across all configs
    loader, dataset = build_test_loader(test_h5, batch_size)
    print(f"Test set: {len(dataset)} samples\n")

    results = {}
    for cfg in CONFIGS:
        ckpt_path = os.path.join(checkpoint_dir, cfg["checkpoint_file"])
        if not os.path.exists(ckpt_path):
            print(f"WARNING: Checkpoint not found: {ckpt_path} -- skipping {cfg['name']}")
            continue

        print(f"{'-' * 60}")
        print(f"Evaluating {cfg['label']} ({cfg['name']})")
        print(f"  Checkpoint: {ckpt_path}")

        model, model_cfg, ckpt = load_model_from_checkpoint(ckpt_path, device)
        use_cfa = model_cfg.get("use_cfa", False)
        print(f"  use_cfa={use_cfa}, epoch={ckpt['epoch']}, "
              f"train_best_f1={ckpt['best_f1']:.4f}")

        metrics = evaluate_checkpoint(model, loader, device, use_cfa=use_cfa)
        results[cfg["name"]] = metrics
        print_metrics(metrics, cfg["label"])

        # Free GPU memory between configs
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if len(results) < 2:
        print("ERROR: Need at least Config B and Config C to produce proof.")
        sys.exit(1)

    # ── Comparison table ──────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print("  ABLATION COMPARISON: CFA vs Baseline")
    print(f"{'=' * 72}")
    print(f"  {'Config':<16s} | {'F1':>6s} | {'Pairwise Acc':>12s} | "
          f"{'Per-CWE F1 (avg)':>16s} | {'Worst CWE F1':>12s}")
    print(f"  {'-' * 16}-+-{'-' * 6}-+-{'-' * 12}-+-{'-' * 16}-+-{'-' * 12}")

    for cfg in CONFIGS:
        name = cfg["name"]
        if name not in results:
            continue
        m = results[name]
        pairwise = (
            f"{m['pairwise_contrast_accuracy']:.4f}"
            if m.get("n_valid_pairs", 0) > 0
            else "N/A"
        )
        print(
            f"  {cfg['label']:<16s} | {m['f1']:6.4f} | {pairwise:>12s} | "
            f"{m['avg_cwe_f1']:16.4f} | {m['worst_group_f1']:12.4f}"
        )
    print(f"{'=' * 72}\n")

    # ── Proof result ──────────────────────────────────────────────
    config_b_f1 = results.get("B_plus_ggnn", {}).get("f1", 0.0)
    config_c_f1 = results.get("C_plus_cfa", {}).get("f1", 0.0)
    config_d_f1 = results.get("D_plus_tpg", {}).get("f1", 0.0)
    cfa_delta = config_c_f1 - config_b_f1

    # Pairwise accuracy from Config C (the CFA proof config)
    pairwise_acc = results.get("C_plus_cfa", {}).get(
        "pairwise_contrast_accuracy", 0.0
    )
    per_cwe_dict = results.get("C_plus_cfa", {}).get("per_cwe_f1", {})

    proof_positive = cfa_delta >= 0.03  # 3 percentage points

    result = {
        "config_B_f1": config_b_f1,
        "config_C_f1": config_c_f1,
        "config_D_f1": config_d_f1,
        "cfa_delta": cfa_delta,
        "pairwise_accuracy": pairwise_acc,
        "per_cwe_f1": per_cwe_dict,
        "worst_group_f1": results.get("C_plus_cfa", {}).get("worst_group_f1", 0.0),
        "proof_positive": proof_positive,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "test_samples": len(dataset),
    }

    # Add Config D details if available
    if "D_plus_tpg" in results:
        result["tpg_delta"] = config_d_f1 - config_c_f1
        result["config_D_pairwise"] = results["D_plus_tpg"].get(
            "pairwise_contrast_accuracy", 0.0
        )

    # Save proof JSON
    proof_dir = Path("proof")
    proof_dir.mkdir(exist_ok=True)
    proof_path = proof_dir / "cfa_proof_result.json"
    proof_path.write_text(json.dumps(result, indent=2))

    print(f"PROOF POSITIVE: {result['proof_positive']}")
    print(f"CFA delta: +{result['cfa_delta']:.3f} F1")
    print(f"Proof saved to: {proof_path}")

    return result


def main():
    p = argparse.ArgumentParser(
        description="StreamGuard CFA ablation comparison (B vs C vs D)"
    )
    p.add_argument("--test-h5", required=True, help="Path to test.h5")
    p.add_argument("--checkpoint-dir", required=True,
                   help="Directory containing best_model_*.pt checkpoints")
    p.add_argument("--batch-size", type=int, default=8)
    args = p.parse_args()

    run_ablation(args.test_h5, args.checkpoint_dir, args.batch_size)


if __name__ == "__main__":
    main()
