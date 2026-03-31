# training/scripts/model/run_ablations.py
#
# Phase 4 Ablation Study Runner (P4-S7).
#
# Runs all 6 ablation configs sequentially, collects test metrics,
# prints Paper Table 2, and validates CFA/TPG novelty proofs.
#
# MANDATORY INVARIANTS (breaking any invalidates the paper):
#   1. seed=42 for ALL configs
#   2. Same train.h5/val.h5/test.h5 for ALL configs
#   3. test.h5 evaluated ONCE per config AFTER training
#   4. All runs logged to same MLflow experiment "streamguard_m2_ablation"
#
# Risk mitigations:
#   R-05: All configs share same HDF5 paths (asserted at startup)
#   R-06: All configs share seed=42 (asserted at startup)
#   R-15: Config B uses dedicated single_conv (fixed in model.py)
#   R-16: Config A bypasses GGNN + cross-attention (use_graph=False in model.py)
#
# Usage:
#   python -m training.scripts.model.run_ablations \
#       --dry-run                          # smoke test all configs (2 epochs each)
#   python -m training.scripts.model.run_ablations \
#       --configs A_baseline B_plus_ggnn   # run specific configs
#   python -m training.scripts.model.run_ablations  # full run (all 6 configs)

import argparse
import json
import os
from pathlib import Path

import numpy as np
from loguru import logger

from training.scripts.model.train import train


# ─────────────────────────────────────────────────────────────────────────────
# Config definitions
# ─────────────────────────────────────────────────────────────────────────────

BASE_CONFIG = {
    "base_model":              "microsoft/codebert-base",
    "node_feature_dim":        824,
    "batch_size_graphs":       8,
    "gradient_accumulation":   4,
    "max_seq_len":             512,
    "epochs":                  20,
    "early_stopping_patience": 5,
    "seed":                    42,
    "weight_decay":            0.01,
    "lr_codebert":             2e-5,
    "lr_ggnn_fusion":          1e-4,
    "warmup_ratio":            0.1,
    "grad_clip":               1.0,
    "freeze_codebert_layers":  9,
    "num_cwe_classes":         12,
    "train_h5":                "training/data/final/train.h5",
    "val_h5":                  "training/data/final/val.h5",
    "test_h5":                 "training/data/final/test.h5",
    "checkpoint_dir":          "training/checkpoints/",
    "mlflow_experiment":       "streamguard_m2_ablation",
    "resume":                  True,
    # Loss weights
    "lambda_ce":               1.0,
    "lambda_cfa":              0.5,
    "lambda_sev":              0.1,
    "cfa_margin":              0.5,
}

ABLATION_CONFIGS = {

    # Config A: CodeBERT sequence only — no graph
    "A_baseline": {
        **BASE_CONFIG,
        "config_name":         "A_baseline",
        "use_graph":           False,
        "type_aware_edges":    True,
        "use_cfa":             False,
        "lambda_cfa":          0.0,
        "lambda_sev":          0.0,
        "use_interproc":       False,
        "cpg_components":      ["AST", "CFG", "DFG"],
    },

    # Config B: CodeBERT + type-BLIND GGNN, 3-CPG, no CFA
    "B_plus_ggnn": {
        **BASE_CONFIG,
        "config_name":         "B_plus_ggnn",
        "use_graph":           True,
        "type_aware_edges":    False,
        "use_cfa":             False,
        "lambda_cfa":          0.0,
        "lambda_sev":          0.0,
        "use_interproc":       False,
        "cpg_components":      ["AST", "CFG", "DFG"],
    },

    # Config B': CodeBERT + type-AWARE GGNN, 3-CPG, no CFA
    "B_prime_type_aware": {
        **BASE_CONFIG,
        "config_name":         "B_prime_type_aware",
        "use_graph":           True,
        "type_aware_edges":    True,
        "use_cfa":             False,
        "lambda_cfa":          0.0,
        "lambda_sev":          0.0,
        "use_interproc":       False,
        "cpg_components":      ["AST", "CFG", "DFG"],
    },

    # Config C: B' + CFA contrastive training (PRIMARY NOVELTY TEST)
    "C_plus_cfa": {
        **BASE_CONFIG,
        "config_name":         "C_plus_cfa",
        "use_graph":           True,
        "type_aware_edges":    True,
        "use_cfa":             True,
        "lambda_cfa":          0.5,
        "lambda_sev":          0.1,
        "use_interproc":       False,
        "cpg_components":      ["AST", "CFG", "DFG"],
    },

    # Config D: C + TPG (4-CPG)
    "D_plus_tpg": {
        **BASE_CONFIG,
        "config_name":         "D_plus_tpg",
        "use_graph":           True,
        "type_aware_edges":    True,
        "use_cfa":             True,
        "lambda_cfa":          0.5,
        "lambda_sev":          0.1,
        "use_interproc":       False,
        "cpg_components":      ["AST", "CFG", "DFG", "TPG"],
    },

    # Config E: Full StreamGuard (D + inter-proc)
    "E_full": {
        **BASE_CONFIG,
        "config_name":         "E_full",
        "use_graph":           True,
        "type_aware_edges":    True,
        "use_cfa":             True,
        "lambda_cfa":          0.5,
        "lambda_sev":          0.1,
        "use_interproc":       True,
        "cpg_components":      ["AST", "CFG", "DFG", "TPG"],
    },
}

# Canonical ordering for table output
CONFIG_ORDER = [
    "A_baseline", "B_plus_ggnn", "B_prime_type_aware",
    "C_plus_cfa", "D_plus_tpg", "E_full",
]


# ─────────────────────────────────────────────────────────────────────────────
# Invariant assertions (R-05, R-06)
# ─────────────────────────────────────────────────────────────────────────────

def assert_ablation_invariants():
    """Verify all configs share seed, data paths, and experiment name."""
    for name, cfg in ABLATION_CONFIGS.items():
        assert cfg["seed"] == BASE_CONFIG["seed"], \
            f"R-06: {name} has seed={cfg['seed']}, expected {BASE_CONFIG['seed']}"
        assert cfg["train_h5"] == BASE_CONFIG["train_h5"], \
            f"R-05: {name} has different train_h5"
        assert cfg["val_h5"] == BASE_CONFIG["val_h5"], \
            f"R-05: {name} has different val_h5"
        assert cfg["test_h5"] == BASE_CONFIG["test_h5"], \
            f"R-05: {name} has different test_h5"
        assert cfg["mlflow_experiment"] == BASE_CONFIG["mlflow_experiment"], \
            f"R-32: {name} has different mlflow_experiment"


# ─────────────────────────────────────────────────────────────────────────────
# Ablation runner
# ─────────────────────────────────────────────────────────────────────────────

def run_ablations(
    configs_to_run: list = None,
    dry_run: bool = False,
):
    """
    Run ablation configs in sequence. Each config trains independently
    with same seed and data. Results saved to results/ablation_table.json.

    Args:
        configs_to_run: List of config names (default: all 6).
        dry_run: If True, train 2 epochs per config for smoke test.

    Returns:
        dict: {config_name: {best_val_f1, test_f1, test_fpr, ...}}
    """
    assert_ablation_invariants()

    if configs_to_run is None:
        configs_to_run = list(CONFIG_ORDER)

    results = {}

    for config_name in configs_to_run:
        if config_name not in ABLATION_CONFIGS:
            logger.warning(f"Unknown config: {config_name}. Skipping.")
            continue

        config = dict(ABLATION_CONFIGS[config_name])

        if dry_run:
            config["dry_run"] = True
            logger.info(f"DRY RUN mode: 2 epochs, 1 batch for {config_name}")

        logger.info(f"\n{'=' * 60}")
        logger.info(f"RUNNING ABLATION: {config_name}")
        logger.info(f"  use_graph={config['use_graph']}, "
                     f"type_aware={config['type_aware_edges']}, "
                     f"use_cfa={config['use_cfa']}, "
                     f"lambda_cfa={config['lambda_cfa']}, "
                     f"use_interproc={config['use_interproc']}")
        logger.info(f"{'=' * 60}")

        try:
            best_val_f1, test_metrics = train(config)
            results[config_name] = {
                "best_val_f1": best_val_f1,
                **{f"test_{k}": v for k, v in test_metrics.items()
                   if isinstance(v, (int, float))},
            }
            results[config_name]["per_cwe_f1"] = test_metrics.get("per_cwe_f1", {})
            logger.info(
                f"Config {config_name} complete. "
                f"Val F1: {best_val_f1:.4f}, "
                f"Test F1: {test_metrics.get('f1', 'N/A')}"
            )
        except Exception as e:
            logger.error(f"Config {config_name} FAILED: {e}")
            results[config_name] = {"error": str(e)}

    # ── Save results ─────────────────────────────────────────────────
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    table_path = results_dir / "ablation_table.json"

    serializable = _make_serializable(results)
    with open(table_path, "w") as f:
        json.dump(serializable, f, indent=2)

    # ── Print summary ────────────────────────────────────────────────
    print_ablation_table(results)
    logger.info(f"Results saved to {table_path}")

    return results


def _make_serializable(results: dict) -> dict:
    """Convert numpy types to Python natives for JSON serialization."""
    out = {}
    for k, v in results.items():
        if isinstance(v, dict):
            out[k] = _make_serializable(v)
        elif isinstance(v, (np.floating, np.integer)):
            out[k] = float(v)
        else:
            out[k] = v
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Table output
# ─────────────────────────────────────────────────────────────────────────────

def print_ablation_table(results: dict):
    """Print ablation results as a formatted table (paper Table 2)."""
    print(f"\n{'=' * 80}")
    print("ABLATION STUDY RESULTS (Paper Table 2)")
    print(f"{'=' * 80}")
    print(f"{'Config':<24} {'F1':>6} {'FPR':>6} {'FNR':>6} "
          f"{'Pairwise':>10} {'Worst-CWE':>10}")
    print("-" * 80)

    for config_name in CONFIG_ORDER:
        if config_name not in results:
            continue
        r = results[config_name]
        if "error" in r:
            print(f"{config_name:<24} ERROR: {r['error']}")
            continue

        f1    = r.get("test_f1",                0.0)
        fpr   = r.get("test_fpr",               0.0)
        fnr   = r.get("test_fnr",               0.0)
        pair  = r.get("test_pairwise_accuracy",  0.0)
        worst = r.get("test_worst_cwe_f1",       0.0)

        print(f"{config_name:<24} {f1:>6.4f} {fpr:>6.4f} {fnr:>6.4f} "
              f"{pair:>10.4f} {worst:>10.4f}")

    print("=" * 80)

    # ── CFA proof: Config C > Config B' ──────────────────────────────
    if "C_plus_cfa" in results and "B_prime_type_aware" in results:
        c_r = results["C_plus_cfa"]
        b_r = results["B_prime_type_aware"]
        if "error" not in c_r and "error" not in b_r:
            c_f1 = c_r.get("test_f1", 0)
            b_f1 = b_r.get("test_f1", 0)
            if c_f1 > b_f1:
                print(f"\u2713 CFA PROOF POSITIVE: "
                      f"Config C ({c_f1:.4f}) > Config B' ({b_f1:.4f})")
            else:
                print(f"\u2717 CFA PROOF NEGATIVE: "
                      f"Config C ({c_f1:.4f}) <= Config B' ({b_f1:.4f})")

    # ── TPG proof: Config D > Config C ───────────────────────────────
    if "D_plus_tpg" in results and "C_plus_cfa" in results:
        d_r = results["D_plus_tpg"]
        c_r = results["C_plus_cfa"]
        if "error" not in d_r and "error" not in c_r:
            d_f1 = d_r.get("test_f1", 0)
            c_f1 = c_r.get("test_f1", 0)
            if d_f1 > c_f1:
                print(f"\u2713 TPG PROOF POSITIVE: "
                      f"Config D ({d_f1:.4f}) > Config C ({c_f1:.4f})")
            else:
                print(f"\u2717 TPG ADDS LITTLE: "
                      f"Config D ({d_f1:.4f}) <= Config C ({c_f1:.4f})")

    print()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="StreamGuard Phase 4 Ablation Study Runner",
    )
    parser.add_argument(
        "--configs", nargs="+", default=None,
        choices=list(ABLATION_CONFIGS.keys()),
        help="Which configs to run (default: all 6)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Train 2 epochs per config for smoke test",
    )
    args = parser.parse_args()

    run_ablations(args.configs, dry_run=args.dry_run)
