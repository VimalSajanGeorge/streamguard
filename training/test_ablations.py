#!/usr/bin/env python3
"""
Ablation Test Script for Weighting Strategies

Tests 7 combinations of sampler + class_weights + focal_loss.
Runs quick test (10 epochs) for each, compares F1/precision/recall.

Usage:
    python training/test_ablations.py

Requirements:
    - Preprocessed data in data/processed/codexglue/
    - StreamGuard training script installed
"""

import subprocess
import json
import sys
from pathlib import Path
import pandas as pd

# Test combinations
COMBINATIONS = [
    {'name': 'baseline', 'sampler': False, 'mult': 1.0, 'focal': False},
    {'name': 'sampler_only', 'sampler': True, 'mult': 1.0, 'focal': False},
    {'name': 'weights_only', 'sampler': False, 'mult': 1.5, 'focal': False},
    {'name': 'focal_only', 'sampler': False, 'mult': 1.0, 'focal': True},
    {'name': 'sampler_weights', 'sampler': True, 'mult': 1.5, 'focal': False},
    {'name': 'sampler_focal', 'sampler': True, 'mult': 1.0, 'focal': True},
    {'name': 'all_three', 'sampler': True, 'mult': 1.5, 'focal': True},  # Will auto-adjust
]


def run_experiment(combo: dict, data_dir: Path) -> dict:
    """
    Run training with specific combination, return metrics.

    Args:
        combo: Configuration dictionary
        data_dir: Data directory path

    Returns:
        Dictionary with results
    """
    # Base command
    cmd = [
        sys.executable,  # Use current Python interpreter
        'training/train_transformer.py',
        '--train-data', str(data_dir / 'train.jsonl'),
        '--val-data', str(data_dir / 'valid.jsonl'),
        '--quick-test',
        '--epochs', '10',
        '--batch-size', '8',
        '--seed', '42'
    ]

    # Add flags based on combination
    if combo['sampler']:
        cmd.append('--use-weighted-sampler')
    if combo['mult'] > 1.0:
        cmd.extend(['--weight-multiplier', str(combo['mult'])])
    if combo['focal']:
        cmd.append('--focal-loss')

    # Output directory for this combo
    output_dir = Path(f"models/ablation_{combo['name']}")
    cmd.extend(['--output-dir', str(output_dir)])

    print(f"\n{'='*70}")
    print(f"Running: {combo['name']}")
    print(f"{'='*70}")
    print(f"Config:")
    print(f"  Sampler: {combo['sampler']}")
    print(f"  Weight multiplier: {combo['mult']}")
    print(f"  Focal loss: {combo['focal']}")
    print(f"Command: {' '.join(cmd[:5])} ... (see above for full command)")

    # Run training
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
    except subprocess.CalledProcessError as e:
        print(f"[!] Training failed with exit code {e.returncode}")
        return None

    # Load metrics from CSV
    csv_path = output_dir / 'metrics_history.csv'
    if not csv_path.exists():
        print(f"[!] Metrics file not found: {csv_path}")
        return None

    df = pd.read_csv(csv_path)

    # Get best epoch metrics
    best_idx = df['val_f1'].idxmax()
    best_metrics = df.iloc[best_idx]

    # Calculate prediction ratios
    pred_vuln_ratio = best_metrics['pred_vuln'] / (best_metrics['pred_vuln'] + best_metrics['pred_safe'])
    actual_vuln_ratio = best_metrics['actual_vuln'] / (best_metrics['actual_vuln'] + best_metrics['actual_safe'])

    return {
        'name': combo['name'],
        'sampler': combo['sampler'],
        'weight_mult': combo['mult'],
        'focal': combo['focal'],
        'best_epoch': int(best_metrics['epoch']),
        'f1': float(best_metrics['val_f1']),
        'accuracy': float(best_metrics['val_acc']),
        'precision': float(best_metrics['val_precision']),
        'recall': float(best_metrics['val_recall']),
        'pred_vuln_ratio': float(pred_vuln_ratio),
        'actual_vuln_ratio': float(actual_vuln_ratio),
        'final_lr': float(best_metrics['lr'])
    }


def main():
    print("""
╔═══════════════════════════════════════════════════════════════════╗
║          StreamGuard Ablation Test: Weighting Strategies         ║
╚═══════════════════════════════════════════════════════════════════╝

Testing 7 combinations of:
  - WeightedRandomSampler (balances batches)
  - Class weights (loss weighting)
  - Focal Loss (hard example focus)

This will take approximately 70-100 minutes (10 min × 7 runs).
""")

    # Check data exists
    data_dir = Path('data/processed/codexglue')
    if not data_dir.exists():
        print(f"[!] ERROR: Data directory not found: {data_dir}")
        print(f"    Please ensure preprocessed data is available.")
        sys.exit(1)

    if not (data_dir / 'train.jsonl').exists():
        print(f"[!] ERROR: train.jsonl not found in {data_dir}")
        sys.exit(1)

    # Run experiments
    results = []
    for i, combo in enumerate(COMBINATIONS, 1):
        print(f"\n[{i}/{len(COMBINATIONS)}] Testing: {combo['name']}")
        try:
            result = run_experiment(combo, data_dir)
            if result:
                results.append(result)
                print(f"[+] Complete: F1={result['f1']:.4f}, Precision={result['precision']:.4f}, Recall={result['recall']:.4f}")
        except Exception as e:
            print(f"[!] Failed: {combo['name']} - {e}")
            import traceback
            traceback.print_exc()

    if not results:
        print("\n[!] No results collected. All experiments failed.")
        sys.exit(1)

    # Save results
    results_df = pd.DataFrame(results)
    results_csv = Path('ablation_results.csv')
    results_df.to_csv(results_csv, index=False)
    print(f"\n[+] Results saved to: {results_csv}")

    # Print summary table
    print(f"\n{'='*70}")
    print("ABLATION TEST RESULTS")
    print(f"{'='*70}\n")

    # Format table
    print(results_df.to_string(index=False))

    # Find best config
    best_f1_idx = results_df['f1'].idxmax()
    best = results_df.iloc[best_f1_idx]

    print(f"\n{'='*70}")
    print("BEST CONFIGURATION")
    print(f"{'='*70}")
    print(f"  Name: {best['name']}")
    print(f"  Config: Sampler={best['sampler']}, Weight={best['weight_mult']}, Focal={best['focal']}")
    print(f"  F1: {best['f1']:.4f}")
    print(f"  Precision: {best['precision']:.4f}")
    print(f"  Recall: {best['recall']:.4f}")
    print(f"  Pred vulnerable ratio: {best['pred_vuln_ratio']:.2%}")
    print(f"  Actual vulnerable ratio: {best['actual_vuln_ratio']:.2%}")

    # Analysis
    print(f"\n{'='*70}")
    print("ANALYSIS")
    print(f"{'='*70}")

    # Check if triple weighting helped
    all_three = results_df[results_df['name'] == 'all_three']
    if not all_three.empty:
        all_three_f1 = all_three['f1'].values[0]
        baseline_f1 = results_df[results_df['name'] == 'baseline']['f1'].values[0]
        improvement = (all_three_f1 - baseline_f1) / baseline_f1 * 100

        print(f"  Triple weighting (auto-adjusted):")
        print(f"    F1: {all_three_f1:.4f}")
        print(f"    vs Baseline: {improvement:+.1f}%")

    # Check balance
    print(f"\n  Prediction balance:")
    for _, row in results_df.iterrows():
        balance_score = abs(row['pred_vuln_ratio'] - row['actual_vuln_ratio'])
        status = "✓" if balance_score < 0.1 else "!"
        print(f"    {status} {row['name']:20s}: {row['pred_vuln_ratio']:.2%} (actual: {row['actual_vuln_ratio']:.2%})")

    print(f"\n[+] Ablation test complete!")


if __name__ == '__main__':
    main()
