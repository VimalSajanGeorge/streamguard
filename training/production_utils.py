"""
Production Training Utilities for Jupyter Notebook

This module provides simple wrapper functions for running multi-seed production
training from Jupyter notebook cells.

Usage from notebook:
    from training.production_utils import run_transformer_production
    results = run_transformer_production(
        train_data='data/processed/codexglue/train.jsonl',
        val_data='data/processed/codexglue/val.jsonl',
        output_dir='training/outputs/transformer_v17',
        seeds=[42, 2025, 7]
    )
"""

import sys
import json
import subprocess
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from datetime import datetime


def run_multi_seed_training(
    script_path: str,
    train_data: str,
    val_data: str,
    output_dir: str,
    seeds: List[int] = [42, 2025, 7],
    extra_args: List[str] = None
) -> Dict[str, Any]:
    """
    Run multi-seed training and aggregate results.

    Args:
        script_path: Path to training script
        train_data: Path to training data
        val_data: Path to validation data
        output_dir: Base output directory
        seeds: List of random seeds
        extra_args: Additional command-line arguments

    Returns:
        Dictionary with aggregated results
    """
    results_all_seeds = []
    base_output = Path(output_dir)
    base_output.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"PRODUCTION MODE: MULTI-SEED TRAINING")
    print(f"{'='*80}")
    print(f"Script: {script_path}")
    print(f"Seeds: {seeds}")
    print(f"Output: {output_dir}")
    print(f"{'='*80}\n")

    for seed_idx, seed in enumerate(seeds):
        print(f"\n{'='*80}")
        print(f"TRAINING WITH SEED: {seed} ({seed_idx + 1}/{len(seeds)})")
        print(f"{'='*80}\n")

        # Create seed-specific output directory
        seed_output_dir = base_output / f"seed_{seed}"
        seed_output_dir.mkdir(parents=True, exist_ok=True)

        # Build command
        cmd = [
            sys.executable,  # Python interpreter
            script_path,
            f"--train-data={train_data}",
            f"--val-data={val_data}",
            f"--output-dir={seed_output_dir}",
            f"--seed={seed}",
            "--mixed-precision",
            "--find-lr"
        ]

        if extra_args:
            cmd.extend(extra_args)

        # Run training
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=False,  # Show output in real-time
                text=True
            )

            # Load result from metadata
            metadata_file = seed_output_dir / "experiment_config.json"
            if metadata_file.exists():
                with open(metadata_file) as f:
                    metadata = json.load(f)

                # Try to find best F1 from training logs or metadata
                # For now, we'll use a placeholder
                best_f1 = 0.0  # This should be extracted from logs

                results_all_seeds.append({
                    "seed": seed,
                    "best_f1": best_f1,
                    "status": "success"
                })

                print(f"\n[+] Seed {seed} complete.")
            else:
                print(f"\n[!] Warning: No metadata found for seed {seed}")
                results_all_seeds.append({
                    "seed": seed,
                    "status": "no_metadata"
                })

        except subprocess.CalledProcessError as e:
            print(f"\n[!] Training failed for seed {seed}: {e}")
            results_all_seeds.append({
                "seed": seed,
                "status": "failed",
                "error": str(e)
            })

    # Aggregate results
    print(f"\n{'='*80}")
    print("PRODUCTION TRAINING COMPLETE")
    print(f"{'='*80}\n")

    valid_results = [r for r in results_all_seeds if r.get('status') == 'success']

    summary = {
        "model": Path(script_path).stem.replace('train_', ''),
        "timestamp": datetime.now().isoformat(),
        "seeds": seeds,
        "results": results_all_seeds
    }

    if valid_results:
        f1_scores = [r['best_f1'] for r in valid_results]
        summary["mean_f1"] = float(np.mean(f1_scores))
        summary["std_f1"] = float(np.std(f1_scores))
        summary["best_seed"] = max(valid_results, key=lambda x: x['best_f1'])['seed']
        summary["best_f1"] = max(f1_scores)

        print(f"Results across {len(valid_results)} seeds:")
        for r in valid_results:
            print(f"  Seed {r['seed']}: F1 = {r['best_f1']:.4f}")
        print(f"\nMean F1: {summary['mean_f1']:.4f} ± {summary['std_f1']:.4f}")

    # Save summary
    summary_path = base_output / "production_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n[+] Production summary saved: {summary_path}")

    return summary


def run_transformer_production(
    train_data: str = "data/processed/codexglue/train.jsonl",
    val_data: str = "data/processed/codexglue/val.jsonl",
    output_dir: str = "training/outputs/transformer_v17",
    seeds: List[int] = [42, 2025, 7],
    epochs: int = 10,
    batch_size: int = 64
) -> Dict[str, Any]:
    """
    Run production Transformer training with multiple seeds.

    Args:
        train_data: Path to training data
        val_data: Path to validation data
        output_dir: Output directory
        seeds: List of seeds to use
        epochs: Number of epochs per seed
        batch_size: Batch size

    Returns:
        Dictionary with aggregated results
    """
    extra_args = [
        f"--epochs={epochs}",
        f"--batch-size={batch_size}",
        "--use-weighted-sampler"
    ]

    return run_multi_seed_training(
        script_path="training/train_transformer.py",
        train_data=train_data,
        val_data=val_data,
        output_dir=output_dir,
        seeds=seeds,
        extra_args=extra_args
    )


def run_gnn_production(
    train_data: str = "data/processed/graphs/train",
    val_data: str = "data/processed/graphs/val",
    output_dir: str = "training/outputs/gnn_v17",
    seeds: List[int] = [42, 2025, 7],
    epochs: int = 15,
    batch_size: int = 64
) -> Dict[str, Any]:
    """
    Run production GNN training with multiple seeds.

    Args:
        train_data: Path to training graphs
        val_data: Path to validation graphs
        output_dir: Output directory
        seeds: List of seeds to use
        epochs: Number of epochs per seed
        batch_size: Batch size

    Returns:
        Dictionary with aggregated results
    """
    extra_args = [
        f"--epochs={epochs}",
        f"--batch-size={batch_size}",
        "--use-weighted-sampler",
        "--focal-loss"
    ]

    return run_multi_seed_training(
        script_path="training/train_gnn.py",
        train_data=train_data,
        val_data=val_data,
        output_dir=output_dir,
        seeds=seeds,
        extra_args=extra_args
    )


def run_fusion_production(
    train_data: str = "data/processed/codexglue/train.jsonl",
    val_data: str = "data/processed/codexglue/val.jsonl",
    transformer_checkpoint: str = "training/outputs/transformer_v17/seed_42/best_model.pt",
    gnn_checkpoint: str = "training/outputs/gnn_v17/seed_42/best_model.pt",
    output_dir: str = "training/outputs/fusion_v17",
    seeds: List[int] = [42, 2025, 7],
    epochs: int = 12,
    batch_size: int = 32
) -> Dict[str, Any]:
    """
    Run production Fusion training with multiple seeds.

    Args:
        train_data: Path to training data
        val_data: Path to validation data
        transformer_checkpoint: Path to pretrained Transformer
        gnn_checkpoint: Path to pretrained GNN
        output_dir: Output directory
        seeds: List of seeds to use
        epochs: Number of epochs per seed
        batch_size: Batch size

    Returns:
        Dictionary with aggregated results
    """
    extra_args = [
        f"--epochs={epochs}",
        f"--batch-size={batch_size}",
        f"--transformer-checkpoint={transformer_checkpoint}",
        f"--gnn-checkpoint={gnn_checkpoint}"
    ]

    return run_multi_seed_training(
        script_path="training/train_fusion.py",
        train_data=train_data,
        val_data=val_data,
        output_dir=output_dir,
        seeds=seeds,
        extra_args=extra_args
    )


def display_production_summary():
    """
    Display production summary from all models.
    """
    print("\n" + "="*80)
    print("PRODUCTION TRAINING SUMMARY")
    print("="*80 + "\n")

    models = {
        "Transformer": "training/outputs/transformer_v17/production_summary.json",
        "GNN": "training/outputs/gnn_v17/production_summary.json",
        "Fusion": "training/outputs/fusion_v17/production_summary.json"
    }

    all_summaries = {}

    for model_name, summary_path in models.items():
        path = Path(summary_path)
        if path.exists():
            with open(path) as f:
                summary = json.load(f)
                all_summaries[model_name] = summary

                print(f"{model_name}:")
                if 'mean_f1' in summary:
                    print(f"  Mean F1: {summary['mean_f1']:.4f} ± {summary['std_f1']:.4f}")
                    print(f"  Best F1: {summary['best_f1']:.4f} (seed {summary['best_seed']})")
                else:
                    print("  Status: Training incomplete or failed")
                print()
        else:
            print(f"{model_name}: Not found\n")

    if all_summaries:
        # Find best overall
        valid_models = {k: v for k, v in all_summaries.items() if 'mean_f1' in v}
        if valid_models:
            best_model = max(valid_models.items(), key=lambda x: x[1]['mean_f1'])
            print(f"{'='*80}")
            print(f"Best Model: {best_model[0]}")
            print(f"Best F1: {best_model[1]['mean_f1']:.4f}")
            print(f"{'='*80}\n")

    return all_summaries
