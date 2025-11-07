"""
Cell 54: Unified Metrics & Export

Aggregates training metrics across all models (Transformer, GNN, Fusion)
and all seeds, generates comparison plots, and exports production summary.

Features:
- Load metrics from all 9 runs (3 models × 3 seeds)
- Compute mean/std for F1, Recall, Precision, Accuracy
- Generate comparison visualizations
- Export production_summary.json

Story Points: 5
Status: Production-Ready
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any

# Visualization imports (optional)
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("[!] Matplotlib/Seaborn not available. Skipping plots.")


# ==================== CONFIGURATION ====================

print("=" * 80)
print("CELL 54: UNIFIED METRICS & EXPORT")
print("=" * 80)

MODELS = ["transformer_v17", "gnn_v17", "fusion_v17"]
SEEDS = [42, 2025, 7]
OUTPUT_BASE = Path("training/outputs")
SUMMARY_DIR = OUTPUT_BASE / "production_summary"
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)


# ==================== DATA LOADING ====================

def load_model_results(model_name: str) -> List[Dict[str, Any]]:
    """Load results for a specific model across all seeds."""
    model_dir = OUTPUT_BASE / f"{model_name}_production"

    if not model_dir.exists():
        print(f"[!] Model directory not found: {model_dir}")
        return []

    results = []
    for seed in SEEDS:
        metadata_file = model_dir / f"seed_{seed}" / "training_metadata.json"

        if not metadata_file.exists():
            print(f"[!] Metadata not found: {metadata_file}")
            continue

        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                results.append(metadata)
        except Exception as e:
            print(f"[!] Failed to load {metadata_file}: {e}")

    return results


def extract_metrics(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Extract metrics into a DataFrame."""
    data = []

    for result in results:
        seed = result.get('seed', 'unknown')
        best_f1 = result.get('best_f1', 0.0)

        # Get final epoch metrics
        metrics_history = result.get('metrics_history', [])
        if metrics_history:
            final_metrics = metrics_history[-1]
        else:
            final_metrics = {}

        data.append({
            'seed': seed,
            'f1': best_f1,
            'accuracy': final_metrics.get('accuracy', 0.0),
            'precision': final_metrics.get('precision', 0.0),
            'recall': final_metrics.get('recall', 0.0)
        })

    return pd.DataFrame(data)


# ==================== AGGREGATION ====================

def aggregate_all_models() -> Dict[str, Any]:
    """Aggregate metrics across all models and seeds."""
    summary = {
        "timestamp": datetime.now().isoformat(),
        "models": {},
        "best_overall_model": None,
        "best_overall_f1": 0.0
    }

    for model in MODELS:
        print(f"\n[+] Loading results for {model}...")
        results = load_model_results(model)

        if not results:
            print(f"[!] No results found for {model}")
            continue

        # Extract metrics
        df = extract_metrics(results)

        # Compute statistics
        mean_f1 = df['f1'].mean()
        std_f1 = df['f1'].std()
        best_seed = df.loc[df['f1'].idxmax(), 'seed']
        best_f1 = df['f1'].max()

        model_summary = {
            "mean_f1": float(mean_f1),
            "std_f1": float(std_f1),
            "mean_accuracy": float(df['accuracy'].mean()),
            "mean_precision": float(df['precision'].mean()),
            "mean_recall": float(df['recall'].mean()),
            "seeds": SEEDS,
            "best_seed": int(best_seed),
            "best_f1": float(best_f1),
            "results_per_seed": df.to_dict('records')
        }

        summary["models"][model] = model_summary

        # Track best overall
        if best_f1 > summary["best_overall_f1"]:
            summary["best_overall_f1"] = best_f1
            summary["best_overall_model"] = model

        print(f"    Mean F1: {mean_f1:.4f} ± {std_f1:.4f}")
        print(f"    Best F1: {best_f1:.4f} (seed {best_seed})")

    return summary


# ==================== VISUALIZATION ====================

def generate_comparison_plots(summary: Dict[str, Any]) -> None:
    """Generate comparison plots."""
    if not PLOTTING_AVAILABLE:
        print("[!] Plotting not available. Skipping visualizations.")
        return

    print("\n[+] Generating comparison plots...")

    # Prepare data for plotting
    plot_data = []
    for model, stats in summary["models"].items():
        for result in stats["results_per_seed"]:
            plot_data.append({
                "Model": model,
                "Seed": result["seed"],
                "F1 Score": result["f1"]
            })

    df = pd.DataFrame(plot_data)

    # Set style
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 6))

    # Box plot
    plt.subplot(1, 2, 1)
    sns.boxplot(data=df, x="Model", y="F1 Score", palette="Set2")
    plt.title("F1 Score Distribution by Model")
    plt.xticks(rotation=15)
    plt.ylabel("F1 Score")

    # Line plot (by seed)
    plt.subplot(1, 2, 2)
    for seed in SEEDS:
        seed_data = df[df["Seed"] == seed]
        plt.plot(seed_data["Model"], seed_data["F1 Score"], marker='o', label=f"Seed {seed}")

    plt.title("F1 Score by Seed")
    plt.xlabel("Model")
    plt.ylabel("F1 Score")
    plt.xticks(rotation=15)
    plt.legend()

    plt.tight_layout()
    plt.savefig(SUMMARY_DIR / "metrics_comparison.png", dpi=150)
    print(f"    Saved: metrics_comparison.png")


# ==================== EXPORT ====================

def export_summary(summary: Dict[str, Any]) -> None:
    """Export summary to JSON and Markdown."""

    # JSON export
    summary_file = SUMMARY_DIR / "production_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n[+] Exported JSON: {summary_file}")

    # Markdown report
    md_lines = [
        "# StreamGuard Production Training Summary",
        f"\n**Generated:** {summary['timestamp']}\n",
        "## Results by Model\n"
    ]

    for model, stats in summary["models"].items():
        md_lines.append(f"### {model}\n")
        md_lines.append(f"- **Mean F1:** {stats['mean_f1']:.4f} ± {stats['std_f1']:.4f}")
        md_lines.append(f"- **Best F1:** {stats['best_f1']:.4f} (seed {stats['best_seed']})")
        md_lines.append(f"- **Mean Accuracy:** {stats['mean_accuracy']:.4f}")
        md_lines.append(f"- **Mean Precision:** {stats['mean_precision']:.4f}")
        md_lines.append(f"- **Mean Recall:** {stats['mean_recall']:.4f}\n")

    md_lines.append(f"## Best Overall Model\n")
    md_lines.append(f"**{summary['best_overall_model']}** with F1 = {summary['best_overall_f1']:.4f}\n")

    md_file = SUMMARY_DIR / "production_report.md"
    with open(md_file, 'w') as f:
        f.write('\n'.join(md_lines))

    print(f"[+] Exported Markdown: {md_file}")


# ==================== MAIN ====================

def main():
    """Main aggregation function."""

    print("\n[+] Aggregating metrics from all models and seeds...")

    # Aggregate
    summary = aggregate_all_models()

    if not summary["models"]:
        print("\n[!] No models found. Please run Cells 51-53 first.")
        return

    # Visualize
    generate_comparison_plots(summary)

    # Export
    export_summary(summary)

    print(f"\n{'=' * 80}")
    print("METRICS AGGREGATION COMPLETE")
    print(f"{'=' * 80}")
    print(f"\nBest Model: {summary['best_overall_model']} (F1 = {summary['best_overall_f1']:.4f})")
    print(f"All results saved to: {SUMMARY_DIR}")


if __name__ == "__main__":
    main()
