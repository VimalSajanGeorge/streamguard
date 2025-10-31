"""
Automatic Training Visualization Script

Reads metrics_history.csv and generates comprehensive plots:
- Loss curves (train vs val)
- F1 score progression
- Prediction distribution over time
- Learning rate schedule
- Accuracy/Precision/Recall curves

Usage:
    python training/visualize_training.py --metrics-csv models/transformer/metrics_history.csv
    python training/visualize_training.py --model-dir models/transformer
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def load_metrics(csv_path: Path) -> pd.DataFrame:
    """
    Load metrics CSV file.

    Args:
        csv_path: Path to metrics_history.csv

    Returns:
        DataFrame with metrics
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"[+] Loaded {len(df)} epochs from {csv_path}")
    print(f"    Columns: {', '.join(df.columns)}")

    return df


def plot_loss_curves(df: pd.DataFrame, output_dir: Path):
    """Plot training and validation loss curves."""
    plt.figure(figsize=(10, 6))

    plt.plot(df['epoch'], df['train_loss'], marker='o', label='Train Loss', linewidth=2)
    plt.plot(df['epoch'], df['val_loss'], marker='s', label='Val Loss', linewidth=2)

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = output_dir / 'loss_curves.png'
    plt.savefig(output_path, dpi=150)
    print(f"[+] Saved: {output_path}")
    plt.close()


def plot_f1_progression(df: pd.DataFrame, output_dir: Path):
    """Plot F1 score progression over epochs."""
    plt.figure(figsize=(10, 6))

    plt.plot(df['epoch'], df['val_f1'], marker='o', label='F1 (Vulnerable)', linewidth=2, color='green')

    # Mark best F1
    best_f1_idx = df['val_f1'].idxmax()
    best_f1 = df.loc[best_f1_idx, 'val_f1']
    best_epoch = df.loc[best_f1_idx, 'epoch']

    plt.axhline(best_f1, color='red', linestyle='--', alpha=0.5, label=f'Best F1: {best_f1:.4f} (Epoch {best_epoch})')
    plt.scatter([best_epoch], [best_f1], color='red', s=100, zorder=5)

    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('F1 Score Progression (Vulnerable Class)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = output_dir / 'f1_progression.png'
    plt.savefig(output_path, dpi=150)
    print(f"[+] Saved: {output_path}")
    plt.close()


def plot_metrics_overview(df: pd.DataFrame, output_dir: Path):
    """Plot accuracy, precision, recall together."""
    plt.figure(figsize=(10, 6))

    plt.plot(df['epoch'], df['val_acc'], marker='o', label='Accuracy', linewidth=2)
    plt.plot(df['epoch'], df['val_precision'], marker='s', label='Precision', linewidth=2)
    plt.plot(df['epoch'], df['val_recall'], marker='^', label='Recall', linewidth=2)
    plt.plot(df['epoch'], df['val_f1'], marker='d', label='F1', linewidth=2)

    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Validation Metrics Overview')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1.05])
    plt.tight_layout()

    output_path = output_dir / 'metrics_overview.png'
    plt.savefig(output_path, dpi=150)
    print(f"[+] Saved: {output_path}")
    plt.close()


def plot_prediction_distribution(df: pd.DataFrame, output_dir: Path):
    """Plot prediction distribution over time (vulnerable vs safe)."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Subplot 1: Absolute counts
    ax1.plot(df['epoch'], df['pred_vuln'], marker='o', label='Predicted Vulnerable', linewidth=2, color='red')
    ax1.plot(df['epoch'], df['pred_safe'], marker='s', label='Predicted Safe', linewidth=2, color='green')
    ax1.plot(df['epoch'], df['actual_vuln'], marker='', linestyle='--', label='Actual Vulnerable', alpha=0.5, color='darkred')
    ax1.plot(df['epoch'], df['actual_safe'], marker='', linestyle='--', label='Actual Safe', alpha=0.5, color='darkgreen')

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Count')
    ax1.set_title('Prediction Distribution: Absolute Counts')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Subplot 2: Ratios (percentage)
    total_pred = df['pred_vuln'] + df['pred_safe']
    vuln_ratio = (df['pred_vuln'] / total_pred) * 100
    safe_ratio = (df['pred_safe'] / total_pred) * 100

    total_actual = df['actual_vuln'] + df['actual_safe']
    actual_vuln_ratio = (df['actual_vuln'] / total_actual) * 100

    ax2.plot(df['epoch'], vuln_ratio, marker='o', label='Predicted Vulnerable %', linewidth=2, color='red')
    ax2.plot(df['epoch'], safe_ratio, marker='s', label='Predicted Safe %', linewidth=2, color='green')
    ax2.axhline(actual_vuln_ratio.iloc[0], color='darkred', linestyle='--', alpha=0.5,
                label=f'Actual Vulnerable % ({actual_vuln_ratio.iloc[0]:.1f}%)')

    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Percentage (%)')
    ax2.set_title('Prediction Distribution: Percentage')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 105])

    plt.tight_layout()

    output_path = output_dir / 'prediction_distribution.png'
    plt.savefig(output_path, dpi=150)
    print(f"[+] Saved: {output_path}")
    plt.close()


def plot_lr_schedule(df: pd.DataFrame, output_dir: Path):
    """Plot learning rate schedule over epochs."""
    plt.figure(figsize=(10, 6))

    plt.plot(df['epoch'], df['lr'], marker='o', linewidth=2, color='purple')

    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = output_dir / 'lr_schedule.png'
    plt.savefig(output_path, dpi=150)
    print(f"[+] Saved: {output_path}")
    plt.close()


def plot_all_in_one(df: pd.DataFrame, output_dir: Path):
    """Create a comprehensive dashboard with all metrics."""
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # 1. Loss curves
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(df['epoch'], df['train_loss'], marker='o', label='Train Loss', linewidth=2)
    ax1.plot(df['epoch'], df['val_loss'], marker='s', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Curves')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. F1 Score
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(df['epoch'], df['val_f1'], marker='o', label='F1 (Vulnerable)', linewidth=2, color='green')
    best_f1_idx = df['val_f1'].idxmax()
    best_f1 = df.loc[best_f1_idx, 'val_f1']
    best_epoch = df.loc[best_f1_idx, 'epoch']
    ax2.axhline(best_f1, color='red', linestyle='--', alpha=0.5)
    ax2.scatter([best_epoch], [best_f1], color='red', s=100, zorder=5)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('F1 Score')
    ax2.set_title(f'F1 Progression (Best: {best_f1:.4f} @ Epoch {best_epoch})')
    ax2.grid(True, alpha=0.3)

    # 3. Metrics overview
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(df['epoch'], df['val_acc'], marker='o', label='Accuracy', linewidth=2)
    ax3.plot(df['epoch'], df['val_precision'], marker='s', label='Precision', linewidth=2)
    ax3.plot(df['epoch'], df['val_recall'], marker='^', label='Recall', linewidth=2)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Score')
    ax3.set_title('Validation Metrics')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 1.05])

    # 4. Prediction distribution (percentage)
    ax4 = fig.add_subplot(gs[1, 1])
    total_pred = df['pred_vuln'] + df['pred_safe']
    vuln_ratio = (df['pred_vuln'] / total_pred) * 100
    safe_ratio = (df['pred_safe'] / total_pred) * 100
    ax4.plot(df['epoch'], vuln_ratio, marker='o', label='Predicted Vulnerable %', linewidth=2, color='red')
    ax4.plot(df['epoch'], safe_ratio, marker='s', label='Predicted Safe %', linewidth=2, color='green')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Percentage (%)')
    ax4.set_title('Prediction Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, 105])

    # 5. Learning rate schedule
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.plot(df['epoch'], df['lr'], marker='o', linewidth=2, color='purple')
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('Learning Rate')
    ax5.set_title('Learning Rate Schedule')
    ax5.set_yscale('log')
    ax5.grid(True, alpha=0.3)

    # 6. Summary statistics (text)
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.axis('off')

    # Compute summary stats
    final_epoch = df['epoch'].iloc[-1]
    final_f1 = df['val_f1'].iloc[-1]
    final_acc = df['val_acc'].iloc[-1]
    max_f1 = df['val_f1'].max()
    max_f1_epoch = df.loc[df['val_f1'].idxmax(), 'epoch']

    summary_text = f"""
    Training Summary
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Total Epochs: {int(final_epoch)}

    Final Metrics (Epoch {int(final_epoch)}):
      • F1 Score: {final_f1:.4f}
      • Accuracy: {final_acc:.4f}
      • Precision: {df['val_precision'].iloc[-1]:.4f}
      • Recall: {df['val_recall'].iloc[-1]:.4f}

    Best Performance:
      • Best F1: {max_f1:.4f} (Epoch {int(max_f1_epoch)})
      • Best Accuracy: {df['val_acc'].max():.4f}

    Training Stability:
      • Min Val Loss: {df['val_loss'].min():.4f}
      • Final LR: {df['lr'].iloc[-1]:.2e}
    """

    ax6.text(0.1, 0.5, summary_text, transform=ax6.transAxes,
             fontsize=11, verticalalignment='center', family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    fig.suptitle('StreamGuard Training Dashboard', fontsize=16, fontweight='bold')

    output_path = output_dir / 'training_dashboard.png'
    plt.savefig(output_path, dpi=150)
    print(f"[+] Saved comprehensive dashboard: {output_path}")
    plt.close()


def generate_all_plots(metrics_csv: Path, output_dir: Path):
    """
    Generate all training visualization plots.

    Args:
        metrics_csv: Path to metrics_history.csv
        output_dir: Directory to save plots
    """
    print(f"\n{'='*70}")
    print(f"GENERATING TRAINING VISUALIZATIONS")
    print(f"{'='*70}")

    # Load metrics
    df = load_metrics(metrics_csv)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate plots
    print(f"\n[*] Generating plots...")
    plot_loss_curves(df, output_dir)
    plot_f1_progression(df, output_dir)
    plot_metrics_overview(df, output_dir)
    plot_prediction_distribution(df, output_dir)
    plot_lr_schedule(df, output_dir)
    plot_all_in_one(df, output_dir)

    print(f"\n{'='*70}")
    print(f"[+] All plots generated successfully!")
    print(f"[+] Output directory: {output_dir}")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description='Visualize training metrics')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--metrics-csv', type=Path,
                       help='Path to metrics_history.csv file')
    group.add_argument('--model-dir', type=Path,
                       help='Model directory (will look for metrics_history.csv inside)')

    parser.add_argument('--output-dir', type=Path, default=None,
                        help='Output directory for plots (default: same as model dir)')

    args = parser.parse_args()

    # Determine metrics CSV path
    if args.metrics_csv:
        metrics_csv = args.metrics_csv
        default_output_dir = metrics_csv.parent / 'plots'
    else:
        metrics_csv = args.model_dir / 'metrics_history.csv'
        default_output_dir = args.model_dir / 'plots'

    # Determine output directory
    output_dir = args.output_dir if args.output_dir else default_output_dir

    # Check if CSV exists
    if not metrics_csv.exists():
        print(f"[!] ERROR: Metrics file not found: {metrics_csv}")
        print(f"[!] Make sure training has completed and metrics_history.csv exists")
        return 1

    # Generate plots
    try:
        generate_all_plots(metrics_csv, output_dir)
        return 0
    except Exception as e:
        print(f"[!] ERROR: Failed to generate plots: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
