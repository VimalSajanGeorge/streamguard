#!/usr/bin/env python3
"""
Add A100 production training cells to StreamGuard_Complete_Training.ipynb
Adds cells 50-56 (3 markdown + 3 code cells for Transformer, GNN, Fusion)
"""

import json
import sys
from pathlib import Path

def create_markdown_cell(source):
    """Create a markdown cell"""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source
    }

def create_code_cell(source):
    """Create a code cell"""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source
    }

def add_production_cells(notebook_path):
    """Add A100 production training cells to notebook"""

    # Read notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    print(f"[*] Current notebook has {len(nb['cells'])} cells")

    # Define new cells
    new_cells = []

    # Cell 50: Production Training Header
    new_cells.append(create_markdown_cell([
        "# 🚀 A100 PRODUCTION TRAINING\n",
        "\n",
        "**Configuration:**\n",
        "- GPU: A100 (40GB)\n",
        "- Phase 1 Features: ✅ Enabled (weighted sampler, focal loss, class weighting)\n",
        "- LR Finder: ❌ Disabled (using manual LR values)\n",
        "- Estimated Total Time: 5-7 hours\n",
        "\n",
        "**Training Sequence:**\n",
        "1. Cell 52: Transformer (30-45 min)\n",
        "2. Cell 54: GNN (1-2 hours)\n",
        "3. Cell 56: Fusion (3-5 hours)\n",
        "\n",
        "**Manual Learning Rates:**\n",
        "- Transformer: 2e-5\n",
        "- GNN: 1e-3\n",
        "- Fusion: 1e-3"
    ]))

    # Cell 51: Transformer Training Info
    new_cells.append(create_markdown_cell([
        "## Cell 52: Transformer Production Training\n",
        "\n",
        "**Configuration:**\n",
        "- Epochs: 15\n",
        "- Batch Size: 64\n",
        "- Learning Rate: 2e-5 (manual)\n",
        "- Mixed Precision: Enabled\n",
        "- Expected Time: 30-45 minutes\n",
        "- Target F1: 0.85-0.90\n",
        "\n",
        "**Phase 1 Features:**\n",
        "- ✅ Weighted Sampler (inverse-frequency)\n",
        "- ✅ Class Weight Multiplier: 1.5\n",
        "- ✅ Focal Loss (gamma=2.0)\n",
        "- ✅ Triple Weighting Auto-Adjustment\n",
        "- ✅ Collapse Detection\n",
        "- ✅ Enhanced Metadata"
    ]))

    # Cell 52: Transformer Training Code
    new_cells.append(create_code_cell([
        "# A100 Production Training - Transformer\n",
        "import os\n",
        "os.chdir('/content/streamguard')\n",
        "\n",
        "print(\"=\"*70)\n",
        "print(\"A100 PRODUCTION TRAINING - TRANSFORMER\")\n",
        "print(\"=\"*70)\n",
        "print(\"Configuration: A100 OPTIMIZED\")\n",
        "print(\"Expected Duration: 30-45 minutes\")\n",
        "print(\"=\"*70)\n",
        "\n",
        "!python training/train_transformer.py \\\n",
        "  --train-data data/processed/codexglue/train.jsonl \\\n",
        "  --val-data data/processed/codexglue/valid.jsonl \\\n",
        "  --test-data data/processed/codexglue/test.jsonl \\\n",
        "  --output-dir training/outputs/transformer/production \\\n",
        "  --epochs 15 \\\n",
        "  --batch-size 64 \\\n",
        "  --max-seq-len 512 \\\n",
        "  --lr 2e-5 \\\n",
        "  --weight-decay 0.01 \\\n",
        "  --warmup-ratio 0.15 \\\n",
        "  --dropout 0.1 \\\n",
        "  --early-stopping-patience 3 \\\n",
        "  --mixed-precision \\\n",
        "  --use-weighted-sampler \\\n",
        "  --weight-multiplier 1.5 \\\n",
        "  --focal-loss \\\n",
        "  --focal-gamma 2.0 \\\n",
        "  --seed 42\n",
        "\n",
        "print(\"\\n\" + \"=\"*70)\n",
        "print(\"✅ Transformer training complete!\")\n",
        "print(\"=\"*70)\n",
        "print(\"\\n💡 Next Steps:\")\n",
        "print(\"  1. Check metrics: training/outputs/transformer/production/metrics_history.csv\")\n",
        "print(\"  2. Best model: training/outputs/transformer/production/checkpoints/best_model.pt\")\n",
        "print(\"  3. Metadata: training/outputs/transformer/production/best_model_metadata.json\")\n",
        "print(\"  4. Proceed to GNN training (Cell 54)\")\n",
        "print(\"=\"*70)"
    ]))

    # Cell 53: GNN Training Info
    new_cells.append(create_markdown_cell([
        "## Cell 54: GNN Production Training\n",
        "\n",
        "**Configuration:**\n",
        "- Epochs: 200 (early stopping at ~150)\n",
        "- Batch Size: 128 (auto-adjusted by --auto-batch-size)\n",
        "- Learning Rate: 1e-3 (manual)\n",
        "- Expected Time: 1-2 hours\n",
        "- Target F1: 0.82-0.88\n",
        "\n",
        "**Phase 1 Features:**\n",
        "- ✅ Weighted Sampler (inverse-frequency)\n",
        "- ✅ Class Weight Multiplier: 1.5\n",
        "- ✅ Focal Loss (gamma=2.0)\n",
        "- ✅ Triple Weighting Auto-Adjustment\n",
        "- ✅ Collapse Detection (after epoch 2)\n",
        "- ✅ Auto Batch Size (graph-aware)"
    ]))

    # Cell 54: GNN Training Code
    new_cells.append(create_code_cell([
        "# A100 Production Training - GNN\n",
        "import os\n",
        "os.chdir('/content/streamguard')\n",
        "\n",
        "print(\"=\"*70)\n",
        "print(\"A100 PRODUCTION TRAINING - GNN\")\n",
        "print(\"=\"*70)\n",
        "print(\"Configuration: A100 OPTIMIZED\")\n",
        "print(\"Expected Duration: 1-2 hours\")\n",
        "print(\"=\"*70)\n",
        "\n",
        "!python training/train_gnn.py \\\n",
        "  --train-data data/processed/codexglue/train.jsonl \\\n",
        "  --val-data data/processed/codexglue/valid.jsonl \\\n",
        "  --test-data data/processed/codexglue/test.jsonl \\\n",
        "  --output-dir training/outputs/gnn/production \\\n",
        "  --epochs 200 \\\n",
        "  --batch-size 128 \\\n",
        "  --hidden-dim 256 \\\n",
        "  --num-layers 4 \\\n",
        "  --lr 1e-3 \\\n",
        "  --weight-decay 1e-4 \\\n",
        "  --dropout 0.3 \\\n",
        "  --early-stopping-patience 15 \\\n",
        "  --auto-batch-size \\\n",
        "  --use-weighted-sampler \\\n",
        "  --weight-multiplier 1.5 \\\n",
        "  --focal-loss \\\n",
        "  --focal-gamma 2.0 \\\n",
        "  --seed 42\n",
        "\n",
        "print(\"\\n\" + \"=\"*70)\n",
        "print(\"✅ GNN training complete!\")\n",
        "print(\"=\"*70)\n",
        "print(\"\\n💡 Next Steps:\")\n",
        "print(\"  1. Check metrics: training/outputs/gnn/production/metrics_history.csv\")\n",
        "print(\"  2. Best model: training/outputs/gnn/production/checkpoints/best_model.pt\")\n",
        "print(\"  3. Metadata: training/outputs/gnn/production/best_model_metadata.json\")\n",
        "print(\"  4. Proceed to Fusion training (Cell 56)\")\n",
        "print(\"=\"*70)"
    ]))

    # Cell 55: Fusion Training Info
    new_cells.append(create_markdown_cell([
        "## Cell 56: Fusion Production Training\n",
        "\n",
        "**Configuration:**\n",
        "- N-Folds: 10 (OOF generation)\n",
        "- Epochs: 50 (fusion layer)\n",
        "- Learning Rate: 1e-3\n",
        "- Expected Time: 3-5 hours (10-fold OOF + fusion)\n",
        "- Target F1: 0.88-0.93 (ensemble boost)\n",
        "\n",
        "**Prerequisites:**\n",
        "- ✅ Transformer trained (Cell 52)\n",
        "- ✅ GNN trained (Cell 54)\n",
        "\n",
        "**Training Process:**\n",
        "1. OOF Generation (2-4 hours):\n",
        "   - 10-fold cross-validation\n",
        "   - Fine-tune Transformer (3 epochs/fold)\n",
        "   - Fine-tune GNN (10 epochs/fold)\n",
        "   - Generate out-of-fold predictions\n",
        "2. Fusion Training (5-10 minutes):\n",
        "   - Train learned weights (transformer_weight, gnn_weight)\n",
        "   - Train fusion MLP (4 → 4 → 2)\n",
        "   - Optimize on OOF predictions"
    ]))

    # Cell 56: Fusion Training Code
    new_cells.append(create_code_cell([
        "# A100 Production Training - Fusion\n",
        "import os\n",
        "os.chdir('/content/streamguard')\n",
        "\n",
        "print(\"=\"*70)\n",
        "print(\"A100 PRODUCTION TRAINING - FUSION\")\n",
        "print(\"=\"*70)\n",
        "print(\"Configuration: A100 OPTIMIZED\")\n",
        "print(\"Expected Duration: 3-5 hours (10-fold OOF + fusion)\")\n",
        "print(\"=\"*70)\n",
        "\n",
        "!python training/train_fusion.py \\\n",
        "  --train-data data/processed/codexglue/train.jsonl \\\n",
        "  --val-data data/processed/codexglue/valid.jsonl \\\n",
        "  --test-data data/processed/codexglue/test.jsonl \\\n",
        "  --output-dir training/outputs/fusion/production \\\n",
        "  --transformer-checkpoint training/outputs/transformer/production/checkpoints/best_model.pt \\\n",
        "  --gnn-checkpoint training/outputs/gnn/production/checkpoints/best_model.pt \\\n",
        "  --n-folds 10 \\\n",
        "  --epochs 50 \\\n",
        "  --lr 1e-3 \\\n",
        "  --seed 42\n",
        "\n",
        "print(\"\\n\" + \"=\"*70)\n",
        "print(\"✅ Fusion training complete!\")\n",
        "print(\"=\"*70)\n",
        "print(\"\\n💡 Next Steps:\")\n",
        "print(\"  1. Best fusion model: training/outputs/fusion/production/best_fusion.pt\")\n",
        "print(\"  2. OOF predictions cached: training/outputs/fusion/production/oof_predictions.npz\")\n",
        "print(\"  3. Run evaluation on test set\")\n",
        "print(\"  4. Compare Transformer vs GNN vs Fusion metrics\")\n",
        "print(\"=\"*70)\n",
        "\n",
        "print(\"\\n\" + \"=\"*70)\n",
        "print(\"📊 PRODUCTION TRAINING SUMMARY\")\n",
        "print(\"=\"*70)\n",
        "print(\"All models trained successfully!\")\n",
        "print(\"\\nModel Checkpoints:\")\n",
        "print(\"  - Transformer: training/outputs/transformer/production/checkpoints/best_model.pt\")\n",
        "print(\"  - GNN: training/outputs/gnn/production/checkpoints/best_model.pt\")\n",
        "print(\"  - Fusion: training/outputs/fusion/production/best_fusion.pt\")\n",
        "print(\"\\nMetrics:\")\n",
        "print(\"  - Transformer: training/outputs/transformer/production/metrics_history.csv\")\n",
        "print(\"  - GNN: training/outputs/gnn/production/metrics_history.csv\")\n",
        "print(\"\\nNext: Run evaluation and generate comparison reports\")\n",
        "print(\"=\"*70)"
    ]))

    # Add new cells to notebook
    nb['cells'].extend(new_cells)

    print(f"[+] Added {len(new_cells)} new cells (cells 50-56)")
    print(f"[+] New total: {len(nb['cells'])} cells")

    # Write updated notebook
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print(f"[+] Notebook updated successfully: {notebook_path}")

    # Print summary
    print("\n" + "="*70)
    print("SUMMARY: A100 Production Training Cells Added")
    print("="*70)
    print("Cell 50: Markdown - Production Training Header")
    print("Cell 51: Markdown - Transformer Training Info")
    print("Cell 52: Code     - Transformer Training (epochs=15, batch=64, lr=2e-5)")
    print("Cell 53: Markdown - GNN Training Info")
    print("Cell 54: Code     - GNN Training (epochs=200, batch=128, lr=1e-3)")
    print("Cell 55: Markdown - Fusion Training Info")
    print("Cell 56: Code     - Fusion Training (n_folds=10, epochs=50, lr=1e-3)")
    print("="*70)
    print("\nConfiguration:")
    print("  - GPU: A100 (40GB)")
    print("  - Phase 1 Features: All enabled")
    print("  - LR Finder: Disabled (manual LR values)")
    print("  - Output Dir: training/outputs/{model}/production")
    print("  - Estimated Total Time: 5-7 hours")
    print("="*70)

if __name__ == "__main__":
    notebook_path = Path(__file__).parent.parent / "StreamGuard_Complete_Training.ipynb"

    if not notebook_path.exists():
        print(f"[!] ERROR: Notebook not found: {notebook_path}")
        sys.exit(1)

    add_production_cells(notebook_path)
    print("\n[+] Done! Ready to run production training on A100.")
