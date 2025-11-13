"""
Model Evaluation with Statistical Significance Testing

Features:
- Multiple runs with different seeds
- Bootstrap confidence intervals
- Paired t-tests for model comparison
- Comprehensive metrics reporting
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report, roc_auc_score
)
from scipy import stats

try:
    from train_transformer import (
        EnhancedSQLIntentTransformer, CodeDataset as TransformerDataset,
        set_seed
    )
    from train_gnn import EnhancedTaintFlowGNN, GraphDataset as GNNDataset
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"[!] Import error: {e}")
    IMPORTS_AVAILABLE = False


def bootstrap_confidence_interval(
    values: List[float],
    confidence: float = 0.95,
    n_bootstrap: int = 1000
) -> Tuple[float, float]:
    """
    Compute bootstrap confidence interval.

    Args:
        values: List of metric values
        confidence: Confidence level
        n_bootstrap: Number of bootstrap samples

    Returns:
        (lower_bound, upper_bound)
    """
    values = np.array(values)
    bootstrap_means = []

    for _ in range(n_bootstrap):
        sample = np.random.choice(values, size=len(values), replace=True)
        bootstrap_means.append(np.mean(sample))

    alpha = 1 - confidence
    lower = np.percentile(bootstrap_means, alpha/2 * 100)
    upper = np.percentile(bootstrap_means, (1 - alpha/2) * 100)

    return lower, upper


def evaluate_single_run(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    model_type: str = "transformer"
) -> Dict[str, float]:
    """
    Evaluate model on dataset (single run).

    Args:
        model: Model instance
        dataloader: Data loader
        device: Device
        model_type: 'transformer', 'gnn', or 'fusion'

    Returns:
        Metrics dictionary
    """
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in dataloader:
            if model_type == "transformer":
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                logits = model(input_ids, attention_mask)

            elif model_type == "gnn":
                data = batch.to(device)
                logits = model(data)
                labels = data.y

            elif model_type == "fusion":
                # Fusion requires both inputs
                # (This is simplified - actual implementation would need both models)
                raise NotImplementedError("Fusion evaluation needs special handling")

            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of vulnerable class

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', pos_label=1
    )

    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
        all_labels, all_preds, average=None, labels=[0, 1]
    )

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()

    # ROC AUC
    try:
        roc_auc = roc_auc_score(all_labels, all_probs)
    except:
        roc_auc = 0.0

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'f1_vulnerable': f1_per_class[1],
        'f1_safe': f1_per_class[0],
        'precision_vulnerable': precision_per_class[1],
        'recall_vulnerable': recall_per_class[1],
        'true_positives': int(tp),
        'false_positives': int(fp),
        'true_negatives': int(tn),
        'false_negatives': int(fn),
        'roc_auc': roc_auc
    }

    return metrics


def evaluate_multiple_runs(
    model_checkpoint: Path,
    test_data_path: Path,
    tokenizer: any,
    model_type: str,
    n_runs: int,
    device: torch.device
) -> Dict[str, List[float]]:
    """
    Evaluate model with multiple seeds.

    Args:
        model_checkpoint: Path to model checkpoint
        test_data_path: Path to test data
        tokenizer: Tokenizer (for transformer)
        model_type: Model type
        n_runs: Number of runs with different seeds
        device: Device

    Returns:
        Dictionary of metric lists
    """
    print(f"\n[*] Evaluating {model_type} with {n_runs} runs")

    all_metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'f1_vulnerable': [],
        'roc_auc': []
    }

    for run in range(n_runs):
        seed = 42 + run
        set_seed(seed)

        print(f"    Run {run + 1}/{n_runs} (seed={seed})...")

        # Load model
        if model_type == "transformer":
            model = EnhancedSQLIntentTransformer()
            checkpoint = torch.load(model_checkpoint, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)

            # Load data
            dataset = TransformerDataset(test_data_path, tokenizer, 512, False)
            dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

        elif model_type == "gnn":
            model = EnhancedTaintFlowGNN()
            checkpoint = torch.load(model_checkpoint, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)

            # Load data
            dataset = GNNDataset(test_data_path, False)
            from torch_geometric.data import DataLoader as PyGDataLoader
            dataloader = PyGDataLoader(dataset, batch_size=32, shuffle=False)

        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Evaluate
        metrics = evaluate_single_run(model, dataloader, device, model_type)

        # Store metrics
        for key in all_metrics.keys():
            all_metrics[key].append(metrics[key])

    return all_metrics


def compare_models_with_significance(
    model1_metrics: Dict[str, List[float]],
    model2_metrics: Dict[str, List[float]],
    model1_name: str,
    model2_name: str
) -> Dict[str, Dict]:
    """
    Compare two models with statistical significance testing.

    Args:
        model1_metrics: Metrics for model 1
        model2_metrics: Metrics for model 2
        model1_name: Name of model 1
        model2_name: Name of model 2

    Returns:
        Comparison results
    """
    print(f"\n[*] Statistical Comparison: {model1_name} vs {model2_name}")
    print("="*70)

    comparison = {}

    for metric_name in model1_metrics.keys():
        values1 = np.array(model1_metrics[metric_name])
        values2 = np.array(model2_metrics[metric_name])

        # Means
        mean1 = np.mean(values1)
        mean2 = np.mean(values2)

        # Standard deviations
        std1 = np.std(values1)
        std2 = np.std(values2)

        # Bootstrap confidence intervals
        ci1_lower, ci1_upper = bootstrap_confidence_interval(values1)
        ci2_lower, ci2_upper = bootstrap_confidence_interval(values2)

        # Paired t-test
        t_stat, p_value = stats.ttest_rel(values2, values1)  # Test if model2 > model1

        # Improvement
        improvement = mean2 - mean1
        improvement_pct = (improvement / mean1) * 100 if mean1 > 0 else 0

        # Significance
        is_significant = p_value < 0.05

        comparison[metric_name] = {
            f'{model1_name}_mean': mean1,
            f'{model1_name}_std': std1,
            f'{model1_name}_ci': (ci1_lower, ci1_upper),
            f'{model2_name}_mean': mean2,
            f'{model2_name}_std': std2,
            f'{model2_name}_ci': (ci2_lower, ci2_upper),
            'improvement': improvement,
            'improvement_pct': improvement_pct,
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': is_significant
        }

        # Print
        print(f"\n{metric_name.upper()}:")
        print(f"  {model1_name}: {mean1:.4f} ± {std1:.4f} (95% CI: [{ci1_lower:.4f}, {ci1_upper:.4f}])")
        print(f"  {model2_name}: {mean2:.4f} ± {std2:.4f} (95% CI: [{ci2_lower:.4f}, {ci2_upper:.4f}])")
        print(f"  Improvement: {improvement:+.4f} ({improvement_pct:+.1f}%)")
        print(f"  p-value: {p_value:.4f} {'✓ SIGNIFICANT' if is_significant else '✗ Not significant'}")

    return comparison


def main():
    parser = argparse.ArgumentParser(description="Evaluate Models with Statistical Testing")

    # Models
    parser.add_argument('--transformer-checkpoint', type=Path, required=True)
    parser.add_argument('--gnn-checkpoint', type=Path, default=None)
    parser.add_argument('--fusion-checkpoint', type=Path, default=None)

    # Data
    parser.add_argument('--test-data', type=Path, required=True)

    # Evaluation
    parser.add_argument('--n-runs', type=int, default=5, help='Number of runs with different seeds')
    parser.add_argument('--compare', action='store_true', help='Compare transformer vs GNN')

    # Output
    parser.add_argument('--output', type=Path, default=Path('evaluation_results.json'))

    args = parser.parse_args()

    if not IMPORTS_AVAILABLE:
        print("[!] Required imports not available. Exiting.")
        return 1

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[+] Using device: {device}")

    # Tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('microsoft/codebert-base')

    # Evaluate Transformer
    transformer_metrics = evaluate_multiple_runs(
        args.transformer_checkpoint,
        args.test_data,
        tokenizer,
        "transformer",
        args.n_runs,
        device
    )

    # Evaluate GNN (if provided)
    gnn_metrics = None
    if args.gnn_checkpoint:
        gnn_metrics = evaluate_multiple_runs(
            args.gnn_checkpoint,
            args.test_data,
            tokenizer,
            "gnn",
            args.n_runs,
            device
        )

    # Compare models
    comparison_results = None
    if args.compare and gnn_metrics:
        comparison_results = compare_models_with_significance(
            transformer_metrics,
            gnn_metrics,
            "Transformer",
            "GNN"
        )

    # Save results
    results = {
        'transformer': {
            metric: {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'values': [float(v) for v in values],
                'ci_95': list(bootstrap_confidence_interval(values))
            }
            for metric, values in transformer_metrics.items()
        }
    }

    if gnn_metrics:
        results['gnn'] = {
            metric: {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'values': [float(v) for v in values],
                'ci_95': list(bootstrap_confidence_interval(values))
            }
            for metric, values in gnn_metrics.items()
        }

    if comparison_results:
        results['comparison'] = {
            metric: {
                k: (float(v) if isinstance(v, (int, float, np.number)) else
                    [float(x) for x in v] if isinstance(v, tuple) else v)
                for k, v in data.items()
            }
            for metric, data in comparison_results.items()
        }

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n[+] Results saved to: {args.output}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
