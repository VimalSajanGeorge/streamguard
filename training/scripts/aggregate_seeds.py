"""
Seed-level ensemble aggregator for Transformer, GNN, and Fusion outputs.

Combines per-seed validation logits (or recomputes them) to produce an
ensemble metrics report with threshold sweep and balanced accuracy stats.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from torch.utils.data import DataLoader

try:
    from training.models.backbones import load_backbone
    from training.train_transformer import (
        EnhancedSQLIntentTransformer,
        CodeDataset,
        collect_validation_outputs as transformer_collect_outputs,
        dump_validation_logits as _  # noqa: F401 (ensures import side-effects)
    )
    from training.train_gnn import (
        EnhancedTaintFlowGNN,
        GraphDataset,
        collect_validation_outputs as gnn_collect_outputs
    )
    from training.train_fusion import (
        FusionDataset,
        FusionHead,
        FeatureExtractor,
        fusion_collate
    )
except ModuleNotFoundError:
    # Allow execution in environments where training package isn't on PYTHONPATH
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from training.models.backbones import load_backbone  # type: ignore
    from training.train_transformer import (  # type: ignore
        EnhancedSQLIntentTransformer,
        CodeDataset,
        collect_validation_outputs as transformer_collect_outputs
    )
    from training.train_gnn import (  # type: ignore
        EnhancedTaintFlowGNN,
        GraphDataset,
        collect_validation_outputs as gnn_collect_outputs
    )
    from training.train_fusion import (  # type: ignore
        FusionDataset,
        FusionHead,
        FeatureExtractor,
        fusion_collate
    )

try:
    from torch_geometric.loader import DataLoader as PyGDataLoader
except ImportError:  # pragma: no cover
    PyGDataLoader = None


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@dataclass
class SeedLogits:
    seed: str
    probs: np.ndarray
    labels: np.ndarray


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate per-seed validation logits into an ensemble report.")
    parser.add_argument('--track', choices=['transformer', 'gnn', 'fusion'], required=True,
                        help='Model family to aggregate')
    parser.add_argument('--input-dir', type=Path, required=True,
                        help='Path to outputs directory (e.g., training/outputs/transformer_v17)')
    parser.add_argument('--pattern', type=str, default='seed_*/metrics.json',
                        help='Glob pattern for per-seed metrics files (default: seed_*/metrics.json)')
    parser.add_argument('--output', type=Path, default=None,
                        help='Where to write ensemble_metrics.json (default: <input-dir>/ensemble_metrics.json)')
    parser.add_argument('--recompute', action='store_true',
                        help='If logits are missing, recompute them from best checkpoints (requires val data)')
    parser.add_argument('--val-data', type=Path, default=None,
                        help='Validation JSONL (CodeXGLUE) for transformer/fusion recompute')
    parser.add_argument('--graph-val-data', type=Path, default=None,
                        help='Validation graphs JSONL for GNN recompute')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size when recomputing logits')
    parser.add_argument('--max-seq-len', type=int, default=512, help='Max sequence length for transformer/fusion recompute')
    parser.add_argument('--pad-to-multiple-of', type=int, default=None,
                        help='Pad to multiple-of when tokenizing (recompute only)')
    parser.add_argument('--model-name', choices=['codebert', 'graphcodebert', 'unixcoder'], default='codebert',
                        help='Backbone key to use when recomputing transformer/fusion logits')
    parser.add_argument('--gnn-encoder-features', choices=['none', 'cls', 'token'], default='none',
                        help='Encoder feature mode for GNN recompute (default: none)')
    parser.add_argument('--gnn-encoder-feature-dim', type=int, default=768,
                        help='Encoder feature dim for GNN recompute (default: 768)')
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir: Path = args.input_dir
    metrics_paths = sorted(input_dir.glob(args.pattern))
    if not metrics_paths:
        raise SystemExit(f"No metrics found under {input_dir} matching pattern '{args.pattern}'")

    per_seed_stats: List[Dict[str, Optional[float]]] = []
    logits: List[SeedLogits] = []

    for metrics_path in metrics_paths:
        seed_dir = metrics_path.parent
        seed_name = seed_dir.name
        seed_id = seed_name.split('_')[-1]
        try:
            metrics = json.loads(metrics_path.read_text())
        except json.JSONDecodeError as exc:
            print(f"[warn] Failed to parse {metrics_path}: {exc}")
            metrics = {}

        f1_best = metrics.get('f1_at_best_threshold') or metrics.get('best_f1_vulnerable')
        per_seed_stats.append({'seed': seed_id, 'f1_at_best_threshold': f1_best})

        probs, labels = load_saved_logits(seed_dir)
        if probs is None and args.recompute:
            try:
                probs, labels = recompute_logits(args.track, seed_dir, args)
            except Exception as exc:  # pragma: no cover - recompute optional
                print(f"[warn] Failed to recompute logits for {seed_name}: {exc}")
                probs = labels = None

        if probs is None or labels is None:
            print(f"[warn] Skipping {seed_name}: validation logits unavailable.")
            continue

        logits.append(SeedLogits(seed=seed_id, probs=probs, labels=labels))

    if not logits:
        raise SystemExit("No seeds with validation logits were found. "
                         "Rerun training with --dump-val-logits or pass --recompute with proper data paths.")

    validate_label_alignment(logits)
    ensemble_probs = np.mean(np.stack([entry.probs for entry in logits], axis=0), axis=0)
    labels = logits[0].labels

    sweep, best_entry, best_conf = run_threshold_sweep(ensemble_probs, labels)
    if best_entry is None or best_conf is None:
        raise SystemExit("Unable to compute ensemble metrics (empty probabilities).")

    per_seed_values = [stat['f1_at_best_threshold'] for stat in per_seed_stats if stat['f1_at_best_threshold'] is not None]
    mean_f1 = float(np.mean(per_seed_values)) if per_seed_values else None
    std_f1 = float(np.std(per_seed_values)) if per_seed_values else None

    payload = {
        'track': args.track,
        'num_seeds': len(per_seed_stats),
        'num_seeds_with_logits': len(logits),
        'f1_at_best_threshold': float(best_entry['f1']),
        'best_threshold_by_f1': float(best_entry['threshold']),
        'balanced_accuracy_at_best_threshold': float(best_entry['balanced_accuracy']),
        'confusion_matrix_best_threshold': best_conf,
        'threshold_sweep': sweep,
        'per_seed_f1_at_best': per_seed_stats,
        'per_seed_mean_f1': mean_f1,
        'per_seed_std_f1': std_f1,
        'timestamp': datetime.now(timezone.utc).isoformat()
    }

    output_path = args.output or (input_dir / 'ensemble_metrics.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)

    print(f"[ok] Ensemble metrics saved to {output_path}")
    print(f"    Seeds aggregated: {len(logits)} | Ensemble F1@best: {payload['f1_at_best_threshold']:.4f} "
          f"(threshold={payload['best_threshold_by_f1']:.2f})")


def load_saved_logits(seed_dir: Path) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    probs_path = seed_dir / 'val_probs.npy'
    labels_path = seed_dir / 'val_labels.npy'
    if not probs_path.exists() or not labels_path.exists():
        return None, None
    return np.load(probs_path), np.load(labels_path)


def recompute_logits(track: str, seed_dir: Path, args) -> Tuple[np.ndarray, np.ndarray]:
    if track == 'transformer':
        return recompute_transformer_logits(seed_dir, args)
    if track == 'gnn':
        return recompute_gnn_logits(seed_dir, args)
    if track == 'fusion':
        return recompute_fusion_logits(seed_dir, args)
    raise ValueError(f"Unsupported track: {track}")


def recompute_transformer_logits(seed_dir: Path, args) -> Tuple[np.ndarray, np.ndarray]:
    if args.val_data is None:
        raise ValueError("--val-data is required to recompute transformer logits")
    checkpoint = find_best_checkpoint(seed_dir, prefer_subdir=False)
    tokenizer, encoder, _, pooling = load_backbone(args.model_name)
    state = torch.load(checkpoint, map_location=DEVICE)
    state_dict = state['model_state_dict'] if 'model_state_dict' in state else state
    use_features = any(key.startswith('feature_projection') for key in state_dict.keys())
    hidden_dim = state_dict['classifier.0.weight'].shape[1]

    model = EnhancedSQLIntentTransformer(
        encoder=encoder,
        hidden_dim=hidden_dim,
        dropout=0.0,
        use_features=use_features,
        pooling=pooling
    )
    model.load_state_dict(state_dict, strict=True)
    model.to(DEVICE)

    dataset = CodeDataset(
        args.val_data,
        tokenizer,
        max_seq_len=args.max_seq_len,
        use_weights=False,
        use_features=use_features,
        pad_to_multiple_of=args.pad_to_multiple_of
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    outputs = transformer_collect_outputs(model, loader, DEVICE)
    return np.array(outputs['probs'], dtype=np.float32), np.array(outputs['labels'], dtype=np.int64)


def infer_gnn_config(state_dict: Dict[str, torch.Tensor]) -> Dict[str, Optional[int]]:
    config: Dict[str, Optional[int]] = {}
    embedding_weight = state_dict.get('embedding.weight')
    if embedding_weight is not None:
        config['node_vocab_size'] = embedding_weight.shape[0]
        config['embedding_dim'] = embedding_weight.shape[1]
    else:
        config['node_vocab_size'] = 1000
        config['embedding_dim'] = 128

    first_conv = state_dict.get('convs.0.lin_l.weight')
    config['hidden_dim'] = first_conv.shape[0] if first_conv is not None else 256
    conv_layers = {name.split('.')[1] for name in state_dict.keys() if name.startswith('convs')}
    config['num_layers'] = max(len(conv_layers), 1)
    config['dropout'] = 0.3
    proj_weight = state_dict.get('input_proj.weight')
    config['precomputed_feature_dim'] = proj_weight.shape[1] if proj_weight is not None else None
    return config


def recompute_gnn_logits(seed_dir: Path, args) -> Tuple[np.ndarray, np.ndarray]:
    if args.graph_val_data is None:
        raise ValueError("--graph-val-data is required to recompute GNN logits")
    if PyGDataLoader is None:
        raise ImportError("torch_geometric is required to recompute GNN logits")

    checkpoint = find_best_checkpoint(seed_dir, prefer_subdir=False)
    state = torch.load(checkpoint, map_location=DEVICE)
    state_dict = state['model_state_dict'] if 'model_state_dict' in state else state
    config = infer_gnn_config(state_dict)

    model = EnhancedTaintFlowGNN(
        node_vocab_size=config['node_vocab_size'],
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        precomputed_feature_dim=config['precomputed_feature_dim']
    )
    model.load_state_dict(state_dict, strict=True)
    model.to(DEVICE)

    dataset = GraphDataset(
        args.graph_val_data,
        use_weights=False,
        encoder_features=args.gnn_encoder_features,
        encoder_feature_dim=args.gnn_encoder_feature_dim
    )
    loader = PyGDataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    outputs = gnn_collect_outputs(model, loader, DEVICE)
    return np.array(outputs['probs'], dtype=np.float32), np.array(outputs['labels'], dtype=np.int64)


def load_transformer_branch_for_fusion(model_key: str, checkpoint_path: str) -> Tuple[Any, EnhancedSQLIntentTransformer, int]:
    tokenizer, encoder, _, pooling = load_backbone(model_key)
    state = torch.load(checkpoint_path, map_location=DEVICE)
    state_dict = state['model_state_dict'] if 'model_state_dict' in state else state
    use_features = any(key.startswith('feature_projection') for key in state_dict.keys())
    hidden_dim = state_dict['classifier.0.weight'].shape[1]
    model = EnhancedSQLIntentTransformer(
        encoder=encoder,
        hidden_dim=hidden_dim,
        dropout=0.0,
        use_features=use_features,
        pooling=pooling
    )
    model.load_state_dict(state_dict, strict=True)
    model.to(DEVICE)
    model.eval()
    return tokenizer, model, hidden_dim


def load_gnn_branch_from_checkpoint(checkpoint_path: str) -> Tuple[EnhancedTaintFlowGNN, int]:
    state = torch.load(checkpoint_path, map_location=DEVICE)
    state_dict = state['model_state_dict'] if 'model_state_dict' in state else state
    config = infer_gnn_config(state_dict)
    model = EnhancedTaintFlowGNN(
        node_vocab_size=config['node_vocab_size'],
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        precomputed_feature_dim=config['precomputed_feature_dim']
    )
    model.load_state_dict(state_dict, strict=True)
    model.to(DEVICE)
    model.eval()
    graph_dim = model.classifier_in_dim + model.graph_feature_dim
    return model, graph_dim


def recompute_fusion_logits(seed_dir: Path, args) -> Tuple[np.ndarray, np.ndarray]:
    if args.val_data is None:
        raise ValueError("--val-data is required to recompute fusion logits")
    metadata_path = seed_dir / 'fusion_training_metadata.json'
    if not metadata_path.exists():
        raise FileNotFoundError(f"{metadata_path} missing; cannot recompute fusion logits")

    metadata = json.loads(metadata_path.read_text())
    branches = metadata.get('branches', {})
    model_key = branches.get('model_name', args.model_name)
    transformer_ckpt = branches.get('transformer_checkpoint')
    gnn_ckpt = branches.get('gnn_checkpoint')

    tokenizer = None
    transformer_model = None
    if transformer_ckpt and Path(transformer_ckpt).exists():
        tokenizer, transformer_model, _ = load_transformer_branch_for_fusion(model_key, transformer_ckpt)
    else:
        tokenizer = load_backbone(model_key)[0]

    gnn_model = None
    if gnn_ckpt and Path(gnn_ckpt).exists():
        gnn_model, _ = load_gnn_branch_from_checkpoint(gnn_ckpt)

    checkpoint = find_best_checkpoint(seed_dir, prefer_subdir=True)
    state = torch.load(checkpoint, map_location=DEVICE)
    head_state = state.get('head_state_dict', state)
    hidden_dim = head_state['net.0.weight'].shape[0]
    input_dim = head_state['net.0.weight'].shape[1]
    dropout = metadata.get('hyperparameters', {}).get('fusion_dropout', 0.25)

    head = FusionHead(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout)
    head.load_state_dict(head_state)
    head.to(DEVICE)
    head.eval()

    enable_gnn = gnn_model is not None
    dataset = FusionDataset(
        args.val_data,
        tokenizer,
        max_seq_len=args.max_seq_len,
        pad_to_multiple_of=args.pad_to_multiple_of,
        enable_gnn=enable_gnn
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=fusion_collate
    )
    feature_extractor = FeatureExtractor(transformer_model, gnn_model)

    probs: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            feats = feature_extractor(batch, DEVICE)
            logits = head(feats)
            batch_probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            probs.append(batch_probs)
            labels.append(batch['label'].cpu().numpy())

    return np.concatenate(probs).astype(np.float32), np.concatenate(labels).astype(np.int64)


def find_best_checkpoint(seed_dir: Path, prefer_subdir: bool) -> Path:
    """Locate the best checkpoint file."""
    if prefer_subdir:
        candidate = seed_dir / 'checkpoints' / 'best_model.pt'
        if candidate.exists():
            return candidate
    candidate = seed_dir / 'best_model.pt'
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"best_model.pt not found under {seed_dir}")


def validate_label_alignment(logits: Sequence[SeedLogits]):
    reference = logits[0].labels
    for entry in logits[1:]:
        if entry.labels.shape != reference.shape or not np.array_equal(entry.labels, reference):
            raise ValueError("Validation labels differ between seeds. Ensure datasets are aligned.")


def run_threshold_sweep(probs: np.ndarray, labels: np.ndarray) -> Tuple[List[Dict[str, float]], Optional[Dict[str, float]], Optional[Dict[str, int]]]:
    thresholds = np.round(np.arange(0.30, 0.7001, 0.01), 2)
    sweep: List[Dict[str, float]] = []
    best_entry: Optional[Dict[str, float]] = None
    best_conf: Optional[Dict[str, int]] = None

    for thr in thresholds:
        preds = (probs >= thr).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average='binary', pos_label=1, zero_division=0
        )
        cm = confusion_matrix(labels, preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        balanced = compute_balanced_accuracy(tn, fp, fn, tp)
        entry = {
            'threshold': float(thr),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'balanced_accuracy': float(balanced)
        }
        sweep.append(entry)
        if best_entry is None or entry['f1'] > best_entry['f1']:
            best_entry = entry
            best_conf = {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}

    return sweep, best_entry, best_conf


def compute_balanced_accuracy(tn: int, fp: int, fn: int, tp: int) -> float:
    def safe(num, den):
        return num / den if den > 0 else 0.0
    tpr = safe(tp, tp + fn)
    tnr = safe(tn, tn + fp)
    return (tpr + tnr) / 2.0


if __name__ == '__main__':
    main()
