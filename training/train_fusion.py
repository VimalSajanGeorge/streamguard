"""
Late-fusion training baseline that combines Transformer and GNN representations.
"""

import argparse
import json
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast, GradScaler
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)
from transformers import (
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup
)

try:
    from training.train_transformer import (
        EnhancedSQLIntentTransformer,
        set_seed,
        get_git_commit,
        compute_file_checksum
    )
except ModuleNotFoundError:  # pragma: no cover
    from train_transformer import (
        EnhancedSQLIntentTransformer,
        set_seed,
        get_git_commit,
        compute_file_checksum
    )

try:
    from training.train_gnn import EnhancedTaintFlowGNN, TORCH_GEOMETRIC_AVAILABLE
except ModuleNotFoundError:  # pragma: no cover
    from train_gnn import EnhancedTaintFlowGNN, TORCH_GEOMETRIC_AVAILABLE

try:
    from training.losses.focal_loss import FocalLoss
except ModuleNotFoundError:  # pragma: no cover
    from losses.focal_loss import FocalLoss

try:
    from training.models.backbones import load_backbone
except ModuleNotFoundError:  # pragma: no cover
    from models.backbones import load_backbone

try:
    from torch_geometric.data import Data, Batch
except ImportError:  # pragma: no cover
    Data = None
    Batch = None


@dataclass
class FusionSample:
    code: str
    label: int
    weight: float
    tokens: Optional[List[int]]
    graph: Optional[Any]
    graph_features: Optional[Any]


class FusionDataset(Dataset):
    def __init__(
        self,
        data_path: Path,
        tokenizer,
        max_seq_len: int = 512,
        pad_to_multiple_of: Optional[int] = None,
        enable_gnn: bool = True
    ):
        if enable_gnn and (not TORCH_GEOMETRIC_AVAILABLE or Data is None):
            raise ImportError("PyTorch Geometric is required for GNN branch")

        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.pad_to_multiple_of = pad_to_multiple_of
        self.enable_gnn = enable_gnn
        self.pad_token_id = getattr(tokenizer, 'pad_token_id', 1) if tokenizer else 1
        self.samples: List[FusionSample] = []

        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                code = record.get('code') or record.get('func') or ''
                tokens = record.get('tokens')
                label = int(record.get('label', record.get('target', 0)))
                weight = float(record.get('weight', 1.0))

                graph_obj = None
                graph_features = None
                if enable_gnn:
                    graph_obj, graph_features = self._build_graph(record, label)

                self.samples.append(
                    FusionSample(
                        code=code,
                        label=label,
                        weight=weight,
                        tokens=tokens,
                        graph=graph_obj,
                        graph_features=graph_features
                    )
                )

    def _build_graph(self, record: Dict[str, Any], label: int):
        ast_nodes = record.get('ast_nodes', [])
        edge_index = record.get('edge_index', [])
        if not ast_nodes:
            tokens = [t for t in (record.get('code') or '').split() if t]
            if not tokens:
                ast_nodes = [0]
            else:
                ast_nodes = [abs(hash(t)) % 1000 for t in tokens]
                edge_index = [[i, i + 1] for i in range(len(ast_nodes) - 1)]
                edge_index += [[i + 1, i] for i in range(len(ast_nodes) - 1)]

        x = torch.tensor(ast_nodes, dtype=torch.long).unsqueeze(1)
        if edge_index:
            edge_idx = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        else:
            edge_idx = torch.tensor([[0], [0]], dtype=torch.long)

        graph = Data(x=x, edge_index=edge_idx, y=torch.tensor([label], dtype=torch.long))
        graph_features = None
        if 'graph_cls' in record:
            try:
                graph_features = torch.tensor(record['graph_cls'], dtype=torch.float32)
                graph.graph_features = graph_features
            except Exception:
                graph_features = None
        return graph, graph_features

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        if self.tokenizer:
            if sample.tokens:
                token_ids = sample.tokens[: self.max_seq_len]
                if len(token_ids) < self.max_seq_len:
                    pad_len = self.max_seq_len - len(token_ids)
                    token_ids = token_ids + [self.pad_token_id] * pad_len
                token_tensor = torch.tensor(token_ids, dtype=torch.long)
                attention = (token_tensor != self.pad_token_id).long()
            else:
                encoding = self.tokenizer(
                    sample.code,
                    max_length=self.max_seq_len,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt',
                    pad_to_multiple_of=self.pad_to_multiple_of
                )
                token_tensor = encoding['input_ids'].squeeze(0)
                attention = encoding['attention_mask'].squeeze(0)
        else:
            token_tensor = torch.zeros(self.max_seq_len, dtype=torch.long)
            attention = torch.zeros_like(token_tensor)

        graph = None
        if sample.graph is not None:
            graph = sample.graph.clone() if hasattr(sample.graph, 'clone') else sample.graph
            if sample.graph_features is not None:
                if isinstance(sample.graph_features, torch.Tensor):
                    graph.graph_features = sample.graph_features.clone()
                else:
                    graph.graph_features = sample.graph_features

        return {
            'input_ids': token_tensor,
            'attention_mask': attention,
            'graph': graph,
            'label': torch.tensor(sample.label, dtype=torch.long)
        }


def fusion_collate(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])

    graphs = [item['graph'] for item in batch if item['graph'] is not None]
    graph_batch = Batch.from_data_list(graphs) if graphs and Batch is not None else None

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'graph_batch': graph_batch,
            'label': labels
        }


def dump_val_logits(output_dir: Path, probs: np.ndarray, labels: np.ndarray):
    """
    Persist validation logits for ensembling/aggregation workflows.
    """
    if probs.size == 0 or labels.size == 0:
        print("[warn] No validation logits to dump.")
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / 'val_probs.npy', probs.astype(np.float32))
    np.save(output_dir / 'val_labels.npy', labels.astype(np.int64))
    print(f"[+] Saved fusion validation logits under {output_dir}")


class FusionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 512, dropout: float = 0.25, num_labels: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class FeatureExtractor:
    def __init__(self, transformer_model: Optional[EnhancedSQLIntentTransformer], gnn_model: Optional[EnhancedTaintFlowGNN]):
        self.transformer = transformer_model
        self.gnn = gnn_model

    def __call__(self, batch: Dict[str, Any], device: torch.device) -> torch.Tensor:
    features: List[torch.Tensor] = []
        if self.transformer is not None:
            with torch.no_grad():
                feats = self.transformer.extract_features(
                    batch['input_ids'].to(device),
                    batch['attention_mask'].to(device)
                )
                features.append(feats)
        if self.gnn is not None and batch.get('graph_batch') is not None:
            with torch.no_grad():
                g_feats = self.gnn.extract_graph_embedding(batch['graph_batch'].to(device))
                features.append(g_feats)
        if not features:
            raise RuntimeError("No active branches for fusion")
        return torch.cat(features, dim=1)


class EMAHelper:
    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.shadow = {name: param.data.clone() for name, param in model.named_parameters() if param.requires_grad}
        self.backup: Optional[Dict[str, torch.Tensor]] = None

    def update(self, model: nn.Module):
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            assert name in self.shadow
            self.shadow[name].mul_(self.decay).add_(param.data * (1.0 - self.decay))

    def apply_shadow(self, model: nn.Module):
        self.backup = {}
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module):
        if self.backup is None:
            return
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = None


class FusionCheckpointManager:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.checkpoint_dir = output_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save(self, epoch: int, head: nn.Module, optimizer, scheduler, best_metrics: Dict[str, Any], ema_helper: Optional[EMAHelper], is_best: bool):
        payload = {
            'epoch': epoch,
            'head_state_dict': head.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'best_metrics': best_metrics
        }
        if ema_helper:
            payload['ema_shadow'] = {k: v.clone() for k, v in ema_helper.shadow.items()}

        ckpt_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(payload, ckpt_path)
        if is_best:
            torch.save(payload, self.checkpoint_dir / 'best_model.pt')


def build_transformer_branch(args, device: torch.device):
    if args.transformer_checkpoint is None or not Path(args.transformer_checkpoint).exists():
        print('[fusion] Transformer checkpoint not found. Disabling transformer branch.')
        return None, None

    tokenizer, encoder, hidden_size, pooling = load_backbone(args.model_name)
    state = torch.load(args.transformer_checkpoint, map_location=device)
    state_dict = state['model_state_dict'] if 'model_state_dict' in state else state

    use_features = any(k.startswith('feature_projection') for k in state_dict.keys())
    inferred_hidden = state_dict['classifier.0.weight'].shape[1]

    model = EnhancedSQLIntentTransformer(
        encoder=encoder,
        hidden_dim=inferred_hidden,
        dropout=args.transformer_dropout,
        use_features=use_features,
        pooling=pooling
    )
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    return tokenizer, model


def infer_gnn_config(state_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    node_vocab = state_dict.get('embedding.weight').shape[0] if 'embedding.weight' in state_dict else 1000
    embedding_dim = state_dict.get('embedding.weight').shape[1] if 'embedding.weight' in state_dict else 128
    hidden_dim = state_dict.get('convs.0.lin_l.weight').shape[0] if 'convs.0.lin_l.weight' in state_dict else 256
    num_layers = len({k.split('.')[1] for k in state_dict.keys() if k.startswith('convs')})
    dropout = 0.3
    precomputed_dim = state_dict.get('input_proj.weight').shape[1] if 'input_proj.weight' in state_dict else None
    return {
        'node_vocab_size': node_vocab,
        'embedding_dim': embedding_dim,
        'hidden_dim': hidden_dim,
        'num_layers': max(num_layers, 1),
        'dropout': dropout,
        'precomputed_feature_dim': precomputed_dim
    }


def build_gnn_branch(checkpoint_path: Optional[str], device: torch.device):
    if checkpoint_path is None or not Path(checkpoint_path).exists():
        print('[fusion] GNN checkpoint not found. Disabling GNN branch.')
        return None, None
    if not TORCH_GEOMETRIC_AVAILABLE or Data is None:
        print('[fusion] PyTorch Geometric unavailable. Disabling GNN branch.')
        return None, None

    state = torch.load(checkpoint_path, map_location=device)
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
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    graph_dim = model.classifier_in_dim + model.graph_feature_dim
    return model, graph_dim


def build_criterion(args, device):
    if args.focal_loss:
        alpha = torch.tensor([1.0, 1.0], dtype=torch.float32, device=device)
        return FocalLoss(alpha=alpha, gamma=args.focal_gamma)
    return nn.CrossEntropyLoss()


def train_one_epoch(head, dataloader, optimizer, scheduler, criterion, device, feature_extractor, scaler, grad_clip, ema_helper, amp_dtype):
    head.train()
    total_loss = 0.0
    total_samples = 0

    device_type = 'cuda' if device.type == 'cuda' else 'cpu'

    for batch in dataloader:
        labels = batch['label'].to(device)
        features = feature_extractor(batch, device)

        optimizer.zero_grad(set_to_none=True)
        use_amp = scaler is not None
        with autocast(device_type=device_type, dtype=amp_dtype, enabled=use_amp):
            logits = head(features)
            loss = criterion(logits, labels)

        if scaler:
            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(head.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(head.parameters(), grad_clip)
            optimizer.step()

        if scheduler:
            scheduler.step()
        if ema_helper:
            ema_helper.update(head)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / max(total_samples, 1)


def run_validation(head, dataloader, criterion, device, feature_extractor, ema_helper):
    if ema_helper:
        ema_helper.apply_shadow(head)

    head.eval()
    losses = []
    probs = []
    labels = []

    with torch.no_grad():
        for batch in dataloader:
            features = feature_extractor(batch, device)
            label = batch['label'].to(device)
            logits = head(features)
            if criterion:
                loss = criterion(logits, label)
                losses.append(loss.item() * label.size(0))
            prob = torch.softmax(logits, dim=1)[:, 1]
            probs.append(prob.cpu().numpy())
            labels.append(label.cpu().numpy())

    if ema_helper:
        ema_helper.restore(head)

    all_probs = np.concatenate(probs) if probs else np.array([])
    all_labels = np.concatenate(labels) if labels else np.array([])
    avg_loss = sum(losses) / max(sum(len(l) for l in labels), 1) if losses else 0.0
    return avg_loss, all_probs, all_labels


def compute_threshold_metrics(y_true: np.ndarray, probs: np.ndarray) -> Dict[str, Any]:
    if probs.size == 0 or y_true.size == 0:
        default_conf = {'tn': 0, 'fp': 0, 'fn': 0, 'tp': 0}
        return {
            'f1_at_0_5': 0.0,
            'precision_at_0_5': 0.0,
            'recall_at_0_5': 0.0,
            'confusion_matrix_threshold_0_5': default_conf,
            'threshold_sweep': [],
            'best_entry': {
                'threshold': 0.5,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'balanced_accuracy': 0.0,
                'tp': 0,
                'tn': 0,
                'fp': 0,
                'fn': 0
            }
        }
    thresholds = np.linspace(0.05, 0.95, 19)
    sweep = []
    best_entry = None

    for thr in thresholds:
        preds = (probs >= thr).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, preds, average='binary', zero_division=0
        )
        tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
        balanced = 0.5 * (
            (tp / (tp + fn + 1e-8)) + (tn / (tn + fp + 1e-8))
        )
        entry = {
            'threshold': float(thr),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'balanced_accuracy': float(balanced),
            'tp': int(tp),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn)
        }
        sweep.append(entry)
        if best_entry is None or f1 > best_entry['f1']:
            best_entry = entry

    default_thr = 0.5
    default_preds = (probs >= default_thr).astype(int)
    precision_05, recall_05, f1_05, _ = precision_recall_fscore_support(
        y_true, default_preds, average='binary', zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(y_true, default_preds, labels=[0, 1]).ravel()
    default_conf = {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}

    return {
        'f1_at_0_5': float(f1_05),
        'precision_at_0_5': float(precision_05),
        'recall_at_0_5': float(recall_05),
        'confusion_matrix_threshold_0_5': default_conf,
        'threshold_sweep': sweep,
        'best_entry': best_entry
    }


def save_metrics(output_dir: Path, metrics: Dict[str, Any]):
    metrics_path = output_dir / 'metrics.json'
    metrics['timestamp'] = datetime.now(timezone.utc).isoformat()
    with metrics_path.open('w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)


def save_training_metadata(args, output_dir: Path, data_paths: Dict[str, Path], best_metrics: Dict[str, Any]):
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'git_commit': get_git_commit(),
        'seed': args.seed,
        'hyperparameters': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'scheduler': args.scheduler,
            'warmup_ratio': args.warmup_ratio,
            'fusion_hidden_dim': args.fusion_hidden_dim,
            'fusion_dropout': args.fusion_dropout,
            'grad_clip_norm': args.grad_clip_norm,
            'mixed_precision': args.mixed_precision,
            'amp_dtype': args.amp_dtype
        },
        'branches': {
            'transformer_checkpoint': str(args.transformer_checkpoint) if args.transformer_checkpoint else None,
            'gnn_checkpoint': str(args.gnn_checkpoint) if args.gnn_checkpoint else None,
            'model_name': args.model_name
        },
        'dataset_checksums': {
            split: compute_file_checksum(path)
            for split, path in data_paths.items()
        },
        'best_metrics': best_metrics
    }
    with (output_dir / 'fusion_training_metadata.json').open('w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description='Late Fusion Training (Transformer + GNN)')
    parser.add_argument('--train-data', type=Path, required=True)
    parser.add_argument('--val-data', type=Path, required=True)
    parser.add_argument('--transformer-checkpoint', type=str, default=None)
    parser.add_argument('--gnn-checkpoint', type=str, default=None)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--model-name', choices=['codebert', 'graphcodebert', 'unixcoder'], default='codebert')
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--scheduler', choices=['linear', 'cosine', 'none'], default='linear')
    parser.add_argument('--warmup-ratio', type=float, default=0.05)
    parser.add_argument('--mixed-precision', action='store_true')
    parser.add_argument('--amp-dtype', choices=['fp16', 'bf16'], default='fp16')
    parser.add_argument('--grad-clip-norm', type=float, default=1.0)
    parser.add_argument('--fusion-hidden-dim', type=int, default=512)
    parser.add_argument('--fusion-dropout', type=float, default=0.25)
    parser.add_argument('--max-seq-len', type=int, default=512)
    parser.add_argument('--pad-to-multiple-of', type=int, default=None)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--prefetch-factor', type=int, default=2)
    parser.add_argument('--persistent-workers', action='store_true')
    parser.add_argument('--drop-last', action='store_true')
    parser.add_argument('--ema', action='store_true')
    parser.add_argument('--ema-decay', type=float, default=0.995)
    parser.add_argument('--focal-loss', action='store_true')
    parser.add_argument('--focal-gamma', type=float, default=2.0)
    parser.add_argument('--quick-test', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--transformer-dropout', type=float, default=0.1)
    parser.add_argument('--dump-val-logits', action='store_true', help='Persist validation logits for ensembling')
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[fusion] Using device: {device}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_mgr = FusionCheckpointManager(args.output_dir)

    tokenizer, transformer_model = build_transformer_branch(args, device)
    gnn_model, gnn_dim = build_gnn_branch(args.gnn_checkpoint, device)

    if transformer_model is None and gnn_model is None:
        raise RuntimeError('At least one branch (Transformer or GNN) must be available')

    tokenizer = tokenizer or load_backbone('codebert')[0]

    enable_gnn = gnn_model is not None
    train_dataset = FusionDataset(
        args.train_data,
        tokenizer,
        max_seq_len=args.max_seq_len,
        pad_to_multiple_of=args.pad_to_multiple_of,
        enable_gnn=enable_gnn
    )
    val_dataset = FusionDataset(
        args.val_data,
        tokenizer,
        max_seq_len=args.max_seq_len,
        pad_to_multiple_of=args.pad_to_multiple_of,
        enable_gnn=enable_gnn
    )

    if args.quick_test:
        train_dataset.samples = train_dataset.samples[:400]
        val_dataset.samples = val_dataset.samples[:100]

    loader_kwargs = {
        'batch_size': args.batch_size,
        'shuffle': True,
        'num_workers': args.num_workers,
        'pin_memory': device.type == 'cuda',
        'drop_last': args.drop_last,
        'collate_fn': fusion_collate
    }
    if args.num_workers > 0:
        loader_kwargs['prefetch_factor'] = args.prefetch_factor
        loader_kwargs['persistent_workers'] = args.persistent_workers

    train_loader = DataLoader(train_dataset, **loader_kwargs)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == 'cuda',
        collate_fn=fusion_collate
    )

    feature_extractor = FeatureExtractor(transformer_model, gnn_model)
    input_dim = 0
    if transformer_model is not None:
        input_dim += transformer_model.hidden_dim
    if gnn_model is not None:
        input_dim += gnn_dim

    head = FusionHead(input_dim=input_dim, hidden_dim=args.fusion_hidden_dim, dropout=args.fusion_dropout)
    head.to(device)

    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    if total_steps > 0:
        warmup_steps = int(total_steps * args.warmup_ratio)
        if args.scheduler == 'linear':
            scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        elif args.scheduler == 'cosine':
            scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        else:
            scheduler = None
    else:
        scheduler = None

    criterion = build_criterion(args, device)
    scaler = GradScaler() if (args.mixed_precision and device.type == 'cuda') else None
    amp_dtype = torch.float16 if args.amp_dtype == 'fp16' else torch.bfloat16
    ema_helper = EMAHelper(head, args.ema_decay) if args.ema else None

    csv_path = args.output_dir / 'metrics_history.csv'
    with csv_path.open('w', encoding='utf-8') as f:
        f.write('epoch,train_loss,val_loss,f1_at_0_5,f1_at_best,precision_at_0_5,recall_at_0_5,best_threshold,lr\n')

    best_metrics = None
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")
        train_loss = train_one_epoch(
            head,
            train_loader,
            optimizer,
            scheduler,
            criterion,
            device,
            feature_extractor,
            scaler,
            args.grad_clip_norm,
            ema_helper,
            amp_dtype
        )

        val_loss, val_probs, val_labels = run_validation(
            head,
            val_loader,
            criterion,
            device,
            feature_extractor,
            ema_helper
        )
        if args.dump_val_logits:
            dump_val_logits(args.output_dir, val_probs, val_labels)

        metrics = compute_threshold_metrics(val_labels, val_probs)
        best_entry = metrics.get('best_entry') or {}
        best_f1 = best_entry.get('f1', 0.0)
        lr_val = optimizer.param_groups[0]['lr']

        with csv_path.open('a', encoding='utf-8') as f:
            f.write(
                f"{epoch},{train_loss:.4f},{val_loss:.4f},{metrics.get('f1_at_0_5', 0.0):.4f},"
                f"{best_f1:.4f},{metrics.get('precision_at_0_5', 0.0):.4f},{metrics.get('recall_at_0_5', 0.0):.4f},"
                f"{best_entry.get('threshold', 0.5):.3f},{lr_val:.6f}\n"
            )
        print(f"Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f} | F1@0.5: {metrics.get('f1_at_0_5', 0.0):.4f} | F1@best: {best_f1:.4f}")

        improve = (best_metrics is None) or (best_f1 > best_metrics['best_entry']['f1'])
        if improve:
            best_metrics = metrics
            best_epoch = epoch

        checkpoint_mgr.save(epoch, head, optimizer, scheduler, best_metrics or metrics, ema_helper, is_best=improve)

    if best_metrics is None:
        best_metrics = metrics

    summary = {
        'best_epoch': best_epoch,
        'f1_at_0_5': best_metrics.get('f1_at_0_5', 0.0),
        'f1_at_best_threshold': best_metrics['best_entry']['f1'],
        'best_threshold_by_f1': best_metrics['best_entry']['threshold'],
        'balanced_accuracy_at_best_threshold': best_metrics['best_entry']['balanced_accuracy'],
        'confusion_matrix_threshold_0_5': best_metrics.get('confusion_matrix_threshold_0_5'),
        'confusion_matrix_best_threshold': {
            'tn': best_metrics['best_entry']['tn'],
            'fp': best_metrics['best_entry']['fp'],
            'fn': best_metrics['best_entry']['fn'],
            'tp': best_metrics['best_entry']['tp']
        },
        'threshold_sweep': best_metrics.get('threshold_sweep', [])
    }
    save_metrics(args.output_dir, summary)

    data_paths = {'train': args.train_data, 'val': args.val_data}
    save_training_metadata(args, args.output_dir, data_paths, summary)

    print(f"\n[fusion] Training complete. Metrics saved to {args.output_dir}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
