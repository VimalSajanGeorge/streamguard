"""
Graph EDA for PyG .pt graphs produced by create_simple_graph_data.py.

Computes: counts, node/edge stats, label distribution, simple health checks,
and a rough batch-size recommendation based on p95 nodes.

Usage:
  python training/analysis/eda_graphs.py \
    --train-dir data/processed/graphs/train \
    --val-dir data/processed/graphs/val \
    --outdir training/outputs/eda_graphs
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any

import numpy as np

try:
    import torch
    from torch_geometric.data import Data
except Exception as exc:
    raise RuntimeError(
        "PyTorch Geometric is required for graph EDA.\n"
        "Install it to match your Torch/CUDA: https://pytorch-geometric.readthedocs.io/"
    ) from exc

try:
    from training.utils.safe_torch import safe_torch_load
except ImportError:
    safe_torch_load = None


def load_pt_graphs(dir_path: Path, max_files: int | None = None) -> list[Data]:
    graphs: list[Data] = []
    files = sorted([p for p in dir_path.glob('*.pt')])
    if max_files:
        files = files[:max_files]
    loader = safe_torch_load if callable(safe_torch_load) else torch.load
    for p in files:
        try:
            g = loader(p)
            if isinstance(g, Data):
                graphs.append(g)
        except Exception:
            continue
    return graphs


def stats_from_graphs(graphs: list[Data]) -> Dict[str, Any]:
    n = len(graphs)
    if n == 0:
        return {
            'num_graphs': 0,
            'label_fraction': {'safe_0': 0.0, 'vuln_1': 0.0},
            'nodes': {}, 'edges': {},
            'empty_edges': 0, 'tiny_nodes': 0, 'huge_nodes': 0,
        }
    labels = np.array([int(getattr(g.y, 'item')() if hasattr(g.y, 'item') else int(g.y)) for g in graphs], dtype=int)
    pos = int(labels.sum())
    neg = n - pos
    label_frac = {'safe_0': float(neg)/n, 'vuln_1': float(pos)/n}

    node_counts = np.array([int(getattr(g, 'num_nodes', g.x.size(0) if hasattr(g, 'x') else 0)) for g in graphs], dtype=int)
    # edge_index shape [2, E], many graphs are undirected with doubled edges; report both
    edges_raw = np.array([int(getattr(g, 'edge_index').size(1) if hasattr(g, 'edge_index') else 0) for g in graphs], dtype=int)
    edges_undirected = edges_raw // 2

    def desc(arr: np.ndarray) -> Dict[str, Any]:
        if arr.size == 0:
            return {}
        return {
            'mean': float(np.mean(arr)),
            'p50': float(np.percentile(arr, 50)),
            'p75': float(np.percentile(arr, 75)),
            'p90': float(np.percentile(arr, 90)),
            'p95': float(np.percentile(arr, 95)),
            'max': int(np.max(arr)),
        }

    empty_edges = int((edges_raw == 0).sum())
    tiny_nodes = int((node_counts <= 2).sum())
    huge_nodes = int((node_counts >= 1000).sum())

    return {
        'num_graphs': n,
        'label_fraction': label_frac,
        'nodes': desc(node_counts),
        'edges_raw': desc(edges_raw),
        'edges_undirected': desc(edges_undirected),
        'empty_edges': empty_edges,
        'tiny_nodes': tiny_nodes,
        'huge_nodes': huge_nodes,
        'node_counts': node_counts,
    }


def recommend_batch_size(stats: Dict[str, Any], gpu_mem_gb: float = 16.0, hidden_dim: int = 256, safety_margin: float = 0.5) -> int:
    """Back-of-the-envelope batch size suggestion using p95 nodes.
    Rough heuristic: memory ~ (nodes * hidden_dim * 4 bytes). Uses 50% headroom.
    """
    nodes = stats.get('nodes', {})
    p95 = nodes.get('p95', 0.0)
    if p95 <= 0:
        return 8
    bytes_per_param = 4
    mem_per_graph = p95 * hidden_dim * bytes_per_param
    available = gpu_mem_gb * 1e9 * safety_margin
    bs = int(max(1, min(64, available / max(mem_per_graph, 1.0))))
    return bs


def write_summary(outdir: Path, train_stats: Dict[str, Any], val_stats: Dict[str, Any]) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    lines = []
    for name, s in (('train', train_stats), ('val', val_stats)):
        lines.append(f"=== {name} ===")
        lines.append(f"graphs: {s['num_graphs']}")
        lf = s['label_fraction']
        lines.append(f"labels: safe_0={lf['safe_0']:.3f}, vuln_1={lf['vuln_1']:.3f}")
        nodes = s['nodes']
        lines.append(f"nodes: mean={nodes.get('mean',0):.1f} p95={nodes.get('p95',0):.0f} max={nodes.get('max',0)}")
        eu = s['edges_undirected']
        lines.append(f"edges(undir): mean={eu.get('mean',0):.1f} p95={eu.get('p95',0):.0f} max={eu.get('max',0)}")
        lines.append(f"empty_edges={s['empty_edges']} tiny_nodes={s['tiny_nodes']} huge_nodes={s['huge_nodes']}")
        lines.append("")

    # Simple batch-size estimate from train p95
    bs = recommend_batch_size(train_stats)
    lines.append(f"recommended_batch_size (p95-based, 16GB, sd=0.5): {bs}")

    (outdir / 'summary.txt').write_text('\n'.join(lines), encoding='utf-8')


def main():
    ap = argparse.ArgumentParser(description='Graph EDA for PyG .pt graphs')
    ap.add_argument('--train-dir', type=Path, required=True)
    ap.add_argument('--val-dir', type=Path, required=True)
    ap.add_argument('--outdir', type=Path, default=Path('training/outputs/eda_graphs'))
    ap.add_argument('--max-files', type=int, default=None, help='Limit number of .pt files to scan')
    args = ap.parse_args()

    train_graphs = load_pt_graphs(args.train_dir, args.max_files)
    val_graphs = load_pt_graphs(args.val_dir, args.max_files)
    train_stats = stats_from_graphs(train_graphs)
    val_stats = stats_from_graphs(val_graphs)
    write_summary(args.outdir, train_stats, val_stats)
    print(f"[ok] Graph EDA written to {args.outdir/'summary.txt'}")


if __name__ == '__main__':
    main()
