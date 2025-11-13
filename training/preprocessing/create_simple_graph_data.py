"""
Simple Graph Data Creation for StreamGuard GNN Training

Creates basic sequential graphs from code text where:
- Each line of code is a node
- Sequential edges connect consecutive lines (line i → line i+1)
- Node features are random 768-dim vectors (placeholder for CodeBERT embeddings)

This is a MINIMAL implementation to get GNN training started.
For production, you should use proper AST-based graphs.

Usage:
    # Default paths (CodeXGLUE under data/processed/codexglue)
    python training/preprocessing/create_simple_graph_data.py

    # Custom output + limit samples (smoke)
    python training/preprocessing/create_simple_graph_data.py \
      --output data/processed/graphs_sample --max-samples 200

Outputs:
    data/processed/graphs/train/*.pt - Training graph files
    data/processed/graphs/val/*.pt - Validation graph files
"""

import sys
import json
from pathlib import Path
import argparse
from typing import Dict, Any, List
import warnings

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from tqdm.auto import tqdm

# Try importing PyTorch Geometric
try:
    from torch_geometric.data import Data
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    warnings.warn(
        "PyTorch Geometric not available. Install: "
        "pip install torch-geometric"
    )
    sys.exit(1)


def code_to_simple_graph(code_text: str, label: int, graph_id: int) -> Data:
    """
    Convert code text to simple sequential graph.

    Args:
        code_text: Source code as string
        label: Vulnerability label (0=safe, 1=vulnerable)
        graph_id: Unique graph ID

    Returns:
        PyTorch Geometric Data object
    """
    # Split into lines
    lines = [line.strip() for line in code_text.strip().split('\n') if line.strip()]

    if len(lines) == 0:
        # Empty code - create single-node graph
        x = torch.randn(1, 768)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        return Data(
            x=x,
            edge_index=edge_index,
            y=torch.tensor([label], dtype=torch.long),
            graph_id=graph_id,
            num_lines=1
        )

    # Node features: Random 768-dim vectors (placeholder for CodeBERT)
    # In production, replace with actual CodeBERT embeddings
    num_nodes = len(lines)
    x = torch.randn(num_nodes, 768)

    # Sequential edges: line i → line i+1
    edges = []
    for i in range(num_nodes - 1):
        edges.append([i, i + 1])      # Forward edge
        edges.append([i + 1, i])      # Backward edge (undirected)

    if len(edges) == 0:
        # Single line - no edges
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edges, dtype=torch.long).t()

    # Create graph
    graph = Data(
        x=x,
        edge_index=edge_index,
        y=torch.tensor([label], dtype=torch.long),
        graph_id=graph_id,
        num_lines=num_nodes
    )

    return graph


def process_jsonl_to_graphs(
    jsonl_path: Path,
    output_dir: Path,
    split_name: str,
    max_samples: int = None
) -> Dict[str, Any]:
    """
    Convert JSONL dataset to graph .pt files.

    Args:
        jsonl_path: Path to input JSONL file
        output_dir: Output directory for .pt files
        split_name: Name of split ("train" or "val")
        max_samples: Maximum samples to process (None = all)

    Returns:
        Statistics dict
    """
    print(f"\n[*] Processing {split_name} set: {jsonl_path}")

    if not jsonl_path.exists():
        warnings.warn(f"[!] Input file not found: {jsonl_path}")
        return {"error": "File not found"}

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read JSONL
    samples = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    if max_samples:
        samples = samples[:max_samples]

    print(f"[+] Found {len(samples)} samples")

    # Process each sample
    stats = {
        "total": len(samples),
        "saved": 0,
        "failed": 0,
        "label_counts": {0: 0, 1: 0},
        "avg_nodes": 0,
        "avg_edges": 0
    }

    total_nodes = 0
    total_edges = 0

    for idx, sample in enumerate(tqdm(samples, desc=f"Creating {split_name} graphs")):
        try:
            # Extract code and label
            code = sample.get('code', '') or sample.get('func', '')
            label = sample.get('target', 0)

            if not code:
                stats['failed'] += 1
                continue

            # Create graph
            graph = code_to_simple_graph(code, label, graph_id=idx)

            # Save graph
            output_file = output_dir / f"graph_{idx:06d}.pt"
            torch.save(graph, output_file)

            # Update stats
            stats['saved'] += 1
            stats['label_counts'][label] += 1
            total_nodes += graph.num_nodes
            total_edges += graph.edge_index.shape[1] // 2  # Divide by 2 for undirected

        except Exception as e:
            print(f"\n[!] Failed to process sample {idx}: {str(e)}")
            stats['failed'] += 1

    # Compute averages
    if stats['saved'] > 0:
        stats['avg_nodes'] = total_nodes / stats['saved']
        stats['avg_edges'] = total_edges / stats['saved']

    print(f"\n[+] {split_name.upper()} SET COMPLETE")
    print(f"    Saved: {stats['saved']} graphs")
    print(f"    Failed: {stats['failed']}")
    print(f"    Label 0 (safe): {stats['label_counts'][0]}")
    print(f"    Label 1 (vulnerable): {stats['label_counts'][1]}")
    print(f"    Avg nodes per graph: {stats['avg_nodes']:.1f}")
    print(f"    Avg edges per graph: {stats['avg_edges']:.1f}")

    return stats


def main():
    """Main preprocessing."""
    print("=" * 80)
    print("SIMPLE GRAPH DATA CREATION")
    print("=" * 80)

    # Paths
    parser = argparse.ArgumentParser(description="Create simple sequential graphs for GNN training")
    parser.add_argument('--train-jsonl', type=Path, default=Path('data/processed/codexglue/train.jsonl'))
    parser.add_argument('--val-jsonl', type=Path, default=Path('data/processed/codexglue/valid.jsonl'))
    parser.add_argument('--output', type=Path, default=Path('data/processed/graphs'))
    parser.add_argument('--max-samples', type=int, default=None, help='Limit samples per split (smoke)')
    parser.add_argument('--force', action='store_true', help='Overwrite existing outputs')
    args = parser.parse_args()

    BASE_DIR = args.output
    TRAIN_JSONL = args.train_jsonl
    VAL_JSONL = args.val_jsonl

    OUTPUT_DIR = BASE_DIR
    TRAIN_OUTPUT = OUTPUT_DIR / "train"
    VAL_OUTPUT = OUTPUT_DIR / "val"

    # Guard existing outputs unless --force
    if (TRAIN_OUTPUT.exists() or VAL_OUTPUT.exists()) and not args.force:
        print(f"[!] Output directories already exist under {OUTPUT_DIR}. Use --force to overwrite.")
        # Continue to write stats based on existing graphs if present

    # Check input
    if not TRAIN_JSONL.exists() or not VAL_JSONL.exists():
        print(f"\n[!] ERROR: Input files not found. Train={TRAIN_JSONL}, Val={VAL_JSONL}")
        sys.exit(1)

    # Process splits
    train_stats = process_jsonl_to_graphs(TRAIN_JSONL, TRAIN_OUTPUT, split_name="train", max_samples=args.max_samples)
    val_stats = process_jsonl_to_graphs(VAL_JSONL, VAL_OUTPUT, split_name="val", max_samples=args.max_samples)

    # Save metadata
    stats_file = OUTPUT_DIR / "graphs_metadata.json"
    stats = {
        "train": train_stats,
        "val": val_stats,
        "note": "Simple sequential graphs - replace with AST graphs for production",
        "timestamp": __import__('datetime').datetime.now().isoformat()
    }
    stats_file.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)

    print(f"\n[+] Metadata saved to: {stats_file}")
    print(f"\n[+] Graph data ready!")
    print(f"    Train graphs: {TRAIN_OUTPUT}")
    print(f"    Val graphs: {VAL_OUTPUT}")

    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        print(f"\n[!] Preprocessing failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
