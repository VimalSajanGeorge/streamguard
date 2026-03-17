#!/usr/bin/env python3
"""
training/scripts/preprocessing/stage5_embed.py

Stage 5: Node Embedding for StreamGuard pipeline.

Input:  training/data/processed/cpg/{shard}/{sample_id}.json  (one CPG per sample)
Output: training/data/processed/embedded/{shard}/{sample_id}.npz  (node features per sample)

Per-sample pipeline:
  1. Load CPG JSON (nodes + edges)
  2. For each node, extract code statement text
  3. Batch-encode all node code statements through CodeBERT (batch_size=64, max 64 tokens)
  4. Build 824-d feature vector per node:
     [0:768]   CodeBERT [CLS] embedding
     [768:800] Node type one-hot (32 Joern node types)
     [800:808] Taint role one-hot (8 slots: source/sink/sanitizer/propagation/none + 3 reserved)
     [808:812] CPG component one-hot (AST=0, CFG=1, DFG=2, TPG=3)
     [812:824] Structural features (in_degree, out_degree, AST_depth, taint_dist_to_sink x3)
  5. Save as .npz with key "node_features" shape (N, 824) float32

Risk mitigations (from StreamGuard_Stories5to8_RiskAnalysis.docx):
  - R1: Batch CodeBERT inference (batch_size=64) with AMP for throughput
  - R2: Empty code nodes -> zero vector (never skip; preserves graph topology)
  - R3: Max 64 tokens per node (not 512 — nodes are single statements, not functions)
  - R4: Checkpoint/resume via skip-if-exists on output .npz files
  - R5: @torch.inference_mode() to disable grad tracking (lower memory)
  - R6: NaN validation before saving each .npz

Source: docs/New Docs/PREPROCESSING.md §Stage 5
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import torch
from loguru import logger

# ── Feature vector dimensions ────────────────────────────────────
CODEBERT_DIM = 768
NODE_TYPE_DIM = 32
TAINT_ROLE_DIM = 8
CPG_COMPONENT_DIM = 4
STRUCTURAL_DIM = 12
TOTAL_DIM = CODEBERT_DIM + NODE_TYPE_DIM + TAINT_ROLE_DIM + CPG_COMPONENT_DIM + STRUCTURAL_DIM
assert TOTAL_DIM == 824, f"Feature vector must be 824-d, got {TOTAL_DIM}"

# ── CodeBERT config ──────────────────────────────────────────────
CODEBERT_MODEL = "microsoft/codebert-base"
CODEBERT_MAX_TOKENS = 64   # Per-node: single statements, NOT full functions
CODEBERT_BATCH_SIZE = 64   # Batch inference for throughput

# ── Node type vocabulary (32 slots) ──────────────────────────────
# Ordered list of Joern CPG node types. First 17 are observed in our
# SARD dataset; remaining slots reserved for real-world code patterns.
# Index in this list = one-hot position in [768:800].
JOERN_NODE_TYPES: list[str] = [
    "BLOCK",                  # 0
    "CALL",                   # 1
    "CONTROL_STRUCTURE",      # 2
    "FIELD_IDENTIFIER",       # 3
    "IDENTIFIER",             # 4
    "JUMP_TARGET",            # 5
    "LITERAL",                # 6
    "LOCAL",                  # 7
    "METHOD",                 # 8
    "METHOD_PARAMETER_IN",    # 9
    "METHOD_PARAMETER_OUT",   # 10
    "METHOD_REF",             # 11
    "METHOD_RETURN",          # 12
    "MODIFIER",               # 13
    "RETURN",                 # 14
    "TYPE_DECL",              # 15
    "TYPE_REF",               # 16
    "UNKNOWN",                # 17
    # Reserved for real-world / future Joern versions
    "ANNOTATION",             # 18
    "ANNOTATION_LITERAL",     # 19
    "ANNOTATION_PARAMETER",   # 20
    "BINDING",                # 21
    "CLOSURE_BINDING",        # 22
    "COMMENT",                # 23
    "FILE",                   # 24
    "MEMBER",                 # 25
    "META_DATA",              # 26
    "NAMESPACE",              # 27
    "NAMESPACE_BLOCK",        # 28
    "TAG",                    # 29
    "TEMPLATE_DOM",           # 30
    "_RESERVED_31",           # 31
]
assert len(JOERN_NODE_TYPES) == NODE_TYPE_DIM
NODE_TYPE_TO_IDX = {t: i for i, t in enumerate(JOERN_NODE_TYPES)}

# ── Taint role vocabulary (8 slots) ──────────────────────────────
TAINT_ROLES: list[str] = [
    "SOURCE",       # 0
    "SINK",         # 1
    "SANITIZER",    # 2
    "PROPAGATION",  # 3
    "NONE",         # 4
    "_RESERVED_5",  # 5
    "_RESERVED_6",  # 6
    "_RESERVED_7",  # 7
]
assert len(TAINT_ROLES) == TAINT_ROLE_DIM
TAINT_ROLE_TO_IDX = {r: i for i, r in enumerate(TAINT_ROLES)}

# ── CPG edge type map ────────────────────────────────────────────
EDGE_TYPE_MAP = {"AST": 0, "CFG": 1, "DFG": 2, "TPG": 3}


# ── CodeBERT encoder (lazy singleton) ───────────────────────────
class CodeBERTEncoder:
    """Wraps CodeBERT for batched node-level embedding extraction."""

    def __init__(self, model_name: str = CODEBERT_MODEL, device: str | None = None):
        from transformers import AutoModel, AutoTokenizer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading CodeBERT from {model_name} on {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def encode_batch(self, texts: list[str]) -> np.ndarray:
        """
        Encode a batch of code strings, returning [CLS] embeddings.

        Args:
            texts: list of code statements (one per node)

        Returns:
            np.ndarray of shape (len(texts), 768) float32
        """
        if not texts:
            return np.zeros((0, CODEBERT_DIM), dtype=np.float32)

        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=CODEBERT_MAX_TOKENS,
            return_tensors="pt",
        )
        tokens = {k: v.to(self.device) for k, v in tokens.items()}

        with torch.autocast(device_type=self.device.split(":")[0], dtype=torch.float16,
                            enabled=(self.device != "cpu")):
            outputs = self.model(**tokens)

        # [CLS] token embedding — first token of last hidden state
        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        return cls_embeddings.float().cpu().numpy()


# ── Feature construction ─────────────────────────────────────────
def _build_node_type_onehot(label: str) -> np.ndarray:
    """32-d one-hot for Joern node type."""
    vec = np.zeros(NODE_TYPE_DIM, dtype=np.float32)
    idx = NODE_TYPE_TO_IDX.get(label, NODE_TYPE_TO_IDX["UNKNOWN"])
    vec[idx] = 1.0
    return vec


def _build_taint_role_onehot(role: str) -> np.ndarray:
    """8-d one-hot for taint role."""
    vec = np.zeros(TAINT_ROLE_DIM, dtype=np.float32)
    idx = TAINT_ROLE_TO_IDX.get(role, TAINT_ROLE_TO_IDX["NONE"])
    vec[idx] = 1.0
    return vec


def _build_cpg_component_onehot(max_edge_type: int) -> np.ndarray:
    """
    4-d one-hot for dominant CPG component this node participates in.
    Priority: TPG > DFG > CFG > AST (higher signal = dominant).
    max_edge_type: pre-computed max edge type for this node, or -1 if isolated.
    """
    vec = np.zeros(CPG_COMPONENT_DIM, dtype=np.float32)
    if max_edge_type >= 0:
        vec[min(max_edge_type, CPG_COMPONENT_DIM - 1)] = 1.0
    else:
        # Isolated node — default to AST
        vec[0] = 1.0
    return vec


def _precompute_node_max_edge_type(edges: list[dict]) -> dict[str, int]:
    """Pre-compute the highest edge type each node participates in (O(E) once)."""
    node_max: dict[str, int] = {}
    for e in edges:
        src, dst, etype = e["src"], e["dst"], e["type"]
        if src not in node_max or etype > node_max[src]:
            node_max[src] = etype
        if dst not in node_max or etype > node_max[dst]:
            node_max[dst] = etype
    return node_max


def _build_structural_features(
    node_id: str,
    n_nodes: int,
    adj_in: dict[str, list[tuple[str, int]]],
    adj_out: dict[str, list[tuple[str, int]]],
    ast_depth_map: dict[str, int],
    max_depth: int,
    sink_ids: set[str],
    dfg_adj: dict[str, list[str]],
    cfg_adj: dict[str, list[str]],
    any_adj: dict[str, list[str]],
) -> np.ndarray:
    """
    12-d structural features:
    [0]  in_degree (normalized)
    [1]  out_degree (normalized)
    [2]  AST_depth (normalized)
    [3]  taint_dist_to_sink via DFG (BFS hops, normalized) — or 1.0 if unreachable
    [4]  taint_dist_to_sink via CFG
    [5]  taint_dist_to_sink via any edge
    [6-11]  reserved (zero-filled)
    """
    vec = np.zeros(STRUCTURAL_DIM, dtype=np.float32)
    max_degree = max(n_nodes, 1)

    # In-degree, out-degree
    vec[0] = len(adj_in.get(node_id, [])) / max_degree
    vec[1] = len(adj_out.get(node_id, [])) / max_degree

    # AST depth (normalized by max depth)
    vec[2] = ast_depth_map.get(node_id, 0) / max(max_depth, 1)

    # Taint distance to nearest sink (BFS on DFG, CFG, any)
    # Normalized: dist / num_nodes, capped at 1.0 for unreachable
    if sink_ids:
        vec[3] = _bfs_distance(node_id, sink_ids, dfg_adj) / n_nodes
        vec[4] = _bfs_distance(node_id, sink_ids, cfg_adj) / n_nodes
        vec[5] = _bfs_distance(node_id, sink_ids, any_adj) / n_nodes
    else:
        vec[3] = 1.0
        vec[4] = 1.0
        vec[5] = 1.0

    return vec


def _bfs_distance(
    start: str,
    targets: set[str],
    adj: dict[str, list[str]],
    max_dist: int = 200,
) -> float:
    """BFS from start to any target. Returns hop count or max_dist if unreachable."""
    if start in targets:
        return 0.0
    visited = {start}
    queue: deque[tuple[str, int]] = deque([(start, 0)])
    while queue:
        node, dist = queue.popleft()
        if dist >= max_dist:
            break
        for neighbor in adj.get(node, []):
            if neighbor in targets:
                return float(dist + 1)
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))
    return float(max_dist)


def _compute_ast_depths(nodes: list[dict], edges: list[dict]) -> dict[str, int]:
    """Compute AST depth for each node via BFS from AST roots."""
    # Build AST parent→child adjacency
    ast_children: dict[str, list[str]] = defaultdict(list)
    has_parent: set[str] = set()
    node_ids = {n["id"] for n in nodes}

    for e in edges:
        if e["type"] == 0:  # AST
            ast_children[e["src"]].append(e["dst"])
            has_parent.add(e["dst"])

    # AST roots: nodes without AST parent
    roots = [nid for nid in node_ids if nid not in has_parent]
    if not roots:
        # Fallback: METHOD nodes are typically roots
        roots = [n["id"] for n in nodes if n["_label"] == "METHOD"]
    if not roots:
        roots = [nodes[0]["id"]] if nodes else []

    depth_map: dict[str, int] = {}
    queue: deque[tuple[str, int]] = deque()
    for r in roots:
        queue.append((r, 0))
        depth_map[r] = 0

    while queue:
        nid, d = queue.popleft()
        for child in ast_children.get(nid, []):
            if child not in depth_map:
                depth_map[child] = d + 1
                queue.append((child, d + 1))

    # Assign depth 0 to any unvisited nodes
    for n in nodes:
        if n["id"] not in depth_map:
            depth_map[n["id"]] = 0

    return depth_map


def embed_cpg(
    cpg: dict,
    encoder: CodeBERTEncoder,
    batch_size: int = CODEBERT_BATCH_SIZE,
) -> np.ndarray:
    """
    Build 824-d node feature matrix for a single CPG.

    Args:
        cpg: CPG dict with "nodes" and "edges" keys
        encoder: CodeBERT encoder instance
        batch_size: batch size for CodeBERT inference

    Returns:
        np.ndarray of shape (N, 824) float32 where N = len(cpg["nodes"])
    """
    nodes = cpg["nodes"]
    edges = cpg["edges"]
    n_nodes = len(nodes)

    if n_nodes == 0:
        return np.zeros((0, TOTAL_DIM), dtype=np.float32)

    # ── Phase 1: CodeBERT embeddings (batched) ───────────────────
    # Collect code text for each node; empty code → empty string
    code_texts = []
    empty_mask = []  # True if node has no code → will get zero vector
    for node in nodes:
        code = (node.get("code") or "").strip()
        if code:
            code_texts.append(code)
            empty_mask.append(False)
        else:
            code_texts.append("")  # placeholder
            empty_mask.append(True)

    # Encode in batches
    codebert_embeddings = np.zeros((n_nodes, CODEBERT_DIM), dtype=np.float32)
    non_empty_indices = [i for i, is_empty in enumerate(empty_mask) if not is_empty]
    non_empty_texts = [code_texts[i] for i in non_empty_indices]

    if non_empty_texts:
        for batch_start in range(0, len(non_empty_texts), batch_size):
            batch_end = min(batch_start + batch_size, len(non_empty_texts))
            batch_texts = non_empty_texts[batch_start:batch_end]
            batch_embeddings = encoder.encode_batch(batch_texts)
            for j, global_idx in enumerate(non_empty_indices[batch_start:batch_end]):
                codebert_embeddings[global_idx] = batch_embeddings[j]

    # ── Phase 2: Pre-compute ALL graph structures (O(E) once) ─────
    adj_in: dict[str, list[tuple[str, int]]] = defaultdict(list)
    adj_out: dict[str, list[tuple[str, int]]] = defaultdict(list)
    dfg_adj: dict[str, list[str]] = defaultdict(list)
    cfg_adj: dict[str, list[str]] = defaultdict(list)
    any_adj: dict[str, list[str]] = defaultdict(list)

    for e in edges:
        src, dst, etype = e["src"], e["dst"], e["type"]
        adj_in[dst].append((src, etype))
        adj_out[src].append((dst, etype))
        any_adj[src].append(dst)
        if etype == 2:  # DFG
            dfg_adj[src].append(dst)
        elif etype == 1:  # CFG
            cfg_adj[src].append(dst)

    # Pre-compute per-node max edge type for CPG component one-hot
    node_max_edge_type = _precompute_node_max_edge_type(edges)

    # AST depths
    ast_depth_map = _compute_ast_depths(nodes, edges)
    max_depth = max(ast_depth_map.values()) if ast_depth_map else 1

    # Sink node IDs for taint distance
    sink_ids = {n["id"] for n in nodes if n.get("taint_role") == "SINK"}

    # ── Phase 3: Build per-node feature vectors ──────────────────
    features = np.zeros((n_nodes, TOTAL_DIM), dtype=np.float32)

    for i, node in enumerate(nodes):
        nid = node["id"]
        offset = 0

        # [0:768] CodeBERT [CLS]
        features[i, offset:offset + CODEBERT_DIM] = codebert_embeddings[i]
        offset += CODEBERT_DIM

        # [768:800] Node type one-hot
        features[i, offset:offset + NODE_TYPE_DIM] = _build_node_type_onehot(
            node.get("_label", "UNKNOWN")
        )
        offset += NODE_TYPE_DIM

        # [800:808] Taint role one-hot
        features[i, offset:offset + TAINT_ROLE_DIM] = _build_taint_role_onehot(
            node.get("taint_role", "NONE")
        )
        offset += TAINT_ROLE_DIM

        # [808:812] CPG component one-hot
        features[i, offset:offset + CPG_COMPONENT_DIM] = _build_cpg_component_onehot(
            node_max_edge_type.get(nid, -1)
        )
        offset += CPG_COMPONENT_DIM

        # [812:824] Structural features
        features[i, offset:offset + STRUCTURAL_DIM] = _build_structural_features(
            nid, n_nodes, adj_in, adj_out, ast_depth_map, max_depth,
            sink_ids, dfg_adj, cfg_adj, any_adj,
        )
        offset += STRUCTURAL_DIM

        assert offset == TOTAL_DIM

    return features


# ── Output path helpers ──────────────────────────────────────────
def _output_path(output_dir: Path, sample_id: str) -> Path:
    """Sharded output: embedded/{first_2_chars}/{sample_id}.npz"""
    shard = sample_id[:2]
    shard_dir = output_dir / shard
    shard_dir.mkdir(parents=True, exist_ok=True)
    return shard_dir / f"{sample_id}.npz"


# ── Main pipeline ────────────────────────────────────────────────
def run_stage5(
    input_dir: str,
    output_dir: str,
    dry_run: bool = False,
    max_samples: int | None = None,
    device: str | None = None,
    batch_size: int = CODEBERT_BATCH_SIZE,
) -> dict:
    """
    Run Stage 5 node embedding on all CPG JSON files.

    Args:
        input_dir: path to cpg/ directory (sharded)
        output_dir: path to embedded/ directory (sharded output)
        dry_run: if True, just report what would be done
        max_samples: limit number of samples to process
        device: torch device override (default: auto-detect)
        batch_size: CodeBERT batch size

    Returns:
        stats dict
    """
    logger.info("Stage 5: Node Embedding")
    logger.info(f"  Input:  {input_dir}")
    logger.info(f"  Output: {output_dir}")

    inp = Path(input_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Discover all CPG JSON files
    cpg_files: list[Path] = sorted(inp.rglob("*.json"))
    # Exclude non-CPG files (stats, failures, checkpoint)
    cpg_files = [
        f for f in cpg_files
        if f.name not in ("cpg_stats.json", "cpg_failures.jsonl", "checkpoint.json")
    ]

    n_total = len(cpg_files)
    logger.info(f"Found {n_total} CPG files")

    if max_samples is not None:
        cpg_files = cpg_files[:max_samples]

    # Skip already-embedded files (checkpoint/resume)
    to_process: list[Path] = []
    skipped = 0
    for cpg_path in cpg_files:
        sample_id = cpg_path.stem
        out_path = _output_path(out, sample_id)
        if out_path.exists():
            skipped += 1
        else:
            to_process.append(cpg_path)

    n_process = len(to_process)
    logger.info(f"To process: {n_process} (skipped {skipped} already embedded)")

    if dry_run:
        logger.info("[DRY RUN] Would process samples, showing first 5:")
        for cpg_path in to_process[:5]:
            try:
                cpg = json.loads(cpg_path.read_text(encoding="utf-8"))
                n_nodes = len(cpg.get("nodes", []))
                n_edges = len(cpg.get("edges", []))
                logger.info(
                    f"  {cpg_path.stem[:16]}... nodes={n_nodes} edges={n_edges} "
                    f"cwe={cpg.get('cwe', '')} label={cpg.get('label', '')}"
                )
            except Exception as exc:
                logger.info(f"  {cpg_path.stem[:16]}... ERROR: {exc}")
        return {
            "mode": "dry_run",
            "total_cpgs": n_total,
            "would_process": n_process,
            "already_embedded": skipped,
        }

    # Initialize CodeBERT encoder
    encoder = CodeBERTEncoder(model_name=CODEBERT_MODEL, device=device)

    # Process each CPG
    start_time = time.time()
    succeeded = 0
    failed = 0
    nan_rejected = 0
    total_nodes = 0

    for i, cpg_path in enumerate(to_process):
        sample_id = cpg_path.stem

        try:
            cpg = json.loads(cpg_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            logger.debug(f"Failed to load {cpg_path}: {exc}")
            failed += 1
            continue

        try:
            features = embed_cpg(cpg, encoder, batch_size=batch_size)
        except Exception as exc:
            logger.debug(f"Embedding failed for {sample_id}: {exc}")
            failed += 1
            continue

        # Validate: no NaN
        if np.isnan(features).any():
            logger.warning(f"NaN detected in {sample_id} — rejected")
            nan_rejected += 1
            failed += 1
            continue

        # Validate shape
        n_nodes = len(cpg["nodes"])
        assert features.shape == (n_nodes, TOTAL_DIM), (
            f"Shape mismatch: {features.shape} vs ({n_nodes}, {TOTAL_DIM})"
        )

        # Save
        out_path = _output_path(out, sample_id)
        np.savez_compressed(out_path, node_features=features)
        succeeded += 1
        total_nodes += n_nodes

        # Progress
        if (i + 1) % 50 == 0 or (i + 1) == n_process:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            remaining = n_process - (i + 1)
            eta_min = remaining / rate / 60 if rate > 0 else 0
            pct = (skipped + i + 1) / n_total * 100 if n_total > 0 else 0
            print(
                f"[{skipped + i + 1}/{n_total} {pct:.1f}%] "
                f"ok={succeeded} fail={failed} "
                f"rate={rate:.1f}/s ETA={eta_min:.1f}min "
                f"avg_nodes={total_nodes / max(succeeded, 1):.0f}",
                flush=True,
            )

    elapsed = time.time() - start_time

    stats = {
        "total_cpgs": n_total,
        "already_embedded": skipped,
        "processed": n_process,
        "succeeded": succeeded,
        "failed": failed,
        "nan_rejected": nan_rejected,
        "total_nodes_embedded": total_nodes,
        "avg_nodes_per_sample": total_nodes / max(succeeded, 1),
        "elapsed_seconds": round(elapsed, 1),
        "samples_per_second": round(succeeded / max(elapsed, 0.001), 2),
        "feature_dim": TOTAL_DIM,
    }

    # Save stats
    stats_path = out / "embed_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    logger.info(f"Stage 5 complete: {json.dumps(stats, indent=2)}")

    return stats


# ── Verification helpers ─────────────────────────────────────────
def verify_embeddings(output_dir: str, max_check: int = 3) -> None:
    """Print shape and NaN check for first N embedding files."""
    out = Path(output_dir)
    npz_files = sorted(out.rglob("*.npz"))

    if not npz_files:
        print("No .npz files found!")
        return

    print(f"\nVerification: checking {min(max_check, len(npz_files))} of {len(npz_files)} files\n")

    nan_count = 0
    for npz_path in npz_files[:max_check]:
        data = np.load(npz_path)
        features = data["node_features"]
        has_nan = np.isnan(features).any()
        if has_nan:
            nan_count += 1
        print(
            f"  {npz_path.name}: shape={features.shape} "
            f"dtype={features.dtype} "
            f"NaN={has_nan} "
            f"min={features.min():.4f} max={features.max():.4f}"
        )
        assert features.shape[1] == TOTAL_DIM, f"Expected 824 columns, got {features.shape[1]}"
        assert features.dtype == np.float32, f"Expected float32, got {features.dtype}"

    if nan_count == 0:
        print(f"\nAll {min(max_check, len(npz_files))} files: shape (N, 824) float32, no NaN")
    else:
        print(f"\nWARNING: {nan_count} files contain NaN values!")


# ── CLI ──────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Stage 5: Node Embedding — CodeBERT + structural features",
    )
    parser.add_argument(
        "--input",
        default="training/data/processed/cpg/",
        help="Input CPG directory (default: training/data/processed/cpg/)",
    )
    parser.add_argument(
        "--output",
        default="training/data/processed/embedded/",
        help="Output embedding directory (default: training/data/processed/embedded/)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without running CodeBERT",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit number of samples to process",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device (default: auto-detect cuda/cpu)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=CODEBERT_BATCH_SIZE,
        help=f"CodeBERT batch size (default: {CODEBERT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="After embedding, verify first 3 output files",
    )

    args = parser.parse_args()

    stats = run_stage5(
        input_dir=args.input,
        output_dir=args.output,
        dry_run=args.dry_run,
        max_samples=args.max_samples,
        device=args.device,
        batch_size=args.batch_size,
    )

    # Always verify output after a real run (or if --verify flag)
    if not args.dry_run or args.verify:
        verify_embeddings(args.output, max_check=3)


if __name__ == "__main__":
    main()
