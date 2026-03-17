#!/usr/bin/env python3
"""
training/scripts/preprocessing/stage6_graphs.py

Stage 6: Graph Tensor Assembly for StreamGuard pipeline.

Input:  training/data/processed/cpg/{shard}/{sample_id}.json   (CPG structure)
        training/data/processed/embedded/{shard}/{sample_id}.npz (node features)
Output: training/data/graphs/all_graphs.h5  (HDF5 with all PyG-compatible graph data)

Per-sample pipeline:
  1. Load CPG JSON (nodes, edges, metadata)
  2. Load corresponding .npz (node features, shape (N, 824))
  3. Build node_id → contiguous index mapping (0..N-1)
  4. Construct edge_index (2, E) int64 COO format via remapped indices
  5. Extract edge_attr (E,) int64 — edge types {AST=0, CFG=1, DFG=2, TPG=3}
  6. Validate: num_nodes >= 3, edge_index OOB check, no NaN in features
  7. Save to HDF5 with atomic write (.tmp + os.replace)

Risk mitigations (from StreamGuard_Stories5to8_RiskAnalysis.docx):
  - R1: Edge index OOB check — CUDA crash prevention
  - R2: Edge type range check — only {0,1,2,3} allowed; >=4 causes GGNN crash
  - R3: Atomic HDF5 write — prevents corruption on interrupt
  - R4: NaN validation — prevents silent training corruption
  - R5: Trivial graph rejection — num_nodes >= 3
  - R6: feature_dim=824 stored in HDF5 attrs for model loading

Source: docs/New Docs/PREPROCESSING.md §Stage 6
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from loguru import logger

# ── Constants ────────────────────────────────────────────────────
FEATURE_DIM = 824
VALID_EDGE_TYPES = {0, 1, 2, 3}  # AST, CFG, DFG, TPG
MIN_NODES = 3


# ── Load and validate one sample ─────────────────────────────────
def load_sample(
    cpg_path: Path,
    npz_path: Path,
) -> dict | None:
    """
    Load CPG JSON + embedding .npz for one sample.
    Returns a validated dict with all fields needed for HDF5, or None on failure.
    """
    sample_id = cpg_path.stem

    # Load CPG JSON
    try:
        cpg = json.loads(cpg_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        logger.debug(f"Bad CPG JSON for {sample_id}: {exc}")
        return None

    nodes = cpg.get("nodes", [])
    edges = cpg.get("edges", [])

    # Load embeddings
    try:
        data = np.load(npz_path)
        node_features = data["node_features"]
    except Exception as exc:
        logger.debug(f"Bad .npz for {sample_id}: {exc}")
        return None

    n_nodes = len(nodes)

    # ── Validation Gate 1: Trivial graph ─────────────────────────
    if n_nodes < MIN_NODES:
        logger.debug(f"Trivial graph {sample_id}: {n_nodes} nodes < {MIN_NODES}")
        return None

    # ── Validation Gate 2: Feature shape match ───────────────────
    if node_features.shape != (n_nodes, FEATURE_DIM):
        logger.debug(
            f"Feature shape mismatch {sample_id}: "
            f"{node_features.shape} vs ({n_nodes}, {FEATURE_DIM})"
        )
        return None

    # ── Validation Gate 3: No NaN in features ────────────────────
    if np.isnan(node_features).any():
        logger.warning(f"NaN in features for {sample_id} — rejected")
        return None

    # ── Build node_id → contiguous index mapping ─────────────────
    node_id_to_idx = {n["id"]: i for i, n in enumerate(nodes)}

    # ── Build edge_index and edge_attr ───────────────────────────
    src_list: list[int] = []
    dst_list: list[int] = []
    attr_list: list[int] = []

    for e in edges:
        src_id = e["src"]
        dst_id = e["dst"]
        etype = e["type"]

        # Skip edges referencing unknown nodes (defensive)
        if src_id not in node_id_to_idx or dst_id not in node_id_to_idx:
            continue

        # Validation Gate 4: Edge type range
        if etype not in VALID_EDGE_TYPES:
            logger.debug(f"Invalid edge type {etype} in {sample_id} — skipped")
            continue

        src_list.append(node_id_to_idx[src_id])
        dst_list.append(node_id_to_idx[dst_id])
        attr_list.append(etype)

    if not src_list:
        logger.debug(f"No valid edges for {sample_id}")
        return None

    edge_index = np.array([src_list, dst_list], dtype=np.int64)
    edge_attr = np.array(attr_list, dtype=np.int64)

    # ── Validation Gate 5: Edge index OOB ────────────────────────
    if edge_index.max() >= n_nodes:
        logger.warning(
            f"Edge index OOB in {sample_id}: max={edge_index.max()} >= {n_nodes}"
        )
        return None

    return {
        "sample_id": sample_id,
        "label": int(cpg.get("label", 0)),
        "cwe": cpg.get("cwe", ""),
        "pair_id": cpg.get("pair_id", ""),
        "source": cpg.get("source", ""),
        "x": node_features.astype(np.float32),
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "num_nodes": n_nodes,
        "num_edges": len(src_list),
    }


# ── HDF5 writer ──────────────────────────────────────────────────
def write_graphs_h5(
    samples: list[dict],
    output_path: Path,
) -> dict:
    """
    Write all validated graph samples to HDF5.
    Uses atomic write: writes to .tmp then os.replace().

    HDF5 structure:
      /metadata/
        feature_dim (attr) = 824
        num_graphs (attr) = N
        sample_ids = [str, ...]
        labels = [int, ...]
        cwes = [str, ...]
        pair_ids = [str, ...]
      /graphs/{idx}/
        x          = (N_nodes, 824) float32
        edge_index = (2, N_edges) int64
        edge_attr  = (N_edges,) int64
        y          = (1,) int64
    """
    tmp_path = output_path.with_suffix(".h5.tmp")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_graphs = len(samples)

    with h5py.File(tmp_path, "w") as f:
        # Global metadata
        meta = f.create_group("metadata")
        meta.attrs["feature_dim"] = FEATURE_DIM
        meta.attrs["num_graphs"] = n_graphs
        meta.attrs["edge_types"] = 4  # AST, CFG, DFG, TPG

        # String arrays for metadata
        sample_ids = [s["sample_id"] for s in samples]
        labels = [s["label"] for s in samples]
        cwes = [s["cwe"] for s in samples]
        pair_ids = [s["pair_id"] for s in samples]

        dt_str = h5py.special_dtype(vlen=str)
        meta.create_dataset("sample_ids", data=sample_ids, dtype=dt_str)
        meta.create_dataset("labels", data=np.array(labels, dtype=np.int64))
        meta.create_dataset("cwes", data=cwes, dtype=dt_str)
        meta.create_dataset("pair_ids", data=pair_ids, dtype=dt_str)

        # Per-graph data
        graphs_grp = f.create_group("graphs")
        for i, s in enumerate(samples):
            g = graphs_grp.create_group(str(i))
            g.create_dataset("x", data=s["x"], compression="gzip", compression_opts=4)
            g.create_dataset("edge_index", data=s["edge_index"])
            g.create_dataset("edge_attr", data=s["edge_attr"])
            g.create_dataset("y", data=np.array([s["label"]], dtype=np.int64))

    # Atomic rename
    os.replace(tmp_path, output_path)

    return {
        "num_graphs": n_graphs,
        "output_path": str(output_path),
    }


# ── Main pipeline ────────────────────────────────────────────────
def run_stage6(
    cpg_dir: str,
    embed_dir: str,
    output_path: str,
    dry_run: bool = False,
    max_samples: int | None = None,
) -> dict:
    """
    Run Stage 6: assemble PyG-compatible graph tensors into HDF5.
    """
    logger.info("Stage 6: Graph Tensor Assembly")
    logger.info(f"  CPG input:   {cpg_dir}")
    logger.info(f"  Embed input: {embed_dir}")
    logger.info(f"  Output:      {output_path}")

    cpg_root = Path(cpg_dir)
    embed_root = Path(embed_dir)
    out_path = Path(output_path)

    # Discover all CPG files
    cpg_files: list[Path] = sorted(cpg_root.rglob("*.json"))
    cpg_files = [
        f for f in cpg_files
        if f.name not in ("cpg_stats.json", "cpg_failures.jsonl", "checkpoint.json")
    ]

    n_total_cpg = len(cpg_files)
    logger.info(f"Found {n_total_cpg} CPG files")

    if max_samples is not None:
        cpg_files = cpg_files[:max_samples]

    # Match CPG files to embedding files
    matched: list[tuple[Path, Path]] = []
    no_embed = 0
    for cpg_path in cpg_files:
        sample_id = cpg_path.stem
        shard = sample_id[:2]
        npz_path = embed_root / shard / f"{sample_id}.npz"
        if npz_path.exists():
            matched.append((cpg_path, npz_path))
        else:
            no_embed += 1

    n_matched = len(matched)
    logger.info(
        f"Matched {n_matched} CPG+embedding pairs "
        f"({no_embed} CPGs have no embedding yet)"
    )

    if dry_run:
        logger.info("[DRY RUN] Would process, showing first 5:")
        for cpg_path, npz_path in matched[:5]:
            try:
                cpg = json.loads(cpg_path.read_text(encoding="utf-8"))
                data = np.load(npz_path)
                feat = data["node_features"]
                logger.info(
                    f"  {cpg_path.stem[:16]}... "
                    f"nodes={len(cpg.get('nodes', []))} "
                    f"edges={len(cpg.get('edges', []))} "
                    f"embed={feat.shape} "
                    f"cwe={cpg.get('cwe', '')} label={cpg.get('label', '')}"
                )
            except Exception as exc:
                logger.info(f"  {cpg_path.stem[:16]}... ERROR: {exc}")
        return {
            "mode": "dry_run",
            "total_cpgs": n_total_cpg,
            "matched": n_matched,
            "no_embedding": no_embed,
        }

    # Process all matched pairs
    start_time = time.time()
    valid_samples: list[dict] = []
    failed = 0
    reject_reasons: dict[str, int] = {}

    for i, (cpg_path, npz_path) in enumerate(matched):
        result = load_sample(cpg_path, npz_path)
        if result is not None:
            valid_samples.append(result)
        else:
            failed += 1

        if (i + 1) % 500 == 0 or (i + 1) == n_matched:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            print(
                f"[{i + 1}/{n_matched}] valid={len(valid_samples)} "
                f"failed={failed} rate={rate:.0f}/s",
                flush=True,
            )

    if not valid_samples:
        logger.error("No valid samples — nothing to write!")
        return {"num_graphs": 0, "failed": failed}

    # Write HDF5 (atomic)
    logger.info(f"Writing {len(valid_samples)} graphs to {output_path}")
    write_result = write_graphs_h5(valid_samples, out_path)

    elapsed = time.time() - start_time

    # Compute stats
    total_nodes = sum(s["num_nodes"] for s in valid_samples)
    total_edges = sum(s["num_edges"] for s in valid_samples)

    stats = {
        "total_cpgs_found": n_total_cpg,
        "matched_pairs": n_matched,
        "no_embedding": no_embed,
        "valid_graphs": len(valid_samples),
        "failed": failed,
        "total_nodes": total_nodes,
        "total_edges": total_edges,
        "avg_nodes": total_nodes / len(valid_samples),
        "avg_edges": total_edges / len(valid_samples),
        "feature_dim": FEATURE_DIM,
        "elapsed_seconds": round(elapsed, 1),
        "output_path": str(out_path),
    }

    # Save stats alongside the HDF5
    stats_path = out_path.with_name("graph_stats.json")
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    logger.info(f"Stage 6 complete: {json.dumps(stats, indent=2)}")

    return stats


# ── Verification helper ──────────────────────────────────────────
def verify_h5(h5_path: str, max_check: int = 3) -> None:
    """Verify HDF5 structure and print sample graph info."""
    path = Path(h5_path)
    if not path.exists():
        print(f"File not found: {h5_path}")
        return

    with h5py.File(path, "r") as f:
        meta = f["metadata"]
        n_graphs = meta.attrs["num_graphs"]
        feat_dim = meta.attrs["feature_dim"]

        print(f"\n=== HDF5 Verification: {h5_path} ===")
        print(f"  num_graphs:  {n_graphs}")
        print(f"  feature_dim: {feat_dim}")
        assert feat_dim == FEATURE_DIM, f"Expected {FEATURE_DIM}, got {feat_dim}"

        labels = meta["labels"][:]
        n_vuln = int((labels == 1).sum())
        n_safe = int((labels == 0).sum())
        print(f"  labels: {n_vuln} vuln, {n_safe} safe")

        graphs = f["graphs"]
        checked = 0
        for idx in list(graphs.keys())[:max_check]:
            g = graphs[idx]
            x = g["x"][:]
            ei = g["edge_index"][:]
            ea = g["edge_attr"][:]
            y = g["y"][:]

            n_nodes = x.shape[0]
            n_edges = ea.shape[0]

            # Validate
            assert x.shape[1] == FEATURE_DIM
            assert ei.shape == (2, n_edges)
            assert not np.isnan(x).any(), f"NaN in graph {idx}"
            assert n_nodes >= MIN_NODES, f"Trivial graph {idx}"
            if n_edges > 0:
                assert ei.max() < n_nodes, f"Edge OOB in graph {idx}"
                assert ea.max() <= 3, f"Invalid edge type in graph {idx}"

            sample_id = meta["sample_ids"][int(idx)].decode() if isinstance(
                meta["sample_ids"][int(idx)], bytes
            ) else str(meta["sample_ids"][int(idx)])

            print(
                f"\n  Graph {idx} ({sample_id[:16]}...):"
                f"\n    x:          {x.shape} {x.dtype}"
                f"\n    edge_index: {ei.shape} {ei.dtype}"
                f"\n    edge_attr:  {ea.shape} {ea.dtype} types={set(ea.tolist())}"
                f"\n    y:          {y} (label)"
                f"\n    NaN:        {np.isnan(x).any()}"
            )
            checked += 1

        print(f"\n  Verified {checked} graphs: all valid")


# ── CLI ──────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Stage 6: Graph Tensor Assembly — CPG + embeddings → HDF5",
    )
    parser.add_argument(
        "--cpg-dir",
        default="training/data/processed/cpg/",
        help="CPG JSON directory (default: training/data/processed/cpg/)",
    )
    parser.add_argument(
        "--embed-dir",
        default="training/data/processed/embedded/",
        help="Embedding .npz directory (default: training/data/processed/embedded/)",
    )
    parser.add_argument(
        "--output",
        default="training/data/graphs/all_graphs.h5",
        help="Output HDF5 path (default: training/data/graphs/all_graphs.h5)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without building HDF5",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit number of samples",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify output HDF5 after writing",
    )

    args = parser.parse_args()

    stats = run_stage6(
        cpg_dir=args.cpg_dir,
        embed_dir=args.embed_dir,
        output_path=args.output,
        dry_run=args.dry_run,
        max_samples=args.max_samples,
    )

    if not args.dry_run or args.verify:
        verify_h5(args.output, max_check=3)


if __name__ == "__main__":
    main()
