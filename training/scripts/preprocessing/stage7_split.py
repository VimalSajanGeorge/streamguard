#!/usr/bin/env python3
"""
training/scripts/preprocessing/stage7_split.py

Stage 7: CFA-Aware Train/Val/Test Split for StreamGuard pipeline.

Input:  training/data/graphs/all_graphs.h5  (HDF5 from Stage 6)
        --supplement JSONL  (optional, for commit_sha/cve_id grouping + leakage checks)
Output: training/data/final/train.h5, val.h5, test.h5, split_stats.json

Algorithm (Phase 3 — all challenges mitigated):
  1. Load metadata from all_graphs.h5 (~500 KB, not the 3.7 GB graph data)
  2. Optionally load supplement JSONL for commit_sha/cve_id per sample_id
     - These fields exist in the canonical schema but are not carried through
       Stage 4 (CPG) -> Stage 5 (embed) -> Stage 6 (HDF5)
     - If supplement is absent, falls back to pair_id-only grouping
  3. Build split groups with priority hierarchy:
       pair_id > commit_sha > cve_id > singleton
     - Challenge 6: empty string, '0', 'None' treated as absent
  4. Separate paired groups (>= 2 members) from singletons
  5. Split paired groups first (80/10/10 by sample count)
     - Challenge 5: Ensures test/val have enough CFA pairs (min 500)
  6. Distribute singletons proportionally to fill remaining capacity
  7. Assert pair integrity: no pair_id in more than one split (Challenge 1)
  8. Assert zero commit_sha overlap between train and test (data leakage)
  9. Check balance: vuln ratio per split; warn if outside M1 range (Challenge 4)
     - M1 relaxed range: (0.35, 0.65) — SARD inherent ratio is ~0.408
 10. Streaming HDF5 copy: one graph at a time (memory-efficient)
 11. Atomic writes: .tmp + os.replace per output file

Source: docs/New Docs/StreamGuard_Phase3_Preprocessing_Plan.docx Stage 7
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np
from loguru import logger

FEATURE_DIM = 824

# M1 relaxed balance range (Challenge 4: SARD inherent ratio ~0.408)
M1_BALANCE_RANGE = (0.35, 0.65)

# Minimum CFA pairs in test set (Challenge 5)
MIN_TEST_PAIRS = 500

# Sentinel values treated as "no pair_id" (Challenge 6)
EMPTY_PAIR_SENTINELS = {"", "None", "null", "0", "none"}

# Default supplement JSONL path (relative to repo root)
DEFAULT_SUPPLEMENT = "training/data/processed/deduped/samples.jsonl"


# -- Metadata loader ----------------------------------------------------------
def load_metadata(h5_path: Path) -> dict:
    """
    Load only metadata arrays from HDF5 (lightweight, ~500 KB).
    Returns dict with sample_ids, labels, cwes, pair_ids as lists.
    """
    with h5py.File(h5_path, "r") as f:
        meta = f["metadata"]
        num_graphs = int(meta.attrs["num_graphs"])

        raw_ids = meta["sample_ids"][:]
        sample_ids = [
            s.decode("utf-8") if isinstance(s, bytes) else str(s)
            for s in raw_ids
        ]
        labels = meta["labels"][:].tolist()
        raw_cwes = meta["cwes"][:]
        cwes = [
            s.decode("utf-8") if isinstance(s, bytes) else str(s)
            for s in raw_cwes
        ]
        raw_pids = meta["pair_ids"][:]
        pair_ids = [
            s.decode("utf-8") if isinstance(s, bytes) else str(s)
            for s in raw_pids
        ]

    return {
        "num_graphs": num_graphs,
        "sample_ids": sample_ids,
        "labels": labels,
        "cwes": cwes,
        "pair_ids": pair_ids,
    }


# -- Supplement loader (commit_sha / cve_id from JSONL) -----------------------
def load_supplement(
    jsonl_path: Path,
    sample_ids: list[str],
) -> dict[str, dict[str, str]]:
    """
    Read deduped JSONL and extract commit_sha/cve_id for each sample_id
    present in the HDF5 metadata.

    Returns: {sample_id: {"commit_sha": str, "cve_id": str}}

    Fields are in the canonical schema (schema.py) but lost at Stage 4
    because process_sample() only carries sample_id/label/cwe/pair_id/source
    into the CPG JSON.
    """
    wanted = set(sample_ids)
    lookup: dict[str, dict[str, str]] = {}

    with open(jsonl_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = row.get("id", "")
            if sid in wanted:
                lookup[sid] = {
                    "commit_sha": row.get("commit_sha", ""),
                    "cve_id": row.get("cve_id", ""),
                }

    matched = len(lookup)
    logger.info(
        f"Supplement: matched {matched}/{len(wanted)} sample_ids "
        f"({matched / max(len(wanted), 1) * 100:.1f}%)"
    )
    return lookup


# -- Group key with priority hierarchy ----------------------------------------
def _is_valid_pair_id(pid: str) -> bool:
    """
    Check if pair_id is a real group key (Challenge 6).
    Empty strings, 'None', 'null', '0' are treated as absent.
    """
    return bool(pid) and pid not in EMPTY_PAIR_SENTINELS


def _get_group_key(
    idx: int,
    pair_ids: list[str],
    sample_ids: list[str],
    supplement: dict[str, dict[str, str]] | None,
) -> str:
    """
    Determine the grouping key for sample at index idx.
    Priority: pair_id > commit_sha > cve_id > sample_id (singleton).
    Prefixes prevent collisions between key types.
    """
    pid = pair_ids[idx]
    if _is_valid_pair_id(pid):
        return f"pair:{pid}"

    if supplement:
        sid = sample_ids[idx]
        sup = supplement.get(sid, {})
        sha = sup.get("commit_sha", "")
        if sha:
            return f"sha:{sha}"
        cve = sup.get("cve_id", "")
        if cve:
            return f"cve:{cve}"

    return f"single:{sample_ids[idx]}"


# -- Union-Find for SHA merge --------------------------------------------------
class _UnionFind:
    """Lightweight union-find for merging groups that share a commit_sha."""

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def _merge_groups_by_sha(
    groups: list[list[int]],
    sample_ids: list[str],
    supplement: dict[str, dict[str, str]],
) -> list[list[int]]:
    """
    Post-grouping merge: if two groups share a commit_sha (via any member),
    merge them into one super-group. Prevents R-29 SHA leakage where
    pair_id-grouped samples from the same commit end up in different splits.

    Example: pair-A (samples 0,1 from sha-X) and pair-B (sample 2 from sha-X)
    get merged into one group [0,1,2].
    """
    if not groups:
        return groups

    # Map each group index to the SHAs it contains
    uf = _UnionFind(len(groups))
    sha_to_group_idx: dict[str, int] = {}

    for gi, group in enumerate(groups):
        for idx in group:
            sid = sample_ids[idx]
            sha = supplement.get(sid, {}).get("commit_sha", "")
            if sha:
                if sha in sha_to_group_idx:
                    uf.union(gi, sha_to_group_idx[sha])
                else:
                    sha_to_group_idx[sha] = gi

    # Collect merged groups
    root_to_members: dict[int, list[int]] = defaultdict(list)
    for gi, group in enumerate(groups):
        root = uf.find(gi)
        root_to_members[root].extend(group)

    merged = [sorted(members) for members in root_to_members.values()]
    merged.sort(key=lambda g: g[0])

    n_merges = len(groups) - len(merged)
    if n_merges > 0:
        logger.info(f"SHA merge: {len(groups)} groups -> {len(merged)} ({n_merges} merged)")

    return merged


# -- Group builder ------------------------------------------------------------
def build_groups(
    pair_ids: list[str],
    sample_ids: list[str] | None = None,
    supplement: dict[str, dict[str, str]] | None = None,
) -> tuple[list[list[int]], list[list[int]]]:
    """
    Build groups using priority hierarchy: pair_id > commit_sha > cve_id > singleton.

    After initial grouping, performs a SHA-merge pass: any groups whose members
    share a commit_sha are merged into one super-group (R-29 mitigation).

    Returns:
        (paired_groups, singleton_groups) -- both sorted for determinism.
        paired_groups: groups with >= 2 members.
        singleton_groups: single-member groups.

    If sample_ids and supplement are None, falls back to pair_id-only grouping.
    """
    key_to_indices: dict[str, list[int]] = defaultdict(list)

    for i in range(len(pair_ids)):
        if sample_ids is not None and supplement is not None:
            key = _get_group_key(i, pair_ids, sample_ids, supplement)
        else:
            # Fallback: pair_id-only grouping (Phase 1 behavior)
            pid = pair_ids[i]
            if _is_valid_pair_id(pid):
                key = f"pair:{pid}"
            else:
                key = f"single:{i}"
        key_to_indices[key].append(i)

    # Initial split into paired/singleton
    initial_groups: list[list[int]] = []
    singletons: list[list[int]] = []

    for k in sorted(key_to_indices.keys()):
        group = key_to_indices[k]
        if len(group) >= 2:
            initial_groups.append(group)
        else:
            singletons.append(group)

    # SHA-merge pass: merge groups that share commit_sha (R-29)
    if supplement and sample_ids is not None:
        # Merge both paired groups AND singletons, then re-separate
        all_groups = initial_groups + singletons
        all_groups = _merge_groups_by_sha(all_groups, sample_ids, supplement)

        # Re-separate after merge
        paired_groups: list[list[int]] = []
        singletons = []
        for g in all_groups:
            if len(g) >= 2:
                paired_groups.append(g)
            else:
                singletons.append(g)
    else:
        paired_groups = initial_groups

    singletons.sort(key=lambda g: g[0])
    return paired_groups, singletons


# -- Splitter -----------------------------------------------------------------
def split_groups(
    groups: list[list[int]],
    labels: list[int],
    split_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
    max_samples: int | None = None,
) -> dict:
    """
    Split groups into train/val/test by cumulative sample count.

    Accepts a flat list of groups (paired + singletons mixed).
    For the two-phase split (paired first, singletons second),
    use split_paired_and_singletons() instead.

    Returns dict with train_indices, val_indices, test_indices.
    """
    rng = random.Random(seed)

    if max_samples is not None:
        limited_groups = []
        total = 0
        for g in groups:
            if total + len(g) > max_samples:
                break
            limited_groups.append(g)
            total += len(g)
        groups = limited_groups

    shuffled = list(groups)
    rng.shuffle(shuffled)

    total_samples = sum(len(g) for g in shuffled)
    train_target = int(total_samples * split_ratios[0])
    val_target = int(total_samples * split_ratios[1])

    train_groups: list[list[int]] = []
    val_groups: list[list[int]] = []
    test_groups: list[list[int]] = []

    cumulative = 0
    for g in shuffled:
        if cumulative < train_target:
            train_groups.append(g)
        elif cumulative < train_target + val_target:
            val_groups.append(g)
        else:
            test_groups.append(g)
        cumulative += len(g)

    train_idx, val_idx, test_idx = _rebalance(
        train_groups, val_groups, test_groups,
        labels, max_swaps=100,
    )

    _assert_pair_integrity(train_idx, val_idx, test_idx, groups)

    return {
        "train_indices": sorted(train_idx),
        "val_indices": sorted(val_idx),
        "test_indices": sorted(test_idx),
    }


def split_paired_and_singletons(
    paired_groups: list[list[int]],
    singleton_groups: list[list[int]],
    labels: list[int],
    split_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
    max_samples: int | None = None,
    min_test_pairs: int = MIN_TEST_PAIRS,
) -> dict:
    """
    Two-phase split (Challenge 5):
      Phase 1: Split paired groups (CFA pairs stay together)
      Phase 2: Distribute singletons proportionally

    Ensures test and val each have >= min_test_pairs CFA pairs.
    """
    rng = random.Random(seed)

    # Phase 1: Split paired groups
    paired = list(paired_groups)
    rng.shuffle(paired)

    # Apply max_samples cap across both phases
    if max_samples is not None:
        limited_paired = []
        cap_total = 0
        for g in paired:
            if cap_total + len(g) > max_samples:
                break
            limited_paired.append(g)
            cap_total += len(g)
        paired = limited_paired

    total_paired = sum(len(g) for g in paired)
    total_singletons = sum(len(g) for g in singleton_groups)
    total_all = total_paired + total_singletons

    if max_samples is not None:
        total_all = min(total_all, max_samples)

    # For paired groups, target the same ratios
    train_paired_target = int(total_paired * split_ratios[0])
    val_paired_target = int(total_paired * split_ratios[1])

    train_paired: list[list[int]] = []
    val_paired: list[list[int]] = []
    test_paired: list[list[int]] = []

    cumulative = 0
    for g in paired:
        if cumulative < train_paired_target:
            train_paired.append(g)
        elif cumulative < train_paired_target + val_paired_target:
            val_paired.append(g)
        else:
            test_paired.append(g)
        cumulative += len(g)

    # Check minimum pairs in test/val (Challenge 5)
    n_test_pairs = len(test_paired)
    n_val_pairs = len(val_paired)
    if n_test_pairs < min_test_pairs and len(paired) > min_test_pairs * 3:
        logger.warning(
            f"Test has only {n_test_pairs} CFA pairs (need {min_test_pairs}). "
            f"Redistributing from train."
        )
        # Move groups from end of train to test
        while len(test_paired) < min_test_pairs and train_paired:
            test_paired.append(train_paired.pop())
    if n_val_pairs < min_test_pairs and len(paired) > min_test_pairs * 3:
        while len(val_paired) < min_test_pairs and train_paired:
            val_paired.append(train_paired.pop())

    # Phase 2: Distribute singletons proportionally
    singletons = list(singleton_groups)
    rng.shuffle(singletons)

    n_train_paired = sum(len(g) for g in train_paired)
    n_val_paired = sum(len(g) for g in val_paired)
    n_test_paired = sum(len(g) for g in test_paired)

    # Target sizes for each split
    target_train = int(total_all * split_ratios[0])
    target_val = int(total_all * split_ratios[1])

    train_singleton_target = max(0, target_train - n_train_paired)
    val_singleton_target = max(0, target_val - n_val_paired)

    train_singletons: list[list[int]] = []
    val_singletons: list[list[int]] = []
    test_singletons: list[list[int]] = []

    s_cumulative = 0
    for g in singletons:
        if max_samples is not None:
            total_so_far = (
                n_train_paired + n_val_paired + n_test_paired
                + sum(len(sg) for sg in train_singletons)
                + sum(len(sg) for sg in val_singletons)
                + sum(len(sg) for sg in test_singletons)
            )
            if total_so_far + len(g) > max_samples:
                break

        if s_cumulative < train_singleton_target:
            train_singletons.append(g)
        elif s_cumulative < train_singleton_target + val_singleton_target:
            val_singletons.append(g)
        else:
            test_singletons.append(g)
        s_cumulative += len(g)

    # Flatten
    train_idx = (
        [i for g in train_paired for i in g]
        + [i for g in train_singletons for i in g]
    )
    val_idx = (
        [i for g in val_paired for i in g]
        + [i for g in val_singletons for i in g]
    )
    test_idx = (
        [i for g in test_paired for i in g]
        + [i for g in test_singletons for i in g]
    )

    # Rebuild group lists for rebalancing
    train_groups = train_paired + train_singletons
    val_groups = val_paired + val_singletons
    test_groups = test_paired + test_singletons

    train_idx, val_idx, test_idx = _rebalance(
        train_groups, val_groups, test_groups,
        labels, max_swaps=100,
    )

    # Final pair integrity check
    _assert_pair_integrity(train_idx, val_idx, test_idx, paired_groups)

    # Compute per-split pair stats (Challenge 5)
    pair_stats = {}
    for name, idx_list in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        idx_set = set(idx_list)
        n_pairs = sum(1 for g in paired_groups if any(i in idx_set for i in g))
        n_singles = len(idx_list) - sum(
            len(g) for g in paired_groups if any(i in idx_set for i in g)
        )
        pair_stats[name] = {"cfa_pairs": n_pairs, "singletons": n_singles}

    return {
        "train_indices": sorted(train_idx),
        "val_indices": sorted(val_idx),
        "test_indices": sorted(test_idx),
        "pair_stats": pair_stats,
    }


def _vuln_ratio(indices: list[int], labels: list[int]) -> float:
    """Compute fraction of vulnerable (label=1) samples."""
    if not indices:
        return 0.0
    return sum(1 for i in indices if labels[i] == 1) / len(indices)


def _rebalance(
    train_groups: list[list[int]],
    val_groups: list[list[int]],
    test_groups: list[list[int]],
    labels: list[int],
    max_swaps: int = 100,
    target_range: tuple[float, float] = M1_BALANCE_RANGE,
) -> tuple[list[int], list[int], list[int]]:
    """
    Greedy group-swap rebalancing between train and offending split.
    Uses pre-computed vuln counts per group for O(1) ratio simulation.
    M1 relaxed range: (0.35, 0.65) -- Challenge 4.
    """
    splits = {
        "train": train_groups,
        "val": val_groups,
        "test": test_groups,
    }

    for swap_round in range(max_swaps):
        all_ok = True
        for name in ("val", "test"):
            flat = [i for g in splits[name] for i in g]
            ratio = _vuln_ratio(flat, labels)
            if not (target_range[0] <= ratio <= target_range[1]):
                all_ok = False
                _try_swap(splits["train"], splits[name], labels, target_range)
        if all_ok:
            break

    # Check train too
    train_flat = [i for g in splits["train"] for i in g]
    train_ratio = _vuln_ratio(train_flat, labels)
    if not (target_range[0] <= train_ratio <= target_range[1]):
        logger.warning(
            f"Train vuln ratio {train_ratio:.3f} outside {target_range} "
            f"after rebalancing -- proceeding (SARD inherent ratio ~0.408)"
        )

    return (
        [i for g in splits["train"] for i in g],
        [i for g in splits["val"] for i in g],
        [i for g in splits["test"] for i in g],
    )


def _group_vuln_count(group: list[int], labels: list[int]) -> int:
    """Count vulnerable samples in a group."""
    return sum(1 for i in group if labels[i] == 1)


def _try_swap(
    donor_groups: list[list[int]],
    target_groups: list[list[int]],
    labels: list[int],
    target_range: tuple[float, float],
    max_candidates: int = 200,
) -> bool:
    """
    Try to swap one group between donor and target to improve balance.
    Uses pre-computed vuln counts and samples candidates for O(n) performance.
    """
    target_total = sum(len(g) for g in target_groups)
    target_vuln = sum(_group_vuln_count(g, labels) for g in target_groups)
    if target_total == 0:
        return False

    current_ratio = target_vuln / target_total
    target_mid = (target_range[0] + target_range[1]) / 2

    donor_info = [(i, len(g), _group_vuln_count(g, labels)) for i, g in enumerate(donor_groups)]
    target_info = [(i, len(g), _group_vuln_count(g, labels)) for i, g in enumerate(target_groups)]

    rng = random.Random(0)
    d_candidates = donor_info if len(donor_info) <= max_candidates else rng.sample(donor_info, max_candidates)
    t_candidates = target_info if len(target_info) <= max_candidates else rng.sample(target_info, max_candidates)

    best_improvement = 0.0
    best_d_idx = -1
    best_t_idx = -1

    for d_idx, d_size, d_vuln in d_candidates:
        for t_idx, t_size, t_vuln in t_candidates:
            if abs(d_size - t_size) > 5:
                continue

            new_total = target_total - t_size + d_size
            new_vuln = target_vuln - t_vuln + d_vuln
            if new_total == 0:
                continue
            new_ratio = new_vuln / new_total
            improvement = abs(current_ratio - target_mid) - abs(new_ratio - target_mid)

            if improvement > best_improvement:
                best_improvement = improvement
                best_d_idx = d_idx
                best_t_idx = t_idx

    if best_d_idx >= 0 and best_t_idx >= 0:
        donor_groups[best_d_idx], target_groups[best_t_idx] = (
            target_groups[best_t_idx],
            donor_groups[best_d_idx],
        )
        return True
    return False


# -- Integrity assertions -----------------------------------------------------
def _assert_pair_integrity(
    train_idx: list[int],
    val_idx: list[int],
    test_idx: list[int],
    groups: list[list[int]],
) -> None:
    """Assert no group's members are split across multiple splits (Challenge 1)."""
    idx_to_split: dict[int, str] = {}
    for i in train_idx:
        idx_to_split[i] = "train"
    for i in val_idx:
        idx_to_split[i] = "val"
    for i in test_idx:
        idx_to_split[i] = "test"

    for g in groups:
        if len(g) < 2:
            continue
        splits_in_group = {idx_to_split.get(i) for i in g if i in idx_to_split}
        splits_in_group.discard(None)
        if len(splits_in_group) > 1:
            raise ValueError(
                f"Pair integrity violation: group {g} split across {splits_in_group}"
            )


def assert_no_sha_leakage(
    train_idx: list[int],
    test_idx: list[int],
    sample_ids: list[str],
    supplement: dict[str, dict[str, str]],
) -> int:
    """
    Assert zero commit_sha overlap between train and test sets.
    Prevents data leakage from the same CVE fix commit appearing in both.

    Returns the number of overlapping SHAs (0 = pass).
    Raises ValueError if overlap > 0.
    """
    train_shas: set[str] = set()
    test_shas: set[str] = set()

    for i in train_idx:
        sid = sample_ids[i]
        sha = supplement.get(sid, {}).get("commit_sha", "")
        if sha:
            train_shas.add(sha)

    for i in test_idx:
        sid = sample_ids[i]
        sha = supplement.get(sid, {}).get("commit_sha", "")
        if sha:
            test_shas.add(sha)

    overlap = train_shas & test_shas
    if overlap:
        raise ValueError(
            f"Commit SHA leakage: {len(overlap)} SHAs in both train and test. "
            f"First 5: {list(overlap)[:5]}"
        )
    return 0


def assert_no_cve_leakage(
    train_idx: list[int],
    test_idx: list[int],
    sample_ids: list[str],
    supplement: dict[str, dict[str, str]],
) -> int:
    """
    Warn if any cve_id appears in both train and test.
    Returns the number of overlapping CVE IDs.
    Does NOT raise -- CVE overlap is a warning, not a hard failure,
    because the same CVE can have different fix commits.
    """
    train_cves: set[str] = set()
    test_cves: set[str] = set()

    for i in train_idx:
        sid = sample_ids[i]
        cve = supplement.get(sid, {}).get("cve_id", "")
        if cve:
            train_cves.add(cve)

    for i in test_idx:
        sid = sample_ids[i]
        cve = supplement.get(sid, {}).get("cve_id", "")
        if cve:
            test_cves.add(cve)

    overlap = train_cves & test_cves
    if overlap:
        logger.warning(
            f"CVE ID overlap: {len(overlap)} CVEs in both train and test. "
            f"First 5: {list(overlap)[:5]}"
        )
    return len(overlap)


# -- Streaming HDF5 copy ------------------------------------------------------
def write_split_h5(
    src_path: Path,
    dst_path: Path,
    indices: list[int],
    metadata: dict,
) -> None:
    """
    Stream-copy selected graphs from src HDF5 to dst HDF5.
    One graph at a time to avoid loading 3.7 GB into memory.
    Atomic write via .tmp + os.replace.
    """
    tmp_path = dst_path.with_suffix(".h5.tmp")
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    sorted_indices = sorted(indices)
    old_to_new = {old: new for new, old in enumerate(sorted_indices)}

    with h5py.File(src_path, "r") as src, h5py.File(tmp_path, "w") as dst:
        meta = dst.create_group("metadata")
        meta.attrs["feature_dim"] = FEATURE_DIM
        meta.attrs["num_graphs"] = len(sorted_indices)
        meta.attrs["edge_types"] = 4

        dt_str = h5py.special_dtype(vlen=str)
        sample_ids = [metadata["sample_ids"][i] for i in sorted_indices]
        labels_list = [metadata["labels"][i] for i in sorted_indices]
        cwes = [metadata["cwes"][i] for i in sorted_indices]
        pair_ids = [metadata["pair_ids"][i] for i in sorted_indices]

        meta.create_dataset("sample_ids", data=sample_ids, dtype=dt_str)
        meta.create_dataset("labels", data=np.array(labels_list, dtype=np.int64))
        meta.create_dataset("cwes", data=cwes, dtype=dt_str)
        meta.create_dataset("pair_ids", data=pair_ids, dtype=dt_str)

        graphs_grp = dst.create_group("graphs")
        src_graphs = src["graphs"]

        for old_idx in sorted_indices:
            new_idx = old_to_new[old_idx]
            src_g = src_graphs[str(old_idx)]
            dst_g = graphs_grp.create_group(str(new_idx))

            dst_g.create_dataset(
                "x", data=src_g["x"][:],
                compression="gzip", compression_opts=4,
            )
            dst_g.create_dataset("edge_index", data=src_g["edge_index"][:])
            dst_g.create_dataset("edge_attr", data=src_g["edge_attr"][:])
            dst_g.create_dataset("y", data=src_g["y"][:])

    os.replace(tmp_path, dst_path)


# -- Main pipeline -------------------------------------------------------------
def run_stage7(
    input_path: str,
    output_dir: str,
    split_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
    dry_run: bool = False,
    max_samples: int | None = None,
    supplement_path: str | None = None,
) -> dict:
    """
    Run Stage 7: CFA-aware train/val/test split.

    Args:
        supplement_path: path to deduped JSONL for commit_sha/cve_id.
            If None, tries DEFAULT_SUPPLEMENT; if that doesn't exist,
            falls back to pair_id-only grouping.
    """
    logger.info("Stage 7: CFA-Aware Split (Phase 3)")
    logger.info(f"  Input:  {input_path}")
    logger.info(f"  Output: {output_dir}")
    logger.info(f"  Split:  {split_ratios}")
    logger.info(f"  Seed:   {seed}")

    src = Path(input_path)
    out = Path(output_dir)

    if not src.exists():
        raise FileNotFoundError(f"Input HDF5 not found: {src}")

    # Step 1: Load metadata
    metadata = load_metadata(src)
    n_total = metadata["num_graphs"]
    logger.info(f"Loaded metadata: {n_total} graphs")

    # Step 2: Load supplement (commit_sha/cve_id)
    supplement: dict[str, dict[str, str]] | None = None
    sup_path = Path(supplement_path) if supplement_path else Path(DEFAULT_SUPPLEMENT)
    if sup_path.exists():
        logger.info(f"Loading supplement from {sup_path}...")
        supplement = load_supplement(sup_path, metadata["sample_ids"])
    else:
        logger.warning(
            f"Supplement JSONL not found at {sup_path} -- "
            f"falling back to pair_id-only grouping (no commit_sha/cve_id)"
        )

    # Step 3: Build groups with hierarchy
    paired_groups, singleton_groups = build_groups(
        metadata["pair_ids"],
        sample_ids=metadata["sample_ids"],
        supplement=supplement,
    )
    n_paired = len(paired_groups)
    n_singletons = len(singleton_groups)
    n_paired_samples = sum(len(g) for g in paired_groups)

    # Count group types for stats
    group_type_counts = {"pair_id": 0, "commit_sha": 0, "cve_id": 0}
    if supplement:
        for g in paired_groups:
            key = _get_group_key(g[0], metadata["pair_ids"], metadata["sample_ids"], supplement)
            if key.startswith("pair:"):
                group_type_counts["pair_id"] += 1
            elif key.startswith("sha:"):
                group_type_counts["commit_sha"] += 1
            elif key.startswith("cve:"):
                group_type_counts["cve_id"] += 1

    logger.info(
        f"Groups: {n_paired} paired ({n_paired_samples} samples), "
        f"{n_singletons} singletons"
    )
    if supplement:
        logger.info(
            f"  Group types: {group_type_counts['pair_id']} pair_id, "
            f"{group_type_counts['commit_sha']} commit_sha, "
            f"{group_type_counts['cve_id']} cve_id"
        )

    # Step 4: Two-phase split (Challenge 5)
    result = split_paired_and_singletons(
        paired_groups, singleton_groups,
        metadata["labels"],
        split_ratios=split_ratios, seed=seed,
        max_samples=max_samples,
    )

    train_idx = result["train_indices"]
    val_idx = result["val_indices"]
    test_idx = result["test_indices"]
    pair_stats = result.get("pair_stats", {})

    # Step 5: Leakage assertions
    sha_overlap = 0
    cve_overlap = 0
    if supplement:
        assert_no_sha_leakage(
            train_idx, test_idx,
            metadata["sample_ids"], supplement,
        )
        cve_overlap = assert_no_cve_leakage(
            train_idx, test_idx,
            metadata["sample_ids"], supplement,
        )

    # Compute stats
    train_ratio = _vuln_ratio(train_idx, metadata["labels"])
    val_ratio = _vuln_ratio(val_idx, metadata["labels"])
    test_ratio = _vuln_ratio(test_idx, metadata["labels"])

    # Per-CWE breakdown (Challenge 3)
    cwe_counter: Counter = Counter()
    for i in train_idx + val_idx + test_idx:
        cwe_counter[metadata["cwes"][i]] += 1
    cwe_breakdown = dict(cwe_counter.most_common())

    per_split_cwes = {}
    for name, idx_list in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        split_cwe = Counter(metadata["cwes"][i] for i in idx_list)
        per_split_cwes[name] = dict(split_cwe.most_common())

    stats = {
        "total_samples": n_total,
        "train_samples": len(train_idx),
        "val_samples": len(val_idx),
        "test_samples": len(test_idx),
        "train_pct": round(len(train_idx) / max(n_total, 1) * 100, 1),
        "val_pct": round(len(val_idx) / max(n_total, 1) * 100, 1),
        "test_pct": round(len(test_idx) / max(n_total, 1) * 100, 1),
        "train_vuln_ratio": round(train_ratio, 4),
        "val_vuln_ratio": round(val_ratio, 4),
        "test_vuln_ratio": round(test_ratio, 4),
        "m1_balance_range": list(M1_BALANCE_RANGE),
        "paired_groups": n_paired,
        "singletons": n_singletons,
        "paired_samples": n_paired_samples,
        "group_type_counts": group_type_counts,
        "sha_overlap_train_test": sha_overlap,
        "cve_overlap_train_test": cve_overlap,
        "supplement_loaded": supplement is not None,
        "pair_stats_per_split": pair_stats,
        "cwe_breakdown": cwe_breakdown,
        "per_split_cwes": per_split_cwes,
        "total_cwes": len(cwe_breakdown),
        "seed": seed,
        "split_ratios": list(split_ratios),
    }

    logger.info(
        f"Split: train={len(train_idx)} ({stats['train_pct']}%, "
        f"vuln={train_ratio:.3f}), "
        f"val={len(val_idx)} ({stats['val_pct']}%, "
        f"vuln={val_ratio:.3f}), "
        f"test={len(test_idx)} ({stats['test_pct']}%, "
        f"vuln={test_ratio:.3f})"
    )
    if pair_stats:
        for name, ps in pair_stats.items():
            logger.info(
                f"  {name}: {ps['cfa_pairs']} CFA pairs, "
                f"{ps['singletons']} singletons"
            )
    logger.info(f"CWE breakdown: {cwe_breakdown}")

    # Challenge 4: warn if any split outside M1 range
    for name, ratio in [("train", train_ratio), ("val", val_ratio), ("test", test_ratio)]:
        if not (M1_BALANCE_RANGE[0] <= ratio <= M1_BALANCE_RANGE[1]):
            logger.warning(
                f"{name} vuln ratio {ratio:.3f} outside M1 range "
                f"{M1_BALANCE_RANGE} -- review pipeline stages"
            )

    if dry_run:
        logger.info("[DRY RUN] Would write split files. Stats:")
        logger.info(json.dumps(stats, indent=2))
        stats["mode"] = "dry_run"
        return stats

    # Step 7: Streaming HDF5 copy
    start_time = time.time()

    for split_name, indices in [
        ("train", train_idx),
        ("val", val_idx),
        ("test", test_idx),
    ]:
        dst = out / f"{split_name}.h5"
        logger.info(f"Writing {split_name}.h5 ({len(indices)} graphs)...")
        write_split_h5(src, dst, indices, metadata)

    elapsed = time.time() - start_time
    stats["elapsed_seconds"] = round(elapsed, 1)

    # Save stats
    stats_path = out / "split_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    logger.info(f"Stage 7 complete in {elapsed:.1f}s")
    logger.info(f"Stats saved to {stats_path}")

    return stats


# -- CLI -----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Stage 7: CFA-Aware Train/Val/Test Split",
    )
    parser.add_argument(
        "--input",
        default="training/data/graphs/all_graphs.h5",
        help="Input HDF5 path (default: training/data/graphs/all_graphs.h5)",
    )
    parser.add_argument(
        "--output-dir",
        default="training/data/final/",
        help="Output directory (default: training/data/final/)",
    )
    parser.add_argument(
        "--supplement",
        default=None,
        help=(
            "Path to deduped JSONL for commit_sha/cve_id grouping "
            f"(default: {DEFAULT_SUPPLEMENT})"
        ),
    )
    parser.add_argument(
        "--split",
        default="80/10/10",
        help="Split ratios as train/val/test (default: 80/10/10)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show split stats without writing files",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit total samples (for testing)",
    )

    args = parser.parse_args()

    parts = args.split.split("/")
    if len(parts) != 3:
        parser.error("--split must be in format A/B/C (e.g., 80/10/10)")
    ratios = tuple(int(p) / 100 for p in parts)
    if abs(sum(ratios) - 1.0) > 0.01:
        parser.error(f"Split ratios must sum to 100, got {sum(int(p) for p in parts)}")

    run_stage7(
        input_path=args.input,
        output_dir=args.output_dir,
        split_ratios=ratios,
        seed=args.seed,
        dry_run=args.dry_run,
        max_samples=args.max_samples,
        supplement_path=args.supplement,
    )


if __name__ == "__main__":
    main()
