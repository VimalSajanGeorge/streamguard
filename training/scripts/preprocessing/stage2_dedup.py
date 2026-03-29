#!/usr/bin/env python3
"""
training/scripts/preprocessing/stage2_dedup.py

Stage 2: 4-Level Deduplication for StreamGuard pipeline.

Input:  training/data/processed/cleaned/samples.jsonl
Output: training/data/processed/deduped/samples.jsonl

Deduplication levels (in order):
  L1: MD5 hash of whitespace-normalized code (exact dedup)
  L2: CVE-ID dedup — same CVE + file_path kept once
  L3: Commit SHA dedup — same SHA + file_path kept once
  L4: MinHash LSH — 3-group source-aware near-dedup:
      sard       → threshold 0.95  (Juliet flow variants are intentional)
      real_world → threshold 0.80  (cross-collector duplicates of same commit)
      unpaired   → threshold 0.75  (ExploitDB near-copies)

CRITICAL: After L4, repair_broken_pairs() clears orphaned pair_ids where
MinHash removed one side of a CFA pair. Never leave half-pairs in the dataset.

CLI flags: --input, --output, --dry-run, --threshold (overrides per-group defaults)

Source: docs/New Docs/PREPROCESSING.md §Stage 2
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

from datasketch import MinHash, MinHashLSH
from loguru import logger

# ── Constants ─────────────────────────────────────────────────────
SHINGLE_K = 4           # Character shingle size
DEFAULT_THRESHOLD = 0.85
LARGE_DATASET_CUTOFF = 40_000  # Switch to num_perm=64 above this

# Source → (group_name, lsh_threshold)
SOURCE_GROUP: dict[str, tuple[str, float]] = {
    "sard":                ("sard_group",   0.95),
    "exploitdb":           ("unpaired",     0.75),
    "manual":              ("unpaired",     0.75),
    "cve":                 ("real_world",   0.80),
    "github_advisory":     ("real_world",   0.80),
    "osv":                 ("real_world",   0.80),
    "repo":                ("real_world",   0.80),
    "sard_cfa":            ("sard_group",   0.95),
    "exploitdb_cfa":       ("unpaired",     0.75),
    "cve_cfa":             ("real_world",   0.80),
    "github_advisory_cfa": ("real_world",   0.80),
    "osv_cfa":             ("real_world",   0.80),
    "repo_cfa":            ("real_world",   0.80),
}

GROUP_THRESHOLDS: dict[str, float] = {
    "sard_group": 0.95,
    "real_world":  0.80,
    "unpaired":    0.75,
}


# ── Helpers ───────────────────────────────────────────────────────
def md5_normalize(code: str) -> str:
    """MD5 hash of whitespace-normalized code."""
    return hashlib.md5(" ".join(code.split()).encode()).hexdigest()


def _build_minhash(code: str, num_perm: int) -> MinHash:
    """Build MinHash from 4-gram character shingles of whitespace-normalized code."""
    m = MinHash(num_perm=num_perm)
    normalized = " ".join(code.split())
    if len(normalized) < SHINGLE_K:
        m.update(normalized.encode())
        return m
    for i in range(len(normalized) - SHINGLE_K + 1):
        m.update(normalized[i:i + SHINGLE_K].encode())
    return m


def _get_group(sample: dict) -> tuple[str, float]:
    """Return (group_name, threshold) for a sample based on its source."""
    source = sample.get("source", "")
    return SOURCE_GROUP.get(source, ("real_world", 0.80))


# ── Level 1: Exact dedup (MD5 of whitespace-normalized code) ─────
def level1_exact(samples: list[dict]) -> list[dict]:
    """Remove exact duplicates after whitespace normalization."""
    seen: dict[str, int] = {}
    result: list[dict] = []

    for s in samples:
        h = md5_normalize(s["code"])
        if h not in seen:
            seen[h] = len(result)
            result.append(s)

    removed = len(samples) - len(result)
    logger.info(f"L1 exact: {removed} removed")
    return result


# ── Level 2: CVE-ID dedup ────────────────────────────────────────
def level2_cve_id(samples: list[dict]) -> list[dict]:
    """Same CVE from multiple sources -> keep one per (cve_id, file_path)."""
    seen: set[tuple] = set()
    result: list[dict] = []

    for s in samples:
        cve = s.get("cve_id")
        key = (cve, s.get("file_path", "")) if cve else None
        if not key or key not in seen:
            result.append(s)
            if key:
                seen.add(key)

    removed = len(samples) - len(result)
    logger.info(f"L2 CVE-ID: {removed} removed")
    return result


# ── Level 3: Commit SHA dedup ────────────────────────────────────
def level3_commit_sha(samples: list[dict]) -> list[dict]:
    """Same commit + file -> keep one."""
    seen: dict[tuple, bool] = {}
    result: list[dict] = []

    for s in samples:
        sha = s.get("commit_sha")
        key = (sha, s.get("file_path", "")) if sha else None
        if not key or key not in seen:
            result.append(s)
            if key:
                seen[key] = True

    removed = len(samples) - len(result)
    logger.info(f"L3 commit SHA: {removed} removed")
    return result


# ── Level 4: MinHash LSH near-dedup (3-group, source-aware) ──────
def level4_minhash_lsh(
    samples: list[dict],
    threshold: float | None = None,
) -> list[dict]:
    """
    3-group source-aware near-dedup.

    Groups:
      sard_group → 0.95  (Juliet flow variants are intentional)
      real_world → 0.80  (cross-collector duplicates of same commit)
      unpaired   → 0.75  (ExploitDB near-copies)

    If threshold is provided, it overrides per-group defaults.

    CFA pair protection: pair groups are processed atomically — use the
    vulnerable member (label=1) as representative. If it's novel, keep
    ALL members. If near-duplicate, skip ALL members. Never break a pair.
    """
    num_perm = 64 if len(samples) > LARGE_DATASET_CUTOFF else 128

    # Partition by group
    groups: dict[str, list[dict]] = {
        "sard_group": [], "real_world": [], "unpaired": [],
    }
    for s in samples:
        group_name, _ = _get_group(s)
        groups[group_name].append(s)

    result: list[dict] = []

    for group_name, group_samples in groups.items():
        # Use override threshold if provided, else per-group default
        thresh = threshold if threshold is not None else GROUP_THRESHOLDS[group_name]
        lsh = MinHashLSH(threshold=thresh, num_perm=num_perm)
        key_ctr = 0

        # Separate paired vs unpaired within this group
        pair_groups: dict[str, list[dict]] = {}
        unpaired_in_group: list[dict] = []
        for s in group_samples:
            pid = s.get("pair_id", "")
            if pid:
                pair_groups.setdefault(pid, []).append(s)
            else:
                unpaired_in_group.append(s)

        # Process paired groups atomically
        pairs_kept = 0
        pairs_skipped = 0
        for pid, members in pair_groups.items():
            rep = next((m for m in members if m.get("label") == 1), members[0])
            mh = _build_minhash(rep["code"], num_perm)

            if not lsh.query(mh):
                lsh.insert(f"p_{key_ctr}", mh)
                key_ctr += 1
                result.extend(members)
                pairs_kept += 1
            else:
                pairs_skipped += 1

        # Process unpaired samples individually
        unpaired_kept = 0
        for s in unpaired_in_group:
            mh = _build_minhash(s["code"], num_perm)
            if not lsh.query(mh):
                lsh.insert(f"u_{key_ctr}", mh)
                key_ctr += 1
                result.append(s)
                unpaired_kept += 1

        logger.info(
            f"L4 [{group_name}] (threshold={thresh}): "
            f"{len(group_samples)} in, pairs: {pairs_kept} kept / "
            f"{pairs_skipped} skipped, unpaired: "
            f"{unpaired_kept}/{len(unpaired_in_group)} kept"
        )

    return result


# ── Pair repair: clear orphaned pair_ids after L4 ────────────────
def repair_broken_pairs(samples: list[dict]) -> list[dict]:
    """
    CRITICAL — run AFTER level4_minhash_lsh.

    L4 may remove one side of a CFA pair. If a pair_id has only one label
    remaining (not both 0 and 1), clear the pair_id on survivors.
    Never leave a half-pair in the dataset.
    """
    pairs: dict[str, set[int]] = {}
    for s in samples:
        pid = s.get("pair_id", "")
        if pid:
            pairs.setdefault(pid, set()).add(s.get("label", -1))

    broken = {pid for pid, labels in pairs.items() if not ({0, 1} <= labels)}

    if not broken:
        logger.info("Pair repair: 0 pairs repaired (all intact)")
        return samples

    result: list[dict] = []
    for s in samples:
        pid = s.get("pair_id", "")
        if pid and pid in broken:
            s = {**s, "pair_id": ""}
        result.append(s)

    logger.info(f"Pair repair: {len(broken)} pairs repaired (orphaned)")
    return result


# ── Main pipeline ─────────────────────────────────────────────────
def run_stage2(
    input_path: str,
    output_path: str,
    dry_run: bool = False,
    threshold: float | None = None,
) -> dict:
    """
    Run all 4 dedup levels + pair repair sequentially. Returns stats dict.

    threshold: if None, use per-group defaults (0.95/0.80/0.75).
               if float, override all groups with this value.
    """
    logger.info(f"Stage 2: Loading samples from {input_path}")

    inp = Path(input_path)
    if not inp.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    samples: list[dict] = []
    with open(inp, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    n_input = len(samples)
    logger.info(f"Input: {n_input}")

    # L1: Exact dedup
    samples = level1_exact(samples)
    n_after_l1 = len(samples)
    logger.info(f"After L1: {n_after_l1}")

    # L2: CVE-ID dedup
    samples = level2_cve_id(samples)
    n_after_l2 = len(samples)
    logger.info(f"After L2: {n_after_l2}")

    # L3: Commit SHA dedup
    samples = level3_commit_sha(samples)
    n_after_l3 = len(samples)
    logger.info(f"After L3: {n_after_l3}")

    # L4: MinHash LSH near-dedup (3-group, source-aware)
    samples = level4_minhash_lsh(samples, threshold=threshold)
    n_after_l4 = len(samples)
    logger.info(f"After L4: {n_after_l4}")

    # Pair repair
    samples = repair_broken_pairs(samples)
    n_after_repair = len(samples)
    logger.info(f"After pair repair: {n_after_repair}")

    # Write output (atomic: write to .tmp then os.replace)
    if not dry_run:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=str(out.parent), suffix=".tmp", prefix="stage2_"
        )
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                for s in samples:
                    f.write(json.dumps(s) + "\n")
            os.replace(tmp_path, str(out))
            logger.info(f"Output written to {output_path}")
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise
    else:
        logger.info("[DRY RUN] No output written")

    total_removed = n_input - n_after_l4
    pct = (total_removed / n_input * 100) if n_input > 0 else 0.0

    stats = {
        "input": n_input,
        "after_l1_exact": n_after_l1,
        "after_l2_cve": n_after_l2,
        "after_l3_sha": n_after_l3,
        "after_l4_lsh": n_after_l4,
        "after_pair_repair": n_after_repair,
        "output": n_after_repair,
        "total_removed": total_removed,
        "removal_pct": round(pct, 2),
    }

    logger.info(
        f"Stage 2 complete: {n_input} → {n_after_repair} "
        f"({total_removed} removed, {pct:.1f}%)"
    )
    return stats


# ── CLI ───────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Stage 2: 4-Level Deduplication (3-group source-aware)"
    )
    ap.add_argument(
        "--input",
        default="training/data/processed/cleaned/samples.jsonl",
        help="Input JSONL path (Stage 1 output)",
    )
    ap.add_argument(
        "--output",
        default="training/data/processed/deduped/samples.jsonl",
        help="Output JSONL path",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Run dedup levels but do not write output",
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override MinHash LSH threshold for all groups (default: per-group)",
    )

    args = ap.parse_args()
    run_stage2(args.input, args.output, args.dry_run, args.threshold)


if __name__ == "__main__":
    main()
