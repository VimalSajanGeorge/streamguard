#!/usr/bin/env python3
"""
training/scripts/preprocessing/pre_training_audit.py

Pre-Training Audit: 9 automated quality checks + Check 10 (CFA quality)
+ Check 11 (commit_sha leakage) on split HDF5 files.

Designed to catch data issues BEFORE training begins, saving GPU hours.
Exit code 0 = all checks pass, 1 = any check fails.

Checks (M1 thresholds / M2 thresholds):
  1. min_train_samples    >= 5,000 / >= 30,000
  2. vuln_safe_balance    vuln ratio in (0.35, 0.65) / (0.45, 0.55) per split
  3. cwe_diversity        >= 4 CWEs with >= 500 / >= 7 CWEs with >= 500
  4. max_cwe_dominance    no CWE > 0.45 / > 0.40 of total
  5. no_null_code         every graph has x.shape[0] >= 3
  6. test_train_no_overlap  0 overlapping sample_ids
  7. code_length_range    all graphs have 3 <= nodes <= 4096
  8. pair_integrity        no pair_id in multiple splits
  9. min_taint_coverage   >= 0.20 / >= 0.25 fraction with taint nodes
 10. cfa_quality_per_cwe  (optional) per-CWE compile/fix rates
 11. commit_sha_leakage   (optional) 0 SHA overlap between train/test

Source: docs/New Docs/StreamGuard_Phase3_Preprocessing_Plan.docx
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
from loguru import logger

FEATURE_DIM = 824
MIN_NODES = 3
MAX_NODES = 4096

# -- Threshold dictionaries ---------------------------------------------------
REQUIRED_CHECKS_M1 = {
    "min_train_samples":     5_000,
    "vuln_safe_balance":     (0.35, 0.65),
    "cwe_diversity":         4,
    "max_cwe_dominance":     0.45,
    "no_null_code":          0,
    "test_train_no_overlap": 0,
    "code_length_range":     (MIN_NODES, MAX_NODES),
    "pair_integrity":        0,
    "min_taint_coverage":    0.20,
}

REQUIRED_CHECKS_M2 = {
    "min_train_samples":     30_000,
    "vuln_safe_balance":     (0.45, 0.55),
    "cwe_diversity":         7,
    "max_cwe_dominance":     0.40,
    "no_null_code":          0,
    "test_train_no_overlap": 0,
    "code_length_range":     (MIN_NODES, MAX_NODES),
    "pair_integrity":        0,
    "min_taint_coverage":    0.25,
}

# -- CFA quality thresholds (Section 4.3 of CFA research doc) ----------------
CFA_QUALITY_THRESHOLDS: dict[str, dict[str, float]] = {
    "CWE-134": {"compile_rate": 0.95, "fix_signature_rate": 0.90},
    "CWE-120": {"compile_rate": 0.90, "fix_signature_rate": 0.88},
    "CWE-476": {"compile_rate": 0.88, "fix_signature_rate": 0.80},
    "CWE-121": {"compile_rate": 0.83, "fix_signature_rate": 0.75},
    "CWE-122": {"compile_rate": 0.80, "fix_signature_rate": 0.72},
    "CWE-125": {"compile_rate": 0.78, "fix_signature_rate": 0.70},
    "CWE-89":  {"compile_rate": 0.78, "taint_break_rate": 0.72},
    "CWE-78":  {"compile_rate": 0.75, "taint_break_rate": 0.70},
    "CWE-190": {"compile_rate": 0.80, "fix_signature_rate": 0.70},
    "CWE-79":  {"compile_rate": 0.68, "fix_signature_rate": 0.60},
    "CWE-119": {"compile_rate": 0.65, "fix_signature_rate": 0.58},
    "CWE-416": {"compile_rate": 0.60, "fix_signature_rate": 0.52},
}

# Default supplement JSONL path
DEFAULT_SUPPLEMENT = "training/data/processed/deduped/samples.jsonl"


# -- Check result --------------------------------------------------------------
class CheckResult:
    def __init__(self, name: str, passed: bool, detail: str):
        self.name = name
        self.passed = passed
        self.detail = detail

    def __repr__(self):
        status = "PASS" if self.passed else "FAIL"
        return f"[{status}] {self.name}: {self.detail}"


# -- Metadata loader -----------------------------------------------------------
def _load_split_metadata(h5_path: Path) -> dict:
    """Load metadata from a split HDF5 file."""
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
        "path": str(h5_path),
        "num_graphs": num_graphs,
        "sample_ids": sample_ids,
        "labels": labels,
        "cwes": cwes,
        "pair_ids": pair_ids,
    }


def _iter_graph_shapes(h5_path: Path):
    """Yield (graph_idx, num_nodes) for each graph in the HDF5."""
    with h5py.File(h5_path, "r") as f:
        graphs = f["graphs"]
        for idx in sorted(graphs.keys(), key=int):
            x = graphs[idx]["x"]
            yield int(idx), x.shape[0]


def _count_taint_graphs(h5_path: Path) -> tuple[int, int]:
    """
    Count graphs that have at least one taint node.
    Taint roles are at positions [800:804] in 824-d vector.
    Indices 0-3 = SOURCE/SINK/SANITIZER/PROPAGATION (active roles).
    Returns (n_with_taint, n_total).
    """
    n_with_taint = 0
    n_total = 0
    with h5py.File(h5_path, "r") as f:
        graphs = f["graphs"]
        for idx in sorted(graphs.keys(), key=int):
            x = graphs[idx]["x"][:]
            n_total += 1
            # Check if any node has an active taint role (positions 800:804)
            taint_slice = x[:, 800:804]
            if taint_slice.any():
                n_with_taint += 1
    return n_with_taint, n_total


# -- The 9 structural checks --------------------------------------------------
def check_min_train_samples(
    splits: dict[str, dict],
    threshold: int = 5000,
) -> CheckResult:
    """Check 1: train split has >= threshold samples."""
    n = splits["train"]["num_graphs"]
    passed = n >= threshold
    return CheckResult(
        "min_train_samples",
        passed,
        f"train has {n} samples (threshold: >= {threshold})",
    )


def check_vuln_safe_balance(
    splits: dict[str, dict],
    low: float = 0.35,
    high: float = 0.65,
) -> CheckResult:
    """Check 2: vuln ratio in (low, high) for each split."""
    details = []
    all_pass = True
    for name, meta in splits.items():
        labels = meta["labels"]
        if not labels:
            details.append(f"{name}: empty")
            all_pass = False
            continue
        ratio = sum(1 for l in labels if l == 1) / len(labels)
        ok = low <= ratio <= high
        if not ok:
            all_pass = False
        details.append(f"{name}={ratio:.3f} {'OK' if ok else 'FAIL'}")
    return CheckResult(
        "vuln_safe_balance",
        all_pass,
        f"vuln ratios: {', '.join(details)} (range: [{low}, {high}])",
    )


def check_cwe_diversity(
    splits: dict[str, dict],
    min_cwes: int = 5,
    min_samples_per_cwe: int = 500,
) -> CheckResult:
    """Check 3: >= min_cwes CWEs each with >= min_samples_per_cwe samples."""
    all_cwes: list[str] = []
    for meta in splits.values():
        all_cwes.extend(meta["cwes"])

    cwe_counts = Counter(all_cwes)
    qualifying = {cwe: cnt for cwe, cnt in cwe_counts.items() if cnt >= min_samples_per_cwe}
    n_qualifying = len(qualifying)
    passed = n_qualifying >= min_cwes

    top5 = cwe_counts.most_common(5)
    return CheckResult(
        "cwe_diversity",
        passed,
        f"{n_qualifying} CWEs with >= {min_samples_per_cwe} samples "
        f"(threshold: >= {min_cwes}). Top 5: {top5}",
    )


def check_max_cwe_dominance(
    splits: dict[str, dict],
    max_fraction: float = 0.40,
) -> CheckResult:
    """Check 4: no single CWE > max_fraction of total."""
    all_cwes: list[str] = []
    for meta in splits.values():
        all_cwes.extend(meta["cwes"])

    if not all_cwes:
        return CheckResult("max_cwe_dominance", False, "no samples")

    cwe_counts = Counter(all_cwes)
    total = len(all_cwes)
    worst_cwe, worst_count = cwe_counts.most_common(1)[0]
    worst_frac = worst_count / total
    passed = worst_frac <= max_fraction

    return CheckResult(
        "max_cwe_dominance",
        passed,
        f"most common: {worst_cwe} = {worst_frac:.3f} "
        f"({worst_count}/{total}, threshold: <= {max_fraction})",
    )


def check_no_null_code(
    split_paths: dict[str, Path],
) -> CheckResult:
    """Check 5: every graph has x.shape[0] >= MIN_NODES."""
    violations = 0
    total = 0
    for name, path in split_paths.items():
        for idx, n_nodes in _iter_graph_shapes(path):
            total += 1
            if n_nodes < MIN_NODES:
                violations += 1
                if violations <= 3:
                    logger.debug(f"  {name} graph {idx}: {n_nodes} nodes < {MIN_NODES}")

    return CheckResult(
        "no_null_code",
        violations == 0,
        f"{violations} violations out of {total} graphs "
        f"(threshold: x.shape[0] >= {MIN_NODES})",
    )


def check_test_train_no_overlap(
    splits: dict[str, dict],
) -> CheckResult:
    """Check 6: no sample_id overlap between any pair of splits."""
    split_sets: dict[str, set[str]] = {}
    for name, meta in splits.items():
        split_sets[name] = set(meta["sample_ids"])

    overlaps = []
    names = list(split_sets.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            overlap = split_sets[names[i]] & split_sets[names[j]]
            if overlap:
                overlaps.append(
                    f"{names[i]}-{names[j]}: {len(overlap)} overlapping"
                )

    return CheckResult(
        "test_train_no_overlap",
        len(overlaps) == 0,
        f"{len(overlaps)} overlapping pairs" if overlaps else "0 overlap across all splits",
    )


def check_code_length_range(
    split_paths: dict[str, Path],
    min_nodes: int = MIN_NODES,
    max_nodes: int = MAX_NODES,
) -> CheckResult:
    """Check 7: all graphs have min_nodes <= x.shape[0] <= max_nodes."""
    violations = 0
    total = 0
    too_small = 0
    too_large = 0
    for name, path in split_paths.items():
        for idx, n_nodes in _iter_graph_shapes(path):
            total += 1
            if n_nodes < min_nodes:
                too_small += 1
                violations += 1
            elif n_nodes > max_nodes:
                too_large += 1
                violations += 1

    return CheckResult(
        "code_length_range",
        violations == 0,
        f"{violations} violations ({too_small} too small, {too_large} too large) "
        f"out of {total} graphs (range: [{min_nodes}, {max_nodes}])",
    )


def check_pair_integrity(
    splits: dict[str, dict],
) -> CheckResult:
    """Check 8: no pair_id appears in more than one split."""
    pair_to_splits: dict[str, set[str]] = {}
    for name, meta in splits.items():
        for pid in meta["pair_ids"]:
            if pid:  # skip empty pair_ids
                if pid not in pair_to_splits:
                    pair_to_splits[pid] = set()
                pair_to_splits[pid].add(name)

    broken = {pid: s for pid, s in pair_to_splits.items() if len(s) > 1}

    return CheckResult(
        "pair_integrity",
        len(broken) == 0,
        f"{len(broken)} broken pairs" if broken else
        f"0 broken pairs (checked {len(pair_to_splits)} pair_ids)",
    )


def check_min_taint_coverage(
    split_paths: dict[str, Path],
    min_coverage: float = 0.20,
) -> CheckResult:
    """Check 9: >= min_coverage fraction of samples have taint nodes."""
    total_with_taint = 0
    total_samples = 0
    for name, path in split_paths.items():
        n_taint, n_total = _count_taint_graphs(path)
        total_with_taint += n_taint
        total_samples += n_total

    coverage = total_with_taint / max(total_samples, 1)
    passed = coverage >= min_coverage

    return CheckResult(
        "min_taint_coverage",
        passed,
        f"{total_with_taint}/{total_samples} = {coverage:.3f} "
        f"(threshold: >= {min_coverage})",
    )


# -- Check 10: CFA quality per CWE --------------------------------------------

class CfaCheckRow:
    """One row in the CFA quality table."""

    def __init__(
        self,
        check_id: str,
        cwe: str,
        metric: str,
        actual: float,
        threshold: float,
        passed: bool,
    ) -> None:
        self.check_id = check_id
        self.cwe = cwe
        self.metric = metric
        self.actual = actual
        self.threshold = threshold
        self.passed = passed

    def status(self) -> str:
        return "PASS" if self.passed else "FAIL"


def check_cfa_quality_per_cwe(
    dataset_dir: str,
    report_path: str | None = None,
) -> tuple[CheckResult, list[CfaCheckRow]]:
    """
    Check 10: Read cfa_quality_report.json and compare each CWE against
    the thresholds in CFA_QUALITY_THRESHOLDS.

    If cfa_quality_report.json does not exist, returns a PASS CheckResult
    with a warning (Stage 3 may not have run yet -- this check is optional).
    """
    if report_path:
        rp = Path(report_path)
    else:
        base = Path(dataset_dir)
        candidates = [
            base.parent / "with_cfa" / "cfa_quality_report.json",
            base / "cfa_quality_report.json",
            base.parent / "cfa_quality_report.json",
        ]
        rp = next((p for p in candidates if p.exists()), None)  # type: ignore[assignment]

    if rp is None or not Path(rp).exists():
        logger.warning(
            "cfa_quality_report.json not found -- "
            "Stage 3 (CFA generation) may not have run yet. "
            "Skipping CFA quality check (non-blocking)."
        )
        return (
            CheckResult(
                "cfa_quality_per_cwe",
                True,  # PASS: optional check, don't block training
                "SKIPPED -- cfa_quality_report.json not found (Stage 3 not yet run)",
            ),
            [],
        )

    try:
        report = json.loads(Path(rp).read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning(f"Failed to read cfa_quality_report.json: {exc}")
        return (
            CheckResult(
                "cfa_quality_per_cwe",
                True,
                f"SKIPPED -- could not read report: {exc}",
            ),
            [],
        )

    rows: list[CfaCheckRow] = []
    row_idx = 0

    for cwe, thresholds in sorted(CFA_QUALITY_THRESHOLDS.items()):
        cwe_data = report.get(cwe, {})
        if not cwe_data or cwe_data.get("total_attempts", 0) == 0:
            continue

        for metric, threshold in thresholds.items():
            if metric == "taint_break_rate":
                total = cwe_data.get("total_attempts", 0)
                taint_fails = (
                    cwe_data.get("rejection_breakdown", {})
                    .get("taint_path_intact", 0)
                )
                actual = 1.0 - (taint_fails / total) if total > 0 else 1.0
            else:
                actual = cwe_data.get(metric, 0.0)

            passed = actual >= threshold
            row_idx += 1
            rows.append(
                CfaCheckRow(
                    check_id=f"C10.{row_idx}",
                    cwe=cwe,
                    metric=metric,
                    actual=actual,
                    threshold=threshold,
                    passed=passed,
                )
            )

    all_pass = all(r.passed for r in rows)
    n_fail = sum(1 for r in rows if not r.passed)
    n_total = len(rows)

    detail = (
        f"{n_total - n_fail}/{n_total} metric checks passed"
        if rows
        else "no CWE data in report"
    )

    return CheckResult("cfa_quality_per_cwe", all_pass, detail), rows


def _print_cfa_quality_table(rows: list[CfaCheckRow]) -> None:
    """Print the per-CWE CFA quality table to stdout."""
    if not rows:
        return
    sep = "-" * 70
    print(f"\n{sep}")
    print("CFA Quality Check (Check 10)")
    print(sep)
    print(f"{'ID':<8} {'CWE':<10} {'Metric':<22} {'Actual':>7} {'Threshold':>10} {'Status':>6}")
    print(sep)
    for r in rows:
        print(
            f"{r.check_id:<8} {r.cwe:<10} {r.metric:<22} "
            f"{r.actual:>7.3f} {r.threshold:>10.3f} {r.status():>6}"
        )
    print(sep)


# -- Check 11: commit_sha leakage (optional) ----------------------------------
def check_commit_sha_leakage(
    splits: dict[str, dict],
    supplement_path: str | None = None,
) -> CheckResult:
    """
    Check 11: Verify zero commit_sha overlap between train and test.
    Requires supplement JSONL (deduped samples.jsonl) to look up SHAs.
    If supplement is absent, returns PASS with SKIPPED note.
    """
    sup_path = Path(supplement_path) if supplement_path else Path(DEFAULT_SUPPLEMENT)
    if not sup_path.exists():
        return CheckResult(
            "commit_sha_leakage",
            True,
            "SKIPPED -- supplement JSONL not found (no SHA check possible)",
        )

    # Build sample_id -> commit_sha lookup from JSONL
    all_ids: set[str] = set()
    for meta in splits.values():
        all_ids.update(meta["sample_ids"])

    sha_lookup: dict[str, str] = {}
    try:
        with open(sup_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                sid = row.get("id", "")
                if sid in all_ids:
                    sha = row.get("commit_sha", "")
                    if sha:
                        sha_lookup[sid] = sha
    except OSError as exc:
        return CheckResult(
            "commit_sha_leakage",
            True,
            f"SKIPPED -- could not read supplement: {exc}",
        )

    if not sha_lookup:
        return CheckResult(
            "commit_sha_leakage",
            True,
            "0 samples with commit_sha -- no leakage possible",
        )

    train_shas: set[str] = set()
    test_shas: set[str] = set()

    for sid in splits["train"]["sample_ids"]:
        sha = sha_lookup.get(sid, "")
        if sha:
            train_shas.add(sha)

    for sid in splits["test"]["sample_ids"]:
        sha = sha_lookup.get(sid, "")
        if sha:
            test_shas.add(sha)

    overlap = train_shas & test_shas
    n_overlap = len(overlap)

    return CheckResult(
        "commit_sha_leakage",
        n_overlap == 0,
        f"{n_overlap} SHA overlap between train ({len(train_shas)} SHAs) "
        f"and test ({len(test_shas)} SHAs)"
        + (f" -- first 3: {list(overlap)[:3]}" if overlap else ""),
    )


# -- Main audit runner ---------------------------------------------------------
def run_audit(
    dataset_dir: str,
    m1: bool = True,
    cfa_report_path: str | None = None,
    supplement_path: str | None = None,
) -> list[CheckResult]:
    """
    Run all 9 pre-training audit checks + Check 10 (CFA quality)
    + Check 11 (commit_sha leakage).

    Args:
        dataset_dir: path to directory with train.h5, val.h5, test.h5
        m1: if True, use M1 (minimum viable) thresholds; else M2
        cfa_report_path: explicit path to cfa_quality_report.json
        supplement_path: explicit path to deduped JSONL for SHA check

    Returns:
        list of CheckResult (9 structural + 2 optional)
    """
    thresholds = REQUIRED_CHECKS_M1 if m1 else REQUIRED_CHECKS_M2
    tier_label = "M1" if m1 else "M2"

    logger.info("Pre-Training Audit")
    logger.info(f"  Dataset: {dataset_dir}")
    logger.info(f"  Thresholds: {tier_label}")

    base = Path(dataset_dir)

    split_paths: dict[str, Path] = {
        "train": base / "train.h5",
        "val": base / "val.h5",
        "test": base / "test.h5",
    }

    # Verify all files exist
    for name, path in split_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing split file: {path}")

    # Load metadata from all splits
    splits: dict[str, dict] = {}
    for name, path in split_paths.items():
        logger.info(f"Loading {name} metadata from {path}...")
        splits[name] = _load_split_metadata(path)

    # Extract thresholds
    balance_range = thresholds["vuln_safe_balance"]
    code_range = thresholds["code_length_range"]

    logger.info(f"Running checks with {tier_label} thresholds...")
    results: list[CheckResult] = []

    results.append(check_min_train_samples(
        splits, threshold=thresholds["min_train_samples"]))
    results.append(check_vuln_safe_balance(
        splits, low=balance_range[0], high=balance_range[1]))
    results.append(check_cwe_diversity(
        splits, min_cwes=thresholds["cwe_diversity"], min_samples_per_cwe=500))
    results.append(check_max_cwe_dominance(
        splits, max_fraction=thresholds["max_cwe_dominance"]))
    results.append(check_no_null_code(split_paths))
    results.append(check_test_train_no_overlap(splits))
    results.append(check_code_length_range(
        split_paths, min_nodes=code_range[0], max_nodes=code_range[1]))
    results.append(check_pair_integrity(splits))
    results.append(check_min_taint_coverage(
        split_paths, min_coverage=thresholds["min_taint_coverage"]))

    # Check 10: CFA quality per CWE (optional -- PASS if report absent)
    cfa_result, cfa_rows = check_cfa_quality_per_cwe(
        dataset_dir, report_path=cfa_report_path
    )
    results.append(cfa_result)
    _print_cfa_quality_table(cfa_rows)

    # Check 11: commit_sha leakage (optional -- PASS if supplement absent)
    results.append(check_commit_sha_leakage(splits, supplement_path=supplement_path))

    # Per-CWE survival report (Challenge 3 monitoring)
    logger.info("")
    logger.info("Per-CWE sample counts:")
    all_cwes: list[str] = []
    for meta in splits.values():
        all_cwes.extend(meta["cwes"])
    cwe_counts = Counter(all_cwes)
    for cwe, count in cwe_counts.most_common():
        flag = " (LOW)" if count < 200 else ""
        logger.info(f"  {cwe}: {count}{flag}")

    # Report
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"PRE-TRAINING AUDIT RESULTS ({tier_label})")
    logger.info("=" * 60)

    n_pass = 0
    n_fail = 0
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        icon = "+" if r.passed else "-"
        logger.info(f"  [{icon}] {status} {r.name}: {r.detail}")
        if r.passed:
            n_pass += 1
        else:
            n_fail += 1

    logger.info("")
    logger.info(f"  {n_pass}/{len(results)} checks passed ({n_fail} failed)")
    logger.info("=" * 60)

    if n_fail == 0:
        logger.info("PRE-TRAINING AUDIT PASSED -- Safe to begin training.")
    else:
        logger.error("PRE-TRAINING AUDIT FAILED -- Fix before training.")

    return results


# -- CLI -----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Pre-Training Audit: quality checks on split HDF5 data",
    )
    parser.add_argument(
        "--dataset",
        default="training/data/final/",
        help="Directory containing train.h5, val.h5, test.h5",
    )
    parser.add_argument(
        "--m1",
        action="store_true",
        help="Use M1 (minimum viable) thresholds (default)",
    )
    parser.add_argument(
        "--m2",
        action="store_true",
        help="Use M2 (full dataset) thresholds",
    )
    parser.add_argument(
        "--cfa-report",
        default=None,
        help="Explicit path to cfa_quality_report.json",
    )
    parser.add_argument(
        "--supplement",
        default=None,
        help=(
            "Path to deduped JSONL for commit_sha leakage check "
            f"(default: {DEFAULT_SUPPLEMENT})"
        ),
    )

    args = parser.parse_args()

    # M2 takes precedence if both specified; default is M1
    use_m1 = True
    if args.m2:
        use_m1 = False

    results = run_audit(
        dataset_dir=args.dataset,
        m1=use_m1,
        cfa_report_path=args.cfa_report,
        supplement_path=args.supplement,
    )

    # Exit code: 0 = all pass, 1 = any fail
    if all(r.passed for r in results):
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
