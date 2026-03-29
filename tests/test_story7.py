#!/usr/bin/env python3
"""
tests/test_story7.py

Tests for Story 7: CFA-Aware Split + Pre-Training Audit.

Tests both stage7_split.py and pre_training_audit.py using synthetic HDF5 data.
Does NOT require the 3.7 GB production all_graphs.h5.

Covers all 6 identified challenges:
  1. CFA pair split integrity
  2. min_train_samples threshold
  3. CWE diversity
  4. Label balance with M1 relaxed range
  5. Singleton vs paired ratio in test set
  6. Empty/sentinel pair_id handling
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.scripts.preprocessing.stage7_split import (
    EMPTY_PAIR_SENTINELS,
    FEATURE_DIM,
    M1_BALANCE_RANGE,
    _assert_pair_integrity,
    _is_valid_pair_id,
    _vuln_ratio,
    build_groups,
    load_metadata,
    run_stage7,
    split_groups,
    split_paired_and_singletons,
    write_split_h5,
)
from training.scripts.preprocessing.pre_training_audit import (
    CheckResult,
    check_code_length_range,
    check_cwe_diversity,
    check_max_cwe_dominance,
    check_min_taint_coverage,
    check_min_train_samples,
    check_no_null_code,
    check_pair_integrity,
    check_test_train_no_overlap,
    check_vuln_safe_balance,
    run_audit,
)


# ══════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════

def _make_graph(n_nodes=10, n_edges=15, label=1, has_taint=True):
    """Create a synthetic graph dict for HDF5 writing."""
    x = np.random.randn(n_nodes, FEATURE_DIM).astype(np.float32)
    # Zero out taint region first so has_taint flag is authoritative
    x[:, 800:804] = 0.0
    if has_taint:
        x[0, 800] = 1.0
    src = np.random.randint(0, n_nodes, size=n_edges)
    dst = np.random.randint(0, n_nodes, size=n_edges)
    edge_index = np.stack([src, dst], axis=0).astype(np.int64)
    edge_attr = np.random.randint(0, 4, size=n_edges).astype(np.int64)
    return {
        "x": x,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "y": np.array([label], dtype=np.int64),
    }


def _write_h5(
    path: Path,
    sample_ids: list[str],
    labels: list[int],
    cwes: list[str],
    pair_ids: list[str],
    graphs: list[dict] | None = None,
) -> None:
    """Write a synthetic HDF5 file matching stage6 output format."""
    n = len(sample_ids)
    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, "w") as f:
        meta = f.create_group("metadata")
        meta.attrs["feature_dim"] = FEATURE_DIM
        meta.attrs["num_graphs"] = n
        meta.attrs["edge_types"] = 4

        dt_str = h5py.special_dtype(vlen=str)
        meta.create_dataset("sample_ids", data=sample_ids, dtype=dt_str)
        meta.create_dataset("labels", data=np.array(labels, dtype=np.int64))
        meta.create_dataset("cwes", data=cwes, dtype=dt_str)
        meta.create_dataset("pair_ids", data=pair_ids, dtype=dt_str)

        graphs_grp = f.create_group("graphs")
        for i in range(n):
            g = graphs[i] if graphs else _make_graph(label=labels[i])
            grp = graphs_grp.create_group(str(i))
            grp.create_dataset("x", data=g["x"], compression="gzip", compression_opts=4)
            grp.create_dataset("edge_index", data=g["edge_index"])
            grp.create_dataset("edge_attr", data=g["edge_attr"])
            grp.create_dataset("y", data=g["y"])


CWE_LIST = ["CWE-121", "CWE-122", "CWE-190", "CWE-476", "CWE-787", "CWE-416", "CWE-401"]


@pytest.fixture
def synthetic_h5(tmp_path):
    """
    Create a synthetic all_graphs.h5 with 200 samples:
      - 100 paired (50 pairs of 2), 100 singletons
      - ~50% vuln, 7 CWEs
    """
    n = 200
    rng = np.random.RandomState(42)

    sample_ids = [f"sample-{i:04d}" for i in range(n)]
    labels = [int(rng.random() > 0.5) for _ in range(n)]
    cwes = [CWE_LIST[i % len(CWE_LIST)] for i in range(n)]

    pair_ids = []
    for i in range(100):
        pair_ids.append(f"pair-{i // 2:04d}")
    pair_ids.extend(["" for _ in range(100)])

    h5_path = tmp_path / "all_graphs.h5"
    _write_h5(h5_path, sample_ids, labels, cwes, pair_ids)
    return h5_path


@pytest.fixture
def split_dir(tmp_path, synthetic_h5):
    """Run stage7 split on synthetic data and return output dir."""
    out_dir = tmp_path / "final"
    run_stage7(
        input_path=str(synthetic_h5),
        output_dir=str(out_dir),
        seed=42,
    )
    return out_dir


# ══════════════════════════════════════════════════════════════════
# Stage 7: Metadata Loading
# ══════════════════════════════════════════════════════════════════

class TestLoadMetadata:

    def test_loads_correct_count(self, synthetic_h5):
        meta = load_metadata(synthetic_h5)
        assert meta["num_graphs"] == 200

    def test_loads_sample_ids(self, synthetic_h5):
        meta = load_metadata(synthetic_h5)
        assert len(meta["sample_ids"]) == 200
        assert meta["sample_ids"][0] == "sample-0000"

    def test_loads_labels(self, synthetic_h5):
        meta = load_metadata(synthetic_h5)
        assert len(meta["labels"]) == 200
        assert all(l in (0, 1) for l in meta["labels"])

    def test_loads_cwes(self, synthetic_h5):
        meta = load_metadata(synthetic_h5)
        assert len(meta["cwes"]) == 200
        assert all(c.startswith("CWE-") for c in meta["cwes"])

    def test_loads_pair_ids(self, synthetic_h5):
        meta = load_metadata(synthetic_h5)
        assert len(meta["pair_ids"]) == 200
        assert meta["pair_ids"][0] != ""
        assert meta["pair_ids"][199] == ""


# ══════════════════════════════════════════════════════════════════
# Challenge 6: Empty/Sentinel Pair ID Handling
# ══════════════════════════════════════════════════════════════════

class TestPairIdValidation:

    def test_empty_string_is_invalid(self):
        assert not _is_valid_pair_id("")

    def test_none_string_is_invalid(self):
        assert not _is_valid_pair_id("None")

    def test_null_string_is_invalid(self):
        assert not _is_valid_pair_id("null")

    def test_zero_string_is_invalid(self):
        assert not _is_valid_pair_id("0")

    def test_real_pair_id_is_valid(self):
        assert _is_valid_pair_id("pair-0001")

    def test_uuid_is_valid(self):
        assert _is_valid_pair_id("abcd1234-5678-90ef")

    def test_all_sentinels(self):
        for sentinel in EMPTY_PAIR_SENTINELS:
            assert not _is_valid_pair_id(sentinel)


# ══════════════════════════════════════════════════════════════════
# Stage 7: Group Builder (returns tuple now)
# ══════════════════════════════════════════════════════════════════

class TestBuildGroups:

    def test_paired_and_singletons_separated(self):
        pair_ids = ["p1", "p1", "p2", "p2", "", ""]
        paired, singletons = build_groups(pair_ids)
        assert len(paired) == 2
        assert len(singletons) == 2

    def test_singletons_are_singleton(self):
        pair_ids = ["", "", ""]
        paired, singletons = build_groups(pair_ids)
        assert len(paired) == 0
        assert len(singletons) == 3

    def test_all_paired(self):
        pair_ids = ["p1", "p1", "p2", "p2"]
        paired, singletons = build_groups(pair_ids)
        assert len(paired) == 2
        assert len(singletons) == 0

    def test_deterministic_order(self):
        pair_ids = ["z", "z", "a", "a", "", ""]
        p1, s1 = build_groups(pair_ids)
        p2, s2 = build_groups(pair_ids)
        assert p1 == p2
        assert s1 == s2
        # "a" group should come before "z" group
        assert 2 in p1[0] or 3 in p1[0]

    def test_triplet_group(self):
        pair_ids = ["p1", "p1", "p1"]
        paired, singletons = build_groups(pair_ids)
        assert len(paired) == 1
        assert len(paired[0]) == 3

    def test_empty_input(self):
        paired, singletons = build_groups([])
        assert paired == []
        assert singletons == []

    def test_all_same_pair(self):
        pair_ids = ["p1"] * 10
        paired, singletons = build_groups(pair_ids)
        assert len(paired) == 1
        assert len(paired[0]) == 10

    def test_single_member_pair_becomes_singleton(self):
        """A pair_id with only 1 member should become a singleton."""
        pair_ids = ["p1", "p2", "p2"]
        paired, singletons = build_groups(pair_ids)
        assert len(paired) == 1  # only p2
        assert len(singletons) == 1  # p1 becomes singleton

    def test_sentinel_none_treated_as_empty(self):
        """Challenge 6: 'None' should not be treated as a valid pair_id."""
        pair_ids = ["None", "None", "real-pair", "real-pair"]
        paired, singletons = build_groups(pair_ids)
        assert len(paired) == 1  # only real-pair
        assert len(singletons) == 2  # both "None" are singletons

    def test_sentinel_zero_treated_as_empty(self):
        """Challenge 6: '0' should not be treated as a valid pair_id."""
        pair_ids = ["0", "0", "pair-1", "pair-1"]
        paired, singletons = build_groups(pair_ids)
        assert len(paired) == 1
        assert len(singletons) == 2


# ══════════════════════════════════════════════════════════════════
# Stage 7: Split Logic (simple flat list version)
# ══════════════════════════════════════════════════════════════════

class TestSplitGroups:

    def test_80_10_10_approximate(self):
        n = 200
        pair_ids = ["" for _ in range(n)]
        labels = [i % 2 for i in range(n)]
        _, singletons = build_groups(pair_ids)
        groups = singletons  # all singletons

        result = split_groups(groups, labels)
        train = result["train_indices"]
        val = result["val_indices"]
        test = result["test_indices"]

        assert abs(len(train) / n - 0.8) < 0.05
        assert abs(len(val) / n - 0.1) < 0.05
        assert abs(len(test) / n - 0.1) < 0.05

    def test_no_sample_lost(self):
        n = 100
        pair_ids = ["" for _ in range(n)]
        labels = [i % 2 for i in range(n)]
        _, singletons = build_groups(pair_ids)

        result = split_groups(singletons, labels)
        all_idx = (
            set(result["train_indices"])
            | set(result["val_indices"])
            | set(result["test_indices"])
        )
        assert all_idx == set(range(n))

    def test_no_overlap(self):
        n = 100
        pair_ids = ["" for _ in range(n)]
        labels = [i % 2 for i in range(n)]
        _, singletons = build_groups(pair_ids)

        result = split_groups(singletons, labels)
        train = set(result["train_indices"])
        val = set(result["val_indices"])
        test = set(result["test_indices"])

        assert len(train & val) == 0
        assert len(train & test) == 0
        assert len(val & test) == 0

    def test_deterministic_with_seed(self):
        n = 100
        pair_ids = ["" for _ in range(n)]
        labels = [i % 2 for i in range(n)]
        _, singletons = build_groups(pair_ids)

        r1 = split_groups(singletons, labels, seed=42)
        r2 = split_groups(singletons, labels, seed=42)
        assert r1["train_indices"] == r2["train_indices"]

    def test_different_seed_different_split(self):
        n = 100
        pair_ids = ["" for _ in range(n)]
        labels = [i % 2 for i in range(n)]
        _, singletons = build_groups(pair_ids)

        r1 = split_groups(singletons, labels, seed=42)
        r2 = split_groups(singletons, labels, seed=99)
        assert r1["train_indices"] != r2["train_indices"]

    def test_max_samples_limit(self):
        n = 200
        pair_ids = ["" for _ in range(n)]
        labels = [i % 2 for i in range(n)]
        _, singletons = build_groups(pair_ids)

        result = split_groups(singletons, labels, max_samples=50)
        total = (
            len(result["train_indices"])
            + len(result["val_indices"])
            + len(result["test_indices"])
        )
        assert total <= 50

    def test_custom_split_ratios(self):
        n = 200
        pair_ids = ["" for _ in range(n)]
        labels = [i % 2 for i in range(n)]
        _, singletons = build_groups(pair_ids)

        result = split_groups(singletons, labels, split_ratios=(0.7, 0.15, 0.15))
        train = result["train_indices"]
        assert abs(len(train) / n - 0.7) < 0.05


# ══════════════════════════════════════════════════════════════════
# Challenge 1: CFA Pair Integrity in Two-Phase Split
# ══════════════════════════════════════════════════════════════════

class TestTwoPhaseSplitPairIntegrity:

    def test_pairs_stay_together(self):
        """Challenge 1: paired samples must be in the same split."""
        pair_ids = []
        for i in range(50):
            pair_ids.extend([f"pair-{i:04d}"] * 2)
        pair_ids.extend([""] * 100)  # 100 singletons
        labels = [i % 2 for i in range(200)]

        paired, singletons = build_groups(pair_ids)
        result = split_paired_and_singletons(paired, singletons, labels)

        train = set(result["train_indices"])
        val = set(result["val_indices"])
        test = set(result["test_indices"])

        for i in range(0, 100, 2):
            if i in train:
                assert i + 1 in train
            elif i in val:
                assert i + 1 in val
            else:
                assert i in test and i + 1 in test

    def test_pair_stats_reported(self):
        """Challenge 5: pair_stats should report CFA pairs per split."""
        pair_ids = []
        for i in range(50):
            pair_ids.extend([f"pair-{i:04d}"] * 2)
        pair_ids.extend([""] * 100)
        labels = [i % 2 for i in range(200)]

        paired, singletons = build_groups(pair_ids)
        result = split_paired_and_singletons(paired, singletons, labels)

        assert "pair_stats" in result
        for name in ("train", "val", "test"):
            assert "cfa_pairs" in result["pair_stats"][name]
            assert "singletons" in result["pair_stats"][name]


# ══════════════════════════════════════════════════════════════════
# Stage 7: Vuln Ratio Helper
# ══════════════════════════════════════════════════════════════════

class TestVulnRatio:

    def test_all_vuln(self):
        assert _vuln_ratio([0, 1, 2], [1, 1, 1]) == 1.0

    def test_all_safe(self):
        assert _vuln_ratio([0, 1, 2], [0, 0, 0]) == 0.0

    def test_half(self):
        assert _vuln_ratio([0, 1], [0, 1]) == 0.5

    def test_empty(self):
        assert _vuln_ratio([], [0, 1]) == 0.0


# ══════════════════════════════════════════════════════════════════
# Stage 7: Pair Integrity Assert
# ══════════════════════════════════════════════════════════════════

class TestPairIntegrityAssert:

    def test_valid_split_passes(self):
        groups = [[0, 1], [2, 3], [4]]
        _assert_pair_integrity([0, 1], [2, 3], [4], groups)

    def test_broken_pair_raises(self):
        groups = [[0, 1], [2, 3]]
        with pytest.raises(ValueError, match="Pair integrity violation"):
            _assert_pair_integrity([0], [1, 2, 3], [], groups)


# ══════════════════════════════════════════════════════════════════
# Stage 7: HDF5 Streaming Copy
# ══════════════════════════════════════════════════════════════════

class TestWriteSplitH5:

    def test_writes_correct_subset(self, synthetic_h5, tmp_path):
        meta = load_metadata(synthetic_h5)
        indices = [0, 5, 10, 15, 20]
        dst = tmp_path / "subset.h5"

        write_split_h5(synthetic_h5, dst, indices, meta)
        assert dst.exists()

        with h5py.File(dst, "r") as f:
            assert f["metadata"].attrs["num_graphs"] == 5
            assert f["metadata"].attrs["feature_dim"] == FEATURE_DIM
            labels = f["metadata"]["labels"][:].tolist()
            expected_labels = [meta["labels"][i] for i in sorted(indices)]
            assert labels == expected_labels

    def test_graphs_readable(self, synthetic_h5, tmp_path):
        meta = load_metadata(synthetic_h5)
        indices = [0, 1, 2]
        dst = tmp_path / "test_read.h5"
        write_split_h5(synthetic_h5, dst, indices, meta)

        with h5py.File(dst, "r") as f:
            for idx in range(3):
                g = f["graphs"][str(idx)]
                x = g["x"][:]
                assert x.shape[1] == FEATURE_DIM
                assert x.shape[0] >= 3

    def test_atomic_write(self, synthetic_h5, tmp_path):
        meta = load_metadata(synthetic_h5)
        dst = tmp_path / "atomic.h5"
        tmp_file = dst.with_suffix(".h5.tmp")

        write_split_h5(synthetic_h5, dst, [0], meta)
        assert dst.exists()
        assert not tmp_file.exists()

    def test_empty_indices(self, synthetic_h5, tmp_path):
        meta = load_metadata(synthetic_h5)
        dst = tmp_path / "empty.h5"
        write_split_h5(synthetic_h5, dst, [], meta)
        assert dst.exists()

        with h5py.File(dst, "r") as f:
            assert f["metadata"].attrs["num_graphs"] == 0


# ══════════════════════════════════════════════════════════════════
# Stage 7: Full Pipeline (run_stage7)
# ══════════════════════════════════════════════════════════════════

class TestRunStage7:

    def test_creates_three_files(self, synthetic_h5, tmp_path):
        out_dir = tmp_path / "final"
        run_stage7(str(synthetic_h5), str(out_dir), seed=42)
        assert (out_dir / "train.h5").exists()
        assert (out_dir / "val.h5").exists()
        assert (out_dir / "test.h5").exists()
        assert (out_dir / "split_stats.json").exists()

    def test_stats_json_content(self, synthetic_h5, tmp_path):
        out_dir = tmp_path / "final"
        stats = run_stage7(str(synthetic_h5), str(out_dir), seed=42)
        assert stats["total_samples"] == 200
        assert stats["train_samples"] + stats["val_samples"] + stats["test_samples"] == 200
        assert stats["seed"] == 42
        assert "cwe_breakdown" in stats
        assert "pair_stats_per_split" in stats

    def test_dry_run_no_files(self, synthetic_h5, tmp_path):
        out_dir = tmp_path / "final"
        stats = run_stage7(str(synthetic_h5), str(out_dir), dry_run=True)
        assert stats["mode"] == "dry_run"
        assert not (out_dir / "train.h5").exists()

    def test_missing_input_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            run_stage7(str(tmp_path / "nonexistent.h5"), str(tmp_path / "out"))

    def test_max_samples(self, synthetic_h5, tmp_path):
        out_dir = tmp_path / "final"
        stats = run_stage7(str(synthetic_h5), str(out_dir), max_samples=50)
        total = stats["train_samples"] + stats["val_samples"] + stats["test_samples"]
        assert total <= 50

    def test_pair_integrity_in_output(self, synthetic_h5, tmp_path):
        """Challenge 1: verify no pair_id in multiple output splits."""
        out_dir = tmp_path / "final"
        run_stage7(str(synthetic_h5), str(out_dir), seed=42)

        pair_to_split = {}
        for split_name in ("train", "val", "test"):
            with h5py.File(out_dir / f"{split_name}.h5", "r") as f:
                pids = f["metadata"]["pair_ids"][:]
                for pid in pids:
                    pid_str = pid.decode("utf-8") if isinstance(pid, bytes) else str(pid)
                    if pid_str:
                        if pid_str in pair_to_split:
                            assert pair_to_split[pid_str] == split_name, (
                                f"pair_id {pid_str} in both "
                                f"{pair_to_split[pid_str]} and {split_name}"
                            )
                        pair_to_split[pid_str] = split_name

    def test_no_sample_overlap(self, synthetic_h5, tmp_path):
        out_dir = tmp_path / "final"
        run_stage7(str(synthetic_h5), str(out_dir), seed=42)

        all_ids: list[str] = []
        for split_name in ("train", "val", "test"):
            with h5py.File(out_dir / f"{split_name}.h5", "r") as f:
                ids = f["metadata"]["sample_ids"][:]
                for sid in ids:
                    s = sid.decode("utf-8") if isinstance(sid, bytes) else str(sid)
                    all_ids.append(s)

        assert len(all_ids) == len(set(all_ids))
        assert len(all_ids) == 200

    def test_total_graphs_preserved(self, synthetic_h5, tmp_path):
        out_dir = tmp_path / "final"
        run_stage7(str(synthetic_h5), str(out_dir), seed=42)

        total = 0
        for split_name in ("train", "val", "test"):
            with h5py.File(out_dir / f"{split_name}.h5", "r") as f:
                total += f["metadata"].attrs["num_graphs"]
        assert total == 200


# ══════════════════════════════════════════════════════════════════
# Audit: Check 1 — min_train_samples
# ══════════════════════════════════════════════════════════════════

class TestCheckMinTrainSamples:

    def test_pass(self):
        splits = {"train": {"num_graphs": 10000}}
        r = check_min_train_samples(splits, threshold=5000)
        assert r.passed

    def test_fail(self):
        splits = {"train": {"num_graphs": 100}}
        r = check_min_train_samples(splits, threshold=5000)
        assert not r.passed

    def test_exact_threshold(self):
        splits = {"train": {"num_graphs": 5000}}
        r = check_min_train_samples(splits, threshold=5000)
        assert r.passed


# ══════════════════════════════════════════════════════════════════
# Audit: Check 2 — vuln_safe_balance (M1 relaxed: 0.35-0.65)
# ══════════════════════════════════════════════════════════════════

class TestCheckVulnSafeBalance:

    def test_balanced_passes(self):
        splits = {
            "train": {"labels": [0, 1] * 50},
            "val": {"labels": [0, 1] * 10},
            "test": {"labels": [0, 1] * 10},
        }
        r = check_vuln_safe_balance(splits)
        assert r.passed

    def test_imbalanced_fails(self):
        splits = {
            "train": {"labels": [1] * 90 + [0] * 10},  # 0.9 ratio
            "val": {"labels": [0, 1] * 10},
            "test": {"labels": [0, 1] * 10},
        }
        r = check_vuln_safe_balance(splits)
        assert not r.passed

    def test_m1_boundary_035(self):
        """Challenge 4: M1 relaxed lower bound is 0.35."""
        splits = {
            "train": {"labels": [1] * 35 + [0] * 65},  # 0.35
            "val": {"labels": [1] * 35 + [0] * 65},
            "test": {"labels": [1] * 35 + [0] * 65},
        }
        r = check_vuln_safe_balance(splits, low=0.35, high=0.65)
        assert r.passed

    def test_m1_boundary_065(self):
        """Challenge 4: M1 relaxed upper bound is 0.65."""
        splits = {
            "train": {"labels": [1] * 65 + [0] * 35},  # 0.65
            "val": {"labels": [1] * 65 + [0] * 35},
            "test": {"labels": [1] * 65 + [0] * 35},
        }
        r = check_vuln_safe_balance(splits, low=0.35, high=0.65)
        assert r.passed

    def test_sard_ratio_408_passes_m1(self):
        """Challenge 4: SARD inherent ratio ~0.408 should pass M1."""
        n = 1000
        n_vuln = 408
        splits = {
            "train": {"labels": [1] * n_vuln + [0] * (n - n_vuln)},
            "val": {"labels": [1] * 41 + [0] * 59},
            "test": {"labels": [1] * 41 + [0] * 59},
        }
        r = check_vuln_safe_balance(splits, low=0.35, high=0.65)
        assert r.passed


# ══════════════════════════════════════════════════════════════════
# Audit: Check 3 — cwe_diversity
# ══════════════════════════════════════════════════════════════════

class TestCheckCWEDiversity:

    def test_diverse_passes(self):
        cwes = []
        for cwe in CWE_LIST:
            cwes.extend([cwe] * 600)
        splits = {"train": {"cwes": cwes}}
        r = check_cwe_diversity(splits, min_cwes=5, min_samples_per_cwe=500)
        assert r.passed

    def test_not_diverse_fails(self):
        splits = {"train": {"cwes": ["CWE-121"] * 1000}}
        r = check_cwe_diversity(splits, min_cwes=5, min_samples_per_cwe=500)
        assert not r.passed

    def test_cwes_below_min_samples(self):
        cwes = []
        for cwe in CWE_LIST:
            cwes.extend([cwe] * 100)
        splits = {"train": {"cwes": cwes}}
        r = check_cwe_diversity(splits, min_cwes=5, min_samples_per_cwe=500)
        assert not r.passed

    def test_m1_relaxed_4_cwes(self):
        """Challenge 3: M1 allows 4 CWEs (CWE-416/476 may be underrepresented)."""
        cwes = []
        for cwe in ["CWE-121", "CWE-122", "CWE-190", "CWE-787"]:
            cwes.extend([cwe] * 600)
        splits = {"train": {"cwes": cwes}}
        r = check_cwe_diversity(splits, min_cwes=4, min_samples_per_cwe=500)
        assert r.passed


# ══════════════════════════════════════════════════════════════════
# Audit: Check 4 — max_cwe_dominance
# ══════════════════════════════════════════════════════════════════

class TestCheckMaxCWEDominance:

    def test_no_dominance_passes(self):
        cwes = []
        for cwe in CWE_LIST:
            cwes.extend([cwe] * 100)
        splits = {"train": {"cwes": cwes}}
        r = check_max_cwe_dominance(splits, max_fraction=0.40)
        assert r.passed

    def test_one_cwe_dominates_fails(self):
        splits = {"train": {"cwes": ["CWE-121"] * 500 + ["CWE-122"] * 100}}
        r = check_max_cwe_dominance(splits, max_fraction=0.40)
        assert not r.passed

    def test_empty_fails(self):
        splits = {"train": {"cwes": []}}
        r = check_max_cwe_dominance(splits, max_fraction=0.40)
        assert not r.passed


# ══════════════════════════════════════════════════════════════════
# Audit: Check 5 — no_null_code
# ══════════════════════════════════════════════════════════════════

class TestCheckNoNullCode:

    def test_all_valid_passes(self, tmp_path):
        ids = [f"s{i}" for i in range(5)]
        _write_h5(
            tmp_path / "train.h5", ids,
            [1] * 5, ["CWE-121"] * 5, [""] * 5,
        )
        r = check_no_null_code({"train": tmp_path / "train.h5"})
        assert r.passed

    def test_tiny_graph_fails(self, tmp_path):
        tiny_graph = _make_graph(n_nodes=2, n_edges=1, label=1)
        _write_h5(
            tmp_path / "train.h5",
            ["tiny"], [1], ["CWE-121"], [""],
            graphs=[tiny_graph],
        )
        r = check_no_null_code({"train": tmp_path / "train.h5"})
        assert not r.passed


# ══════════════════════════════════════════════════════════════════
# Audit: Check 6 — test_train_no_overlap
# ══════════════════════════════════════════════════════════════════

class TestCheckNoOverlap:

    def test_no_overlap_passes(self):
        splits = {
            "train": {"sample_ids": ["a", "b", "c"]},
            "val": {"sample_ids": ["d", "e"]},
            "test": {"sample_ids": ["f", "g"]},
        }
        r = check_test_train_no_overlap(splits)
        assert r.passed

    def test_overlap_fails(self):
        splits = {
            "train": {"sample_ids": ["a", "b", "c"]},
            "val": {"sample_ids": ["c", "d"]},
            "test": {"sample_ids": ["e"]},
        }
        r = check_test_train_no_overlap(splits)
        assert not r.passed

    def test_triple_overlap(self):
        splits = {
            "train": {"sample_ids": ["x"]},
            "val": {"sample_ids": ["x"]},
            "test": {"sample_ids": ["x"]},
        }
        r = check_test_train_no_overlap(splits)
        assert not r.passed


# ══════════════════════════════════════════════════════════════════
# Audit: Check 7 — code_length_range
# ══════════════════════════════════════════════════════════════════

class TestCheckCodeLengthRange:

    def test_normal_passes(self, tmp_path):
        _write_h5(
            tmp_path / "train.h5",
            ["s1", "s2"], [1, 0], ["CWE-121"] * 2, [""] * 2,
        )
        r = check_code_length_range({"train": tmp_path / "train.h5"})
        assert r.passed

    def test_too_large_fails(self, tmp_path):
        huge_graph = _make_graph(n_nodes=5000, n_edges=100, label=1)
        _write_h5(
            tmp_path / "train.h5",
            ["huge"], [1], ["CWE-121"], [""],
            graphs=[huge_graph],
        )
        r = check_code_length_range({"train": tmp_path / "train.h5"})
        assert not r.passed


# ══════════════════════════════════════════════════════════════════
# Audit: Check 8 — pair_integrity
# ══════════════════════════════════════════════════════════════════

class TestCheckPairIntegrity:

    def test_intact_passes(self):
        splits = {
            "train": {"pair_ids": ["p1", "p1", "p2", "p2"]},
            "val": {"pair_ids": ["p3", "p3"]},
            "test": {"pair_ids": ["p4", "p4"]},
        }
        r = check_pair_integrity(splits)
        assert r.passed

    def test_broken_fails(self):
        splits = {
            "train": {"pair_ids": ["p1"]},
            "val": {"pair_ids": ["p1"]},
            "test": {"pair_ids": []},
        }
        r = check_pair_integrity(splits)
        assert not r.passed

    def test_empty_pair_ids_ignored(self):
        splits = {
            "train": {"pair_ids": ["", "", "p1", "p1"]},
            "val": {"pair_ids": ["", ""]},
            "test": {"pair_ids": [""]},
        }
        r = check_pair_integrity(splits)
        assert r.passed


# ══════════════════════════════════════════════════════════════════
# Audit: Check 9 — min_taint_coverage
# ══════════════════════════════════════════════════════════════════

class TestCheckMinTaintCoverage:

    def test_good_coverage_passes(self, tmp_path):
        _write_h5(
            tmp_path / "train.h5",
            [f"s{i}" for i in range(10)],
            [1] * 10, ["CWE-121"] * 10, [""] * 10,
        )
        r = check_min_taint_coverage({"train": tmp_path / "train.h5"}, min_coverage=0.20)
        assert r.passed

    def test_no_taint_fails(self, tmp_path):
        graphs = [_make_graph(has_taint=False) for _ in range(10)]
        _write_h5(
            tmp_path / "train.h5",
            [f"s{i}" for i in range(10)],
            [1] * 10, ["CWE-121"] * 10, [""] * 10,
            graphs=graphs,
        )
        r = check_min_taint_coverage({"train": tmp_path / "train.h5"}, min_coverage=0.20)
        assert not r.passed


# ══════════════════════════════════════════════════════════════════
# Audit: Full run_audit Integration
# ══════════════════════════════════════════════════════════════════

class TestRunAudit:

    def _make_audit_dataset(self, tmp_path, n_train=100, n_val=15, n_test=15):
        out = tmp_path / "audit_data"

        rng = np.random.RandomState(42)
        for split_name, n in [("train", n_train), ("val", n_val), ("test", n_test)]:
            ids = [f"{split_name}-{i:04d}" for i in range(n)]
            labels = [int(rng.random() > 0.5) for _ in range(n)]
            cwes = [CWE_LIST[i % len(CWE_LIST)] for i in range(n)]
            pair_ids = [f"{split_name}-pair-{i // 2:04d}" for i in range(0, n, 1)]
            _write_h5(out / f"{split_name}.h5", ids, labels, cwes, pair_ids)

        return out

    def test_returns_9_results(self, tmp_path):
        dataset = self._make_audit_dataset(tmp_path)
        results = run_audit(str(dataset))
        assert len(results) >= 9  # 9 structural + optional Check 10 (CFA quality)
        assert all(isinstance(r, CheckResult) for r in results)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            run_audit(str(tmp_path / "nonexistent"))

    def test_check_names_are_correct(self, tmp_path):
        dataset = self._make_audit_dataset(tmp_path)
        results = run_audit(str(dataset))
        names = [r.name for r in results]
        assert "min_train_samples" in names
        assert "vuln_safe_balance" in names
        assert "cwe_diversity" in names
        assert "max_cwe_dominance" in names
        assert "no_null_code" in names
        assert "test_train_no_overlap" in names
        assert "code_length_range" in names
        assert "pair_integrity" in names
        assert "min_taint_coverage" in names


# ══════════════════════════════════════════════════════════════════
# Integration: stage7 -> audit pipeline
# ══════════════════════════════════════════════════════════════════

class TestStage7ToAudit:

    def test_split_then_audit(self, synthetic_h5, tmp_path):
        """Run full split, then audit — verify pipeline works end-to-end."""
        out_dir = tmp_path / "final"
        stats = run_stage7(str(synthetic_h5), str(out_dir), seed=42)

        results = run_audit(str(out_dir))
        assert len(results) >= 9  # 9 structural + optional Check 10

        result_map = {r.name: r for r in results}

        # These should always pass for our synthetic data
        assert result_map["test_train_no_overlap"].passed
        assert result_map["pair_integrity"].passed
        assert result_map["no_null_code"].passed
        assert result_map["code_length_range"].passed
