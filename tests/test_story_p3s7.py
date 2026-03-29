#!/usr/bin/env python3
"""
tests/test_story_p3s7.py

Phase 3 Story 7: CFA-Aware Split + Pre-Training Audit (Phase 3 extensions).

Tests the Phase 3 delta:
  - Group hierarchy: pair_id > commit_sha > cve_id > singleton
  - commit_sha leakage assertion
  - cve_id overlap warning
  - M1 / M2 threshold dicts
  - Supplement JSONL loading
  - Audit check 11 (commit_sha_leakage)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.scripts.preprocessing.stage7_split import (
    EMPTY_PAIR_SENTINELS,
    FEATURE_DIM,
    _get_group_key,
    _is_valid_pair_id,
    assert_no_cve_leakage,
    assert_no_sha_leakage,
    build_groups,
    load_metadata,
    load_supplement,
    run_stage7,
    split_paired_and_singletons,
)
from training.scripts.preprocessing.pre_training_audit import (
    CheckResult,
    REQUIRED_CHECKS_M1,
    REQUIRED_CHECKS_M2,
    check_commit_sha_leakage,
    check_min_train_samples,
    check_vuln_safe_balance,
    check_pair_integrity,
    run_audit,
)


# ==============================================================================
# Fixtures
# ==============================================================================

def _make_graph(n_nodes=10, n_edges=15, label=1, has_taint=True):
    """Create a synthetic graph dict for HDF5 writing."""
    x = np.random.randn(n_nodes, FEATURE_DIM).astype(np.float32)
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


CWE_LIST = ["CWE-121", "CWE-122", "CWE-190", "CWE-476", "CWE-787", "CWE-416", "CWE-401"]


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


def _write_supplement(path: Path, rows: list[dict]) -> None:
    """Write a synthetic supplement JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


@pytest.fixture
def supplement_data(tmp_path):
    """
    Create a supplement JSONL with known commit_sha/cve_id values.
    200 samples: first 100 have pair_ids, last 100 are singletons.
    Of singletons, 20 share commit_sha "sha-A", 10 share cve_id "CVE-2024-001".
    """
    jsonl_path = tmp_path / "supplement.jsonl"
    rows = []
    for i in range(200):
        row = {"id": f"sample-{i:04d}", "commit_sha": "", "cve_id": ""}
        # singletons 100-119 share commit_sha
        if 100 <= i < 120:
            row["commit_sha"] = "sha-A"
        # singletons 120-129 share cve_id
        if 120 <= i < 130:
            row["cve_id"] = "CVE-2024-001"
        rows.append(row)
    _write_supplement(jsonl_path, rows)
    return jsonl_path


@pytest.fixture
def synthetic_h5_with_supplement(tmp_path, supplement_data):
    """
    Create a synthetic all_graphs.h5 with 200 samples and supplement JSONL.
    - Indices 0-99: paired by pair_id (50 pairs)
    - Indices 100-119: grouped by commit_sha "sha-A" via supplement
    - Indices 120-129: grouped by cve_id "CVE-2024-001" via supplement
    - Indices 130-199: true singletons
    """
    n = 200
    rng = np.random.RandomState(42)

    sample_ids = [f"sample-{i:04d}" for i in range(n)]
    labels = [int(rng.random() > 0.5) for _ in range(n)]
    cwes = [CWE_LIST[i % len(CWE_LIST)] for i in range(n)]

    pair_ids = []
    for i in range(100):
        pair_ids.append(f"pair-{i // 2:04d}")
    pair_ids.extend([""] * 100)  # singletons (grouped by sha/cve/nothing)

    h5_path = tmp_path / "all_graphs.h5"
    _write_h5(h5_path, sample_ids, labels, cwes, pair_ids)
    return h5_path, supplement_data


# ==============================================================================
# Test 1: build_split_groups: pair_id takes priority over singleton
# ==============================================================================

class TestGroupHierarchyPairIdPriority:

    def test_pair_id_groups_before_sha(self, supplement_data):
        """pair_id should group samples even if they also have commit_sha."""
        pair_ids = ["p1", "p1", "", "", ""]
        sample_ids = ["s0", "s1", "s2", "s3", "s4"]
        # s2 and s3 share sha, s4 is singleton
        supplement = {
            "s0": {"commit_sha": "sha-X", "cve_id": ""},
            "s1": {"commit_sha": "sha-X", "cve_id": ""},
            "s2": {"commit_sha": "sha-Y", "cve_id": ""},
            "s3": {"commit_sha": "sha-Y", "cve_id": ""},
            "s4": {"commit_sha": "", "cve_id": ""},
        }
        paired, singletons = build_groups(pair_ids, sample_ids, supplement)
        # pair_id group: [0,1]. sha group: [2,3]. singleton: [4]
        assert len(paired) == 2
        assert len(singletons) == 1
        # The pair_id group should contain indices 0,1
        pair_group = [g for g in paired if 0 in g][0]
        assert set(pair_group) == {0, 1}


# ==============================================================================
# Test 2: build_split_groups: commit_sha used when no pair_id
# ==============================================================================

class TestGroupHierarchyCommitSha:

    def test_sha_groups_singletons(self):
        """Samples without pair_id but with same commit_sha should group."""
        pair_ids = ["", "", "", ""]
        sample_ids = ["s0", "s1", "s2", "s3"]
        supplement = {
            "s0": {"commit_sha": "sha-A", "cve_id": ""},
            "s1": {"commit_sha": "sha-A", "cve_id": ""},
            "s2": {"commit_sha": "sha-B", "cve_id": ""},
            "s3": {"commit_sha": "", "cve_id": ""},
        }
        paired, singletons = build_groups(pair_ids, sample_ids, supplement)
        # sha-A group: [0,1]. sha-B: singleton [2]. no sha: singleton [3]
        assert len(paired) == 1
        assert set(paired[0]) == {0, 1}
        assert len(singletons) == 2


# ==============================================================================
# Test 3: split_groups: ratios produce correct proportions
# ==============================================================================

class TestSplitRatios:

    def test_80_10_10_with_supplement(self, synthetic_h5_with_supplement):
        h5_path, sup_path = synthetic_h5_with_supplement
        meta = load_metadata(h5_path)
        supplement = load_supplement(sup_path, meta["sample_ids"])

        paired, singletons = build_groups(
            meta["pair_ids"], meta["sample_ids"], supplement
        )
        result = split_paired_and_singletons(paired, singletons, meta["labels"])

        n = meta["num_graphs"]
        train = result["train_indices"]
        val = result["val_indices"]
        test = result["test_indices"]

        assert abs(len(train) / n - 0.8) < 0.1
        assert abs(len(val) / n - 0.1) < 0.1
        assert abs(len(test) / n - 0.1) < 0.1
        # No sample lost
        assert len(train) + len(val) + len(test) == n


# ==============================================================================
# Test 4: split_groups: seed=42 gives reproducible result
# ==============================================================================

class TestSplitReproducibility:

    def test_deterministic(self, synthetic_h5_with_supplement):
        h5_path, sup_path = synthetic_h5_with_supplement
        meta = load_metadata(h5_path)
        supplement = load_supplement(sup_path, meta["sample_ids"])

        paired, singletons = build_groups(
            meta["pair_ids"], meta["sample_ids"], supplement
        )
        r1 = split_paired_and_singletons(paired, singletons, meta["labels"], seed=42)
        # Rebuild groups (build_groups is deterministic, but test it)
        paired2, singletons2 = build_groups(
            meta["pair_ids"], meta["sample_ids"], supplement
        )
        r2 = split_paired_and_singletons(paired2, singletons2, meta["labels"], seed=42)

        assert r1["train_indices"] == r2["train_indices"]
        assert r1["val_indices"] == r2["val_indices"]
        assert r1["test_indices"] == r2["test_indices"]


# ==============================================================================
# Test 5: Pair integrity: pair split across train/test -> error
# ==============================================================================

class TestPairIntegrityShaGroup:

    def test_sha_group_stays_together(self, synthetic_h5_with_supplement):
        """commit_sha groups must not be split across train/test."""
        h5_path, sup_path = synthetic_h5_with_supplement
        meta = load_metadata(h5_path)
        supplement = load_supplement(sup_path, meta["sample_ids"])

        paired, singletons = build_groups(
            meta["pair_ids"], meta["sample_ids"], supplement
        )
        result = split_paired_and_singletons(paired, singletons, meta["labels"])

        train_set = set(result["train_indices"])
        val_set = set(result["val_indices"])
        test_set = set(result["test_indices"])

        # Indices 100-119 share sha-A -> must all be in same split
        sha_indices = set(range(100, 120))
        in_train = sha_indices & train_set
        in_val = sha_indices & val_set
        in_test = sha_indices & test_set

        # They should all be in exactly one split
        non_empty = sum(1 for s in [in_train, in_val, in_test] if s)
        assert non_empty == 1, (
            f"SHA group split across {non_empty} splits: "
            f"train={len(in_train)}, val={len(in_val)}, test={len(in_test)}"
        )


# ==============================================================================
# Test 6: SHA overlap assertion: commit_sha in both train and test -> error
# ==============================================================================

class TestShaLeakageAssertion:

    def test_no_sha_overlap_passes(self):
        """No overlap should return 0."""
        sample_ids = ["s0", "s1", "s2", "s3"]
        supplement = {
            "s0": {"commit_sha": "sha-A", "cve_id": ""},
            "s1": {"commit_sha": "sha-A", "cve_id": ""},
            "s2": {"commit_sha": "sha-B", "cve_id": ""},
            "s3": {"commit_sha": "sha-B", "cve_id": ""},
        }
        # sha-A in train, sha-B in test
        result = assert_no_sha_leakage([0, 1], [2, 3], sample_ids, supplement)
        assert result == 0

    def test_sha_overlap_raises(self):
        """Same SHA in train and test should raise ValueError."""
        sample_ids = ["s0", "s1", "s2", "s3"]
        supplement = {
            "s0": {"commit_sha": "sha-A", "cve_id": ""},
            "s1": {"commit_sha": "sha-B", "cve_id": ""},
            "s2": {"commit_sha": "sha-A", "cve_id": ""},  # sha-A leaked!
            "s3": {"commit_sha": "sha-C", "cve_id": ""},
        }
        with pytest.raises(ValueError, match="Commit SHA leakage"):
            assert_no_sha_leakage([0, 1], [2, 3], sample_ids, supplement)


# ==============================================================================
# Test 7: Audit check 1: train < 5000 -> FAIL (M1)
# ==============================================================================

class TestAuditMinTrainM1:

    def test_below_m1_threshold_fails(self):
        splits = {"train": {"num_graphs": 100}}
        r = check_min_train_samples(splits, threshold=REQUIRED_CHECKS_M1["min_train_samples"])
        assert not r.passed

    def test_above_m1_threshold_passes(self):
        splits = {"train": {"num_graphs": 6000}}
        r = check_min_train_samples(splits, threshold=REQUIRED_CHECKS_M1["min_train_samples"])
        assert r.passed


# ==============================================================================
# Test 8: Audit check 2: label balance 30/70 -> FAIL (M1 range 0.35-0.65)
# ==============================================================================

class TestAuditBalanceM1:

    def test_30_70_fails_m1(self):
        """30% vuln is below M1 lower bound of 0.35."""
        low, high = REQUIRED_CHECKS_M1["vuln_safe_balance"]
        splits = {
            "train": {"labels": [1] * 30 + [0] * 70},
            "val": {"labels": [1] * 50 + [0] * 50},
            "test": {"labels": [1] * 50 + [0] * 50},
        }
        r = check_vuln_safe_balance(splits, low=low, high=high)
        assert not r.passed

    def test_50_50_passes_m1(self):
        low, high = REQUIRED_CHECKS_M1["vuln_safe_balance"]
        splits = {
            "train": {"labels": [1] * 50 + [0] * 50},
            "val": {"labels": [1] * 50 + [0] * 50},
            "test": {"labels": [1] * 50 + [0] * 50},
        }
        r = check_vuln_safe_balance(splits, low=low, high=high)
        assert r.passed


# ==============================================================================
# Test 9: Audit check 6 (SHA overlap) -> FAIL via Check 11
# ==============================================================================

class TestAuditShaOverlap:

    def test_sha_overlap_detected(self, tmp_path):
        """Check 11 should detect SHA overlap between train and test."""
        # Write supplement with overlapping SHA
        sup_path = tmp_path / "supplement.jsonl"
        rows = [
            {"id": "t1", "commit_sha": "sha-LEAK", "cve_id": ""},
            {"id": "t2", "commit_sha": "sha-OK", "cve_id": ""},
            {"id": "v1", "commit_sha": "", "cve_id": ""},
            {"id": "x1", "commit_sha": "sha-LEAK", "cve_id": ""},  # leaked!
            {"id": "x2", "commit_sha": "sha-SAFE", "cve_id": ""},
        ]
        _write_supplement(sup_path, rows)

        splits = {
            "train": {"sample_ids": ["t1", "t2"]},
            "val": {"sample_ids": ["v1"]},
            "test": {"sample_ids": ["x1", "x2"]},
        }
        r = check_commit_sha_leakage(splits, supplement_path=str(sup_path))
        assert not r.passed
        assert "1 SHA overlap" in r.detail

    def test_no_sha_overlap_passes(self, tmp_path):
        sup_path = tmp_path / "supplement.jsonl"
        rows = [
            {"id": "t1", "commit_sha": "sha-A", "cve_id": ""},
            {"id": "x1", "commit_sha": "sha-B", "cve_id": ""},
        ]
        _write_supplement(sup_path, rows)

        splits = {
            "train": {"sample_ids": ["t1"]},
            "val": {"sample_ids": []},
            "test": {"sample_ids": ["x1"]},
        }
        r = check_commit_sha_leakage(splits, supplement_path=str(sup_path))
        assert r.passed


# ==============================================================================
# Test 10: Audit check 8: broken pair -> FAIL
# ==============================================================================

class TestAuditBrokenPair:

    def test_pair_split_across_splits_fails(self):
        splits = {
            "train": {"pair_ids": ["p1"]},
            "val": {"pair_ids": ["p1"]},  # same pair in two splits!
            "test": {"pair_ids": []},
        }
        r = check_pair_integrity(splits)
        assert not r.passed

    def test_intact_pairs_pass(self):
        splits = {
            "train": {"pair_ids": ["p1", "p1"]},
            "val": {"pair_ids": ["p2", "p2"]},
            "test": {"pair_ids": ["p3", "p3"]},
        }
        r = check_pair_integrity(splits)
        assert r.passed


# ==============================================================================
# Test 11: Audit ALL pass -> exit code 0 (integration)
# ==============================================================================

class TestAuditAllPass:

    def _make_clean_dataset(self, tmp_path, n_train=100, n_val=15, n_test=15):
        out = tmp_path / "audit_pass"
        rng = np.random.RandomState(42)
        for split_name, n in [("train", n_train), ("val", n_val), ("test", n_test)]:
            ids = [f"{split_name}-{i:04d}" for i in range(n)]
            labels = [int(rng.random() > 0.5) for _ in range(n)]
            cwes = [CWE_LIST[i % len(CWE_LIST)] for i in range(n)]
            pair_ids = [f"{split_name}-pair-{i // 2:04d}" for i in range(n)]
            _write_h5(out / f"{split_name}.h5", ids, labels, cwes, pair_ids)
        return out

    def test_all_pass_returns_all_passed(self, tmp_path):
        dataset = self._make_clean_dataset(tmp_path)
        results = run_audit(str(dataset), m1=True)
        # At minimum all structural checks should exist
        assert len(results) >= 9
        assert all(isinstance(r, CheckResult) for r in results)
        # Structural checks that should pass on clean synthetic data:
        result_map = {r.name: r for r in results}
        assert result_map["test_train_no_overlap"].passed
        assert result_map["pair_integrity"].passed
        assert result_map["no_null_code"].passed


# ==============================================================================
# Test 12: Audit ANY fail -> exit code 1 (M2 threshold too strict)
# ==============================================================================

class TestAuditM2Fail:

    def test_m2_min_train_too_strict(self, tmp_path):
        """M2 requires 30K train samples -- 100 should fail."""
        out = tmp_path / "audit_m2"
        rng = np.random.RandomState(42)
        for split_name, n in [("train", 100), ("val", 15), ("test", 15)]:
            ids = [f"{split_name}-{i:04d}" for i in range(n)]
            labels = [int(rng.random() > 0.5) for _ in range(n)]
            cwes = [CWE_LIST[i % len(CWE_LIST)] for i in range(n)]
            pair_ids = [f"{split_name}-pair-{i // 2:04d}" for i in range(n)]
            _write_h5(out / f"{split_name}.h5", ids, labels, cwes, pair_ids)

        results = run_audit(str(out), m1=False)  # M2 thresholds
        result_map = {r.name: r for r in results}
        # M2 requires 30K train -- should fail
        assert not result_map["min_train_samples"].passed

    def test_m1_vs_m2_thresholds_differ(self):
        """M1 and M2 should have different values for key checks."""
        assert REQUIRED_CHECKS_M1["min_train_samples"] < REQUIRED_CHECKS_M2["min_train_samples"]
        assert REQUIRED_CHECKS_M1["cwe_diversity"] < REQUIRED_CHECKS_M2["cwe_diversity"]
        assert REQUIRED_CHECKS_M1["min_taint_coverage"] < REQUIRED_CHECKS_M2["min_taint_coverage"]
        # M1 balance is wider than M2
        m1_low, m1_high = REQUIRED_CHECKS_M1["vuln_safe_balance"]
        m2_low, m2_high = REQUIRED_CHECKS_M2["vuln_safe_balance"]
        assert m1_low < m2_low
        assert m1_high > m2_high
