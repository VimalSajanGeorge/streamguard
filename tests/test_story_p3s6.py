#!/usr/bin/env python3
"""
tests/test_story_p3s6.py

Phase 3 Stage 6 tests: Extensions to graph tensor assembly.

Tests:
  1. pair_id saved as HDF5 per-graph attribute and readable on reload
  2. Contiguous node index remap: edge_index.max() < num_nodes
  3. Gate 4: edge_type=4 in ALL edges -> entire graph rejected
  4. Post-write pair check: broken pair logged as warning (not error)
  5. Per-CWE distribution in graph_stats.json
  6. All 4 per-graph attrs (pair_id, cwe, source, sample_id) round-trip
  7. Rejection log written with gate reason
  8. Gate 6: all-dangling edges -> graph rejected
  9. Gate 7: no edges at all -> graph rejected
 10. run_stage6 integration with check_pairs=True
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.scripts.preprocessing.stage6_graphs import (
    FEATURE_DIM,
    MIN_NODES,
    VALID_EDGE_TYPES,
    check_pair_integrity,
    load_sample,
    run_stage6,
    verify_h5,
    write_graphs_h5,
)


# ── Helpers ──────────────────────────────────────────────────────

def _make_cpg(sample_id, label, cwe, pair_id="", source="sard", n_nodes=5):
    """Create a minimal valid CPG dict."""
    nodes = [
        {"id": str(i), "_label": "CALL", "code": f"func_{i}()", "name": f"func_{i}",
         "line": i, "type_full": "", "taint_role": "NONE"}
        for i in range(n_nodes)
    ]
    edges = [
        {"src": str(i), "dst": str(i + 1), "type": 0, "label": "AST"}
        for i in range(n_nodes - 1)
    ]
    # Add a CFG edge
    if n_nodes >= 3:
        edges.append({"src": "0", "dst": "2", "type": 1, "label": "CFG"})
    return {
        "sample_id": sample_id,
        "label": label,
        "cwe": cwe,
        "pair_id": pair_id,
        "source": source,
        "nodes": nodes,
        "edges": edges,
    }


def _write_cpg_and_npz(tmp_path, cpg_dict, n_nodes=None):
    """Write CPG JSON + random .npz to disk, return (cpg_path, npz_path)."""
    sample_id = cpg_dict["sample_id"]
    if n_nodes is None:
        n_nodes = len(cpg_dict["nodes"])
    features = np.random.randn(n_nodes, FEATURE_DIM).astype(np.float32)

    cpg_path = tmp_path / f"{sample_id}.json"
    cpg_path.write_text(json.dumps(cpg_dict), encoding="utf-8")
    npz_path = tmp_path / f"{sample_id}.npz"
    np.savez_compressed(npz_path, node_features=features)
    return cpg_path, npz_path


def _make_sample_dict(sample_id, label, cwe, pair_id="", source="test", n_nodes=10, n_edges=15):
    """Create a validated sample dict for write_graphs_h5."""
    x = np.random.randn(n_nodes, FEATURE_DIM).astype(np.float32)
    src = np.random.randint(0, n_nodes, size=n_edges)
    dst = np.random.randint(0, n_nodes, size=n_edges)
    edge_index = np.stack([src, dst], axis=0).astype(np.int64)
    edge_attr = np.random.randint(0, 4, size=n_edges).astype(np.int64)
    return {
        "sample_id": sample_id,
        "label": label,
        "cwe": cwe,
        "pair_id": pair_id,
        "source": source,
        "x": x,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "num_nodes": n_nodes,
        "num_edges": n_edges,
    }


# ══════════════════════════════════════════════════════════════════
# Test 1: pair_id saved as HDF5 per-graph attribute
# ══════════════════════════════════════════════════════════════════

class TestPairIdAttr:

    def test_pair_id_written_and_readable(self, tmp_path):
        """pair_id is saved as HDF5 attr on each graph group."""
        samples = [
            _make_sample_dict("s1", 1, "CWE-121", pair_id="pair-abc-123"),
            _make_sample_dict("s2", 0, "CWE-121", pair_id="pair-abc-123"),
            _make_sample_dict("s3", 1, "CWE-122", pair_id=""),
        ]
        h5_path = tmp_path / "test.h5"
        write_graphs_h5(samples, h5_path)

        with h5py.File(h5_path, "r") as f:
            g0 = f["graphs"]["0"]
            assert g0.attrs["pair_id"] == "pair-abc-123"
            g1 = f["graphs"]["1"]
            assert g1.attrs["pair_id"] == "pair-abc-123"
            g2 = f["graphs"]["2"]
            assert g2.attrs["pair_id"] == ""

    def test_all_four_attrs_roundtrip(self, tmp_path):
        """All per-graph attrs (pair_id, cwe, source, sample_id) survive write/read."""
        samples = [
            _make_sample_dict("uuid-001", 1, "CWE-190", pair_id="pair-x", source="cve"),
        ]
        h5_path = tmp_path / "test.h5"
        write_graphs_h5(samples, h5_path)

        with h5py.File(h5_path, "r") as f:
            g = f["graphs"]["0"]
            assert g.attrs["pair_id"] == "pair-x"
            assert g.attrs["cwe"] == "CWE-190"
            assert g.attrs["source"] == "cve"
            assert g.attrs["sample_id"] == "uuid-001"


# ══════════════════════════════════════════════════════════════════
# Test 2: Contiguous node index remap
# ══════════════════════════════════════════════════════════════════

class TestContiguousRemap:

    def test_edge_index_max_less_than_num_nodes(self, tmp_path):
        """After remap, edge_index.max() must be < num_nodes."""
        cpg = {
            "sample_id": "remap-test",
            "label": 1, "cwe": "CWE-121", "pair_id": "", "source": "sard",
            "nodes": [
                {"id": "100", "_label": "METHOD", "code": "f()", "name": "f",
                 "line": 1, "type_full": "", "taint_role": "NONE"},
                {"id": "200", "_label": "CALL", "code": "g()", "name": "g",
                 "line": 2, "type_full": "", "taint_role": "NONE"},
                {"id": "300", "_label": "CALL", "code": "h()", "name": "h",
                 "line": 3, "type_full": "", "taint_role": "NONE"},
                {"id": "400", "_label": "RETURN", "code": "return", "name": "",
                 "line": 4, "type_full": "", "taint_role": "NONE"},
            ],
            "edges": [
                {"src": "100", "dst": "200", "type": 0, "label": "AST"},
                {"src": "200", "dst": "300", "type": 1, "label": "CFG"},
                {"src": "300", "dst": "400", "type": 2, "label": "DFG"},
                {"src": "100", "dst": "400", "type": 0, "label": "AST"},
            ],
        }
        n_nodes = len(cpg["nodes"])
        features = np.random.randn(n_nodes, FEATURE_DIM).astype(np.float32)
        cpg_path = tmp_path / "remap-test.json"
        cpg_path.write_text(json.dumps(cpg), encoding="utf-8")
        npz_path = tmp_path / "remap-test.npz"
        np.savez_compressed(npz_path, node_features=features)

        result, gate = load_sample(cpg_path, npz_path)
        assert result is not None
        assert gate == ""
        # Node IDs were 100, 200, 300, 400 -> remapped to 0, 1, 2, 3
        assert result["edge_index"].max() < n_nodes
        assert result["edge_index"].min() >= 0


# ══════════════════════════════════════════════════════════════════
# Test 3: Gate 4 — edge_type >= 4 causes rejection when ALL edges bad
# ══════════════════════════════════════════════════════════════════

class TestGate4EdgeType:

    def test_all_edges_bad_type_rejects_graph(self, tmp_path):
        """If ALL edges have type >= 4, the graph is rejected via gate4."""
        cpg = {
            "sample_id": "gate4-test",
            "label": 1, "cwe": "CWE-121", "pair_id": "", "source": "sard",
            "nodes": [
                {"id": "1", "_label": "METHOD", "code": "f()", "name": "f",
                 "line": 1, "type_full": "", "taint_role": "NONE"},
                {"id": "2", "_label": "CALL", "code": "g()", "name": "g",
                 "line": 2, "type_full": "", "taint_role": "NONE"},
                {"id": "3", "_label": "CALL", "code": "h()", "name": "h",
                 "line": 3, "type_full": "", "taint_role": "NONE"},
            ],
            "edges": [
                {"src": "1", "dst": "2", "type": 4, "label": "CDG"},
                {"src": "2", "dst": "3", "type": 5, "label": "DOMINATE"},
            ],
        }
        features = np.random.randn(3, FEATURE_DIM).astype(np.float32)
        cpg_path = tmp_path / "gate4-test.json"
        cpg_path.write_text(json.dumps(cpg), encoding="utf-8")
        npz_path = tmp_path / "gate4-test.npz"
        np.savez_compressed(npz_path, node_features=features)

        result, gate = load_sample(cpg_path, npz_path)
        assert result is None
        assert gate == "gate4_edge_type"

    def test_mixed_good_bad_edges_keeps_good(self, tmp_path):
        """If some edges have valid types and some don't, valid ones are kept."""
        cpg = {
            "sample_id": "mixed-edge-test",
            "label": 0, "cwe": "CWE-122", "pair_id": "", "source": "sard",
            "nodes": [
                {"id": "1", "_label": "METHOD", "code": "f()", "name": "f",
                 "line": 1, "type_full": "", "taint_role": "NONE"},
                {"id": "2", "_label": "CALL", "code": "g()", "name": "g",
                 "line": 2, "type_full": "", "taint_role": "NONE"},
                {"id": "3", "_label": "CALL", "code": "h()", "name": "h",
                 "line": 3, "type_full": "", "taint_role": "NONE"},
            ],
            "edges": [
                {"src": "1", "dst": "2", "type": 0, "label": "AST"},
                {"src": "2", "dst": "3", "type": 7, "label": "BOGUS"},
                {"src": "1", "dst": "3", "type": 1, "label": "CFG"},
            ],
        }
        features = np.random.randn(3, FEATURE_DIM).astype(np.float32)
        cpg_path = tmp_path / "mixed-edge-test.json"
        cpg_path.write_text(json.dumps(cpg), encoding="utf-8")
        npz_path = tmp_path / "mixed-edge-test.npz"
        np.savez_compressed(npz_path, node_features=features)

        result, gate = load_sample(cpg_path, npz_path)
        assert result is not None
        assert result["num_edges"] == 2
        assert set(result["edge_attr"].tolist()) == {0, 1}


# ══════════════════════════════════════════════════════════════════
# Test 4: Post-write pair integrity check
# ══════════════════════════════════════════════════════════════════

class TestPairIntegrityCheck:

    def test_valid_pairs(self, tmp_path):
        """All pairs have both labels -> 0 broken."""
        samples = [
            _make_sample_dict("s1", 1, "CWE-121", pair_id="pair-1"),
            _make_sample_dict("s2", 0, "CWE-121", pair_id="pair-1"),
            _make_sample_dict("s3", 1, "CWE-122", pair_id="pair-2"),
            _make_sample_dict("s4", 0, "CWE-122", pair_id="pair-2"),
        ]
        h5_path = tmp_path / "test.h5"
        write_graphs_h5(samples, h5_path)

        result = check_pair_integrity(h5_path)
        assert result["total_pairs"] == 2
        assert result["valid_pairs"] == 2
        assert result["broken_pairs"] == 0

    def test_broken_pair_missing_safe(self, tmp_path):
        """Pair with only label=1 is broken."""
        samples = [
            _make_sample_dict("s1", 1, "CWE-121", pair_id="pair-broken"),
            _make_sample_dict("s2", 1, "CWE-121", pair_id="pair-broken"),
            _make_sample_dict("s3", 0, "CWE-122", pair_id="pair-ok"),
            _make_sample_dict("s4", 1, "CWE-122", pair_id="pair-ok"),
        ]
        h5_path = tmp_path / "test.h5"
        write_graphs_h5(samples, h5_path)

        result = check_pair_integrity(h5_path)
        assert result["total_pairs"] == 2
        assert result["broken_pairs"] == 1
        assert "pair-broken" in result["broken_detail"]

    def test_empty_pair_id_ignored(self, tmp_path):
        """Samples with empty/sentinel pair_id are not counted as pairs."""
        samples = [
            _make_sample_dict("s1", 1, "CWE-121", pair_id=""),
            _make_sample_dict("s2", 0, "CWE-121", pair_id="None"),
            _make_sample_dict("s3", 1, "CWE-122", pair_id="0"),
        ]
        h5_path = tmp_path / "test.h5"
        write_graphs_h5(samples, h5_path)

        result = check_pair_integrity(h5_path)
        assert result["total_pairs"] == 0
        assert result["broken_pairs"] == 0


# ══════════════════════════════════════════════════════════════════
# Test 5: Per-CWE distribution in stats
# ══════════════════════════════════════════════════════════════════

class TestCWEDistribution:

    def test_cwe_distribution_in_run_stage6(self, tmp_path):
        """run_stage6 includes cwe_distribution in returned stats."""
        cpg_dir = tmp_path / "cpg"
        embed_dir = tmp_path / "embed"
        cpg_dir.mkdir()
        embed_dir.mkdir()

        # Create 3 samples across 2 CWEs
        for sid, label, cwe in [("aa01", 1, "CWE-121"), ("aa02", 0, "CWE-121"), ("bb01", 1, "CWE-190")]:
            cpg = _make_cpg(sid, label, cwe)
            shard = sid[:2]
            (cpg_dir / shard).mkdir(exist_ok=True)
            (cpg_dir / shard / f"{sid}.json").write_text(json.dumps(cpg), encoding="utf-8")
            n_nodes = len(cpg["nodes"])
            features = np.random.randn(n_nodes, FEATURE_DIM).astype(np.float32)
            (embed_dir / shard).mkdir(exist_ok=True)
            np.savez_compressed(embed_dir / shard / f"{sid}.npz", node_features=features)

        h5_path = tmp_path / "out.h5"
        stats = run_stage6(
            cpg_dir=str(cpg_dir),
            embed_dir=str(embed_dir),
            output_path=str(h5_path),
        )

        assert "cwe_distribution" in stats
        assert stats["cwe_distribution"]["CWE-121"] == 2
        assert stats["cwe_distribution"]["CWE-190"] == 1

        # Also verify graph_stats.json was written
        stats_file = h5_path.with_name("graph_stats.json")
        assert stats_file.exists()
        saved = json.loads(stats_file.read_text(encoding="utf-8"))
        assert saved["cwe_distribution"]["CWE-121"] == 2


# ══════════════════════════════════════════════════════════════════
# Test 6: Gate 6 — all dangling edges
# ══════════════════════════════════════════════════════════════════

class TestGate6Dangling:

    def test_all_dangling_rejects(self, tmp_path):
        """If ALL edges reference non-existent nodes, reject via gate6."""
        cpg = {
            "sample_id": "gate6-test",
            "label": 1, "cwe": "CWE-121", "pair_id": "", "source": "sard",
            "nodes": [
                {"id": "1", "_label": "METHOD", "code": "f()", "name": "f",
                 "line": 1, "type_full": "", "taint_role": "NONE"},
                {"id": "2", "_label": "CALL", "code": "g()", "name": "g",
                 "line": 2, "type_full": "", "taint_role": "NONE"},
                {"id": "3", "_label": "CALL", "code": "h()", "name": "h",
                 "line": 3, "type_full": "", "taint_role": "NONE"},
            ],
            "edges": [
                {"src": "999", "dst": "888", "type": 0, "label": "AST"},
                {"src": "777", "dst": "666", "type": 1, "label": "CFG"},
            ],
        }
        features = np.random.randn(3, FEATURE_DIM).astype(np.float32)
        cpg_path = tmp_path / "gate6-test.json"
        cpg_path.write_text(json.dumps(cpg), encoding="utf-8")
        npz_path = tmp_path / "gate6-test.npz"
        np.savez_compressed(npz_path, node_features=features)

        result, gate = load_sample(cpg_path, npz_path)
        assert result is None
        assert gate == "gate6_dangling"


# ══════════════════════════════════════════════════════════════════
# Test 7: Gate 7 — no edges at all
# ══════════════════════════════════════════════════════════════════

class TestGate7NoEdges:

    def test_no_edges_rejects(self, tmp_path):
        """Graph with zero edges is rejected via gate7."""
        cpg = {
            "sample_id": "gate7-test",
            "label": 0, "cwe": "CWE-122", "pair_id": "", "source": "sard",
            "nodes": [
                {"id": "1", "_label": "METHOD", "code": "f()", "name": "f",
                 "line": 1, "type_full": "", "taint_role": "NONE"},
                {"id": "2", "_label": "CALL", "code": "g()", "name": "g",
                 "line": 2, "type_full": "", "taint_role": "NONE"},
                {"id": "3", "_label": "CALL", "code": "h()", "name": "h",
                 "line": 3, "type_full": "", "taint_role": "NONE"},
            ],
            "edges": [],
        }
        features = np.random.randn(3, FEATURE_DIM).astype(np.float32)
        cpg_path = tmp_path / "gate7-test.json"
        cpg_path.write_text(json.dumps(cpg), encoding="utf-8")
        npz_path = tmp_path / "gate7-test.npz"
        np.savez_compressed(npz_path, node_features=features)

        result, gate = load_sample(cpg_path, npz_path)
        assert result is None
        assert gate == "gate7_no_edges"


# ══════════════════════════════════════════════════════════════════
# Test 8: Rejection log written
# ══════════════════════════════════════════════════════════════════

class TestRejectionLog:

    def test_rejection_log_created(self, tmp_path):
        """run_stage6 creates graph_rejected.log with gate reasons."""
        cpg_dir = tmp_path / "cpg"
        embed_dir = tmp_path / "embed"
        cpg_dir.mkdir()
        embed_dir.mkdir()

        # One valid sample
        cpg_ok = _make_cpg("aa01", 1, "CWE-121")
        shard = "aa"
        (cpg_dir / shard).mkdir(exist_ok=True)
        (cpg_dir / shard / "aa01.json").write_text(json.dumps(cpg_ok), encoding="utf-8")
        n = len(cpg_ok["nodes"])
        feat = np.random.randn(n, FEATURE_DIM).astype(np.float32)
        (embed_dir / shard).mkdir(exist_ok=True)
        np.savez_compressed(embed_dir / shard / "aa01.npz", node_features=feat)

        # One trivial sample (2 nodes -> gate1)
        cpg_bad = {
            "sample_id": "aa02",
            "label": 0, "cwe": "CWE-121", "pair_id": "", "source": "sard",
            "nodes": [
                {"id": "1", "_label": "METHOD", "code": "f()", "name": "f",
                 "line": 1, "type_full": "", "taint_role": "NONE"},
                {"id": "2", "_label": "CALL", "code": "g()", "name": "g",
                 "line": 2, "type_full": "", "taint_role": "NONE"},
            ],
            "edges": [{"src": "1", "dst": "2", "type": 0, "label": "AST"}],
        }
        (cpg_dir / shard / "aa02.json").write_text(json.dumps(cpg_bad), encoding="utf-8")
        feat2 = np.random.randn(2, FEATURE_DIM).astype(np.float32)
        np.savez_compressed(embed_dir / shard / "aa02.npz", node_features=feat2)

        h5_path = tmp_path / "out.h5"
        stats = run_stage6(
            cpg_dir=str(cpg_dir),
            embed_dir=str(embed_dir),
            output_path=str(h5_path),
        )

        assert stats["valid_graphs"] == 1
        assert stats["failed"] == 1
        assert "gate1_trivial" in stats["reject_reasons"]

        # Check log file
        log_path = h5_path.with_name("graph_rejected.log")
        assert log_path.exists()
        lines = log_path.read_text(encoding="utf-8").strip().split("\n")
        assert lines[0] == "sample_id\tgate"
        assert "aa02" in lines[1]
        assert "gate1_trivial" in lines[1]


# ══════════════════════════════════════════════════════════════════
# Test 9: run_stage6 with check_pairs integration
# ══════════════════════════════════════════════════════════════════

class TestRunStage6WithPairCheck:

    def test_check_pairs_adds_to_stats(self, tmp_path):
        """run_stage6(check_pairs=True) adds pair_integrity to stats."""
        cpg_dir = tmp_path / "cpg"
        embed_dir = tmp_path / "embed"
        cpg_dir.mkdir()
        embed_dir.mkdir()

        for sid, label, cwe, pid in [
            ("aa01", 1, "CWE-121", "pair-1"),
            ("aa02", 0, "CWE-121", "pair-1"),
        ]:
            cpg = _make_cpg(sid, label, cwe, pair_id=pid)
            shard = sid[:2]
            (cpg_dir / shard).mkdir(exist_ok=True)
            (cpg_dir / shard / f"{sid}.json").write_text(json.dumps(cpg), encoding="utf-8")
            n = len(cpg["nodes"])
            feat = np.random.randn(n, FEATURE_DIM).astype(np.float32)
            (embed_dir / shard).mkdir(exist_ok=True)
            np.savez_compressed(embed_dir / shard / f"{sid}.npz", node_features=feat)

        h5_path = tmp_path / "out.h5"
        stats = run_stage6(
            cpg_dir=str(cpg_dir),
            embed_dir=str(embed_dir),
            output_path=str(h5_path),
            check_pairs=True,
        )

        assert "pair_integrity" in stats
        assert stats["pair_integrity"]["total_pairs"] == 1
        assert stats["pair_integrity"]["valid_pairs"] == 1
        assert stats["pair_integrity"]["broken_pairs"] == 0


# ══════════════════════════════════════════════════════════════════
# Test 10: verify_h5 prints new attrs without error
# ══════════════════════════════════════════════════════════════════

class TestVerifyH5WithAttrs:

    def test_verify_with_attrs(self, tmp_path, capsys):
        """verify_h5 reads per-graph attrs without crashing."""
        samples = [
            _make_sample_dict("uuid-abc", 1, "CWE-121", pair_id="pair-xyz", source="sard"),
        ]
        h5_path = tmp_path / "test.h5"
        write_graphs_h5(samples, h5_path)

        verify_h5(str(h5_path), max_check=1)
        captured = capsys.readouterr()
        assert "CWE-121" in captured.out
        assert "pair-xyz" in captured.out
        assert "sard" in captured.out
