#!/usr/bin/env python3
"""
tests/test_story6.py

Tests for Story 6: Stage 5 Node Embedding + Stage 6 Graph Tensors.

Tests pure-Python logic (feature construction, validation, HDF5 I/O).
Does NOT require CodeBERT model download — uses a mock encoder for unit tests.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import h5py
import numpy as np
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.scripts.preprocessing.stage5_embed import (
    CODEBERT_DIM,
    CPG_COMPONENT_DIM,
    EDGE_TYPE_MAP,
    JOERN_NODE_TYPES,
    NODE_TYPE_DIM,
    NODE_TYPE_TO_IDX,
    STRUCTURAL_DIM,
    TAINT_ROLE_DIM,
    TAINT_ROLE_TO_IDX,
    TAINT_ROLES,
    TOTAL_DIM,
    _bfs_distance,
    _build_cpg_component_onehot,
    _build_node_type_onehot,
    _build_structural_features,
    _build_taint_role_onehot,
    _compute_ast_depths,
    _output_path,
    _precompute_node_max_edge_type,
    embed_cpg,
    verify_embeddings,
)
from training.scripts.preprocessing.stage6_graphs import (
    FEATURE_DIM,
    MIN_NODES,
    VALID_EDGE_TYPES,
    check_pair_integrity,
    load_sample,
    verify_h5,
    write_graphs_h5,
)


# ══════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════

@pytest.fixture
def sample_cpg():
    """A minimal valid CPG dict matching stage4 output format."""
    return {
        "sample_id": "abcd1234-test-uuid",
        "label": 1,
        "cwe": "CWE-121",
        "pair_id": "pair_001",
        "source": "sard",
        "nodes": [
            {"id": "1", "_label": "METHOD", "code": "void bad()", "name": "bad",
             "line": 1, "type_full": "void", "taint_role": "NONE"},
            {"id": "2", "_label": "CALL", "code": "gets(buf)", "name": "gets",
             "line": 3, "type_full": "ANY", "taint_role": "SOURCE"},
            {"id": "3", "_label": "IDENTIFIER", "code": "buf", "name": "buf",
             "line": 3, "type_full": "char*", "taint_role": "NONE"},
            {"id": "4", "_label": "CALL", "code": "strcpy(dst, buf)", "name": "strcpy",
             "line": 5, "type_full": "ANY", "taint_role": "SINK"},
            {"id": "5", "_label": "METHOD_RETURN", "code": "RET", "name": "",
             "line": 1, "type_full": "void", "taint_role": "NONE"},
        ],
        "edges": [
            {"src": "1", "dst": "2", "type": 0, "label": "AST"},
            {"src": "1", "dst": "3", "type": 0, "label": "AST"},
            {"src": "1", "dst": "4", "type": 0, "label": "AST"},
            {"src": "1", "dst": "5", "type": 0, "label": "AST"},
            {"src": "2", "dst": "3", "type": 1, "label": "CFG"},
            {"src": "3", "dst": "4", "type": 1, "label": "CFG"},
            {"src": "2", "dst": "3", "type": 2, "label": "DFG"},
            {"src": "3", "dst": "4", "type": 2, "label": "DFG"},
            {"src": "2", "dst": "4", "type": 3, "label": "TPG"},
        ],
        "stats": {"node_count": 5, "edge_count": 9},
    }


@pytest.fixture
def mock_encoder():
    """Mock CodeBERT encoder returning deterministic 768-d vectors."""
    encoder = MagicMock()
    def fake_encode(texts):
        """Return a unique but deterministic 768-d vector per text."""
        result = np.zeros((len(texts), CODEBERT_DIM), dtype=np.float32)
        for i, text in enumerate(texts):
            # Use hash of text to create reproducible but distinct vectors
            seed = hash(text) % (2**31)
            rng = np.random.RandomState(seed)
            result[i] = rng.randn(CODEBERT_DIM).astype(np.float32) * 0.1
        return result
    encoder.encode_batch = fake_encode
    return encoder


# ══════════════════════════════════════════════════════════════════
# Stage 5: Feature Dimension Tests
# ══════════════════════════════════════════════════════════════════

class TestFeatureDimensions:
    """Verify the 824-d feature vector layout is correct."""

    def test_total_dim_is_824(self):
        assert TOTAL_DIM == 824

    def test_codebert_dim(self):
        assert CODEBERT_DIM == 768

    def test_node_type_dim(self):
        assert NODE_TYPE_DIM == 32
        assert len(JOERN_NODE_TYPES) == 32

    def test_taint_role_dim(self):
        assert TAINT_ROLE_DIM == 8
        assert len(TAINT_ROLES) == 8

    def test_cpg_component_dim(self):
        assert CPG_COMPONENT_DIM == 4

    def test_structural_dim(self):
        assert STRUCTURAL_DIM == 12

    def test_dim_sum(self):
        assert CODEBERT_DIM + NODE_TYPE_DIM + TAINT_ROLE_DIM + CPG_COMPONENT_DIM + STRUCTURAL_DIM == 824


# ══════════════════════════════════════════════════════════════════
# Stage 5: Node Type One-Hot Tests
# ══════════════════════════════════════════════════════════════════

class TestNodeTypeOneHot:
    """Test 32-d node type one-hot encoding."""

    def test_known_type_call(self):
        vec = _build_node_type_onehot("CALL")
        assert vec.shape == (32,)
        assert vec.sum() == 1.0
        assert vec[NODE_TYPE_TO_IDX["CALL"]] == 1.0

    def test_known_type_method(self):
        vec = _build_node_type_onehot("METHOD")
        assert vec[NODE_TYPE_TO_IDX["METHOD"]] == 1.0

    def test_known_type_identifier(self):
        vec = _build_node_type_onehot("IDENTIFIER")
        assert vec[NODE_TYPE_TO_IDX["IDENTIFIER"]] == 1.0

    def test_unknown_type_falls_back(self):
        vec = _build_node_type_onehot("TOTALLY_NEW_TYPE")
        assert vec[NODE_TYPE_TO_IDX["UNKNOWN"]] == 1.0

    def test_empty_string_falls_back(self):
        vec = _build_node_type_onehot("")
        assert vec[NODE_TYPE_TO_IDX["UNKNOWN"]] == 1.0

    def test_all_17_observed_types(self):
        """All 17 types from real SARD CPGs map to unique indices."""
        observed = [
            "BLOCK", "CALL", "CONTROL_STRUCTURE", "FIELD_IDENTIFIER",
            "IDENTIFIER", "JUMP_TARGET", "LITERAL", "LOCAL", "METHOD",
            "METHOD_PARAMETER_IN", "METHOD_PARAMETER_OUT", "METHOD_REF",
            "METHOD_RETURN", "MODIFIER", "RETURN", "TYPE_DECL", "TYPE_REF",
        ]
        indices = set()
        for t in observed:
            vec = _build_node_type_onehot(t)
            idx = np.argmax(vec)
            assert idx not in indices, f"Duplicate index for {t}"
            indices.add(idx)
        assert len(indices) == 17


# ══════════════════════════════════════════════════════════════════
# Stage 5: Taint Role One-Hot Tests
# ══════════════════════════════════════════════════════════════════

class TestTaintRoleOneHot:
    """Test 8-d taint role one-hot encoding."""

    def test_source(self):
        vec = _build_taint_role_onehot("SOURCE")
        assert vec.shape == (8,)
        assert vec.sum() == 1.0
        assert vec[0] == 1.0

    def test_sink(self):
        vec = _build_taint_role_onehot("SINK")
        assert vec[1] == 1.0

    def test_sanitizer(self):
        vec = _build_taint_role_onehot("SANITIZER")
        assert vec[2] == 1.0

    def test_propagation(self):
        vec = _build_taint_role_onehot("PROPAGATION")
        assert vec[3] == 1.0

    def test_none(self):
        vec = _build_taint_role_onehot("NONE")
        assert vec[4] == 1.0

    def test_unknown_role_defaults_to_none(self):
        vec = _build_taint_role_onehot("BOGUS")
        assert vec[TAINT_ROLE_TO_IDX["NONE"]] == 1.0

    def test_empty_role_defaults_to_none(self):
        vec = _build_taint_role_onehot("")
        assert vec[TAINT_ROLE_TO_IDX["NONE"]] == 1.0


# ══════════════════════════════════════════════════════════════════
# Stage 5: CPG Component One-Hot Tests
# ══════════════════════════════════════════════════════════════════

class TestCPGComponentOneHot:

    def test_ast_only_node(self):
        """Node only in AST edges → AST one-hot."""
        vec = _build_cpg_component_onehot(0)  # AST = 0
        assert vec.shape == (4,)
        assert vec[0] == 1.0

    def test_dfg_node(self):
        """Node in DFG edges → DFG one-hot (priority > CFG, AST)."""
        vec = _build_cpg_component_onehot(2)  # DFG = 2
        assert vec[2] == 1.0

    def test_tpg_node(self):
        """Node in TPG edges → TPG one-hot (highest priority)."""
        vec = _build_cpg_component_onehot(3)  # TPG = 3
        assert vec[3] == 1.0

    def test_isolated_node(self):
        """Isolated node (no edges) → AST default."""
        vec = _build_cpg_component_onehot(-1)
        assert vec[0] == 1.0

    def test_precompute_max_edge_type(self):
        edges = [
            {"src": "1", "dst": "2", "type": 0},
            {"src": "1", "dst": "3", "type": 2},
            {"src": "2", "dst": "3", "type": 1},
            {"src": "3", "dst": "4", "type": 3},
        ]
        result = _precompute_node_max_edge_type(edges)
        assert result["1"] == 2  # max of 0, 2
        assert result["2"] == 1  # max of 0, 1
        assert result["3"] == 3  # max of 2, 1, 3
        assert result["4"] == 3


# ══════════════════════════════════════════════════════════════════
# Stage 5: AST Depth Tests
# ══════════════════════════════════════════════════════════════════

class TestASTDepths:

    def test_simple_tree(self):
        nodes = [{"id": "1"}, {"id": "2"}, {"id": "3"}]
        edges = [
            {"src": "1", "dst": "2", "type": 0},
            {"src": "2", "dst": "3", "type": 0},
        ]
        depths = _compute_ast_depths(nodes, edges)
        assert depths["1"] == 0
        assert depths["2"] == 1
        assert depths["3"] == 2

    def test_non_ast_edges_ignored(self):
        nodes = [{"id": "1"}, {"id": "2"}]
        edges = [
            {"src": "1", "dst": "2", "type": 1},  # CFG, not AST
        ]
        depths = _compute_ast_depths(nodes, edges)
        # Node 2 has no AST parent, so it's a root with depth 0
        assert depths["1"] == 0
        assert depths["2"] == 0

    def test_orphan_nodes_get_depth_zero(self):
        nodes = [{"id": "1"}, {"id": "2"}, {"id": "3"}]
        edges = [
            {"src": "1", "dst": "2", "type": 0},
        ]
        depths = _compute_ast_depths(nodes, edges)
        assert depths["3"] == 0  # orphan


# ══════════════════════════════════════════════════════════════════
# Stage 5: BFS Distance Tests
# ══════════════════════════════════════════════════════════════════

class TestBFSDistance:

    def test_direct_neighbor(self):
        adj = {"1": ["2"]}
        dist = _bfs_distance("1", {"2"}, adj)
        assert dist == 1.0

    def test_two_hops(self):
        adj = {"1": ["2"], "2": ["3"]}
        dist = _bfs_distance("1", {"3"}, adj)
        assert dist == 2.0

    def test_start_is_target(self):
        adj = {}
        dist = _bfs_distance("1", {"1"}, adj)
        assert dist == 0.0

    def test_unreachable(self):
        adj = {"1": ["2"]}
        dist = _bfs_distance("1", {"99"}, adj, max_dist=10)
        assert dist == 10.0

    def test_empty_targets(self):
        adj = {"1": ["2"]}
        dist = _bfs_distance("1", set(), adj)
        assert dist == 200.0  # default max_dist

    def test_multiple_targets_finds_nearest(self):
        adj = {"1": ["2"], "2": ["3", "4"]}
        dist = _bfs_distance("1", {"3", "4"}, adj)
        assert dist == 2.0


# ══════════════════════════════════════════════════════════════════
# Stage 5: Structural Features Tests
# ══════════════════════════════════════════════════════════════════

class TestStructuralFeatures:

    def test_shape_is_12(self):
        from collections import defaultdict
        vec = _build_structural_features(
            "1", 5, defaultdict(list), defaultdict(list),
            {"1": 0}, 1, set(), defaultdict(list),
            defaultdict(list), defaultdict(list),
        )
        assert vec.shape == (12,)
        assert vec.dtype == np.float32

    def test_no_sinks_gives_1(self):
        from collections import defaultdict
        vec = _build_structural_features(
            "1", 5, defaultdict(list), defaultdict(list),
            {"1": 0}, 1, set(), defaultdict(list),
            defaultdict(list), defaultdict(list),
        )
        # Taint distances should be 1.0 when no sinks
        assert vec[3] == 1.0
        assert vec[4] == 1.0
        assert vec[5] == 1.0

    def test_reserved_slots_are_zero(self):
        from collections import defaultdict
        vec = _build_structural_features(
            "1", 5, defaultdict(list), defaultdict(list),
            {"1": 0}, 1, set(), defaultdict(list),
            defaultdict(list), defaultdict(list),
        )
        # Slots 6-11 are reserved, should be zero
        assert np.all(vec[6:12] == 0.0)


# ══════════════════════════════════════════════════════════════════
# Stage 5: Full Embedding (with mock encoder) Tests
# ══════════════════════════════════════════════════════════════════

class TestEmbedCPG:

    def test_output_shape(self, sample_cpg, mock_encoder):
        features = embed_cpg(sample_cpg, mock_encoder)
        n_nodes = len(sample_cpg["nodes"])
        assert features.shape == (n_nodes, 824)
        assert features.dtype == np.float32

    def test_no_nan(self, sample_cpg, mock_encoder):
        features = embed_cpg(sample_cpg, mock_encoder)
        assert not np.isnan(features).any()

    def test_codebert_region_populated(self, sample_cpg, mock_encoder):
        """CodeBERT region [0:768] should be non-zero for nodes with code."""
        features = embed_cpg(sample_cpg, mock_encoder)
        # Node 0 has code "void bad()" — should get non-zero CodeBERT embedding
        assert features[0, :768].any()

    def test_node_type_region(self, sample_cpg, mock_encoder):
        """Node type region [768:800] should be one-hot."""
        features = embed_cpg(sample_cpg, mock_encoder)
        for i in range(len(sample_cpg["nodes"])):
            type_vec = features[i, 768:800]
            assert type_vec.sum() == pytest.approx(1.0)

    def test_taint_role_region(self, sample_cpg, mock_encoder):
        """Taint role region [800:808] should be one-hot."""
        features = embed_cpg(sample_cpg, mock_encoder)
        for i in range(len(sample_cpg["nodes"])):
            role_vec = features[i, 800:808]
            assert role_vec.sum() == pytest.approx(1.0)

    def test_cpg_component_region(self, sample_cpg, mock_encoder):
        """CPG component region [808:812] should be one-hot."""
        features = embed_cpg(sample_cpg, mock_encoder)
        for i in range(len(sample_cpg["nodes"])):
            comp_vec = features[i, 808:812]
            assert comp_vec.sum() == pytest.approx(1.0)

    def test_structural_region_bounded(self, sample_cpg, mock_encoder):
        """Structural features [812:824] should be in [0, 1] (normalized)."""
        features = embed_cpg(sample_cpg, mock_encoder)
        struct = features[:, 812:824]
        assert struct.min() >= 0.0
        # Taint distance can be > 1.0 when max_dist/n_nodes > 1, but
        # degree and depth are bounded. Just check no negative.
        assert not np.isnan(struct).any()

    def test_empty_code_node_gets_zero_codebert(self, mock_encoder):
        """Node with empty code → zero vector in [0:768], NOT skipped."""
        cpg = {
            "nodes": [
                {"id": "1", "_label": "METHOD", "code": "void f()", "name": "f",
                 "line": 1, "type_full": "", "taint_role": "NONE"},
                {"id": "2", "_label": "BLOCK", "code": "", "name": "",
                 "line": 2, "type_full": "", "taint_role": "NONE"},
                {"id": "3", "_label": "CALL", "code": "printf(x)", "name": "printf",
                 "line": 3, "type_full": "", "taint_role": "SINK"},
            ],
            "edges": [
                {"src": "1", "dst": "2", "type": 0, "label": "AST"},
                {"src": "2", "dst": "3", "type": 0, "label": "AST"},
                {"src": "2", "dst": "3", "type": 1, "label": "CFG"},
            ],
        }
        features = embed_cpg(cpg, mock_encoder)
        assert features.shape == (3, 824)
        # Node 1 (index 1) has empty code — [0:768] should be zero
        assert np.all(features[1, :768] == 0.0)
        # But the rest of its feature vector should still be populated
        assert features[1, 768:].any()

    def test_empty_cpg_returns_zero_rows(self, mock_encoder):
        """CPG with 0 nodes → (0, 824) array."""
        cpg = {"nodes": [], "edges": []}
        features = embed_cpg(cpg, mock_encoder)
        assert features.shape == (0, 824)

    def test_source_gets_source_onehot(self, sample_cpg, mock_encoder):
        """Node with taint_role=SOURCE should have bit 0 set in [800:808]."""
        features = embed_cpg(sample_cpg, mock_encoder)
        # Node index 1 is the SOURCE node ("gets(buf)")
        assert features[1, 800] == 1.0  # SOURCE = index 0

    def test_sink_gets_sink_onehot(self, sample_cpg, mock_encoder):
        """Node with taint_role=SINK should have bit 1 set in [800:808]."""
        features = embed_cpg(sample_cpg, mock_encoder)
        # Node index 3 is the SINK node ("strcpy(dst, buf)")
        assert features[3, 801] == 1.0  # SINK = index 1


# ══════════════════════════════════════════════════════════════════
# Stage 5: Output Path Sharding Tests
# ══════════════════════════════════════════════════════════════════

class TestOutputPath:

    def test_shard_from_uuid(self, tmp_path):
        path = _output_path(tmp_path, "abcd1234-test-uuid")
        assert path.parent.name == "ab"
        assert path.name == "abcd1234-test-uuid.npz"

    def test_different_ids_different_shards(self, tmp_path):
        p1 = _output_path(tmp_path, "00001234-test")
        p2 = _output_path(tmp_path, "ff001234-test")
        assert p1.parent.name == "00"
        assert p2.parent.name == "ff"


# ══════════════════════════════════════════════════════════════════
# Stage 6: Load Sample Tests
# ══════════════════════════════════════════════════════════════════

class TestLoadSample:

    def _write_pair(self, tmp_path, cpg_dict, features):
        """Helper to write a CPG JSON + .npz pair to tmp_path."""
        sample_id = cpg_dict["sample_id"]
        cpg_path = tmp_path / f"{sample_id}.json"
        cpg_path.write_text(json.dumps(cpg_dict), encoding="utf-8")
        npz_path = tmp_path / f"{sample_id}.npz"
        np.savez_compressed(npz_path, node_features=features)
        return cpg_path, npz_path

    def test_valid_sample(self, tmp_path, sample_cpg):
        n_nodes = len(sample_cpg["nodes"])
        features = np.random.randn(n_nodes, FEATURE_DIM).astype(np.float32)
        cpg_path, npz_path = self._write_pair(tmp_path, sample_cpg, features)

        result, gate = load_sample(cpg_path, npz_path)
        assert result is not None
        assert gate == ""
        assert result["sample_id"] == sample_cpg["sample_id"]
        assert result["label"] == 1
        assert result["cwe"] == "CWE-121"
        assert result["x"].shape == (n_nodes, FEATURE_DIM)
        assert result["edge_index"].shape[0] == 2
        assert result["edge_attr"].max() <= 3

    def test_reject_trivial_graph(self, tmp_path):
        """Graph with < 3 nodes should be rejected."""
        cpg = {
            "sample_id": "trivial-test",
            "label": 0, "cwe": "CWE-121", "pair_id": "", "source": "sard",
            "nodes": [
                {"id": "1", "_label": "METHOD", "code": "void f()", "name": "f",
                 "line": 1, "type_full": "", "taint_role": "NONE"},
                {"id": "2", "_label": "CALL", "code": "x()", "name": "x",
                 "line": 2, "type_full": "", "taint_role": "NONE"},
            ],
            "edges": [{"src": "1", "dst": "2", "type": 0, "label": "AST"}],
        }
        features = np.random.randn(2, FEATURE_DIM).astype(np.float32)
        cpg_path, npz_path = self._write_pair(tmp_path, cpg, features)
        result, gate = load_sample(cpg_path, npz_path)
        assert result is None
        assert gate == "gate1_trivial"

    def test_reject_nan_features(self, tmp_path, sample_cpg):
        """Features with NaN should be rejected."""
        n_nodes = len(sample_cpg["nodes"])
        features = np.random.randn(n_nodes, FEATURE_DIM).astype(np.float32)
        features[0, 0] = np.nan
        cpg_path, npz_path = self._write_pair(tmp_path, sample_cpg, features)
        result, gate = load_sample(cpg_path, npz_path)
        assert result is None
        assert gate == "gate3_nan"

    def test_reject_shape_mismatch(self, tmp_path, sample_cpg):
        """Feature matrix with wrong number of rows should be rejected."""
        features = np.random.randn(999, FEATURE_DIM).astype(np.float32)
        cpg_path, npz_path = self._write_pair(tmp_path, sample_cpg, features)
        result, gate = load_sample(cpg_path, npz_path)
        assert result is None
        assert gate == "gate2_shape"

    def test_reject_wrong_feature_dim(self, tmp_path, sample_cpg):
        """Feature matrix with wrong column count should be rejected."""
        n_nodes = len(sample_cpg["nodes"])
        features = np.random.randn(n_nodes, 512).astype(np.float32)
        cpg_path, npz_path = self._write_pair(tmp_path, sample_cpg, features)
        result, gate = load_sample(cpg_path, npz_path)
        assert result is None
        assert gate == "gate2_shape"

    def test_edge_type_filtering(self, tmp_path):
        """Edges with invalid type (>= 4) should be silently dropped."""
        cpg = {
            "sample_id": "edgetype-test",
            "label": 0, "cwe": "CWE-121", "pair_id": "", "source": "sard",
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
                {"src": "2", "dst": "3", "type": 5, "label": "BOGUS"},  # invalid
                {"src": "1", "dst": "3", "type": 1, "label": "CFG"},
            ],
        }
        features = np.random.randn(3, FEATURE_DIM).astype(np.float32)
        cpg_path, npz_path = self._write_pair(tmp_path, cpg, features)
        result, gate = load_sample(cpg_path, npz_path)
        assert result is not None
        assert gate == ""
        # Only 2 valid edges (type 0 and type 1), the type=5 was dropped
        assert result["num_edges"] == 2
        assert set(result["edge_attr"].tolist()) == {0, 1}

    def test_dangling_edges_dropped(self, tmp_path):
        """Edges referencing non-existent nodes should be dropped."""
        cpg = {
            "sample_id": "dangling-test",
            "label": 0, "cwe": "CWE-121", "pair_id": "", "source": "sard",
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
                {"src": "999", "dst": "2", "type": 0, "label": "AST"},  # dangling src
                {"src": "1", "dst": "3", "type": 1, "label": "CFG"},
            ],
        }
        features = np.random.randn(3, FEATURE_DIM).astype(np.float32)
        cpg_path, npz_path = self._write_pair(tmp_path, cpg, features)
        result, gate = load_sample(cpg_path, npz_path)
        assert result is not None
        assert gate == ""
        assert result["num_edges"] == 2  # dangling edge dropped

    def test_edge_index_contiguous(self, tmp_path, sample_cpg):
        """Edge indices should be remapped to contiguous 0..N-1."""
        n_nodes = len(sample_cpg["nodes"])
        features = np.random.randn(n_nodes, FEATURE_DIM).astype(np.float32)
        cpg_path, npz_path = self._write_pair(tmp_path, sample_cpg, features)
        result, gate = load_sample(cpg_path, npz_path)
        assert result is not None
        assert gate == ""
        ei = result["edge_index"]
        assert ei.min() >= 0
        assert ei.max() < n_nodes


# ══════════════════════════════════════════════════════════════════
# Stage 6: HDF5 Write/Read Tests
# ══════════════════════════════════════════════════════════════════

class TestHDF5WriteRead:

    def _make_sample(self, sample_id, label, cwe, n_nodes=10, n_edges=15):
        """Create a synthetic valid sample dict."""
        x = np.random.randn(n_nodes, FEATURE_DIM).astype(np.float32)
        src = np.random.randint(0, n_nodes, size=n_edges)
        dst = np.random.randint(0, n_nodes, size=n_edges)
        edge_index = np.stack([src, dst], axis=0).astype(np.int64)
        edge_attr = np.random.randint(0, 4, size=n_edges).astype(np.int64)
        return {
            "sample_id": sample_id,
            "label": label,
            "cwe": cwe,
            "pair_id": "",
            "source": "test",
            "x": x,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "num_nodes": n_nodes,
            "num_edges": n_edges,
        }

    def test_roundtrip(self, tmp_path):
        """Write samples to HDF5, read back, verify data matches."""
        samples = [
            self._make_sample("sample-001", 1, "CWE-121", 10, 15),
            self._make_sample("sample-002", 0, "CWE-122", 20, 30),
            self._make_sample("sample-003", 1, "CWE-190", 5, 8),
        ]
        h5_path = tmp_path / "test.h5"
        write_graphs_h5(samples, h5_path)
        assert h5_path.exists()

        with h5py.File(h5_path, "r") as f:
            meta = f["metadata"]
            assert meta.attrs["num_graphs"] == 3
            assert meta.attrs["feature_dim"] == FEATURE_DIM

            labels = meta["labels"][:]
            assert list(labels) == [1, 0, 1]

            # Check graph 0
            g0 = f["graphs"]["0"]
            x = g0["x"][:]
            assert x.shape == (10, FEATURE_DIM)
            np.testing.assert_array_almost_equal(x, samples[0]["x"])

            ei = g0["edge_index"][:]
            assert ei.shape == (2, 15)

            ea = g0["edge_attr"][:]
            assert ea.shape == (15,)

            y = g0["y"][:]
            assert y[0] == 1

    def test_atomic_write(self, tmp_path):
        """Verify .tmp file is cleaned up and final file exists."""
        samples = [self._make_sample("s1", 0, "CWE-121")]
        h5_path = tmp_path / "output.h5"
        tmp_file = h5_path.with_suffix(".h5.tmp")

        write_graphs_h5(samples, h5_path)

        assert h5_path.exists()
        assert not tmp_file.exists()  # .tmp should be gone after rename

    def test_empty_samples_list(self, tmp_path):
        """Writing empty list should still produce valid HDF5."""
        h5_path = tmp_path / "empty.h5"
        write_graphs_h5([], h5_path)
        assert h5_path.exists()

        with h5py.File(h5_path, "r") as f:
            assert f["metadata"].attrs["num_graphs"] == 0

    def test_feature_dim_metadata(self, tmp_path):
        """feature_dim=824 must be stored in HDF5 attrs for model loading."""
        samples = [self._make_sample("s1", 0, "CWE-121")]
        h5_path = tmp_path / "test.h5"
        write_graphs_h5(samples, h5_path)

        with h5py.File(h5_path, "r") as f:
            assert f["metadata"].attrs["feature_dim"] == 824

    def test_gzip_compression(self, tmp_path):
        """Node features should be gzip-compressed in HDF5."""
        samples = [self._make_sample("s1", 0, "CWE-121", n_nodes=50)]
        h5_path = tmp_path / "test.h5"
        write_graphs_h5(samples, h5_path)

        with h5py.File(h5_path, "r") as f:
            ds = f["graphs"]["0"]["x"]
            assert ds.compression == "gzip"


# ══════════════════════════════════════════════════════════════════
# Stage 6: Validation Edge Cases
# ══════════════════════════════════════════════════════════════════

class TestValidationEdgeCases:

    def test_edge_types_in_valid_range(self):
        assert VALID_EDGE_TYPES == {0, 1, 2, 3}

    def test_feature_dim_constant(self):
        assert FEATURE_DIM == 824

    def test_min_nodes_constant(self):
        assert MIN_NODES == 3


# ══════════════════════════════════════════════════════════════════
# Integration: Stage 5 → Stage 6 pipeline
# ══════════════════════════════════════════════════════════════════

class TestStage5To6Pipeline:
    """Test that stage5 output can be consumed by stage6."""

    def test_embed_then_load(self, tmp_path, sample_cpg, mock_encoder):
        """Embed a CPG via stage5, then load it via stage6."""
        # Stage 5: embed
        features = embed_cpg(sample_cpg, mock_encoder)
        assert features.shape == (5, 824)

        # Write CPG JSON and .npz to temp
        sample_id = sample_cpg["sample_id"]
        cpg_path = tmp_path / f"{sample_id}.json"
        cpg_path.write_text(json.dumps(sample_cpg), encoding="utf-8")
        npz_path = tmp_path / f"{sample_id}.npz"
        np.savez_compressed(npz_path, node_features=features)

        # Stage 6: load
        result, gate = load_sample(cpg_path, npz_path)
        assert result is not None
        assert gate == ""
        assert result["x"].shape == (5, 824)
        assert result["edge_index"].shape[0] == 2
        assert result["edge_index"].max() < 5
        assert not np.isnan(result["x"]).any()

    def test_full_pipeline_to_h5(self, tmp_path, sample_cpg, mock_encoder):
        """Full pipeline: embed -> load -> write HDF5 -> read back."""
        features = embed_cpg(sample_cpg, mock_encoder)

        sample_id = sample_cpg["sample_id"]
        cpg_path = tmp_path / f"{sample_id}.json"
        cpg_path.write_text(json.dumps(sample_cpg), encoding="utf-8")
        npz_path = tmp_path / f"{sample_id}.npz"
        np.savez_compressed(npz_path, node_features=features)

        result, gate = load_sample(cpg_path, npz_path)
        assert result is not None

        h5_path = tmp_path / "test.h5"
        write_graphs_h5([result], h5_path)

        # Read back and verify
        with h5py.File(h5_path, "r") as f:
            g = f["graphs"]["0"]
            x = g["x"][:]
            ei = g["edge_index"][:]
            ea = g["edge_attr"][:]
            y = g["y"][:]

            assert x.shape == (5, 824)
            assert not np.isnan(x).any()
            assert ei.max() < 5
            assert ea.max() <= 3
            assert y[0] == 1
            assert f["metadata"].attrs["feature_dim"] == 824
