#!/usr/bin/env python3
"""
tests/test_p4s1_model.py

Phase 4 Story 1: Full 4-CPG Model (model.py) verification.

Mandatory smoke tests:
  1. Full forward pass — correct output shapes
  2. Edge type isolation — TPG-only != AST-only outputs
  3. Missing edge types — no crash, no NaN with zero DFG edges

Additional checks:
  - GroupNorm (not BatchNorm1d) in GGNN normalization layers
  - Checkpoint round-trip: save + load + forward gives same logits
  - save_checkpoint config dict includes 'ggnn_type': 'per_edge_type_gated'
  - Inter-proc stub: use_interproc=True creates interproc_proj Linear layer
  - return_intermediates=True: h_nodes shape is (N, 256)
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.scripts.model.model import (
    StreamGuardModel,
    load_checkpoint,
    save_checkpoint,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_graph(num_nodes: int, num_edges: int, edge_types=None, seed=0):
    """Create a synthetic PyG Data object."""
    torch.manual_seed(seed)
    x = torch.randn(num_nodes, 824)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    if edge_types is None:
        edge_attr = torch.randint(0, 4, (num_edges,))
    else:
        edge_attr = edge_types
    y = torch.tensor([1])
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


def _make_batch(*graphs):
    """Batch a list of graphs."""
    return Batch.from_data_list(list(graphs))


@pytest.fixture(scope="module")
def model():
    """Shared model for read-only tests (no gradient updates)."""
    m = StreamGuardModel()
    m.eval()
    return m


# ═══════════════════════════════════════════════════════════════════════════
# Test 1 — Full forward pass (mandatory smoke test)
# ═══════════════════════════════════════════════════════════════════════════

class TestForwardPass:
    """Verify output dict keys and shapes with a 2-graph batch."""

    def test_output_shapes(self, model):
        g1 = _make_graph(10, 20, seed=1)
        g2 = _make_graph(8, 15, seed=2)
        batch = _make_batch(g1, g2)

        with torch.no_grad():
            out = model(batch)

        assert out["logits"].shape == (2, 2), f"logits: {out['logits'].shape}"
        assert out["cwe_logits"].shape == (2, 12), f"cwe: {out['cwe_logits'].shape}"
        assert out["severity_score"].shape == (2,), f"sev: {out['severity_score'].shape}"
        assert out["embedding"].shape == (2, 128), f"embed: {out['embedding'].shape}"

    def test_single_graph_batch(self, model):
        """batch_size=1 must work (GroupNorm requirement)."""
        g = _make_graph(5, 10, seed=3)
        batch = _make_batch(g)

        with torch.no_grad():
            out = model(batch)

        assert out["logits"].shape == (1, 2)
        assert out["severity_score"].shape == (1,)

    def test_output_keys(self, model):
        g = _make_graph(6, 12, seed=4)
        batch = _make_batch(g)

        with torch.no_grad():
            out = model(batch)

        required_keys = {"logits", "cwe_logits", "severity_score", "embedding"}
        assert required_keys.issubset(out.keys()), f"Missing keys: {required_keys - out.keys()}"

    def test_no_nan_in_outputs(self, model):
        g1 = _make_graph(10, 20, seed=5)
        g2 = _make_graph(8, 15, seed=6)
        batch = _make_batch(g1, g2)

        with torch.no_grad():
            out = model(batch)

        for key, val in out.items():
            assert not torch.isnan(val).any(), f"NaN in {key}"

    def test_large_graph(self, model):
        """200-node graph (max CPG size) must not crash or NaN."""
        g = _make_graph(200, 500, seed=7)
        batch = _make_batch(g)

        with torch.no_grad():
            out = model(batch)

        assert out["logits"].shape == (1, 2)
        assert not torch.isnan(out["logits"]).any()


# ═══════════════════════════════════════════════════════════════════════════
# Test 2 — Edge type isolation (mandatory smoke test)
# ═══════════════════════════════════════════════════════════════════════════

class TestEdgeTypeIsolation:
    """Prove per-edge-type GatedGraphConv produces different outputs per type."""

    def test_ast_vs_tpg_different(self, model):
        """Same graph, ALL edges AST (0) vs ALL edges TPG (3) → different outputs."""
        torch.manual_seed(42)
        num_nodes, num_edges = 10, 20
        x = torch.randn(num_nodes, 824)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))

        # All AST
        d_ast = Data(
            x=x.clone(), edge_index=edge_index.clone(),
            edge_attr=torch.zeros(num_edges, dtype=torch.long), y=torch.tensor([1]),
        )
        # All TPG
        d_tpg = Data(
            x=x.clone(), edge_index=edge_index.clone(),
            edge_attr=torch.full((num_edges,), 3, dtype=torch.long), y=torch.tensor([1]),
        )

        batch_ast = _make_batch(d_ast)
        batch_tpg = _make_batch(d_tpg)

        with torch.no_grad():
            out_ast = model(batch_ast)
            out_tpg = model(batch_tpg)

        # Outputs must differ — if types were ignored, they'd be identical
        assert not torch.allclose(out_ast["logits"], out_tpg["logits"]), \
            "AST-only and TPG-only outputs are identical — edge types NOT differentiated"

    def test_cfg_vs_dfg_different(self, model):
        """CFG (1) vs DFG (2) should also produce different outputs."""
        torch.manual_seed(99)
        num_nodes, num_edges = 8, 15
        x = torch.randn(num_nodes, 824)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))

        d_cfg = Data(
            x=x.clone(), edge_index=edge_index.clone(),
            edge_attr=torch.ones(num_edges, dtype=torch.long), y=torch.tensor([0]),
        )
        d_dfg = Data(
            x=x.clone(), edge_index=edge_index.clone(),
            edge_attr=torch.full((num_edges,), 2, dtype=torch.long), y=torch.tensor([0]),
        )

        with torch.no_grad():
            out_cfg = model(_make_batch(d_cfg))
            out_dfg = model(_make_batch(d_dfg))

        assert not torch.allclose(out_cfg["logits"], out_dfg["logits"]), \
            "CFG-only and DFG-only outputs are identical"


# ═══════════════════════════════════════════════════════════════════════════
# Test 3 — Missing edge types (mandatory smoke test)
# ═══════════════════════════════════════════════════════════════════════════

class TestMissingEdgeTypes:
    """Graphs with missing edge types must not crash or produce NaN."""

    def test_no_dfg_edges(self, model):
        """Zero DFG (type 2) edges — common for simple safe functions."""
        torch.manual_seed(10)
        num_nodes, num_edges = 8, 15
        # Only AST(0), CFG(1), TPG(3) — no DFG(2)
        types = torch.tensor([0, 0, 0, 1, 1, 1, 0, 3, 3, 0, 1, 0, 3, 1, 0])
        assert (types == 2).sum() == 0, "Test setup error: DFG edges present"

        g = _make_graph(num_nodes, num_edges, edge_types=types, seed=10)
        batch = _make_batch(g)

        with torch.no_grad():
            out = model(batch)

        assert out["logits"].shape == (1, 2)
        assert not torch.isnan(out["logits"]).any(), "NaN with missing DFG edges"

    def test_no_tpg_edges(self, model):
        """Zero TPG (type 3) edges."""
        types = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
        g = _make_graph(6, 10, edge_types=types, seed=11)

        with torch.no_grad():
            out = model(_make_batch(g))

        assert not torch.isnan(out["logits"]).any(), "NaN with missing TPG edges"

    def test_only_ast_edges(self, model):
        """Only AST edges — all other types missing."""
        types = torch.zeros(12, dtype=torch.long)
        g = _make_graph(7, 12, edge_types=types, seed=12)

        with torch.no_grad():
            out = model(_make_batch(g))

        assert not torch.isnan(out["logits"]).any(), "NaN with only AST edges"

    def test_zero_edges(self, model):
        """Graph with no edges at all."""
        x = torch.randn(5, 824)
        g = Data(
            x=x,
            edge_index=torch.zeros(2, 0, dtype=torch.long),
            edge_attr=torch.zeros(0, dtype=torch.long),
            y=torch.tensor([0]),
        )
        batch = _make_batch(g)

        with torch.no_grad():
            out = model(batch)

        assert out["logits"].shape == (1, 2)
        assert not torch.isnan(out["logits"]).any(), "NaN with zero edges"


# ═══════════════════════════════════════════════════════════════════════════
# GroupNorm check
# ═══════════════════════════════════════════════════════════════════════════

class TestGroupNorm:
    """Verify GroupNorm (not BatchNorm1d) in GGNN normalization layers."""

    def test_ggnn_norm_is_groupnorm(self, model):
        for i, norm_layer in enumerate(model.ggnn_norm):
            assert isinstance(norm_layer, nn.GroupNorm), \
                f"ggnn_norm[{i}] is {type(norm_layer).__name__}, expected GroupNorm"

    def test_ggnn_norm_not_batchnorm(self, model):
        for i, norm_layer in enumerate(model.ggnn_norm):
            assert not isinstance(norm_layer, nn.BatchNorm1d), \
                f"ggnn_norm[{i}] is BatchNorm1d — breaks at batch_size=1"

    def test_groupnorm_config(self, model):
        """32 groups, 256 channels → 8 channels per group."""
        gn = model.ggnn_norm[0]
        assert gn.num_groups == 32, f"Expected 32 groups, got {gn.num_groups}"
        assert gn.num_channels == 256, f"Expected 256 channels, got {gn.num_channels}"

    def test_three_norm_layers(self, model):
        assert len(model.ggnn_norm) == 3, f"Expected 3 norm layers, got {len(model.ggnn_norm)}"


# ═══════════════════════════════════════════════════════════════════════════
# Checkpoint round-trip
# ═══════════════════════════════════════════════════════════════════════════

class TestCheckpoint:
    """Save + load + forward gives same logits; config dict complete."""

    def test_checkpoint_round_trip(self):
        """Save, load, and verify forward pass gives identical logits."""
        torch.manual_seed(42)
        m1 = StreamGuardModel()
        m1.eval()

        g = _make_graph(8, 15, seed=42)
        batch = _make_batch(g)

        with torch.no_grad():
            out1 = m1(batch)

        # Save
        optimizer = torch.optim.Adam(m1.parameters(), lr=1e-4)
        config = {"base_model": "microsoft/codebert-base", "seed": 42}
        metrics = {"val_f1": 0.85}

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "test_ckpt.pt")
            save_checkpoint(m1, optimizer, epoch=5, metrics=metrics, config=config, path=ckpt_path)

            assert os.path.exists(ckpt_path), "Checkpoint file not created"

            # Load
            m2, opt_state, epoch, loaded_metrics, loaded_config = load_checkpoint(
                ckpt_path, device="cpu"
            )
            m2.eval()

            with torch.no_grad():
                out2 = m2(batch)

        assert torch.allclose(out1["logits"], out2["logits"], atol=1e-6), \
            "Logits differ after checkpoint round-trip"
        assert epoch == 5
        assert loaded_metrics["val_f1"] == 0.85

    def test_checkpoint_config_has_ggnn_type(self):
        """save_checkpoint must include 'ggnn_type': 'per_edge_type_gated'."""
        m = StreamGuardModel()
        optimizer = torch.optim.Adam(m.parameters(), lr=1e-4)
        config = {"base_model": "microsoft/codebert-base"}

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "test_ckpt.pt")
            save_checkpoint(m, optimizer, epoch=0, metrics={}, config=config, path=ckpt_path)

            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        assert "config" in ckpt, "Checkpoint missing 'config' key"
        assert ckpt["config"]["ggnn_type"] == "per_edge_type_gated", \
            f"ggnn_type = {ckpt['config'].get('ggnn_type')}"

    def test_checkpoint_config_fields(self):
        """Verify all required config fields are present."""
        m = StreamGuardModel()
        optimizer = torch.optim.Adam(m.parameters(), lr=1e-4)
        config = {"base_model": "microsoft/codebert-base", "seed": 42}

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "test_ckpt.pt")
            save_checkpoint(m, optimizer, epoch=1, metrics={}, config=config, path=ckpt_path)

            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        cfg = ckpt["config"]
        required_fields = {
            "codebert_model", "node_feature_dim", "use_interproc",
            "freeze_codebert_layers", "num_edge_types", "num_cwe_classes",
            "ggnn_layers", "ggnn_hidden", "ggnn_type", "cpg_components",
            "ablation_config", "seed",
        }
        missing = required_fields - set(cfg.keys())
        assert not missing, f"Missing config fields: {missing}"
        assert cfg["node_feature_dim"] == 824
        assert cfg["num_edge_types"] == 4
        assert cfg["ggnn_layers"] == 3
        assert cfg["ggnn_hidden"] == 256
        assert cfg["cpg_components"] == ["AST", "CFG", "DFG", "TPG"]

    def test_atomic_write(self):
        """Checkpoint write uses atomic tmp + replace (no .tmp file left)."""
        m = StreamGuardModel()
        optimizer = torch.optim.Adam(m.parameters(), lr=1e-4)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "test_ckpt.pt")
            save_checkpoint(m, optimizer, epoch=0, metrics={}, config={}, path=ckpt_path)

            tmp_path = ckpt_path + ".tmp"
            assert not os.path.exists(tmp_path), ".tmp file not cleaned up"
            assert os.path.exists(ckpt_path), "Final checkpoint not created"


# ═══════════════════════════════════════════════════════════════════════════
# Inter-proc stub
# ═══════════════════════════════════════════════════════════════════════════

class TestInterProc:
    """Config E inter-procedural context stub."""

    def test_interproc_false_no_proj(self, model):
        """Default model (use_interproc=False) has no interproc_proj."""
        assert not hasattr(model, "interproc_proj"), \
            "interproc_proj should not exist when use_interproc=False"

    def test_interproc_true_creates_proj(self):
        """use_interproc=True creates interproc_proj Linear layer."""
        m = StreamGuardModel(use_interproc=True)
        assert hasattr(m, "interproc_proj"), "Missing interproc_proj"
        assert isinstance(m.interproc_proj, nn.Linear), \
            f"interproc_proj is {type(m.interproc_proj).__name__}, expected Linear"
        assert m.interproc_proj.in_features == 768, "interproc_proj input should be 768"
        assert m.interproc_proj.out_features == 256, "interproc_proj output should be 256"

    def test_interproc_forward_no_crash(self):
        """Forward pass with use_interproc=True and no callee data doesn't crash."""
        m = StreamGuardModel(use_interproc=True)
        m.eval()
        g = _make_graph(6, 10, seed=20)
        batch = _make_batch(g)

        with torch.no_grad():
            out = m(batch)

        assert out["logits"].shape == (1, 2)

    def test_interproc_fused_dim(self):
        """With interproc, fusion MLP input should be 1536 (1280 + 256)."""
        m = StreamGuardModel(use_interproc=True)
        # First layer of fusion_mlp should accept 1536-d input
        first_linear = m.fusion_mlp[0]
        assert first_linear.in_features == 1536, \
            f"Expected fused_dim=1536 with interproc, got {first_linear.in_features}"


# ═══════════════════════════════════════════════════════════════════════════
# return_intermediates
# ═══════════════════════════════════════════════════════════════════════════

class TestReturnIntermediates:
    """return_intermediates=True exposes internal tensors."""

    def test_h_nodes_shape(self, model):
        """h_nodes should be (N, 256) where N = total nodes in batch."""
        g1 = _make_graph(10, 20, seed=30)
        g2 = _make_graph(8, 15, seed=31)
        batch = _make_batch(g1, g2)

        with torch.no_grad():
            out = model(batch, return_intermediates=True)

        assert "h_nodes" in out, "Missing h_nodes in intermediates"
        assert out["h_nodes"].shape == (18, 256), \
            f"h_nodes shape {out['h_nodes'].shape}, expected (18, 256)"

    def test_graph_embed_shape(self, model):
        g1 = _make_graph(10, 20, seed=32)
        g2 = _make_graph(8, 15, seed=33)
        batch = _make_batch(g1, g2)

        with torch.no_grad():
            out = model(batch, return_intermediates=True)

        assert "graph_embed" in out
        assert out["graph_embed"].shape == (2, 256)

    def test_fused_embedding_shape(self, model):
        g = _make_graph(6, 10, seed=34)
        batch = _make_batch(g)

        with torch.no_grad():
            out = model(batch, return_intermediates=True)

        assert "fused_embedding" in out
        assert out["fused_embedding"].shape == (1, 1280)

    def test_bert_cls_shape(self, model):
        g = _make_graph(6, 10, seed=35)
        batch = _make_batch(g)

        with torch.no_grad():
            out = model(batch, return_intermediates=True)

        assert "bert_cls" in out
        # Without input_ids, bert_cls is zeros (B, 768)
        assert out["bert_cls"].shape == (1, 768)

    def test_intermediates_not_returned_by_default(self, model):
        g = _make_graph(6, 10, seed=36)
        batch = _make_batch(g)

        with torch.no_grad():
            out = model(batch, return_intermediates=False)

        assert "h_nodes" not in out
        assert "graph_embed" not in out
        assert "fused_embedding" not in out
        assert "bert_cls" not in out


# ═══════════════════════════════════════════════════════════════════════════
# Architecture constants
# ═══════════════════════════════════════════════════════════════════════════

class TestArchitectureConstants:
    """Verify class-level constants match the spec."""

    def test_class_constants(self, model):
        assert model.CODEBERT_DIM == 768
        assert model.GGNN_HIDDEN == 256
        assert model.GGNN_LAYERS == 3
        assert model.NUM_EDGE_TYPES == 4
        assert model.FUSED_DIM == 1280
        assert model.MLP_HIDDEN == 512
        assert model.MLP_OUT == 128
        assert model.NUM_CWE == 12

    def test_12_ggnn_convolutions(self, model):
        """4 edge types × 3 layers = 12 GatedGraphConv modules total."""
        total = sum(len(layer_convs) for layer_convs in model.edge_type_convs)
        assert total == 12, f"Expected 12 GatedGraphConv modules, got {total}"

    def test_node_feature_dim_stored(self, model):
        assert model.node_feature_dim == 824

    def test_node_proj_dimensions(self, model):
        assert model.node_proj.in_features == 824
        assert model.node_proj.out_features == 256

    def test_output_head_dimensions(self, model):
        assert model.binary_head.in_features == 128
        assert model.binary_head.out_features == 2
        assert model.cwe_head.in_features == 128
        assert model.cwe_head.out_features == 12
        assert model.severity_head.in_features == 128
        assert model.severity_head.out_features == 1


# ═══════════════════════════════════════════════════════════════════════════
# Edge validation
# ═══════════════════════════════════════════════════════════════════════════

class TestEdgeValidation:
    """Invalid edge types must raise ValueError."""

    def test_invalid_edge_type_raises(self, model):
        types = torch.tensor([0, 1, 2, 5, 0])  # 5 is invalid
        g = _make_graph(5, 5, edge_types=types, seed=40)
        batch = _make_batch(g)

        with pytest.raises(ValueError, match="Invalid edge_attr"):
            with torch.no_grad():
                model(batch)

    def test_valid_edge_types_pass(self, model):
        types = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
        g = _make_graph(6, 8, edge_types=types, seed=41)
        batch = _make_batch(g)

        with torch.no_grad():
            out = model(batch)

        assert out["logits"].shape == (1, 2)


# ═══════════════════════════════════════════════════════════════════════════
# freeze_codebert_layers
# ═══════════════════════════════════════════════════════════════════════════

class TestFreezeLayers:
    """freeze_codebert_layers param freezes lower CodeBERT layers."""

    def test_freeze_9_layers(self):
        m = StreamGuardModel(freeze_codebert_layers=9)
        for i, layer in enumerate(m.codebert.encoder.layer):
            for p in layer.parameters():
                if i < 9:
                    assert not p.requires_grad, f"Layer {i} should be frozen"
                else:
                    assert p.requires_grad, f"Layer {i} should be trainable"

    def test_freeze_0_all_trainable(self):
        m = StreamGuardModel(freeze_codebert_layers=0)
        for i, layer in enumerate(m.codebert.encoder.layer):
            for p in layer.parameters():
                assert p.requires_grad, f"Layer {i} should be trainable"
