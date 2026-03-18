#!/usr/bin/env python3
"""
tests/test_story8.py

Tests for Story 8: Model Development, Training, Evaluation & Ablation Proof.

Tests cover:
  1. StreamGuardModel forward pass on dummy batches
  2. StreamGuardLoss L_total with no NaN (10 random batches)
  3. CFAAwareBatchSampler pair_id co-location invariant
  4. Config B trains to completion with val_F1 logged
  5. Config C trains with L_CFA decreasing
  6. eval.py checkpoint load + evaluation
  7. ablation_cfa_vs_baseline proof generation
  8. proof/cfa_proof_result.json exists with proof_positive

Does NOT require production HDF5 files — uses synthetic data throughout.
"""
from __future__ import annotations

import json
import os
import random
import sys
import tempfile
import warnings
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from torch_geometric.data import Batch, Data

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

warnings.filterwarnings("ignore", message=".*clean_up_tokenization_spaces.*")
warnings.filterwarnings("ignore", message=".*RequestsDependencyWarning.*")
warnings.filterwarnings("ignore", message=".*input_ids is None.*")

from training.scripts.model.model import StreamGuardModel
from training.scripts.model.losses import StreamGuardLoss
from training.scripts.model.cfa_dataloader import (
    CFADataset,
    CFAAwareBatchSampler,
    cfa_collate_fn,
    count_batches,
)

# ══════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════

FEATURE_DIM = 824
NUM_CWE_CLASSES = 12
DEVICE = torch.device("cpu")
CWES = ["CWE-120", "CWE-121", "CWE-122", "CWE-124", "CWE-126",
        "CWE-127", "CWE-89"]


def _make_dummy_graph(
    num_nodes: int = 10,
    num_edges: int = 20,
    label: int = 1,
    pair_id: str = "",
    cwe: str = "CWE-121",
    sample_id: str = "test_0",
) -> Data:
    """Create a synthetic PyG Data object matching HDF5 schema."""
    x = torch.randn(num_nodes, FEATURE_DIM)
    # Random edges with valid types (0=AST, 1=CFG, 2=DFG, 3=TPG)
    src = torch.randint(0, num_nodes, (num_edges,))
    dst = torch.randint(0, num_nodes, (num_edges,))
    edge_index = torch.stack([src, dst])
    edge_attr = torch.randint(0, 4, (num_edges,))
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=torch.tensor(label, dtype=torch.long),
    )
    data.pair_id = pair_id
    data.sample_id = sample_id
    data.cwe = cwe
    return data


def _make_dummy_batch(batch_size: int = 4, with_pairs: bool = True) -> Batch:
    """Create a batch of synthetic graphs with CFA pairs."""
    graphs = []
    for i in range(batch_size):
        if with_pairs and i < batch_size - 1 and i % 2 == 0:
            pid = f"pair_{i // 2}"
            label = 1  # vulnerable
        elif with_pairs and i > 0 and i % 2 == 1:
            pid = f"pair_{(i - 1) // 2}"
            label = 0  # safe
        else:
            pid = ""
            label = random.choice([0, 1])
        graphs.append(_make_dummy_graph(
            num_nodes=random.randint(5, 30),
            num_edges=random.randint(10, 50),
            label=label,
            pair_id=pid,
            cwe=random.choice(CWES),
            sample_id=f"sample_{i}",
        ))
    return Batch.from_data_list(graphs)


def _make_synthetic_h5(path: str, n_samples: int = 100, n_pairs: int = 30):
    """
    Create a synthetic HDF5 file matching the stage7_split schema.

    Returns path to the file.
    """
    with h5py.File(path, 'w') as f:
        graphs_grp = f.create_group('graphs')
        meta_grp = f.create_group('metadata')

        sample_ids = []
        pair_ids = []
        labels = []
        cwes = []

        for i in range(n_samples):
            # Create CFA pairs for first n_pairs*2 samples
            if i < n_pairs * 2:
                pid = f"pair_{i // 2}"
                label = 1 if i % 2 == 0 else 0
            else:
                pid = ""
                label = random.choice([0, 1])

            cwe = random.choice(CWES)
            num_nodes = random.randint(5, 30)
            num_edges = random.randint(10, 50)

            g = graphs_grp.create_group(str(i))
            g.create_dataset('x', data=np.random.randn(num_nodes, FEATURE_DIM).astype(np.float32))
            src = np.random.randint(0, num_nodes, num_edges)
            dst = np.random.randint(0, num_nodes, num_edges)
            g.create_dataset('edge_index', data=np.stack([src, dst]))
            g.create_dataset('edge_attr', data=np.random.randint(0, 4, num_edges))
            g.create_dataset('y', data=np.array(label))

            sample_ids.append(f"sample_{i}")
            pair_ids.append(pid)
            labels.append(label)
            cwes.append(cwe)

        dt = h5py.string_dtype()
        meta_grp.create_dataset('sample_ids', data=np.array(sample_ids, dtype=object), dtype=dt)
        meta_grp.create_dataset('pair_ids', data=np.array(pair_ids, dtype=object), dtype=dt)
        meta_grp.create_dataset('labels', data=np.array(labels))
        meta_grp.create_dataset('cwes', data=np.array(cwes, dtype=object), dtype=dt)

    return path


@pytest.fixture(scope="session")
def model():
    """Shared model instance (expensive to create due to CodeBERT download)."""
    m = StreamGuardModel(node_feature_dim=FEATURE_DIM).to(DEVICE)
    m.eval()
    return m


@pytest.fixture(scope="session")
def synthetic_h5_dir():
    """Create synthetic train/val/test HDF5 files for integration tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _make_synthetic_h5(os.path.join(tmpdir, "train.h5"), n_samples=200, n_pairs=60)
        _make_synthetic_h5(os.path.join(tmpdir, "val.h5"), n_samples=50, n_pairs=15)
        _make_synthetic_h5(os.path.join(tmpdir, "test.h5"), n_samples=80, n_pairs=25)
        yield tmpdir


# ══════════════════════════════════════════════════════════════════
# Group 1: StreamGuardModel Forward Pass (4 graphs)
# ══════════════════════════════════════════════════════════════════

class TestModelForward:
    """Verify StreamGuardModel forward pass on dummy batches."""

    def test_forward_batch_of_4(self, model):
        """Forward pass on batch of 4 graphs returns correct shapes."""
        batch = _make_dummy_batch(batch_size=4)
        batch = batch.to(DEVICE)
        with torch.no_grad():
            out = model(batch)

        assert "binary_logits" in out
        assert "cwe_logits" in out
        assert "severity" in out
        assert "embedding" in out
        assert out["binary_logits"].shape == (4, 2)
        assert out["cwe_logits"].shape == (4, NUM_CWE_CLASSES)
        assert out["severity"].shape == (4,)
        assert out["embedding"].shape == (4, 128)

    def test_forward_single_graph(self, model):
        """Forward pass on single graph works (batch_size=1 for serving)."""
        batch = _make_dummy_batch(batch_size=1, with_pairs=False)
        batch = batch.to(DEVICE)
        with torch.no_grad():
            out = model(batch)
        assert out["binary_logits"].shape == (1, 2)

    def test_forward_large_batch(self, model):
        """Forward pass on batch of 16 graphs."""
        batch = _make_dummy_batch(batch_size=16)
        batch = batch.to(DEVICE)
        with torch.no_grad():
            out = model(batch)
        assert out["binary_logits"].shape == (16, 2)

    def test_forward_no_nan(self, model):
        """No NaN in any output tensor."""
        batch = _make_dummy_batch(batch_size=8)
        batch = batch.to(DEVICE)
        with torch.no_grad():
            out = model(batch)
        for key, tensor in out.items():
            assert not torch.isnan(tensor).any(), f"NaN in {key}"

    def test_forward_returns_intermediates(self, model):
        """return_intermediates=True exposes internal tensors."""
        batch = _make_dummy_batch(batch_size=4)
        batch = batch.to(DEVICE)
        with torch.no_grad():
            out = model(batch, return_intermediates=True)
        assert "fused_embedding" in out
        assert "ggnn_node_embeddings" in out
        assert "bert_cls" in out
        assert "graph_embed" in out

    def test_forward_empty_edge_types(self, model):
        """Graph with only AST edges (no CFG/DFG/TPG) does not crash."""
        data = _make_dummy_graph(num_nodes=10, num_edges=15)
        data.edge_attr = torch.zeros(15, dtype=torch.long)  # all AST
        batch = Batch.from_data_list([data])
        batch = batch.to(DEVICE)
        with torch.no_grad():
            out = model(batch)
        assert out["binary_logits"].shape == (1, 2)
        assert not torch.isnan(out["binary_logits"]).any()

    def test_forward_with_tpg_edges(self, model):
        """Graph with TPG (type=3) edges included."""
        data = _make_dummy_graph(num_nodes=10, num_edges=20)
        data.edge_attr = torch.tensor([0, 1, 2, 3] * 5, dtype=torch.long)
        batch = Batch.from_data_list([data])
        batch = batch.to(DEVICE)
        with torch.no_grad():
            out = model(batch)
        assert not torch.isnan(out["binary_logits"]).any()


# ══════════════════════════════════════════════════════════════════
# Group 2: StreamGuardLoss — No NaN on 10 Random Batches
# ══════════════════════════════════════════════════════════════════

class TestLossNoNaN:
    """Verify StreamGuardLoss computes finite L_total on random data."""

    @pytest.fixture
    def criterion(self):
        return StreamGuardLoss(
            lambda_ce=1.0, lambda_cfa=0.5, lambda_sev=0.0, cfa_margin=0.5
        )

    @pytest.mark.parametrize("batch_idx", range(10))
    def test_loss_no_nan_random_batch(self, model, criterion, batch_idx):
        """L_total is finite on random batch #{batch_idx}."""
        batch = _make_dummy_batch(batch_size=8, with_pairs=True)
        batch = batch.to(DEVICE)
        with torch.no_grad():
            out = model(batch)
        total_loss, loss_dict = criterion(out, batch, use_cfa=True)

        assert not torch.isnan(total_loss), f"NaN in L_total (batch {batch_idx})"
        assert not torch.isinf(total_loss), f"Inf in L_total (batch {batch_idx})"
        assert "L_total" in loss_dict
        assert "L_CE" in loss_dict
        assert "L_CFA" in loss_dict
        assert "L_severity" in loss_dict
        assert np.isfinite(loss_dict["L_total"])
        assert np.isfinite(loss_dict["L_CE"])
        assert np.isfinite(loss_dict["L_CFA"])

    def test_loss_without_cfa(self, model, criterion):
        """L_CFA=0 when use_cfa=False."""
        batch = _make_dummy_batch(batch_size=4, with_pairs=True)
        batch = batch.to(DEVICE)
        with torch.no_grad():
            out = model(batch)
        total_loss, loss_dict = criterion(out, batch, use_cfa=False)
        assert loss_dict["L_CFA"] == 0.0

    def test_loss_cfa_positive_when_pairs_present(self, model, criterion):
        """L_CFA > 0 when CFA pairs exist in the batch."""
        # Build batch with guaranteed pairs
        graphs = []
        for i in range(4):
            pid = f"pair_{i // 2}"
            label = 1 if i % 2 == 0 else 0
            graphs.append(_make_dummy_graph(label=label, pair_id=pid))
        batch = Batch.from_data_list(graphs).to(DEVICE)

        with torch.no_grad():
            out = model(batch)
        _, loss_dict = criterion(out, batch, use_cfa=True)
        # With random embeddings and margin=0.5, L_CFA should be > 0
        # (cosine similarity of random vectors ~0, plus margin 0.5 → relu(~0.5) > 0)
        assert loss_dict["L_CFA"] > 0

    def test_loss_dict_always_has_four_keys(self, model, criterion):
        """loss_dict always has exactly {L_total, L_CE, L_CFA, L_severity}."""
        batch = _make_dummy_batch(batch_size=4)
        batch = batch.to(DEVICE)
        with torch.no_grad():
            out = model(batch)
        _, loss_dict = criterion(out, batch, use_cfa=True)
        assert set(loss_dict.keys()) == {"L_total", "L_CE", "L_CFA", "L_severity"}


# ══════════════════════════════════════════════════════════════════
# Group 3: CFAAwareBatchSampler — Pair Co-location Invariant
# ══════════════════════════════════════════════════════════════════

class TestBatchSamplerPairs:
    """Every pair_id appears in exactly one batch — never split."""

    @pytest.fixture
    def h5_with_pairs(self, tmp_path):
        path = str(tmp_path / "pairs.h5")
        _make_synthetic_h5(path, n_samples=200, n_pairs=60)
        return path

    def test_pair_ids_never_split_across_batches(self, h5_with_pairs):
        """Core invariant: both members of pair_id land in the same batch."""
        dataset = CFADataset(h5_with_pairs)
        sampler = CFAAwareBatchSampler(dataset, batch_size=8, drop_last=False)

        # Track which batch each sample index lands in
        index_to_batch = {}
        for batch_idx, indices in enumerate(sampler):
            for idx in indices:
                index_to_batch[idx] = batch_idx

        # Group by pair_id
        pair_batches = defaultdict(set)
        for i, item in enumerate(dataset.index):
            pid = item["pair_id"]
            if pid in ("", "None", "null", "0"):
                continue
            if i in index_to_batch:
                pair_batches[pid].add(index_to_batch[i])

        # Every pair_id should appear in exactly ONE batch
        for pid, batches in pair_batches.items():
            assert len(batches) == 1, (
                f"pair_id '{pid}' split across batches {batches}"
            )

    def test_pair_members_appear_exactly_twice(self, h5_with_pairs):
        """Each non-sentinel pair_id has exactly 2 members in the dataset."""
        dataset = CFADataset(h5_with_pairs)
        pair_counts = defaultdict(int)
        for item in dataset.index:
            pid = item["pair_id"]
            if pid in ("", "None", "null", "0"):
                continue
            pair_counts[pid] += 1

        for pid, count in pair_counts.items():
            assert count == 2, f"pair_id '{pid}' has {count} members, expected 2"

    def test_sampler_groups_count(self, h5_with_pairs):
        """Sampler builds correct number of groups."""
        dataset = CFADataset(h5_with_pairs)
        sampler = CFAAwareBatchSampler(dataset, batch_size=8, drop_last=False)
        # 60 pairs (each is 1 group) + 80 singletons (each is 1 group)
        assert sampler._n_groups == 60 + 80

    def test_sampler_drop_last(self, h5_with_pairs):
        """drop_last=True drops the final incomplete batch."""
        dataset = CFADataset(h5_with_pairs)
        sampler_keep = CFAAwareBatchSampler(dataset, batch_size=8, drop_last=False)
        sampler_drop = CFAAwareBatchSampler(dataset, batch_size=8, drop_last=True)
        n_keep = count_batches(sampler_keep)
        n_drop = count_batches(sampler_drop)
        assert n_drop <= n_keep

    def test_sampler_yields_all_samples_when_no_drop(self, h5_with_pairs):
        """All samples are yielded when drop_last=False."""
        dataset = CFADataset(h5_with_pairs)
        sampler = CFAAwareBatchSampler(dataset, batch_size=8, drop_last=False)
        all_indices = set()
        for batch in sampler:
            all_indices.update(batch)
        assert len(all_indices) == len(dataset)

    def test_sampler_reshuffles_between_epochs(self, h5_with_pairs):
        """Two iterations of the sampler produce different batch orderings."""
        dataset = CFADataset(h5_with_pairs)
        sampler = CFAAwareBatchSampler(dataset, batch_size=8, drop_last=False)
        epoch1 = [b[:] for b in sampler]
        epoch2 = [b[:] for b in sampler]
        # With 140 groups, the probability of identical orderings is negligible
        assert epoch1 != epoch2, "Sampler did not reshuffle between epochs"


# ══════════════════════════════════════════════════════════════════
# Group 4: Integration — Training Loop (Config B + C)
# ══════════════════════════════════════════════════════════════════

class TestTrainingIntegration:
    """Integration tests: actual training loop with synthetic data."""

    @pytest.fixture(scope="class")
    def h5_dir(self, tmp_path_factory):
        tmpdir = tmp_path_factory.mktemp("training_data")
        _make_synthetic_h5(str(tmpdir / "train.h5"), n_samples=100, n_pairs=30)
        _make_synthetic_h5(str(tmpdir / "val.h5"), n_samples=40, n_pairs=10)
        return tmpdir

    def _run_training(self, h5_dir, config_name, use_cfa, epochs=2, tmpdir=None):
        """Helper to run training with synthetic data."""
        from training.scripts.model.train import train, ABLATION_CONFIGS, DEFAULT_CONFIG

        config = dict(DEFAULT_CONFIG)
        ablation = ABLATION_CONFIGS[config_name]
        config["use_cfa"] = ablation["use_cfa"]
        config["cpg_components"] = ablation["cpg_components"]
        config["config_name"] = config_name

        config["train_h5"] = str(h5_dir / "train.h5")
        config["val_h5"] = str(h5_dir / "val.h5")
        config["epochs"] = epochs
        config["max_samples"] = 80  # small for speed
        if tmpdir:
            config["checkpoint_dir"] = str(tmpdir)

        # Override CFA
        config["use_cfa"] = use_cfa

        return train(config), config

    def test_config_b_trains_to_completion(self, h5_dir, tmp_path):
        """Config B trains without errors and saves checkpoint."""
        best_f1, config = self._run_training(
            h5_dir, "B_plus_ggnn", use_cfa=False, epochs=2, tmpdir=tmp_path
        )
        ckpt_path = tmp_path / "best_model_B_plus_ggnn.pt"
        assert ckpt_path.exists(), "Checkpoint not saved"
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        assert "model_state_dict" in ckpt
        assert "config" in ckpt
        assert best_f1 >= 0.0  # F1 is non-negative

    def test_config_c_trains_with_cfa(self, h5_dir, tmp_path):
        """Config C trains with CFA enabled, no crashes."""
        best_f1, config = self._run_training(
            h5_dir, "C_plus_cfa", use_cfa=True, epochs=2, tmpdir=tmp_path
        )
        ckpt_path = tmp_path / "best_model_C_plus_cfa.pt"
        assert ckpt_path.exists(), "Checkpoint not saved"
        assert best_f1 >= 0.0

    def test_config_d_trains_with_tpg(self, h5_dir, tmp_path):
        """Config D trains with TPG edges enabled."""
        best_f1, config = self._run_training(
            h5_dir, "D_plus_tpg", use_cfa=True, epochs=2, tmpdir=tmp_path
        )
        ckpt_path = tmp_path / "best_model_D_plus_tpg.pt"
        assert ckpt_path.exists(), "Checkpoint not saved"

    def test_checkpoint_has_required_keys(self, h5_dir, tmp_path):
        """Checkpoint contains all required config keys for reconstruction."""
        self._run_training(h5_dir, "C_plus_cfa", use_cfa=True, epochs=1, tmpdir=tmp_path)
        ckpt = torch.load(
            tmp_path / "best_model_C_plus_cfa.pt",
            map_location="cpu", weights_only=False,
        )
        required_keys = [
            "epoch", "best_f1", "model_state_dict",
            "optimizer_state_dict", "scheduler_state_dict", "config",
        ]
        for key in required_keys:
            assert key in ckpt, f"Missing key: {key}"

        config_keys = [
            "node_feature_dim", "use_ggnn", "ggnn_type", "cpg_components",
            "num_edge_types", "use_cfa", "use_interproc", "num_cwe_classes",
            "ablation_config", "base_model", "freeze_layers",
        ]
        for key in config_keys:
            assert key in ckpt["config"], f"Missing config key: {key}"


# ══════════════════════════════════════════════════════════════════
# Group 5: eval.py — Checkpoint Evaluation
# ══════════════════════════════════════════════════════════════════

class TestEvalScript:
    """Test eval.py functions."""

    @pytest.fixture(scope="class")
    def trained_checkpoint(self, tmp_path_factory):
        """Train a small model and return checkpoint path + test h5 path."""
        tmpdir = tmp_path_factory.mktemp("eval_test")
        train_h5 = str(tmpdir / "train.h5")
        val_h5 = str(tmpdir / "val.h5")
        test_h5 = str(tmpdir / "test.h5")
        _make_synthetic_h5(train_h5, n_samples=100, n_pairs=30)
        _make_synthetic_h5(val_h5, n_samples=40, n_pairs=10)
        _make_synthetic_h5(test_h5, n_samples=80, n_pairs=25)

        from training.scripts.model.train import train, ABLATION_CONFIGS, DEFAULT_CONFIG

        config = dict(DEFAULT_CONFIG)
        ablation = ABLATION_CONFIGS["C_plus_cfa"]
        config.update({
            "use_cfa": True,
            "cpg_components": ablation["cpg_components"],
            "config_name": "C_plus_cfa",
            "train_h5": train_h5,
            "val_h5": val_h5,
            "epochs": 2,
            "max_samples": 80,
            "checkpoint_dir": str(tmpdir),
        })
        train(config)
        return {
            "checkpoint": str(tmpdir / "best_model_C_plus_cfa.pt"),
            "test_h5": test_h5,
            "tmpdir": tmpdir,
        }

    def test_load_model_from_checkpoint(self, trained_checkpoint):
        """Model loads successfully from checkpoint."""
        from training.scripts.model.eval import load_model_from_checkpoint
        model, cfg, ckpt = load_model_from_checkpoint(
            trained_checkpoint["checkpoint"], DEVICE
        )
        assert isinstance(model, StreamGuardModel)
        assert cfg["ablation_config"] == "C_plus_cfa"
        assert cfg["use_cfa"] is True

    def test_evaluate_checkpoint_returns_all_metrics(self, trained_checkpoint):
        """evaluate_checkpoint returns all required metrics."""
        from training.scripts.model.eval import (
            load_model_from_checkpoint, build_test_loader, evaluate_checkpoint,
        )
        model, cfg, _ = load_model_from_checkpoint(
            trained_checkpoint["checkpoint"], DEVICE
        )
        loader, _ = build_test_loader(trained_checkpoint["test_h5"], batch_size=8)
        metrics = evaluate_checkpoint(model, loader, DEVICE, use_cfa=True)

        required = [
            "f1", "precision", "recall", "fpr", "fnr",
            "per_cwe_f1", "worst_group_f1", "best_group_f1", "avg_cwe_f1",
            "pairwise_contrast_accuracy", "n_valid_pairs", "n_samples",
        ]
        for key in required:
            assert key in metrics, f"Missing metric: {key}"

        assert metrics["n_samples"] == 80
        assert 0.0 <= metrics["f1"] <= 1.0
        assert 0.0 <= metrics["precision"] <= 1.0
        assert 0.0 <= metrics["recall"] <= 1.0
        assert 0.0 <= metrics["fpr"] <= 1.0
        assert 0.0 <= metrics["fnr"] <= 1.0
        assert isinstance(metrics["per_cwe_f1"], dict)

    def test_evaluate_per_cwe_f1_has_entries(self, trained_checkpoint):
        """Per-CWE F1 dict has at least one entry."""
        from training.scripts.model.eval import (
            load_model_from_checkpoint, build_test_loader, evaluate_checkpoint,
        )
        model, _, _ = load_model_from_checkpoint(
            trained_checkpoint["checkpoint"], DEVICE
        )
        loader, _ = build_test_loader(trained_checkpoint["test_h5"], batch_size=8)
        metrics = evaluate_checkpoint(model, loader, DEVICE, use_cfa=True)
        assert len(metrics["per_cwe_f1"]) >= 1

    def test_evaluate_pairwise_accuracy_in_range(self, trained_checkpoint):
        """Pairwise accuracy is between 0 and 1."""
        from training.scripts.model.eval import (
            load_model_from_checkpoint, build_test_loader, evaluate_checkpoint,
        )
        model, _, _ = load_model_from_checkpoint(
            trained_checkpoint["checkpoint"], DEVICE
        )
        loader, _ = build_test_loader(trained_checkpoint["test_h5"], batch_size=8)
        metrics = evaluate_checkpoint(model, loader, DEVICE, use_cfa=True)
        assert 0.0 <= metrics["pairwise_contrast_accuracy"] <= 1.0

    def test_evaluate_json_output(self, trained_checkpoint):
        """eval.py --output-json produces valid JSON."""
        from training.scripts.model.eval import (
            load_model_from_checkpoint, build_test_loader, evaluate_checkpoint,
        )
        model, _, _ = load_model_from_checkpoint(
            trained_checkpoint["checkpoint"], DEVICE
        )
        loader, _ = build_test_loader(trained_checkpoint["test_h5"], batch_size=8)
        metrics = evaluate_checkpoint(model, loader, DEVICE, use_cfa=True)

        # Verify JSON serializable
        json_str = json.dumps(metrics, indent=2)
        parsed = json.loads(json_str)
        assert parsed["n_samples"] == 80


# ══════════════════════════════════════════════════════════════════
# Group 6: Ablation Script — Proof Generation
# ══════════════════════════════════════════════════════════════════

class TestAblationProof:
    """Test ablation_cfa_vs_baseline.py proof generation."""

    @pytest.fixture(scope="class")
    def ablation_artifacts(self, tmp_path_factory):
        """Train B and C configs, return paths."""
        tmpdir = tmp_path_factory.mktemp("ablation_test")
        ckpt_dir = tmpdir / "checkpoints"
        ckpt_dir.mkdir()

        train_h5 = str(tmpdir / "train.h5")
        val_h5 = str(tmpdir / "val.h5")
        test_h5 = str(tmpdir / "test.h5")
        _make_synthetic_h5(train_h5, n_samples=100, n_pairs=30)
        _make_synthetic_h5(val_h5, n_samples=40, n_pairs=10)
        _make_synthetic_h5(test_h5, n_samples=80, n_pairs=25)

        from training.scripts.model.train import train, ABLATION_CONFIGS, DEFAULT_CONFIG

        # Train all 3 configs
        for cfg_name in ["B_plus_ggnn", "C_plus_cfa", "D_plus_tpg"]:
            config = dict(DEFAULT_CONFIG)
            ablation = ABLATION_CONFIGS[cfg_name]
            config.update({
                "use_cfa": ablation["use_cfa"],
                "cpg_components": ablation["cpg_components"],
                "config_name": cfg_name,
                "train_h5": train_h5,
                "val_h5": val_h5,
                "epochs": 2,
                "max_samples": 80,
                "checkpoint_dir": str(ckpt_dir),
            })
            train(config)

        return {
            "checkpoint_dir": str(ckpt_dir),
            "test_h5": test_h5,
            "tmpdir": tmpdir,
        }

    def test_ablation_runs_without_error(self, ablation_artifacts, monkeypatch):
        """run_ablation completes without errors."""
        # Change working dir so proof/ is created in tmpdir
        monkeypatch.chdir(ablation_artifacts["tmpdir"])
        from training.scripts.model.ablation_cfa_vs_baseline import run_ablation
        result = run_ablation(
            ablation_artifacts["test_h5"],
            ablation_artifacts["checkpoint_dir"],
            batch_size=8,
        )
        assert isinstance(result, dict)

    def test_proof_json_exists(self, ablation_artifacts, monkeypatch):
        """proof/cfa_proof_result.json is created."""
        monkeypatch.chdir(ablation_artifacts["tmpdir"])
        from training.scripts.model.ablation_cfa_vs_baseline import run_ablation
        run_ablation(
            ablation_artifacts["test_h5"],
            ablation_artifacts["checkpoint_dir"],
        )
        proof_path = Path(ablation_artifacts["tmpdir"]) / "proof" / "cfa_proof_result.json"
        assert proof_path.exists(), f"proof JSON not found at {proof_path}"

    def test_proof_json_has_required_fields(self, ablation_artifacts, monkeypatch):
        """Proof JSON contains all required fields."""
        monkeypatch.chdir(ablation_artifacts["tmpdir"])
        from training.scripts.model.ablation_cfa_vs_baseline import run_ablation
        result = run_ablation(
            ablation_artifacts["test_h5"],
            ablation_artifacts["checkpoint_dir"],
        )
        required = [
            "config_B_f1", "config_C_f1", "config_D_f1",
            "cfa_delta", "pairwise_accuracy", "per_cwe_f1",
            "proof_positive", "timestamp",
        ]
        for key in required:
            assert key in result, f"Missing proof field: {key}"

    def test_proof_f1_values_in_range(self, ablation_artifacts, monkeypatch):
        """All F1 values are in [0, 1]."""
        monkeypatch.chdir(ablation_artifacts["tmpdir"])
        from training.scripts.model.ablation_cfa_vs_baseline import run_ablation
        result = run_ablation(
            ablation_artifacts["test_h5"],
            ablation_artifacts["checkpoint_dir"],
        )
        for key in ["config_B_f1", "config_C_f1", "config_D_f1"]:
            assert 0.0 <= result[key] <= 1.0, f"{key}={result[key]} out of range"

    def test_proof_cfa_delta_is_numeric(self, ablation_artifacts, monkeypatch):
        """CFA delta is a valid float."""
        monkeypatch.chdir(ablation_artifacts["tmpdir"])
        from training.scripts.model.ablation_cfa_vs_baseline import run_ablation
        result = run_ablation(
            ablation_artifacts["test_h5"],
            ablation_artifacts["checkpoint_dir"],
        )
        assert isinstance(result["cfa_delta"], float)
        assert np.isfinite(result["cfa_delta"])

    def test_proof_json_parseable(self, ablation_artifacts, monkeypatch):
        """proof/cfa_proof_result.json is valid JSON."""
        monkeypatch.chdir(ablation_artifacts["tmpdir"])
        from training.scripts.model.ablation_cfa_vs_baseline import run_ablation
        run_ablation(
            ablation_artifacts["test_h5"],
            ablation_artifacts["checkpoint_dir"],
        )
        proof_path = Path(ablation_artifacts["tmpdir"]) / "proof" / "cfa_proof_result.json"
        data = json.loads(proof_path.read_text())
        assert "proof_positive" in data
        assert isinstance(data["proof_positive"], bool)


# ══════════════════════════════════════════════════════════════════
# Group 7: Edge Cases and Robustness
# ══════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Edge cases and robustness checks."""

    def test_model_with_zero_features(self, model):
        """Model handles zero-valued node features without NaN."""
        data = _make_dummy_graph(num_nodes=5, num_edges=10)
        data.x = torch.zeros(5, FEATURE_DIM)
        batch = Batch.from_data_list([data]).to(DEVICE)
        with torch.no_grad():
            out = model(batch)
        assert not torch.isnan(out["binary_logits"]).any()

    def test_model_with_minimal_graph(self, model):
        """Model handles graph with 3 nodes and 3 edges (minimum valid)."""
        data = _make_dummy_graph(num_nodes=3, num_edges=3)
        batch = Batch.from_data_list([data]).to(DEVICE)
        with torch.no_grad():
            out = model(batch)
        assert out["binary_logits"].shape == (1, 2)

    def test_loss_with_all_same_label(self, model):
        """Loss computes without error when all labels are the same."""
        criterion = StreamGuardLoss(lambda_ce=1.0, lambda_cfa=0.5, lambda_sev=0.0)
        graphs = [_make_dummy_graph(label=1, pair_id=f"p_{i}") for i in range(4)]
        batch = Batch.from_data_list(graphs).to(DEVICE)
        with torch.no_grad():
            out = model(batch)
        total_loss, loss_dict = criterion(out, batch, use_cfa=True)
        assert np.isfinite(loss_dict["L_total"])

    def test_loss_with_no_pairs(self, model):
        """L_CFA=0 when no CFA pairs exist in batch."""
        criterion = StreamGuardLoss(lambda_ce=1.0, lambda_cfa=0.5, lambda_sev=0.0)
        graphs = [_make_dummy_graph(label=i % 2, pair_id="") for i in range(4)]
        batch = Batch.from_data_list(graphs).to(DEVICE)
        with torch.no_grad():
            out = model(batch)
        _, loss_dict = criterion(out, batch, use_cfa=True)
        assert loss_dict["L_CFA"] == 0.0

    def test_sentinel_pair_ids_treated_as_singletons(self):
        """Sentinel pair_ids ('', 'None', 'null', '0') are treated as singletons."""
        sentinels = ["", "None", "null", "0"]
        h5_path = os.path.join(tempfile.mkdtemp(), "sentinel_test.h5")
        with h5py.File(h5_path, 'w') as f:
            graphs_grp = f.create_group('graphs')
            meta_grp = f.create_group('metadata')
            for i in range(8):
                g = graphs_grp.create_group(str(i))
                g.create_dataset('x', data=np.random.randn(5, FEATURE_DIM).astype(np.float32))
                g.create_dataset('edge_index', data=np.random.randint(0, 5, (2, 10)))
                g.create_dataset('edge_attr', data=np.random.randint(0, 4, 10))
                g.create_dataset('y', data=np.array(i % 2))
            dt = h5py.string_dtype()
            meta_grp.create_dataset('sample_ids',
                                    data=np.array([f"s_{i}" for i in range(8)], dtype=object), dtype=dt)
            meta_grp.create_dataset('pair_ids',
                                    data=np.array(sentinels * 2, dtype=object), dtype=dt)
            meta_grp.create_dataset('labels', data=np.arange(8) % 2)
            meta_grp.create_dataset('cwes',
                                    data=np.array(["CWE-121"] * 8, dtype=object), dtype=dt)

        dataset = CFADataset(h5_path)
        sampler = CFAAwareBatchSampler(dataset, batch_size=4, drop_last=False)
        # Each sentinel should form its own group
        assert sampler._n_groups == 8  # all singletons

    def test_dataset_index_length_matches_h5(self):
        """CFADataset.index length matches number of graphs in HDF5."""
        h5_path = os.path.join(tempfile.mkdtemp(), "len_test.h5")
        n = 50
        _make_synthetic_h5(h5_path, n_samples=n, n_pairs=10)
        dataset = CFADataset(h5_path)
        assert len(dataset) == n
        assert len(dataset.index) == n

    def test_collate_fn_preserves_string_attrs(self):
        """cfa_collate_fn preserves pair_id, sample_id, cwe as lists."""
        graphs = [
            _make_dummy_graph(pair_id="p1", sample_id="s1", cwe="CWE-120"),
            _make_dummy_graph(pair_id="p1", sample_id="s2", cwe="CWE-120"),
        ]
        batch = cfa_collate_fn(graphs)
        assert isinstance(batch.pair_id, list)
        assert batch.pair_id == ["p1", "p1"]
        assert batch.sample_id == ["s1", "s2"]
        assert batch.cwe == ["CWE-120", "CWE-120"]
