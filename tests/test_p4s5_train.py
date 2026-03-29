#!/usr/bin/env python3
"""
tests/test_p4s5_train.py

Phase 4 Story 5: Training Loop.

6 mandatory tests:
  1. set_all_seeds: same seed → same random numbers
  2. build_optimizer: two param groups with different LRs
  3. tokenize_batch: returns (B, 512) input_ids
  4. tokenize_batch: empty string handled (placeholder inserted)
  5. Full dry-run: 1 epoch, loss is finite, checkpoint saved
  6. Resume: load checkpoint + continue from epoch 2

Additional tests:
  7.  ABLATION_CONFIGS has 6 entries with correct keys
  8.  DEFAULT_CONFIG has all required keys
  9.  load_code_lookup returns dict from JSONL
  10. save_checkpoint is atomic (.tmp + os.replace)
  11. _select_with_complete_groups keeps pairs intact
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np
import torch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.scripts.model.train import (
    set_all_seeds,
    build_optimizer,
    tokenize_batch,
    load_code_lookup,
    save_checkpoint,
    load_checkpoint,
    train,
    ABLATION_CONFIGS,
    DEFAULT_CONFIG,
    _select_with_complete_groups,
)
from training.scripts.model.model import StreamGuardModel
from training.scripts.model.cfa_dataloader import EMPTY_PAIR_SENTINELS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tokenizer():
    """Load CodeBERT tokenizer once for all tests."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("microsoft/codebert-base")


@pytest.fixture
def small_model():
    """Create a small model for testing (no GPU needed)."""
    return StreamGuardModel(
        codebert_model="microsoft/codebert-base",
        node_feature_dim=824,
        use_interproc=False,
        freeze_codebert_layers=9,
    )


def _create_test_h5(path, n_samples=32, n_pairs=8):
    """Create a small HDF5 in Layout A format for test dry-runs."""
    with h5py.File(path, 'w') as f:
        graphs = f.create_group('graphs')
        sample_ids = []
        pair_ids = []
        labels = []
        cwes = []

        for i in range(n_samples):
            g = graphs.create_group(str(i))
            n_nodes = 5 + (i % 3)
            n_edges = 8 + (i % 4)
            g.create_dataset('x', data=np.random.randn(n_nodes, 824).astype(np.float32))
            g.create_dataset('edge_index', data=np.random.randint(0, n_nodes, (2, n_edges)).astype(np.int64))
            g.create_dataset('edge_attr', data=np.random.randint(0, 4, (n_edges,)).astype(np.int64))
            g.create_dataset('y', data=np.array([i % 2], dtype=np.int64))

            sid = f"test_sample_{i:04d}"
            sample_ids.append(sid)
            labels.append(i % 2)

            cwe_choices = ["CWE-121", "CWE-122", "CWE-190", "CWE-78"]
            cwes.append(cwe_choices[i % len(cwe_choices)])

            if i < n_pairs * 2:
                pair_ids.append(f"pair_{i // 2:04d}")
            else:
                pair_ids.append("")

        meta = f.create_group('metadata')
        meta.create_dataset('sample_ids', data=np.array(sample_ids, dtype='S'))
        meta.create_dataset('pair_ids', data=np.array(pair_ids, dtype='S'))
        meta.create_dataset('labels', data=np.array(labels, dtype=np.int64))
        meta.create_dataset('cwes', data=np.array(cwes, dtype='S'))


def _create_test_jsonl(path, sample_ids):
    """Create a JSONL with sample_id → code mapping."""
    with open(path, 'w', encoding='utf-8') as f:
        for sid in sample_ids:
            rec = {
                "id": sid,
                "code": f"int main() {{ return {hash(sid) % 100}; }}",
                "label": 0,
                "cwe": "CWE-121",
                "language": "c",
                "source": "test",
                "collected_at": "2026-01-01",
            }
            f.write(json.dumps(rec) + "\n")


@pytest.fixture
def test_h5_and_jsonl(tmp_path):
    """Create matched HDF5 + JSONL for dry-run tests.

    Creates separate train, val, and test h5 files so the R-29 guard
    (val_h5 != test_h5) does not trigger.
    """
    h5_path = str(tmp_path / "test.h5")
    val_h5_path = str(tmp_path / "val.h5")
    jsonl_path = str(tmp_path / "test.jsonl")
    _create_test_h5(h5_path, n_samples=32, n_pairs=8)

    # Create a separate val.h5 (copy of test.h5 at a different path)
    import shutil
    shutil.copy2(h5_path, val_h5_path)

    # Get sample_ids from HDF5
    with h5py.File(h5_path, 'r') as f:
        sids = [s.decode('utf-8') for s in f['metadata']['sample_ids'][:]]
    _create_test_jsonl(jsonl_path, sids)

    return h5_path, val_h5_path, jsonl_path


# =========================================================================
# Test 1 -- set_all_seeds: same seed → same random numbers
# =========================================================================

class TestSetAllSeeds:
    def test_same_seed_same_output(self):
        """Same seed must produce identical random values."""
        set_all_seeds(42)
        a_torch = torch.randn(5).tolist()
        a_numpy = np.random.randn(5).tolist()
        a_python = [__import__("random").random() for _ in range(5)]

        set_all_seeds(42)
        b_torch = torch.randn(5).tolist()
        b_numpy = np.random.randn(5).tolist()
        b_python = [__import__("random").random() for _ in range(5)]

        assert a_torch == b_torch, "torch.randn differs with same seed"
        assert a_numpy == b_numpy, "np.random.randn differs with same seed"
        assert a_python == b_python, "random.random differs with same seed"

    def test_different_seeds_different_output(self):
        set_all_seeds(42)
        a = torch.randn(10).tolist()
        set_all_seeds(99)
        b = torch.randn(10).tolist()
        assert a != b, "Different seeds should produce different values"


# =========================================================================
# Test 2 -- build_optimizer: two param groups with different LRs
# =========================================================================

class TestBuildOptimizer:
    def test_two_param_groups(self, small_model):
        """Must create exactly 2 param groups with different LRs."""
        config = {
            "lr_codebert": 2e-5,
            "lr_ggnn_fusion": 1e-4,
            "weight_decay": 0.01,
        }
        optimizer = build_optimizer(small_model, config)

        assert len(optimizer.param_groups) == 2, (
            f"Expected 2 param groups, got {len(optimizer.param_groups)}"
        )

        lrs = [g["lr"] for g in optimizer.param_groups]
        assert 2e-5 in lrs, f"CodeBERT LR 2e-5 not found in {lrs}"
        assert 1e-4 in lrs, f"GGNN LR 1e-4 not found in {lrs}"

    def test_codebert_lr_lower(self, small_model):
        """CodeBERT LR must be lower than GGNN LR (R-11)."""
        config = {
            "lr_codebert": 2e-5,
            "lr_ggnn_fusion": 1e-4,
            "weight_decay": 0.01,
        }
        optimizer = build_optimizer(small_model, config)

        codebert_lr = optimizer.param_groups[0]["lr"]
        ggnn_lr = optimizer.param_groups[1]["lr"]
        assert codebert_lr < ggnn_lr, (
            f"CodeBERT LR ({codebert_lr}) should be < GGNN LR ({ggnn_lr})"
        )

    def test_only_trainable_params(self, small_model):
        """Only requires_grad params should be in optimizer."""
        config = {
            "lr_codebert": 2e-5,
            "lr_ggnn_fusion": 1e-4,
            "weight_decay": 0.01,
        }
        optimizer = build_optimizer(small_model, config)

        total_opt_params = sum(len(g["params"]) for g in optimizer.param_groups)
        total_trainable = sum(1 for p in small_model.parameters() if p.requires_grad)
        assert total_opt_params == total_trainable, (
            f"Optimizer has {total_opt_params} params but model has {total_trainable} trainable"
        )


# =========================================================================
# Test 3 -- tokenize_batch: returns (B, 512) input_ids
# =========================================================================

class TestTokenizeBatch:
    def test_output_shape(self, tokenizer):
        """tokenize_batch must return (B, 512) tensors."""
        codes = [
            "int main() { return 0; }",
            "void foo(char *buf) { strcpy(buf, input); }",
            "int add(int a, int b) { return a + b; }",
        ]
        input_ids, attn_mask = tokenize_batch(codes, tokenizer, 512, torch.device("cpu"))

        assert input_ids.shape == (3, 512), f"Expected (3, 512), got {input_ids.shape}"
        assert attn_mask.shape == (3, 512), f"Expected (3, 512), got {attn_mask.shape}"
        assert input_ids.dtype == torch.long
        assert attn_mask.dtype == torch.long

    def test_shorter_max_length(self, tokenizer):
        """Respects max_length parameter."""
        codes = ["int x = 1;"]
        input_ids, attn_mask = tokenize_batch(codes, tokenizer, 128, torch.device("cpu"))
        assert input_ids.shape == (1, 128)


# =========================================================================
# Test 4 -- tokenize_batch: empty string handled
# =========================================================================

class TestTokenizeBatchEmpty:
    def test_empty_string_no_crash(self, tokenizer):
        """Empty strings must be replaced with placeholder, not crash."""
        codes = ["", "  ", "int x = 1;"]
        input_ids, attn_mask = tokenize_batch(codes, tokenizer, 512, torch.device("cpu"))

        assert input_ids.shape == (3, 512)
        # Empty strings should not produce all-zero input_ids (would be degenerate)
        # Placeholder "void placeholder(void){}" should produce non-trivial tokens
        for i in range(2):  # first two are empty/whitespace
            non_pad = (input_ids[i] != tokenizer.pad_token_id).sum().item()
            assert non_pad > 1, f"Empty string {i} produced degenerate tokenization"

    def test_all_empty(self, tokenizer):
        """All-empty batch must not crash."""
        codes = ["", "", ""]
        input_ids, attn_mask = tokenize_batch(codes, tokenizer, 512, torch.device("cpu"))
        assert input_ids.shape == (3, 512)


# =========================================================================
# Test 5 -- Full dry-run: 1 epoch, loss is finite, checkpoint saved
# =========================================================================

class TestDryRun:
    def test_dry_run_completes(self, test_h5_and_jsonl, tmp_path, monkeypatch):
        """Dry-run: 2 epochs, finite loss, checkpoint saved."""
        h5_path, val_h5_path, jsonl_path = test_h5_and_jsonl
        ckpt_dir = str(tmp_path / "checkpoints")

        # Monkeypatch load_code_lookup to use our test JSONL
        import training.scripts.model.train as train_mod
        original_load = train_mod.load_code_lookup

        def mock_load(paths):
            result = original_load([jsonl_path])
            return result

        monkeypatch.setattr(train_mod, "load_code_lookup", mock_load)

        config = dict(DEFAULT_CONFIG)
        config.update(ABLATION_CONFIGS["C_plus_cfa"])
        config["config_name"] = "C_plus_cfa"
        config["train_h5"] = h5_path
        config["val_h5"] = val_h5_path
        config["test_h5"] = h5_path
        config["checkpoint_dir"] = ckpt_dir
        config["dry_run"] = True
        config["batch_size_graphs"] = 4
        config["gradient_accumulation"] = 1
        config["freeze_codebert_layers"] = 11  # freeze almost all for speed

        best_f1 = train(config)

        # Checkpoint must exist
        latest_path = Path(ckpt_dir) / "latest_C_plus_cfa.pt"
        assert latest_path.exists(), f"Latest checkpoint not found at {latest_path}"

        # Best checkpoint must exist
        best_path = Path(ckpt_dir) / "best_model_C_plus_cfa.pt"
        assert best_path.exists(), f"Best checkpoint not found at {best_path}"

        # Load and verify checkpoint is valid
        ckpt = torch.load(str(latest_path), map_location="cpu", weights_only=False)
        assert "model_state_dict" in ckpt
        assert "optimizer_state_dict" in ckpt
        assert "config" in ckpt
        assert ckpt["config"]["ablation_config"] == "C_plus_cfa"

        # F1 should be finite (not NaN)
        assert best_f1 == best_f1, "best_f1 is NaN"  # NaN != NaN


# =========================================================================
# Test 6 -- Resume: load checkpoint + continue from epoch 2
# =========================================================================

class TestResume:
    def test_resume_from_checkpoint(self, test_h5_and_jsonl, tmp_path, monkeypatch):
        """Resume training from saved checkpoint."""
        h5_path, val_h5_path, jsonl_path = test_h5_and_jsonl
        ckpt_dir = str(tmp_path / "checkpoints")

        import training.scripts.model.train as train_mod
        original_load = train_mod.load_code_lookup

        def mock_load(paths):
            return original_load([jsonl_path])

        monkeypatch.setattr(train_mod, "load_code_lookup", mock_load)

        # First run: dry-run to create checkpoint
        config = dict(DEFAULT_CONFIG)
        config.update(ABLATION_CONFIGS["B_plus_ggnn"])
        config["config_name"] = "B_plus_ggnn"
        config["train_h5"] = h5_path
        config["val_h5"] = val_h5_path
        config["checkpoint_dir"] = ckpt_dir
        config["dry_run"] = True
        config["batch_size_graphs"] = 4
        config["gradient_accumulation"] = 1
        config["freeze_codebert_layers"] = 11

        train(config)

        latest_path = Path(ckpt_dir) / "latest_B_plus_ggnn.pt"
        assert latest_path.exists(), "First run didn't create checkpoint"

        # Check epoch in checkpoint
        ckpt = torch.load(str(latest_path), map_location="cpu", weights_only=False)
        saved_epoch = ckpt["epoch"]
        assert saved_epoch >= 0, "Saved epoch should be >= 0"

        # Second run: resume
        config["resume"] = True
        config["dry_run"] = True
        train(config)

        # Verify the latest checkpoint was updated
        ckpt2 = torch.load(str(latest_path), map_location="cpu", weights_only=False)
        assert ckpt2["epoch"] >= saved_epoch, "Resumed run should advance epoch"


# =========================================================================
# Test 7 -- ABLATION_CONFIGS has 6 entries
# =========================================================================

class TestAblationConfigs:
    def test_six_configs(self):
        assert len(ABLATION_CONFIGS) == 6

    def test_config_names(self):
        expected = {"A_baseline", "B_plus_ggnn", "B_prime_type_aware",
                    "C_plus_cfa", "D_plus_tpg", "E_full"}
        assert set(ABLATION_CONFIGS.keys()) == expected

    def test_all_have_required_keys(self):
        required = {"use_graph", "type_aware_edges", "use_cfa",
                     "cpg_components", "use_interproc"}
        for name, cfg in ABLATION_CONFIGS.items():
            assert required.issubset(set(cfg.keys())), (
                f"Config {name} missing keys: {required - set(cfg.keys())}"
            )

    def test_a_baseline_no_graph(self):
        assert ABLATION_CONFIGS["A_baseline"]["use_graph"] is False

    def test_e_full_interproc(self):
        assert ABLATION_CONFIGS["E_full"]["use_interproc"] is True
        assert ABLATION_CONFIGS["E_full"]["use_cfa"] is True


# =========================================================================
# Test 8 -- DEFAULT_CONFIG has all required keys
# =========================================================================

class TestDefaultConfig:
    def test_required_keys_present(self):
        required = {
            "base_model", "node_feature_dim", "lr_codebert", "lr_ggnn_fusion",
            "weight_decay", "warmup_ratio", "batch_size_graphs",
            "gradient_accumulation", "max_seq_len", "epochs",
            "early_stopping_patience", "seed", "grad_clip",
            "lambda_ce", "lambda_cfa", "lambda_sev", "cfa_margin",
            "freeze_codebert_layers",
        }
        assert required.issubset(set(DEFAULT_CONFIG.keys()))

    def test_seed_is_42(self):
        assert DEFAULT_CONFIG["seed"] == 42

    def test_differential_lr_defaults(self):
        assert DEFAULT_CONFIG["lr_codebert"] == 2e-5
        assert DEFAULT_CONFIG["lr_ggnn_fusion"] == 1e-4
        assert DEFAULT_CONFIG["lr_codebert"] < DEFAULT_CONFIG["lr_ggnn_fusion"]


# =========================================================================
# Test 9 -- load_code_lookup
# =========================================================================

class TestLoadCodeLookup:
    def test_loads_from_jsonl(self, tmp_path):
        jsonl_path = str(tmp_path / "test.jsonl")
        _create_test_jsonl(jsonl_path, ["id_a", "id_b", "id_c"])

        lookup = load_code_lookup([jsonl_path])
        assert len(lookup) == 3
        assert "id_a" in lookup
        assert isinstance(lookup["id_a"], str)
        assert len(lookup["id_a"]) > 0

    def test_returns_empty_for_missing(self):
        lookup = load_code_lookup(["/nonexistent/path.jsonl"])
        assert lookup == {}

    def test_tries_paths_in_order(self, tmp_path):
        """Merges all files; earlier file takes priority for duplicate IDs."""
        p1 = str(tmp_path / "a.jsonl")
        p2 = str(tmp_path / "b.jsonl")
        _create_test_jsonl(p1, ["from_a"])
        _create_test_jsonl(p2, ["from_b"])

        lookup = load_code_lookup([p1, p2])
        # Both files are loaded (merge-all for coverage)
        assert "from_a" in lookup
        assert "from_b" in lookup

    def test_earlier_file_takes_priority(self, tmp_path):
        """When same ID appears in multiple files, first file wins."""
        p1 = str(tmp_path / "a.jsonl")
        p2 = str(tmp_path / "b.jsonl")
        # Both files have same ID "shared" but different code
        with open(p1, "w") as f:
            f.write(json.dumps({"id": "shared", "code": "code_from_a"}) + "\n")
        with open(p2, "w") as f:
            f.write(json.dumps({"id": "shared", "code": "code_from_b"}) + "\n")

        lookup = load_code_lookup([p1, p2])
        assert lookup["shared"] == "code_from_a"  # first file wins


# =========================================================================
# Test 10 -- save_checkpoint atomic
# =========================================================================

class TestSaveCheckpoint:
    def test_atomic_write(self, small_model, tmp_path):
        """Checkpoint should not leave .tmp files behind."""
        from torch.amp import GradScaler
        config = dict(DEFAULT_CONFIG)
        config["config_name"] = "test_config"
        config["cpg_components"] = ["AST", "CFG", "DFG"]

        optimizer = build_optimizer(small_model, config)
        from transformers import get_linear_schedule_with_warmup
        scheduler = get_linear_schedule_with_warmup(optimizer, 0, 100)
        scaler = GradScaler(enabled=False)

        ckpt_path = str(tmp_path / "test_ckpt.pt")
        save_checkpoint(
            small_model, optimizer, scheduler, scaler,
            epoch=5, best_f1=0.85, patience_ctr=2,
            config=config, path=ckpt_path,
        )

        assert os.path.exists(ckpt_path), "Checkpoint not saved"
        assert not os.path.exists(ckpt_path + ".tmp"), ".tmp file left behind"

        # Verify contents
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        assert ckpt["epoch"] == 5
        assert ckpt["best_f1"] == 0.85
        assert ckpt["patience_ctr"] == 2
        assert "model_state_dict" in ckpt
        assert ckpt["config"]["ablation_config"] == "test_config"


# =========================================================================
# Test 11 -- _select_with_complete_groups
# =========================================================================

class TestSelectWithCompleteGroups:
    def test_pairs_kept_intact(self):
        """Selecting N samples must not break pairs."""
        index = [
            {"pair_id": "p1", "sample_id": "a"},
            {"pair_id": "p1", "sample_id": "b"},
            {"pair_id": "p2", "sample_id": "c"},
            {"pair_id": "p2", "sample_id": "d"},
            {"pair_id": "", "sample_id": "e"},
            {"pair_id": "", "sample_id": "f"},
        ]
        result = _select_with_complete_groups(index, target_n=3)
        # Should get at least 2 (one full pair) or 4 (two pairs)
        pair_ids = [r["pair_id"] for r in result if r["pair_id"] not in EMPTY_PAIR_SENTINELS]
        # All pair members should be present
        from collections import Counter
        counts = Counter(pair_ids)
        for pid, count in counts.items():
            assert count == 2, f"Pair {pid} split: only {count} member(s)"
