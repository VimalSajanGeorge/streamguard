# tests/test_p4s7_ablations.py
#
# P4-S7: Unit tests for ablation study runner.
# Tests 1-4 verify config invariants (pure dict inspection, no model/data).
# Tests 5-6 mock train() to verify orchestration and JSON output.

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from training.scripts.model.run_ablations import (
    BASE_CONFIG,
    ABLATION_CONFIGS,
    CONFIG_ORDER,
    assert_ablation_invariants,
    run_ablations,
    print_ablation_table,
)


# ── Test 1: BASE_CONFIG has seed=42 ──────────────────────────────────────

def test_base_config_seed_42():
    assert BASE_CONFIG["seed"] == 42


# ── Test 2: All configs share same train_h5/val_h5/test_h5 paths ────────

def test_all_configs_same_data_paths():
    for name, cfg in ABLATION_CONFIGS.items():
        assert cfg["train_h5"] == BASE_CONFIG["train_h5"], \
            f"{name} has different train_h5"
        assert cfg["val_h5"] == BASE_CONFIG["val_h5"], \
            f"{name} has different val_h5"
        assert cfg["test_h5"] == BASE_CONFIG["test_h5"], \
            f"{name} has different test_h5"
        assert cfg["seed"] == 42, \
            f"{name} has seed={cfg['seed']}, expected 42"

    # Also verify the assertion function doesn't raise
    assert_ablation_invariants()


# ── Test 3: Config B has lambda_cfa=0.0 (no contrastive loss) ────────────

def test_config_b_no_cfa():
    cfg = ABLATION_CONFIGS["B_plus_ggnn"]
    assert cfg["lambda_cfa"] == 0.0, "Config B must disable CFA loss"
    assert cfg["use_cfa"] is False, "Config B must have use_cfa=False"
    assert cfg["type_aware_edges"] is False, "Config B must be type-blind"


# ── Test 4: Config C has lambda_cfa=0.5 (CFA enabled) ───────────────────

def test_config_c_cfa_enabled():
    cfg = ABLATION_CONFIGS["C_plus_cfa"]
    assert cfg["lambda_cfa"] == 0.5, "Config C must have lambda_cfa=0.5"
    assert cfg["use_cfa"] is True, "Config C must have use_cfa=True"
    assert cfg["type_aware_edges"] is True, "Config C must be type-aware"


# ── Test 5: Dry-run: run A and B configs for 2 epochs each, no crash ────

def test_dry_run_no_crash(tmp_path):
    """Mock train() to return fake metrics, verify run_ablations orchestrates."""
    fake_test_metrics = {
        "f1": 0.85, "precision": 0.83, "recall": 0.87,
        "fpr": 0.04, "fnr": 0.13, "pairwise_accuracy": 0.80,
        "worst_cwe_f1": 0.72, "per_cwe_f1": {"CWE-89": 0.90},
    }

    call_log = []

    def mock_train(config):
        call_log.append(config["config_name"])
        return 0.86, fake_test_metrics

    with patch("training.scripts.model.run_ablations.train", side_effect=mock_train):
        results = run_ablations(
            configs_to_run=["A_baseline", "B_plus_ggnn"],
            dry_run=True,
        )

    assert "A_baseline" in call_log
    assert "B_plus_ggnn" in call_log
    assert len(call_log) == 2
    assert results["A_baseline"]["best_val_f1"] == 0.86
    assert results["B_plus_ggnn"]["test_f1"] == 0.85


# ── Test 6: Results saved to results/ablation_table.json after run ───────

def test_results_saved_to_json(tmp_path, monkeypatch):
    """Verify ablation_table.json is created with correct structure."""
    # Change working directory so results/ is created in tmp_path
    monkeypatch.chdir(tmp_path)

    fake_test_metrics = {
        "f1": 0.91, "precision": 0.90, "recall": 0.92,
        "fpr": 0.03, "fnr": 0.08, "pairwise_accuracy": 0.88,
        "worst_cwe_f1": 0.80, "per_cwe_f1": {"CWE-120": 0.95},
    }

    def mock_train(config):
        return 0.90, fake_test_metrics

    with patch("training.scripts.model.run_ablations.train", side_effect=mock_train):
        run_ablations(
            configs_to_run=["A_baseline"],
            dry_run=True,
        )

    table_path = tmp_path / "results" / "ablation_table.json"
    assert table_path.exists(), "ablation_table.json not created"

    with open(table_path) as f:
        saved = json.load(f)

    assert "A_baseline" in saved
    assert saved["A_baseline"]["best_val_f1"] == 0.90
    assert saved["A_baseline"]["test_f1"] == 0.91
    assert saved["A_baseline"]["per_cwe_f1"]["CWE-120"] == 0.95
