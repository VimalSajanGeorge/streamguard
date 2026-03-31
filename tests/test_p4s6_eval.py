# tests/test_p4s6_eval.py
#
# P4-S6: Unit tests for evaluation metrics (compute_metrics).
# All 8 tests use synthetic numpy arrays — no model or dataloader required.

import numpy as np
import pytest
from sklearn.metrics import f1_score

from training.scripts.model.eval import compute_metrics


def _empty_cwe_severity(n):
    """Helper: return empty CWE/severity arrays for n samples."""
    return (
        np.zeros(n, dtype=int),       # cwe_preds
        np.full(n, -1, dtype=int),    # cwe_labels (all unknown)
        np.zeros(n, dtype=float),     # severity_preds
        np.full(n, -1.0),             # severity_labels (all skipped)
    )


# ── Test 1: F1 computed correctly on synthetic preds/labels ──────────────

def test_f1_correct():
    # 8 samples: 5 TP, 1 FP, 1 FN, 1 TN
    preds  = np.array([1, 1, 1, 1, 1, 1, 0, 0])
    labels = np.array([1, 1, 1, 1, 1, 0, 1, 0])
    cwe_p, cwe_l, sev_p, sev_l = _empty_cwe_severity(len(preds))

    m = compute_metrics(
        preds, labels, cwe_p, cwe_l, sev_p, sev_l,
        all_pair_results=[], cwe_group_preds={}, cwe_group_labels={},
    )

    expected_f1 = f1_score(labels, preds, zero_division=0)
    assert abs(m["f1"] - expected_f1) < 1e-9
    assert m["precision"] == pytest.approx(5 / 6, abs=1e-9)  # TP / (TP+FP)
    assert m["recall"] == pytest.approx(5 / 6, abs=1e-9)     # TP / (TP+FN)


# ── Test 2: FPR/FNR computed correctly ───────────────────────────────────

def test_fpr_fnr_correct():
    # TN=3, FP=2, FN=1, TP=4
    preds  = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 0])
    labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    cwe_p, cwe_l, sev_p, sev_l = _empty_cwe_severity(len(preds))

    m = compute_metrics(
        preds, labels, cwe_p, cwe_l, sev_p, sev_l,
        all_pair_results=[], cwe_group_preds={}, cwe_group_labels={},
    )

    # FPR = FP / (FP + TN) = 2 / (2 + 3) = 0.4
    assert m["fpr"] == pytest.approx(0.4, abs=1e-9)
    # FNR = FN / (FN + TP) = 1 / (1 + 4) = 0.2
    assert m["fnr"] == pytest.approx(0.2, abs=1e-9)


# ── Test 3: pairwise_accuracy 1.0 when all pairs correctly classified ───

def test_pairwise_accuracy_perfect():
    preds  = np.array([1, 0])
    labels = np.array([1, 0])
    cwe_p, cwe_l, sev_p, sev_l = _empty_cwe_severity(len(preds))

    # All pairs: orig predicted vuln=1 AND CFA predicted safe=0
    pair_results = [True, True, True, True, True]

    m = compute_metrics(
        preds, labels, cwe_p, cwe_l, sev_p, sev_l,
        all_pair_results=pair_results,
        cwe_group_preds={}, cwe_group_labels={},
    )

    assert m["pairwise_accuracy"] == 1.0


# ── Test 4: pairwise_accuracy 0.0 when all pairs wrong ──────────────────

def test_pairwise_accuracy_zero():
    preds  = np.array([1, 0])
    labels = np.array([1, 0])
    cwe_p, cwe_l, sev_p, sev_l = _empty_cwe_severity(len(preds))

    # All pairs wrong
    pair_results = [False, False, False]

    m = compute_metrics(
        preds, labels, cwe_p, cwe_l, sev_p, sev_l,
        all_pair_results=pair_results,
        cwe_group_preds={}, cwe_group_labels={},
    )

    assert m["pairwise_accuracy"] == 0.0


# ── Test 5: worst_group_f1 returns min across CWEs ──────────────────────

def test_worst_group_f1_returns_min():
    # Overall dummy data
    preds  = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    labels = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    cwe_p, cwe_l, sev_p, sev_l = _empty_cwe_severity(len(preds))

    # CWE-89: perfect predictions (F1=1.0) — 6 samples
    # CWE-120: worse predictions (lower F1) — 5 samples
    cwe_group_preds = {
        "CWE-89":  [1, 1, 1, 1, 1, 0],
        "CWE-120": [1, 1, 0, 0, 0],
    }
    cwe_group_labels = {
        "CWE-89":  [1, 1, 1, 1, 1, 0],
        "CWE-120": [1, 1, 1, 1, 0],
    }

    m = compute_metrics(
        preds, labels, cwe_p, cwe_l, sev_p, sev_l,
        all_pair_results=[],
        cwe_group_preds=cwe_group_preds,
        cwe_group_labels=cwe_group_labels,
    )

    f1_89 = f1_score(cwe_group_labels["CWE-89"], cwe_group_preds["CWE-89"], zero_division=0)
    f1_120 = f1_score(cwe_group_labels["CWE-120"], cwe_group_preds["CWE-120"], zero_division=0)

    assert m["worst_group_f1"] == pytest.approx(min(f1_89, f1_120), abs=1e-9)
    assert m["per_cwe_f1"]["CWE-89"] == pytest.approx(f1_89, abs=1e-9)
    assert m["per_cwe_f1"]["CWE-120"] == pytest.approx(f1_120, abs=1e-9)


# ── Test 6: per_cwe_f1 skips CWEs with < 5 samples ──────────────────────

def test_per_cwe_f1_skips_small_groups():
    preds  = np.array([1, 1, 0, 0, 0, 0, 0, 0])
    labels = np.array([1, 1, 0, 0, 0, 0, 0, 0])
    cwe_p, cwe_l, sev_p, sev_l = _empty_cwe_severity(len(preds))

    # CWE-89: 3 samples → below threshold, should be SKIPPED
    # CWE-120: 5 samples → meets threshold, should be INCLUDED
    cwe_group_preds = {
        "CWE-89":  [1, 1, 0],
        "CWE-120": [1, 0, 0, 0, 0],
    }
    cwe_group_labels = {
        "CWE-89":  [1, 1, 0],
        "CWE-120": [1, 0, 0, 0, 0],
    }

    m = compute_metrics(
        preds, labels, cwe_p, cwe_l, sev_p, sev_l,
        all_pair_results=[],
        cwe_group_preds=cwe_group_preds,
        cwe_group_labels=cwe_group_labels,
    )

    assert "CWE-89" not in m["per_cwe_f1"]
    assert "CWE-120" in m["per_cwe_f1"]


# ── Test 7: severity_mae skips samples with label=-1 ────────────────────

def test_severity_mae_skips_negative():
    preds  = np.array([1, 1, 0, 0])
    labels = np.array([1, 1, 0, 0])
    cwe_p = np.zeros(4, dtype=int)
    cwe_l = np.full(4, -1, dtype=int)

    # Severity: 2 valid, 2 skipped (-1)
    sev_preds  = np.array([5.0, 3.0, 7.0, 2.0])
    sev_labels = np.array([4.0, 6.0, -1.0, -1.0])

    m = compute_metrics(
        preds, labels, cwe_p, cwe_l, sev_preds, sev_labels,
        all_pair_results=[], cwe_group_preds={}, cwe_group_labels={},
    )

    # MAE only on first two: |5-4| + |3-6| = 1 + 3 = 4, /2 = 2.0
    assert m["severity_mae"] == pytest.approx(2.0, abs=1e-9)


# ── Test 8: cwe_top1_accuracy computed correctly ─────────────────────────

def test_cwe_top1_accuracy():
    preds  = np.array([1, 1, 0, 0, 1])
    labels = np.array([1, 1, 0, 0, 1])

    # CWE head: 3 correct out of 4 valid (one label is -1 → skipped)
    cwe_preds  = np.array([0, 1, 2, 3, 5])
    cwe_labels = np.array([0, 1, 2, 4, -1])  # idx 3 wrong (3≠4), idx 4 skipped

    sev_p = np.zeros(5, dtype=float)
    sev_l = np.full(5, -1.0)

    m = compute_metrics(
        preds, labels, cwe_preds, cwe_labels, sev_p, sev_l,
        all_pair_results=[], cwe_group_preds={}, cwe_group_labels={},
    )

    # 3 correct out of 4 valid = 0.75
    assert m["cwe_top1_accuracy"] == pytest.approx(0.75, abs=1e-9)
