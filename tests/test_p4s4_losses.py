#!/usr/bin/env python3
"""
tests/test_p4s4_losses.py

Phase 4 Story 4: Composite Loss Function.

8 mandatory tests:
  1. L_CE decreases on random binary task after 10 steps
  2. L_CFA sign: relu(0.8 + 0.5) = 1.3, NOT relu(0.8 - 0.5) = 0.3
  3. L_CFA = 0.0 when cosine_sim = -0.9 (already separated)
  4. L_CFA > 0.0 when cosine_sim = +0.5 (not separated)
  5. frac_separated: 0.0 when close, 1.0 when separated
  6. L_severity skips samples with severity_label = -1
  7. Unequal orig/cfa batch sizes handled via min_len (no crash)
  8. total_loss gradient flows through all active components

Additional tests:
  9.  CWE_LABEL_MAP has 12 entries, LABEL_CWE_MAP is inverse
  10. L_CWE uses label_smoothing
  11. CWE labels with -1 are masked out
  12. Lambda weights scale components correctly
  13. All-None optional inputs → only total in loss_dict
  14. Default hyperparameters match spec
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.scripts.model.losses import (
    StreamGuardLoss,
    CWE_LABEL_MAP,
    LABEL_CWE_MAP,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_outputs(batch_size: int, embedding_dim: int = 128, device: str = "cpu"):
    """Create a fake model output dict."""
    return {
        "logits":         torch.randn(batch_size, 2, device=device),
        "cwe_logits":     torch.randn(batch_size, 12, device=device),
        "severity_score": torch.randn(batch_size, device=device),
        "embedding":      torch.randn(batch_size, embedding_dim, device=device),
    }


def _make_outputs_with_embedding(embedding: torch.Tensor, batch_size: int = None):
    """Create outputs with a specific embedding tensor."""
    B = embedding.size(0) if batch_size is None else batch_size
    return {
        "logits":         torch.randn(B, 2),
        "cwe_logits":     torch.randn(B, 12),
        "severity_score": torch.randn(B),
        "embedding":      embedding,
    }


# =========================================================================
# Test 1 -- L_CE decreases on random binary task after 10 steps
# =========================================================================

class TestLCEDecreases:
    def test_ce_loss_decreases_10_steps(self):
        """L_CE should decrease over 10 gradient steps on a simple task."""
        torch.manual_seed(42)
        loss_fn = StreamGuardLoss(lambda_ce=1.0, lambda_cfa=0.0, lambda_sev=0.0)

        # Learnable logits
        logits = torch.randn(16, 2, requires_grad=True)
        labels = torch.randint(0, 2, (16,))
        optimizer = torch.optim.SGD([logits], lr=0.1)

        losses = []
        for step in range(10):
            outputs = {
                "logits": logits,
                "cwe_logits": torch.randn(16, 12),
                "severity_score": torch.randn(16),
                "embedding": torch.randn(16, 128),
            }
            total, d = loss_fn(outputs, labels=labels)
            losses.append(d["L_CE"])
            optimizer.zero_grad()
            total.backward()
            optimizer.step()

        assert losses[-1] < losses[0], (
            f"L_CE should decrease: first={losses[0]:.4f}, last={losses[-1]:.4f}"
        )


# =========================================================================
# Test 2 -- L_CFA sign: relu(0.8 + 0.5) = 1.3, NOT relu(0.8 - 0.5) = 0.3
# =========================================================================

class TestLCFASign:
    def test_cfa_sign_correct(self):
        """R-02: relu(cosine_sim + margin) must produce 1.3 for sim=0.8, margin=0.5."""
        loss_fn = StreamGuardLoss(cfa_margin=0.5)

        # Create embeddings with known cosine similarity = 0.8
        # Use unit vectors: e1 = [1, 0, ...], e2 = [0.8, 0.6, 0, ...] → sim = 0.8
        B = 1
        dim = 128
        e1 = torch.zeros(B, dim)
        e1[0, 0] = 1.0
        e2 = torch.zeros(B, dim)
        e2[0, 0] = 0.8
        e2[0, 1] = 0.6

        outputs_orig = _make_outputs_with_embedding(e1)
        outputs_cfa = _make_outputs_with_embedding(e2)

        _, d = loss_fn(outputs_orig, outputs_cfa=outputs_cfa, labels=torch.tensor([1]))

        # cosine_sim = 0.8, relu(0.8 + 0.5) = relu(1.3) = 1.3
        expected = 1.3
        assert abs(d["L_CFA"] - expected) < 0.01, (
            f"L_CFA should be ~{expected} (relu(0.8+0.5)), got {d['L_CFA']:.4f}. "
            "CRITICAL: check sign is + not -"
        )

    def test_wrong_sign_would_give_different_value(self):
        """Verify the wrong formula relu(sim - margin) gives 0.3 (which we must NOT match)."""
        # Same setup
        sim = 0.8
        margin = 0.5
        correct = F.relu(torch.tensor(sim + margin)).item()   # 1.3
        wrong   = F.relu(torch.tensor(sim - margin)).item()   # 0.3
        assert correct == pytest.approx(1.3, abs=0.01)
        assert wrong == pytest.approx(0.3, abs=0.01)
        assert correct != pytest.approx(wrong, abs=0.1)


# =========================================================================
# Test 3 -- L_CFA = 0.0 when cosine_sim = -0.9 (well separated)
# =========================================================================

class TestLCFASeparated:
    def test_cfa_zero_when_separated(self):
        """L_CFA = relu(-0.9 + 0.5) = relu(-0.4) = 0.0."""
        loss_fn = StreamGuardLoss(cfa_margin=0.5)

        B, dim = 4, 128
        # Opposite embeddings → cosine_sim ~ -1.0
        e1 = torch.randn(B, dim)
        e1 = F.normalize(e1, dim=-1)
        e2 = -e1  # cosine_sim = -1.0

        outputs_orig = _make_outputs_with_embedding(e1)
        outputs_cfa = _make_outputs_with_embedding(e2)

        _, d = loss_fn(outputs_orig, outputs_cfa=outputs_cfa, labels=torch.randint(0, 2, (B,)))

        assert d["L_CFA"] == pytest.approx(0.0, abs=1e-6), (
            f"L_CFA should be 0.0 when pairs are well-separated, got {d['L_CFA']}"
        )


# =========================================================================
# Test 4 -- L_CFA > 0.0 when cosine_sim = +0.5 (not separated)
# =========================================================================

class TestLCFANotSeparated:
    def test_cfa_positive_when_close(self):
        """L_CFA = relu(0.5 + 0.5) = relu(1.0) = 1.0 > 0."""
        loss_fn = StreamGuardLoss(cfa_margin=0.5)

        B, dim = 4, 128
        # Same embeddings → cosine_sim = 1.0
        e1 = F.normalize(torch.randn(B, dim), dim=-1)
        e2 = e1.clone()  # cosine_sim = 1.0

        outputs_orig = _make_outputs_with_embedding(e1)
        outputs_cfa = _make_outputs_with_embedding(e2)

        _, d = loss_fn(outputs_orig, outputs_cfa=outputs_cfa, labels=torch.randint(0, 2, (B,)))

        assert d["L_CFA"] > 0.0, (
            f"L_CFA should be > 0 when pairs are NOT separated, got {d['L_CFA']}"
        )
        # relu(1.0 + 0.5) = 1.5
        assert d["L_CFA"] == pytest.approx(1.5, abs=0.01), (
            f"L_CFA for identical embeds should be ~1.5, got {d['L_CFA']}"
        )


# =========================================================================
# Test 5 -- frac_separated: 0.0 when close, 1.0 when separated
# =========================================================================

class TestFracSeparated:
    def test_frac_separated_zero_when_close(self):
        """All pairs identical → frac_separated = 0.0."""
        loss_fn = StreamGuardLoss(cfa_margin=0.5)

        B, dim = 8, 128
        e1 = F.normalize(torch.randn(B, dim), dim=-1)
        e2 = e1.clone()

        outputs_orig = _make_outputs_with_embedding(e1)
        outputs_cfa = _make_outputs_with_embedding(e2)

        _, d = loss_fn(outputs_orig, outputs_cfa=outputs_cfa, labels=torch.randint(0, 2, (B,)))

        assert d["frac_separated"] == pytest.approx(0.0, abs=1e-6), (
            f"frac_separated should be 0.0 when all pairs close, got {d['frac_separated']}"
        )

    def test_frac_separated_one_when_opposite(self):
        """All pairs opposite → frac_separated = 1.0."""
        loss_fn = StreamGuardLoss(cfa_margin=0.5)

        B, dim = 8, 128
        e1 = F.normalize(torch.randn(B, dim), dim=-1)
        e2 = -e1  # cosine_sim = -1.0, all < -margin(-0.5)

        outputs_orig = _make_outputs_with_embedding(e1)
        outputs_cfa = _make_outputs_with_embedding(e2)

        _, d = loss_fn(outputs_orig, outputs_cfa=outputs_cfa, labels=torch.randint(0, 2, (B,)))

        assert d["frac_separated"] == pytest.approx(1.0, abs=1e-6), (
            f"frac_separated should be 1.0 when all pairs opposite, got {d['frac_separated']}"
        )


# =========================================================================
# Test 6 -- L_severity skips samples with severity_label = -1
# =========================================================================

class TestSeveritySkip:
    def test_severity_minus_one_skipped(self):
        """R-23: severity_label=-1 samples must be excluded from L_severity."""
        loss_fn = StreamGuardLoss(lambda_sev=0.1)
        B = 8
        outputs = _make_outputs(B)
        labels = torch.randint(0, 2, (B,))

        # All -1 → L_severity should not appear in loss dict
        sev_all_neg = torch.full((B,), -1.0)
        _, d_neg = loss_fn(outputs, labels=labels, severity_labels=sev_all_neg)
        assert "L_severity" not in d_neg, "L_severity should be absent when all labels are -1"

    def test_severity_partial_mask(self):
        """Mix of valid and -1 → L_severity computed only on valid."""
        loss_fn = StreamGuardLoss(lambda_sev=0.1)
        B = 8
        outputs = _make_outputs(B)
        labels = torch.randint(0, 2, (B,))

        # Only first 4 valid
        sev_mixed = torch.tensor([-1, -1, -1, -1, 5.0, 7.0, 3.0, 9.0])
        _, d_mixed = loss_fn(outputs, labels=labels, severity_labels=sev_mixed)
        assert "L_severity" in d_mixed, "L_severity should be present when some labels are valid"

    def test_severity_all_valid(self):
        """All valid → L_severity present."""
        loss_fn = StreamGuardLoss(lambda_sev=0.1)
        B = 4
        outputs = _make_outputs(B)
        labels = torch.randint(0, 2, (B,))
        sev_valid = torch.tensor([5.0, 7.0, 3.0, 9.0])
        _, d = loss_fn(outputs, labels=labels, severity_labels=sev_valid)
        assert "L_severity" in d


# =========================================================================
# Test 7 -- Unequal orig/cfa batch sizes handled (min_len)
# =========================================================================

class TestUnequalBatchSizes:
    def test_unequal_sizes_no_crash(self):
        """orig has 6 samples, CFA has 4 → truncate to min(6,4)=4, no crash."""
        loss_fn = StreamGuardLoss()

        outputs_orig = _make_outputs(6)
        outputs_cfa = _make_outputs(4)
        labels = torch.randint(0, 2, (6,))

        total, d = loss_fn(outputs_orig, outputs_cfa=outputs_cfa, labels=labels)
        assert "L_CFA" in d, "L_CFA should be computed even with unequal sizes"
        assert torch.isfinite(total), "total should be finite"

    def test_unequal_reverse(self):
        """orig has 4, CFA has 8 → truncate to 4."""
        loss_fn = StreamGuardLoss()

        outputs_orig = _make_outputs(4)
        outputs_cfa = _make_outputs(8)
        labels = torch.randint(0, 2, (4,))

        total, d = loss_fn(outputs_orig, outputs_cfa=outputs_cfa, labels=labels)
        assert "L_CFA" in d
        assert torch.isfinite(total)


# =========================================================================
# Test 8 -- total_loss gradient flows through all active components
# =========================================================================

class TestGradientFlow:
    def test_gradient_flows_all_components(self):
        """Gradients must flow through L_CE, L_CFA, L_CWE, L_severity."""
        loss_fn = StreamGuardLoss(lambda_ce=1.0, lambda_cfa=0.5, lambda_sev=0.1)
        B = 4

        # Make outputs with requires_grad
        logits = torch.randn(B, 2, requires_grad=True)
        cwe_logits = torch.randn(B, 12, requires_grad=True)
        sev_score = torch.randn(B, requires_grad=True)
        embedding_orig = torch.randn(B, 128, requires_grad=True)
        embedding_cfa = torch.randn(B, 128, requires_grad=True)

        outputs_orig = {
            "logits": logits,
            "cwe_logits": cwe_logits,
            "severity_score": sev_score,
            "embedding": embedding_orig,
        }
        outputs_cfa = {
            "logits": torch.randn(B, 2),
            "cwe_logits": torch.randn(B, 12),
            "severity_score": torch.randn(B),
            "embedding": embedding_cfa,
        }

        labels = torch.randint(0, 2, (B,))
        cwe_labels = torch.randint(0, 12, (B,))
        sev_labels = torch.tensor([5.0, 7.0, 3.0, 9.0])

        total, d = loss_fn(
            outputs_orig, outputs_cfa=outputs_cfa,
            labels=labels, cwe_labels=cwe_labels, severity_labels=sev_labels,
        )

        total.backward()

        # All four components should have gradients
        assert logits.grad is not None, "L_CE gradient missing"
        assert logits.grad.abs().sum() > 0, "L_CE gradient is zero"

        assert cwe_logits.grad is not None, "L_CWE gradient missing"
        assert cwe_logits.grad.abs().sum() > 0, "L_CWE gradient is zero"

        assert sev_score.grad is not None, "L_severity gradient missing"
        assert sev_score.grad.abs().sum() > 0, "L_severity gradient is zero"

        assert embedding_orig.grad is not None, "L_CFA gradient (orig) missing"
        assert embedding_orig.grad.abs().sum() > 0, "L_CFA gradient (orig) is zero"

        assert embedding_cfa.grad is not None, "L_CFA gradient (cfa) missing"
        assert embedding_cfa.grad.abs().sum() > 0, "L_CFA gradient (cfa) is zero"

    def test_all_loss_components_present(self):
        """When all inputs provided, all loss keys present."""
        loss_fn = StreamGuardLoss()
        B = 4

        outputs_orig = _make_outputs(B)
        outputs_cfa = _make_outputs(B)

        _, d = loss_fn(
            outputs_orig, outputs_cfa=outputs_cfa,
            labels=torch.randint(0, 2, (B,)),
            cwe_labels=torch.randint(0, 12, (B,)),
            severity_labels=torch.tensor([5.0, 7.0, 3.0, 9.0]),
        )

        for key in ["L_CE", "L_CFA", "L_CWE", "L_severity", "frac_separated", "total"]:
            assert key in d, f"Missing key: {key}"


# =========================================================================
# Test 9 -- CWE_LABEL_MAP / LABEL_CWE_MAP
# =========================================================================

class TestCWEMaps:
    def test_cwe_label_map_has_12_entries(self):
        assert len(CWE_LABEL_MAP) == 12

    def test_label_cwe_map_is_inverse(self):
        for cwe, idx in CWE_LABEL_MAP.items():
            assert LABEL_CWE_MAP[idx] == cwe

    def test_indices_are_0_to_11(self):
        assert set(CWE_LABEL_MAP.values()) == set(range(12))

    def test_known_cwes(self):
        expected_cwes = [
            "CWE-89", "CWE-78", "CWE-79", "CWE-119", "CWE-120",
            "CWE-121", "CWE-122", "CWE-125", "CWE-134", "CWE-190",
            "CWE-416", "CWE-476",
        ]
        for cwe in expected_cwes:
            assert cwe in CWE_LABEL_MAP, f"Missing CWE: {cwe}"


# =========================================================================
# Test 10 -- L_CWE uses label smoothing
# =========================================================================

class TestCWELabelSmoothing:
    def test_label_smoothing_configured(self):
        loss_fn = StreamGuardLoss(label_smoothing=0.1)
        assert loss_fn.cwe_loss.label_smoothing == 0.1

    def test_label_smoothing_zero(self):
        loss_fn = StreamGuardLoss(label_smoothing=0.0)
        assert loss_fn.cwe_loss.label_smoothing == 0.0


# =========================================================================
# Test 11 -- CWE labels with -1 masked out
# =========================================================================

class TestCWEMask:
    def test_cwe_all_minus_one_excluded(self):
        loss_fn = StreamGuardLoss()
        B = 4
        outputs = _make_outputs(B)
        cwe_all_neg = torch.full((B,), -1, dtype=torch.long)
        _, d = loss_fn(outputs, labels=torch.randint(0, 2, (B,)), cwe_labels=cwe_all_neg)
        assert "L_CWE" not in d, "L_CWE should be absent when all cwe_labels are -1"

    def test_cwe_partial_valid(self):
        loss_fn = StreamGuardLoss()
        B = 4
        outputs = _make_outputs(B)
        cwe_mixed = torch.tensor([-1, -1, 3, 7], dtype=torch.long)
        _, d = loss_fn(outputs, labels=torch.randint(0, 2, (B,)), cwe_labels=cwe_mixed)
        assert "L_CWE" in d


# =========================================================================
# Test 12 -- Lambda weights scale correctly
# =========================================================================

class TestLambdaWeights:
    def test_lambda_ce_scales(self):
        B = 4
        outputs = _make_outputs(B)
        labels = torch.randint(0, 2, (B,))

        loss_1x = StreamGuardLoss(lambda_ce=1.0, lambda_cfa=0.0, lambda_sev=0.0)
        loss_2x = StreamGuardLoss(lambda_ce=2.0, lambda_cfa=0.0, lambda_sev=0.0)

        t1, d1 = loss_1x(outputs, labels=labels)
        t2, d2 = loss_2x(outputs, labels=labels)

        # total should be ~2x when lambda_ce is doubled (only L_CE active)
        assert abs(t2.item() - 2.0 * t1.item()) < 0.01, (
            f"2x lambda_ce should give 2x total: {t1.item():.4f} vs {t2.item():.4f}"
        )


# =========================================================================
# Test 13 -- All-None optional inputs
# =========================================================================

class TestMinimalInputs:
    def test_no_labels_no_crash(self):
        """No labels, no CFA, no CWE, no severity → total = 0."""
        loss_fn = StreamGuardLoss()
        outputs = _make_outputs(4)
        total, d = loss_fn(outputs)
        assert total.item() == pytest.approx(0.0, abs=1e-6)
        assert "total" in d
        assert "L_CE" not in d

    def test_only_labels(self):
        """Only L_CE computed when only labels provided."""
        loss_fn = StreamGuardLoss()
        outputs = _make_outputs(4)
        _, d = loss_fn(outputs, labels=torch.randint(0, 2, (4,)))
        assert "L_CE" in d
        assert "L_CFA" not in d
        assert "L_severity" not in d


# =========================================================================
# Test 14 -- Default hyperparameters match spec
# =========================================================================

class TestDefaults:
    def test_default_lambdas(self):
        loss_fn = StreamGuardLoss()
        assert loss_fn.lambda_ce == 1.0
        assert loss_fn.lambda_cfa == 0.5
        assert loss_fn.lambda_sev == 0.1

    def test_default_margin(self):
        loss_fn = StreamGuardLoss()
        assert loss_fn.margin == 0.5

    def test_default_label_smoothing(self):
        loss_fn = StreamGuardLoss()
        assert loss_fn.cwe_loss.label_smoothing == 0.1

    def test_huber_delta(self):
        loss_fn = StreamGuardLoss()
        assert loss_fn.sev_loss.delta == 1.0
