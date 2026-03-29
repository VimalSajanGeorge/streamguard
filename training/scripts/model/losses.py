# training/scripts/model/losses.py
#
# Phase 4 StreamGuardLoss: Composite multi-task loss for CFA-paired training.
#
# L_total = λ_ce  * L_CE
#         + λ_cfa * L_CFA
#         + 0.2   * L_CWE
#         + λ_sev * L_severity
#
# L_CE:       CrossEntropyLoss on binary logits (vuln/safe)
# L_CFA:      Cosine-margin contrastive on CFA (vuln, safe) embedding pairs.
#             CORRECT formula: relu(cosine_sim + margin) where margin=0.5
#             WRONG formula:   relu(cosine_sim - margin) ← DO NOT USE
#             R-02 mitigation: sign is critically verified in test_p4s4_losses.py
# L_CWE:     CrossEntropyLoss(label_smoothing=0.1) on CWE head — auxiliary
# L_severity: HuberLoss(delta=1.0) on severity head, -1 labels skipped (R-23)
#
# Research basis: VISION (Egea et al., AIES 2025) — paired GNN training with
# contrastive objective. BCE alone allows spurious correlations; contrastive
# loss forces structural separation between vuln and safe embeddings.

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# CWE label mapping (12 target CWEs)
# ─────────────────────────────────────────────────────────────────────────────

CWE_LABEL_MAP = {
    "CWE-89":  0,   # SQL Injection
    "CWE-78":  1,   # OS Command Injection
    "CWE-79":  2,   # Cross-Site Scripting
    "CWE-119": 3,   # Buffer Overflow (general)
    "CWE-120": 4,   # Buffer Copy No Bound Check
    "CWE-121": 5,   # Stack Buffer Overflow
    "CWE-122": 6,   # Heap Buffer Overflow
    "CWE-125": 7,   # Out-of-Bounds Read
    "CWE-134": 8,   # Uncontrolled Format String
    "CWE-190": 9,   # Integer Overflow
    "CWE-416": 10,  # Use After Free
    "CWE-476": 11,  # NULL Pointer Dereference
}

# Reverse map for eval.py
LABEL_CWE_MAP = {v: k for k, v in CWE_LABEL_MAP.items()}


class StreamGuardLoss(nn.Module):
    """
    Composite loss for CFA-paired training.

    L_total = λ_ce * L_CE + λ_cfa * L_CFA + 0.2 * L_CWE + λ_sev * L_severity

    forward(outputs_orig, outputs_cfa, labels, cwe_labels, severity_labels)
        → (total_loss: Tensor, loss_dict: dict)
    """

    def __init__(
        self,
        lambda_ce:  float = 1.0,
        lambda_cfa: float = 0.5,
        lambda_sev: float = 0.1,
        cfa_margin: float = 0.5,
        num_cwe: int      = 12,
        label_smoothing: float = 0.1,
    ):
        super().__init__()
        self.lambda_ce  = lambda_ce
        self.lambda_cfa = lambda_cfa
        self.lambda_sev = lambda_sev
        self.margin     = cfa_margin

        self.ce_loss  = nn.CrossEntropyLoss()
        self.cwe_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.sev_loss = nn.HuberLoss(delta=1.0)

    def forward(
        self,
        outputs_orig,
        outputs_cfa=None,
        labels=None,
        cwe_labels=None,
        severity_labels=None,
    ):
        """
        Compute composite loss. All components are optional except L_CE.

        Args:
            outputs_orig:    model output dict for original samples
            outputs_cfa:     model output dict for CFA counterparts (None if no pairs)
            labels:          (B,) int64 binary labels 0/1
            cwe_labels:      (B,) int64 CWE class index 0-11 (None or -1 = unknown)
            severity_labels: (B,) float severity 0-10 (-1 = unknown)

        Returns:
            (total_loss: Tensor, loss_dict: dict of component values)
        """
        losses = {}
        device = outputs_orig["logits"].device
        total  = torch.tensor(0.0, device=device, requires_grad=True)

        # ── L_CE: Binary classification ──────────────────────────────────
        if labels is not None:
            l_ce = self.ce_loss(outputs_orig["logits"], labels.to(device))
            losses["L_CE"] = l_ce.item()
            total = total + self.lambda_ce * l_ce

        # ── L_CWE: Multi-class CWE head (auxiliary, weight=0.2) ─────────
        if cwe_labels is not None:
            cwe_labels_dev = cwe_labels.to(device)
            valid = (cwe_labels_dev >= 0) & (cwe_labels_dev < 12)
            if valid.any():
                l_cwe = self.cwe_loss(
                    outputs_orig["cwe_logits"][valid],
                    cwe_labels_dev[valid],
                )
                losses["L_CWE"] = l_cwe.item()
                total = total + 0.2 * l_cwe

        # ── L_CFA: Cosine margin contrastive on (v, v') pairs ───────────
        # CRITICAL R-02: formula is relu(cosine_sim + margin), NOT minus
        # margin = +0.5 → penalizes when cosine_sim > -0.5
        # Goal: push vuln and CFA embeddings to OPPOSITE hemispheres
        if outputs_cfa is not None:
            emb_v  = outputs_orig["embedding"]
            emb_vp = outputs_cfa["embedding"]

            min_len = min(emb_v.size(0), emb_vp.size(0))
            if min_len > 0:
                emb_v  = emb_v[:min_len]
                emb_vp = emb_vp[:min_len]

                cosine_sim = F.cosine_similarity(emb_v, emb_vp, dim=-1)
                l_cfa = F.relu(cosine_sim + self.margin).mean()
                losses["L_CFA"] = l_cfa.item()
                total = total + self.lambda_cfa * l_cfa

                # Diagnostic: fraction of pairs already separated (sim < -margin)
                losses["frac_separated"] = (cosine_sim < -self.margin).float().mean().item()

        # ── L_severity: CVSS proxy regression (HuberLoss) ───────────────
        # R-23 mitigation: skip samples where severity_label == -1
        if severity_labels is not None:
            sev_dev = severity_labels.to(device).float()
            valid_mask = sev_dev >= 0
            if valid_mask.any():
                l_sev = self.sev_loss(
                    outputs_orig["severity_score"][valid_mask],
                    sev_dev[valid_mask],
                )
                losses["L_severity"] = l_sev.item()
                total = total + self.lambda_sev * l_sev

        losses["total"] = total.item()
        return total, losses
