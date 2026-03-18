# training/scripts/model/losses.py
#
# StreamGuardLoss: Composite multi-task loss for CFA-paired training.
#
# L_total = λ_ce * L_CE  +  λ_cfa * L_CFA  +  λ_sev * L_severity
#
#   L_CE:       CrossEntropyLoss on binary_logits (vuln/safe)
#   L_CFA:      Cosine-margin contrastive on CFA (vuln, safe) embedding pairs.
#               sim = cosine_similarity(vuln_emb, safe_emb)
#               L_CFA = mean(relu(sim + margin))   where margin=0.5
#               Pushes paired embeddings to OPPOSITE sides of the embedding space.
#               When no CFA pairs exist in batch (or use_cfa=False): L_CFA = 0.
#   L_severity: MSELoss on severity head output vs severity_score label.
#               Samples with severity < 0 (sentinel for "unknown") are masked out.
#
# Research basis: VISION (Egea et al., AIES 2025) — paired GNN training with
# contrastive objective. BCE alone allows spurious correlations; contrastive
# loss forces structural separation between vuln and safe embeddings.

import torch
import torch.nn as nn
import torch.nn.functional as F


class StreamGuardLoss(nn.Module):
    """
    Composite loss for CFA-paired training.

    L_total = λ_ce * L_CE + λ_cfa * L_CFA + λ_sev * L_severity

    forward(outputs, batch, use_cfa=True) -> (total_loss, loss_dict)

    loss_dict always contains {L_total, L_CE, L_CFA, L_severity} so
    MLflow can log each component separately, even when a component is 0.
    """

    def __init__(
        self,
        lambda_ce:  float = 1.0,
        lambda_cfa: float = 0.5,
        lambda_sev: float = 0.1,
        cfa_margin: float = 0.5,
    ):
        super().__init__()
        self.lambda_ce  = lambda_ce
        self.lambda_cfa = lambda_cfa
        self.lambda_sev = lambda_sev
        self.margin     = cfa_margin

        self.ce_loss  = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()

    def forward(
        self,
        outputs: dict,
        batch,
        use_cfa: bool = True,
    ):
        """
        Args:
            outputs: dict from StreamGuardModel.forward(). Expected keys:
                binary_logits  (B, 2)
                severity       (B,)
                embedding      (B, 128)  — used for CFA contrastive loss
            batch: PyG Batch object. Expected attributes:
                batch.y            (B,) int   — binary labels 0/1
                batch.pair_id      list[str]  — CFA pair identifiers ('' = no pair)
                batch.severity_score (B,) float — CVSS proxy (-1 = unknown), optional
            use_cfa: if False, L_CFA is not computed (set to 0).

        Returns:
            total_loss: scalar Tensor (differentiable)
            loss_dict:  {L_total, L_CE, L_CFA, L_severity} — all float values
                        for MLflow logging.
        """
        device = outputs["binary_logits"].device
        total  = torch.tensor(0.0, device=device)

        # ── L_CE: binary classification ──────────────────────────────────
        labels = batch.y.to(device)
        l_ce   = self.ce_loss(outputs["binary_logits"], labels)
        total  = total + self.lambda_ce * l_ce

        # ── L_CFA: cosine-margin contrastive on CFA pairs ───────────────
        l_cfa_val = 0.0
        if use_cfa:
            l_cfa = self._compute_cfa_loss(outputs, batch, device)
            if l_cfa is not None:
                total     = total + self.lambda_cfa * l_cfa
                l_cfa_val = l_cfa.item()

        # ── L_severity: MSE regression ───────────────────────────────────
        # IMPORTANT (train.py must handle): SARD samples store severity_score=0.0
        # in HDF5. Since 0.0 >= 0 passes valid_mask, the severity head would train
        # 20 epochs learning to predict 0.0 for everything. In train.py, set
        # severity_score = -1 for SARD data so this mask filters it out:
        #   sev_labels[sev_labels == 0.0] = -1.0
        # Or set lambda_sev=0.0 for M1 (no real CVSS scores available).
        l_sev_val = 0.0
        if hasattr(batch, "severity_score") and batch.severity_score is not None:
            sev_labels = batch.severity_score.to(device).float()
            valid_mask = sev_labels >= 0  # -1 sentinel = unknown
            if valid_mask.any():
                l_sev     = self.mse_loss(
                    outputs["severity"][valid_mask],
                    sev_labels[valid_mask],
                )
                total     = total + self.lambda_sev * l_sev
                l_sev_val = l_sev.item()

        loss_dict = {
            "L_total":    total.item(),
            "L_CE":       l_ce.item(),
            "L_CFA":      l_cfa_val,
            "L_severity": l_sev_val,
        }
        return total, loss_dict

    # ──────────────────────────────────────────────────────────────────────
    # Internal: CFA contrastive loss
    # ──────────────────────────────────────────────────────────────────────

    def _compute_cfa_loss(self, outputs, batch, device):
        """
        Extract CFA pairs from batch via pair_id, compute contrastive loss.

        For each pair_id that appears exactly twice:
          - vuln member  (label=1) embedding
          - safe member  (label=0) embedding
          sim = cosine_similarity(vuln_emb, safe_emb)
          per-pair loss = relu(sim + margin)
        L_CFA = mean over all pairs.

        Returns None if no valid pairs found in this batch.
        """
        # pair_id can be a list (from Batch collation) or a tensor attribute
        pair_ids = _get_pair_ids(batch)
        if pair_ids is None:
            return None

        labels    = batch.y.to(device)
        embedding = outputs["embedding"]  # (B, 128)

        # Group indices by pair_id, keep only groups of exactly 2
        from collections import defaultdict
        groups = defaultdict(list)
        for idx, pid in enumerate(pair_ids):
            if pid and pid not in ("", "None", "null", "0"):
                groups[pid].append(idx)

        vuln_embs = []
        safe_embs = []
        for pid, indices in groups.items():
            if len(indices) != 2:
                continue
            i, j = indices
            li, lj = labels[i].item(), labels[j].item()
            # Identify vuln (1) and safe (0) member
            if li == 1 and lj == 0:
                vuln_embs.append(embedding[i])
                safe_embs.append(embedding[j])
            elif li == 0 and lj == 1:
                vuln_embs.append(embedding[j])
                safe_embs.append(embedding[i])
            else:
                # Both same label — malformed pair, skip silently
                continue

        if not vuln_embs:
            return None

        vuln_stack = torch.stack(vuln_embs)  # (P, 128)
        safe_stack = torch.stack(safe_embs)  # (P, 128)

        sim   = F.cosine_similarity(vuln_stack, safe_stack, dim=-1)  # (P,)
        l_cfa = F.relu(sim + self.margin).mean()
        return l_cfa


def _get_pair_ids(batch) -> list | None:
    """
    Extract pair_id list from a PyG Batch.

    PyG Batch collation handles string attributes by concatenating them
    into a list. This helper normalises the various ways pair_id may
    appear (list, tensor, or missing).
    """
    if not hasattr(batch, "pair_id"):
        return None

    pair_ids = batch.pair_id
    if pair_ids is None:
        return None

    # PyG concatenates string attrs into a flat list
    if isinstance(pair_ids, (list, tuple)):
        return [str(p) for p in pair_ids]

    # Tensor of ints/strings — convert
    if isinstance(pair_ids, torch.Tensor):
        return [str(p.item()) for p in pair_ids]

    return None
