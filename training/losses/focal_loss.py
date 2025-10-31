"""
Focal Loss for handling hard negatives and class imbalance.

Reference:
Lin et al., "Focal Loss for Dense Object Detection" (2017)
https://arxiv.org/abs/1708.02002
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss: FL(p) = -(1-p)^gamma * log(p)

    Focuses on hard examples by down-weighting easy ones.
    Useful when:
    - Hard negatives dominate
    - Standard CrossEntropy causes collapse
    - Class imbalance is severe

    Args:
        alpha: Class weights (tensor of shape [num_classes])
        gamma: Focusing parameter (float, typically 1.0-2.5)
               Higher gamma = more focus on hard examples
        reduction: 'mean' | 'sum' | 'none'

    Example:
        >>> alpha = torch.tensor([1.0, 1.5])  # Boost minority class
        >>> criterion = FocalLoss(alpha=alpha, gamma=2.0)
        >>> logits = model(inputs)
        >>> loss = criterion(logits, targets)
    """

    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha  # class weights (tensor)
        self.gamma = gamma  # focusing parameter (1.0-2.5)
        self.reduction = reduction

        if gamma < 0:
            raise ValueError(f"Gamma must be non-negative, got {gamma}")

    def forward(self, inputs, targets):
        """
        Args:
            inputs: Logits tensor of shape [batch_size, num_classes]
            targets: Target labels of shape [batch_size] (class indices)

        Returns:
            Focal loss (scalar if reduction='mean')
        """
        # Standard cross-entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        # p_t: probability of true class
        p_t = torch.exp(-ce_loss)

        # Focal term: (1 - p_t)^gamma
        # When p_t is high (easy example): (1 - p_t)^gamma ≈ 0 → loss down-weighted
        # When p_t is low (hard example): (1 - p_t)^gamma ≈ 1 → loss unchanged
        focal_weight = (1 - p_t) ** self.gamma
        focal_loss = focal_weight * ce_loss

        # Apply class weights (alpha)
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        # Reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:  # 'none'
            return focal_loss

    def __repr__(self):
        return (f"FocalLoss(alpha={'tensor' if self.alpha is not None else None}, "
                f"gamma={self.gamma}, reduction='{self.reduction}')")
