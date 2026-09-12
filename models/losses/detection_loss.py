"""Forgery Detection Loss (L_fd) for MFVLR.

PAPER_SPECIFIED:
- Equation (26): L_fd = (1/b) * sum_u [ -(y^u)^T * log(y_pre^u) ]
- 2-class classification (Real vs Fake).
- Raw logits input y_pre in R^(B x 2), target y in {0, 1}^B.
"""

from typing import Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class ForgeryDetectionLoss(nn.Module):
    """Forgery Detection Cross-Entropy Loss (Eq. 26)."""

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        y_pre: torch.Tensor,
        y_target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute forgery detection loss.

        Args:
            y_pre: Predicted raw detection logits [B, 2]
            y_target: Ground truth label class indices [B] or one-hot [B, 2]

        Returns:
            Scalar loss tensor (or unreduced tensor if reduction="none").
        """
        # Support one-hot target input [B, 2]
        if y_target.ndim == 2 and y_target.shape == y_pre.shape:
            y_target = y_target.argmax(dim=-1)

        return F.cross_entropy(
            input=y_pre,
            target=y_target.long(),
            reduction=self.reduction,
        )
