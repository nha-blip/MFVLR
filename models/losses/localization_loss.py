"""Forgery Localization Loss (L_fl) for MFVLR.

PAPER_SPECIFIED:
- Equation (18): L_fl = (1/b) * sum_u [ -(M^u)^T * log(M_pre^u) ]
- Pixel-wise 2-class cross-entropy loss between predicted mask logits M_pre in R^(B x 2 x 224 x 224)
  and ground-truth binary mask M in {0, 1}^(B x 224 x 224).
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class ForgeryLocalizationLoss(nn.Module):
    """Pixel-level Forgery Localization Cross-Entropy Loss (Eq. 18)."""

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        m_pre: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute pixel-wise localization loss.

        Args:
            m_pre: Predicted mask raw logits [B, 2, 224, 224]
            target_mask: Ground-truth binary mask [B, 224, 224] or [B, 1, 224, 224] in {0, 1}

        Returns:
            Scalar localization loss.
        """
        if target_mask.ndim == 4 and target_mask.size(1) == 1:
            target_mask = target_mask.squeeze(1)  # [B, 224, 224]

        return F.cross_entropy(
            input=m_pre,
            target=target_mask.long(),
            reduction=self.reduction,
        )
