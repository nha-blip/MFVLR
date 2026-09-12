"""Appearance Reconstruction Loss (L_ar) for MFVLR.

PAPER_SPECIFIED:
- Equation (17): L_ar = (1/b) * sum_u (I^u - I_pre^u)^2
- Mean squared error between input appearance image I and reconstructed image I_pre.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class AppearanceReconstructionLoss(nn.Module):
    """Appearance Reconstruction Mean Squared Error Loss (Eq. 17)."""

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        i_pre: torch.Tensor,
        image: torch.Tensor,
    ) -> torch.Tensor:
        """Compute appearance reconstruction loss.

        Args:
            i_pre: Reconstructed appearance image [B, 3, 224, 224] in [0, 1]
            image: Original input image [B, 3, 224, 224] in [0, 1]

        Returns:
            Scalar reconstruction MSE loss.
        """
        return F.mse_loss(
            input=i_pre,
            target=image,
            reduction=self.reduction,
        )
