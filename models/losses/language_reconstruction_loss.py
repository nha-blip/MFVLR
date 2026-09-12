"""Language Reconstruction Loss (L_lr) for MFVLR.

PAPER_SPECIFIED:
- Equation (24): T_pre = T_rec^d @ W_voc^T in R^(n x s)
- Equation (25): L_lr = (1/b) * sum_u sum_x [ -(T_gt^(u,x))^T * log(T_pre^(u,x)) ]
- Token-level categorical cross entropy between predicted vocabulary logits and prompt token IDs.
"""

from typing import Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class LanguageReconstructionLoss(nn.Module):
    """Language Reconstruction Loss over vocabulary tokens (Eq. 24-25)."""

    def __init__(
        self,
        ignore_index: int = -100,
        reduction: str = "mean",
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(
        self,
        t_pre: torch.Tensor,
        target_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Compute language reconstruction loss.

        Args:
            t_pre: Reconstructed vocabulary logits [B, n, s] (e.g. [B, 308, 49408])
            target_token_ids: Ground truth prompt token IDs [B, n]

        Returns:
            Scalar reconstruction loss.
        """
        B, n, s = t_pre.shape

        # Flatten without creating redundant tensor clones
        flat_logits = t_pre.view(-1, s)
        flat_targets = target_token_ids.view(-1).long()

        return F.cross_entropy(
            input=flat_logits,
            target=flat_targets,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
        )
