"""Kullback-Leibler Semantic Alignment Loss (L_kl) for MFVLR.

PAPER_SPECIFIED:
- Equation (19): L_kl = (1/b) * sum_u [ delta(T_l^u)^T * log( delta(T_l^u) / delta(T_lpre^u) ) ]
- Temperature: tau_kl = 0.5
- Teacher distribution: P = softmax(T_l / 0.5)
- Student distribution: Q = softmax(T_lpre / 0.5)
- Direction: D_KL(P || Q) = sum_k P_k * (log(P_k) - log(Q_k)) >= 0
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class KLSemanticAlignmentLoss(nn.Module):
    """KL-divergence feature alignment loss with temperature tau=0.5 (Eq. 19)."""

    def __init__(
        self,
        temperature: float = 0.5,
        reduction: str = "batchmean",
    ):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction

    def forward(
        self,
        t_l: torch.Tensor,
        t_lpre: torch.Tensor,
    ) -> torch.Tensor:
        """Compute KL semantic alignment loss D_KL(P(T_l) || Q(T_lpre)).

        Args:
            t_l: Global language representation from LE [B, 512] (Target / Teacher)
            t_lpre: Predicted language representation from Adapter [B, 512] (Prediction / Student)

        Returns:
            Scalar non-negative KL loss.
        """
        # 1. Cast to float32 to prevent overflow in AMP FP16
        t_l_f32 = t_l.float()
        t_lpre_f32 = t_lpre.float()

        # 2. Scaled logits with tau = 0.5
        scaled_t_l = t_l_f32 / self.temperature
        scaled_t_lpre = t_lpre_f32 / self.temperature

        # 3. Probability distributions in log-space for numerical stability
        log_P = F.log_softmax(scaled_t_l, dim=-1)
        log_Q = F.log_softmax(scaled_t_lpre, dim=-1)

        # 4. D_KL(P || Q) with log_target=True
        if self.reduction in ("batchmean", "mean", "sum"):
            loss = F.kl_div(
                input=log_Q,
                target=log_P,
                log_target=True,
                reduction=self.reduction,
            )
        else:
            loss = F.kl_div(
                input=log_Q,
                target=log_P,
                log_target=True,
                reduction="none",
            ).sum(dim=-1)

        return loss
