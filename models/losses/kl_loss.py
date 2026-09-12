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
        # 1. Scaled logits with tau = 0.5
        scaled_t_l = t_l / self.temperature
        scaled_t_lpre = t_lpre / self.temperature

        # 2. Probability distributions
        # Teacher: P = softmax(T_l / 0.5)
        P = F.softmax(scaled_t_l, dim=-1)
        log_P = F.log_softmax(scaled_t_l, dim=-1)

        # Student: log_Q = log_softmax(T_lpre / 0.5)
        log_Q = F.log_softmax(scaled_t_lpre, dim=-1)

        # 3. D_KL(P || Q) = sum_k P_k * (log_P_k - log_Q_k)
        if self.reduction == "batchmean":
            # PyTorch F.kl_div(input=log_Q, target=P) evaluates sum P * (log P - log Q) / B
            loss = F.kl_div(log_Q, P, reduction="batchmean")
        elif self.reduction == "mean":
            loss = (P * (log_P - log_Q)).sum(dim=-1).mean()
        elif self.reduction == "sum":
            loss = (P * (log_P - log_Q)).sum()
        else:
            loss = (P * (log_P - log_Q)).sum(dim=-1)

        return loss
