"""Cross-Modal Contrastive Loss (L_cmc) for MFVLR.

PAPER_SPECIFIED:
- Equations (20)-(23):
    sim(a, b) = a @ b^T (strictly unnormalized dot product, NO L2 normalization)
    S = I_v @ T_l^T in R^(B x B)
    L_v2l = (1/b) * sum_u [ -log(S_v2l^u(I_v, T_l)) ]
    L_l2v = (1/b) * sum_u [ -log(S_l2v^u(T_l, I_v)) ]
    L_cmc = (L_v2l + L_l2v) / 2
- Trainable temperature parameter tau initialized to 0.07.
"""

import math
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalContrastiveLoss(nn.Module):
    """Bidirectional Cross-Modal Contrastive Loss with unnormalized dot-product similarity (Eq. 20-23)."""

    def __init__(
        self,
        initial_temperature: float = 0.07,
        trainable_temperature: bool = True,
    ):
        super().__init__()
        self.initial_temperature = initial_temperature
        self.trainable_temperature = trainable_temperature

        # Parameterize tau = exp(log_tau) to guarantee strict positivity during training
        log_tau_init = math.log(initial_temperature)
        if trainable_temperature:
            self.log_tau = nn.Parameter(torch.tensor(log_tau_init, dtype=torch.float32))
        else:
            self.register_buffer("log_tau", torch.tensor(log_tau_init, dtype=torch.float32))

    @property
    def tau(self) -> torch.Tensor:
        """Effective positive temperature parameter."""
        return self.log_tau.exp()

    def forward(
        self,
        i_v: torch.Tensor,
        t_l: torch.Tensor,
        return_similarity: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Compute bidirectional cross-modal contrastive loss.

        Args:
            i_v: Global visual feature [B, 512] or [B, 1, 512]
            t_l: Global language feature [B, 512]
            return_similarity: Whether to return similarity matrix S [B, B]

        Returns:
            Scalar CMC loss (and optionally similarity matrix S).
        """
        if i_v.ndim == 3 and i_v.size(1) == 1:
            i_v = i_v.squeeze(1)  # [B, 512]

        B = i_v.size(0)
        assert t_l.size(0) == B, f"Batch size mismatch: I_v has {B}, T_l has {t_l.size(0)}"

        # 1. PAPER_SPECIFIED: Unnormalized dot product similarity matrix S [B, B]
        # sim(a, b) = a @ b^T (NO L2 normalization, NO cosine similarity)
        S = torch.matmul(i_v, t_l.transpose(-2, -1))  # [B, B]

        # 2. Temperature scaling S / tau
        scaled_S = S / self.tau

        # 3. Ground truth matching labels (diagonal pairing: (I_v[u], T_l[u]))
        labels = torch.arange(B, device=i_v.device, dtype=torch.long)

        # 4. Bidirectional contrastive objectives (Eq. 22 & 23)
        # Vision-to-Language: S_v2l
        loss_v2l = F.cross_entropy(scaled_S, labels)
        # Language-to-Vision: S_l2v (transposed similarity)
        loss_l2v = F.cross_entropy(scaled_S.t(), labels)

        # 5. Combined symmetric contrastive loss (Eq. 23 context)
        loss_cmc = (loss_v2l + loss_l2v) / 2.0

        if return_similarity:
            return loss_cmc, S
        return loss_cmc
