"""Feature Adapter Module for visual-to-language prediction (MFVLR).

PAPER_SPECIFIED:
- Adapter projects the fused visual feature I_v in R^(B x 512) into a predicted language representation:
    T_lpre in R^(B x 512)
- Used for semantic alignment with global language feature T_l via KL divergence (Eq. 19).

ASSUMPTION_FROM_PAPER_GAP:
- Implemented as a clean Linear projection (Linear(512, 512)) with configurable hidden layers if desired.
"""

from typing import Optional
import torch
import torch.nn as nn


class Adapter(nn.Module):
    """Feature Adapter projecting visual features to language embedding space."""

    def __init__(
        self,
        in_dim: int = 512,
        out_dim: int = 512,
        bias: bool = True,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.proj = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, i_v: torch.Tensor) -> torch.Tensor:
        """Project visual feature to language feature space.

        Args:
            i_v: Fused visual feature [B, 512] or [B, 1, 512]

        Returns:
            t_lpre: Predicted language feature [B, 512]
        """
        if i_v.ndim == 3 and i_v.size(1) == 1:
            i_v = i_v.squeeze(1)  # [B, 512]

        return self.proj(i_v)
