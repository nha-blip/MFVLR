"""Detection Classification Head for Real/Fake image forgery classification.

PAPER_SPECIFIED:
- Classifies fused visual representation I_v into real vs fake.
- Output: y_pre in R^(B x 2) raw classification logits.
- No softmax applied inside the head.

ASSUMPTION_FROM_PAPER_GAP:
- Implemented as a clean Linear projection (Linear(512, 2)).
"""

from typing import Optional
import torch
import torch.nn as nn


class DetectionHead(nn.Module):
    """Detection Head producing 2-class logits from visual representation I_v."""

    def __init__(
        self,
        in_dim: int = 512,
        num_classes: int = 2,
        bias: bool = True,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.fc = nn.Linear(in_dim, num_classes, bias=bias)

    def forward(self, i_v: torch.Tensor) -> torch.Tensor:
        """Compute detection logits from visual feature.

        Args:
            i_v: Visual feature [B, 512] or [B, 1, 512]

        Returns:
            y_pre: Raw detection logits [B, 2]
        """
        if i_v.ndim == 3 and i_v.size(1) == 1:
            i_v = i_v.squeeze(1)  # [B, 512]

        return self.fc(i_v)
