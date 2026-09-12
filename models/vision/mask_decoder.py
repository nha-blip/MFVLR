"""Mask Decoder (MD) for pixel-level face manipulation localization.

PAPER_SPECIFIED:
- Input: local appearance forgery features I_loc in R^(B x 1024 x 14 x 14)
- Output: predicted mask M_pre in R^(B x f x 224 x 224) where f = 2 (two categories)
- Composed of a U-Net decoder along with a manipulation localization module with a convolutional layer (Section III-C).
- U-Net decoder adopts the exact SAME network and weights as those in AD.

ASSUMPTION_FROM_PAPER_GAP:
- Mask head uses a 3x3 convolution (64 -> 2 channels).
- Returns RAW logits (no softmax/sigmoid inside decoder) for numerical stability with cross-entropy loss.
"""

from typing import List, Optional
import torch
import torch.nn as nn

from models.vision.unet_decoder import UNetDecoderTrunk


class MaskHead(nn.Module):
    """Manipulation localization convolutional head."""

    def __init__(
        self,
        in_channels: int = 64,
        num_classes: int = 2,
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project decoded feature map [B, 64, 224, 224] -> M_pre [B, 2, 224, 224] raw logits."""
        return self.conv(x)


class MaskDecoder(nn.Module):
    """Mask Decoder (MD) combining shared U-Net decoder trunk and localization head."""

    def __init__(
        self,
        decoder_trunk: UNetDecoderTrunk,
        num_classes: int = 2,
    ):
        super().__init__()
        # Direct reference to shared decoder trunk
        self.decoder_trunk = decoder_trunk
        self.head = MaskHead(
            in_channels=decoder_trunk.out_channels,
            num_classes=num_classes,
        )

    def forward(
        self,
        i_loc: torch.Tensor,
        skips: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Decode I_loc to predicted mask logits M_pre.

        Args:
            i_loc: Local feature [B, 1024, 14, 14]
            skips: Optional multi-scale encoder skip list

        Returns:
            M_pre: Predicted mask logits [B, 2, 224, 224] (raw unnormalized logits)
        """
        features = self.decoder_trunk(i_loc, skips=skips)  # [B, 64, 224, 224]
        m_pre = self.head(features)                         # [B, 2, 224, 224]
        return m_pre
