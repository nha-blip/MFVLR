"""Appearance Decoder (AD) for general image appearance reconstruction.

PAPER_SPECIFIED:
- Input: local appearance forgery features I_loc in R^(B x 1024 x 14 x 14)
- Output: predicted/reconstructed appearance image I_pre in R^(B x 3 x 224 x 224)
- Composed of a U-Net decoder along with an appearance reconstruction module with a convolutional layer (Section III-C).
- U-Net decoder shares the SAME network and weights as the Mask Decoder (MD) U-Net decoder.

ASSUMPTION_FROM_PAPER_GAP:
- Appearance head uses a 3x3 convolution (64 -> 3 channels) followed by Sigmoid activation to match the [0, 1] normalized input image range.
"""

from typing import List, Optional
import torch
import torch.nn as nn

from models.vision.unet_decoder import UNetDecoderTrunk


class AppearanceHead(nn.Module):
    """Appearance reconstruction convolutional head."""

    def __init__(
        self,
        in_channels: int = 64,
        out_channels: int = 3,
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        # ASSUMPTION_FROM_PAPER_GAP: Sigmoid activation to constrain output to [0, 1]
        self.activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project decoded feature map [B, 64, 224, 224] -> I_pre [B, 3, 224, 224]."""
        return self.activation(self.conv(x))


class AppearanceDecoder(nn.Module):
    """Appearance Decoder (AD) combining shared U-Net decoder trunk and appearance head."""

    def __init__(
        self,
        decoder_trunk: UNetDecoderTrunk,
        out_channels: int = 3,
    ):
        super().__init__()
        # Direct reference to shared decoder trunk
        self.decoder_trunk = decoder_trunk
        self.head = AppearanceHead(
            in_channels=decoder_trunk.out_channels,
            out_channels=out_channels,
        )

    def forward(
        self,
        i_loc: torch.Tensor,
        skips: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Decode I_loc to reconstructed appearance image I_pre.

        Args:
            i_loc: Local feature [B, 1024, 14, 14]
            skips: Optional multi-scale encoder skip list

        Returns:
            I_pre: Reconstructed appearance image [B, 3, 224, 224] in range [0, 1]
        """
        features = self.decoder_trunk(i_loc, skips=skips)  # [B, 64, 224, 224]
        i_pre = self.head(features)                         # [B, 3, 224, 224]
        return i_pre
